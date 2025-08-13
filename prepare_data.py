import pickle
from os.path import join
import os
import random
import json
from tqdm import tqdm
import datetime
from config import Config, root_dir
import ml_collections
import sklearn.model_selection
import hashlib

seed = Config.global_seed
random.seed(seed)


def load_raw_data():
    """Load raw data from JSON file."""
    with open(Config.data.processed_PDBmmcif_path, 'r') as f:
        return json.load(f)


def filter_assembly(assembly: dict, config: ml_collections.ConfigDict, monomer_rate: float = 0.6) -> bool:
    """
    Filter assembly based on various criteria.
    
    Args:
        assembly (dict): Assembly dictionary
        config (ml_collections.ConfigDict): Configuration object
        monomer_rate (float): Rate for filtering monomers
        
    Returns:
        bool: True if assembly passes all filters
    """
    # first get entities count
    entitiy_count = assembly['entity_count']
    pdb_id = assembly['unique_id']
    if not entitiy_count:
        return False
    
    # case homomeric
    if len(entitiy_count) == 1:
        # case monomeric
        if list(entitiy_count.values())[0] == 1:
            if monomer_rate == 0.0:
                return False
            if random.random() < 1 - monomer_rate:
                return False
        
        # check sequence length
        seq_len = len(assembly[list(entitiy_count.keys())[0]]['sequence'])
        if seq_len < config.data.minimum_seq_len:
            return False
            
        # check if more than 60% of sequence is the unknown amino acid
        if assembly[list(entitiy_count.keys())[0]]['sequence'].count('X') > seq_len * 0.6:
            # print(f'{pdb_id} has more than 60% unknown amino acid')
            return False
            
    # case heteromeric
    else:
        # case too many entities
        if len(entitiy_count) > config.model.num_subunits:
            return False
            
        for k, count in entitiy_count.items():
            # check sequence length
            seq_len = len(assembly[k]['sequence'])
            if seq_len < config.data.minimum_seq_len:
                return False
                
            # check if more than 60% of sequence is the unknown amino acid
            if assembly[k]['sequence'].count('X') > seq_len * 0.6:
                # print(f'{pdb_id} has more than 60% unknown amino acid')
                return False
    
    return True


def is_homomeric(assembly: dict) -> bool:
    """
    Check if the assembly is homomeric.

    Args:
        assembly (dict): Assembly dictionary

    Returns:
        bool: True if the assembly is homomeric.
    """
    return len(assembly['entity_count']) == 1


def is_monomeric(assembly: dict) -> bool:
    """
    Check if the assembly is monomeric.
    
    Args:
        assembly (dict): Assembly dictionary
        
    Returns:
        bool: True if the assembly is monomeric
    """
    return len(assembly['entity_count']) == 1 and list(assembly['entity_count'].values())[0] == 1


def remove_non_protein_assembly(asse_dict: dict, strict: bool = False) -> dict:
    """
    Remove non-protein entities from assembly.
    
    Args:
        asse_dict (dict): Assembly dictionary
        strict (bool): If True, all entities must be proteins
        
    Returns:
        dict: Filtered assembly dictionary
    """
    count_dict = asse_dict['entity_count']
    if not count_dict:
        return {}
        
    is_proteins = []
    for k in count_dict.keys():
        seq_info = asse_dict[k]
        if seq_info['type'] == 'polypeptide(L)':  # or seq_info['type'] == 'polypeptide(D)':
            is_proteins.append(k)
        elif strict:
            return {}
            
    # rm non-protein
    if len(is_proteins) == 0:
        return {}
        
    asse_dict['entity_count'] = {k: count_dict[k] for k in is_proteins}
    not_proteins = set(count_dict.keys()) - set(is_proteins)
    
    # rm non-protein
    for k in not_proteins:
        del asse_dict[k]
        
    return asse_dict


def has_unp_ids(assembly: dict, strict: bool = False) -> bool:
    """
    Check if the assembly has UNP IDs.

    Args:
        assembly (dict): Assembly dictionary
        strict (bool, optional): If True, all entities must have UNP IDs. Defaults to False.

    Returns:
        bool: True if the assembly has UNP IDs.
    """
    db_name_list = []
    for entity_id in assembly['entity_count']:
        db_name = assembly[entity_id]['db_name']
        db_code = assembly[entity_id]['pdbx_db_accession']
        db_name_list.append(db_name)
        
    # check if any is None
    if any([x is None for x in db_name_list]):
        return False
        
    # check exclude
    if strict:
        return all([x == 'UNP' for x in db_name_list])
    else:
        return any([x == 'UNP' for x in db_name_list])


def is_author_defined(assembly: dict) -> bool:
    """
    Check if the author is defined in the assembly.

    Args:
        assembly (dict): Assembly dictionary

    Returns:
        bool: True if the author is defined.
    """
    return 'author' in assembly['details']


def is_exp_defined(assembly: dict) -> bool:
    """
    Check if the experimental support is defined in the assembly.

    Args:
        assembly (dict): Assembly dictionary

    Returns:
        bool: Experimental support is defined.
    """
    exp = assembly['experimental_support']
    if exp == 'none':
        return False
    return exp


def is_after_date(assembly: dict, cut_off_date: str) -> bool:
    """
    Check if the release date is after the cut off date.

    Args:
        assembly (dict): Assembly dictionary
        cut_off_date (str): Cut off date

    Returns:
        bool: True if the release date is after the cut off date.
    """
    release_date = datetime.datetime.strptime(assembly['release_date'], '%Y-%m-%d')
    cut_off_date = datetime.datetime.strptime(cut_off_date, '%Y-%m-%d')
    return release_date > cut_off_date


def apply_first_filter(raw_data: dict) -> dict:
    """
    Apply first round of filtering to raw data.
    
    Args:
        raw_data (dict): Raw data dictionary
        
    Returns:
        dict: Filtered data after first round
    """
    first_filter_data = dict()
    for pdb_id, pdb_dict in raw_data.items():
        # skip no protein
        if not remove_non_protein_assembly(pdb_dict, strict=True):
            continue
            
        if not is_author_defined(pdb_dict):
            continue
            
        if not filter_assembly(pdb_dict, Config, monomer_rate=1.0):
            continue
            
        first_filter_data[pdb_id] = pdb_dict
    
    return first_filter_data


def create_count_mappings(first_filter_data: dict) -> tuple[dict, dict]:
    """
    Create count to label mappings based on entity count frequency.
    
    Args:
        first_filter_data (dict): First filtered data
        
    Returns:
        tuple[dict, dict]: count2label and label2idx mappings
    """
    # check the entity count frequency
    entity_count_freq = dict()
    for pdb_id, pdb_dict in first_filter_data.items():
        entity_count = pdb_dict['entity_count']
        for count in entity_count.values():
            if count not in entity_count_freq:
                entity_count_freq[count] = 0
            entity_count_freq[count] += 1
    
    # for entity count with less than minimum_sample_count, count as -1
    count2label = {0: 0}
    for count, freq in entity_count_freq.items():
        count = int(count)
        if freq < Config.data.minimum_sample_count:
            count2label[count] = -1
        else:
            count2label[count] = count
    
    # sort the count2label
    count2label = dict(sorted(count2label.items(), key=lambda item: item[0]))
    
    # count 2 idx, this would be the label for the count, just check values
    values_set = set(count2label.values())
    label2idx = {v: k for k, v in enumerate(values_set)}
    
    return count2label, label2idx


def save_count_mappings(count2label: dict, label2idx: dict) -> None:
    """
    Save count to label mappings to files.
    
    Args:
        count2label (dict): Count to label mapping
        label2idx (dict): Label to index mapping
    """
    with open(Config.model.count2label, 'w') as f:
        json.dump(count2label, f)
        
    with open(Config.model.label2idx, 'w') as f:
        json.dump(label2idx, f)


def filter_non_monomeric_data(first_filter_data: dict) -> dict:
    """
    Filter out monomeric data from first filtered data.
    
    Args:
        first_filter_data (dict): First filtered data
        
    Returns:
        dict: Data without monomers
    """
    filter_data_noMonomer = {}
    for pdb_id, pdb_dict in first_filter_data.items():
        if not filter_assembly(pdb_dict, Config, monomer_rate=1.0):
            continue
            
        if is_monomeric(pdb_dict):
            continue
            
        filter_data_noMonomer[pdb_id] = pdb_dict
    
    print(f'Number of assemblies in filter_data_noMonomer: {len(filter_data_noMonomer)}')
    return filter_data_noMonomer


def create_sequence_stoichiometry_key(pdb_dict: dict) -> tuple:
    """
    Create a unique key for sequence and stoichiometry combination.
    
    Args:
        pdb_dict (dict): PDB dictionary
        
    Returns:
        tuple: Unique key for sequence and stoichiometry
    """
    # Extract sequences and stoichiometries 
    seqs_sto_pairs = []
    for entity_id, count in pdb_dict['entity_count'].items():
        sequence = pdb_dict[entity_id]['sequence']
        sto = count
        seqs_sto_pairs.append((sequence, sto))
    
    # Sort by sequence, then by stoichiometry for consistent ordering
    seqs_sto_pairs.sort(key=lambda x: (x[0], x[1]))
    return tuple(seqs_sto_pairs)


def split_data_by_date(filter_data_noMonomer: dict) -> tuple[dict, dict, dict]:
    """
    Split data into train, validation, and test sets based on dates.
    
    Args:
        filter_data_noMonomer (dict): Filtered data without monomers
        
    Returns:
        tuple[dict, dict, dict]: train_data, val_data_raw, test_data_raw
    """
    train_data = dict()
    val_data_raw = dict()
    test_data_raw = dict()

    # sometimes people submit the same structure over multiple entries, need to remove redundancy in the test and valid data
    present_seq_sto = dict()
    
    for pdb_id, pdb_dict in filter_data_noMonomer.items():
        seqs_sto_pairs = create_sequence_stoichiometry_key(pdb_dict)
        is_duplicate = False
        
        # Check if this exact sequence+stoichiometry combination has been seen before
        if seqs_sto_pairs in present_seq_sto:
            is_duplicate = True
            present_seq_sto[seqs_sto_pairs].append(pdb_id)
        else:
            # Record this unique sequence+stoichiometry combination
            present_seq_sto[seqs_sto_pairs] = present_seq_sto.get(seqs_sto_pairs, []) + [pdb_id]
        
        # Now process only non-duplicate entries
        if not is_after_date(pdb_dict, Config.data.cut_off_date):
            # train data
            train_data[pdb_id] = pdb_dict
        elif is_after_date(pdb_dict, Config.data.test_cut_off_date):
            # possible test data
            if is_duplicate:
                continue
                
            if is_exp_defined(pdb_dict):
                test_data_raw[pdb_id] = pdb_dict
        elif is_exp_defined(pdb_dict):
            if is_duplicate:
                continue
                
            val_data_raw[pdb_id] = pdb_dict
        else:
            continue
    
    # further filter the test data and valid data
    train_ids = set(train_data.keys())
    val_data_raw = {k: v for k, v in val_data_raw.items() if k not in train_ids}
    test_data_raw = {k: v for k, v in test_data_raw.items() if k not in train_ids}
    
    return train_data, val_data_raw, test_data_raw


def create_cross_validation_folds(train_data: dict) -> list:
    """
    Create cross-validation folds stratified by stoichiometry.
    
    Args:
        train_data (dict): Training data
        
    Returns:
        list: List of fold data dictionaries
    """
    # stratified by the stoichiometry
    id_list = []
    sto_list = []
    sto_counter = dict()
    
    for pdb_id, pdb_dict in train_data.items():
        sto = tuple(sorted([v for v in pdb_dict['entity_count'].values()], reverse=True))
        sto = str(sto)
        sto_counter[sto] = sto_counter.get(sto, 0) + 1
        id_list.append(pdb_id)
        sto_list.append(sto)
    
    # convert less than minimum_sample_count to other
    for i, sto in enumerate(sto_list):
        if sto_counter[sto] < Config.data.minimum_sto_count:
            sto_list[i] = 'other'
    
    # create 5 fold
    kf = sklearn.model_selection.StratifiedKFold(n_splits=Config.data.num_folds, shuffle=True, random_state=seed)
    fold_data = []
    for train_index, test_index in kf.split(id_list, sto_list):
        fold_data.append({
            'train': [id_list[i] for i in train_index],
            'valid': [id_list[i] for i in test_index]
        })
    
    return fold_data


def save_dataset(train_data: dict, val_data_raw: dict, test_data_raw: dict, fold_data: list) -> None:
    """
    Save the complete dataset to pickle file.
    
    Args:
        train_data (dict): Training data
        val_data_raw (dict): Validation data
        test_data_raw (dict): Test data
        fold_data (list): Cross-validation fold data
    """
    save_path = join(Config.data.Dataset, 'StoPredDataset.pkl')
    save_dict = {
        'train_data': train_data,
        'test_data': test_data_raw,
        'valid_data': val_data_raw,
        'fold_data': fold_data,
    }
    # check if casp16_benchmark exists
    if os.path.exists(Config.data.casp16_benchmark_path):
        # also load the casp16_benchmark.json
        with open(Config.data.casp16_benchmark_path, 'r') as f:
            casp16_benchmark = json.load(f)
        print(f'Loaded {len(casp16_benchmark)} casp16 benchmark targets')
        save_dict['casp16_benchmark'] = casp16_benchmark

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)


def create_stoichiometry_mapping(train_data: dict) -> dict:
    """
    Create stoichiometry to index mapping.
    
    Args:
        train_data (dict): Training data
        
    Returns:
        dict: Stoichiometry to index mapping
    """
    sto_list = []
    sto_counter = dict()
    for pdb_id, pdb_dict in train_data.items():
        sto = tuple(sorted([v for v in pdb_dict['entity_count'].values()], reverse=True))
        sto = str(sto)
        sto_counter[sto] = sto_counter.get(sto, 0) + 1
        sto_list.append(sto)
    
    for i, sto in enumerate(sto_list):
        if sto_counter[sto] < Config.data.minimum_sto_count:
            sto_list[i] = 'other'
    
    sto_set = sorted(list(set(sto_list)))
    sto2idx = {sto: idx for idx, sto in enumerate(sto_set)}
    
    with open(Config.model.sto2idx, 'w') as f:
        json.dump(sto2idx, f)
    
    return sto2idx


def extract_and_save_sequences(train_data: dict, val_data_raw: dict, test_data_raw: dict) -> None:
    """
    Extract all unique sequences and save to FASTA file.
    
    Args:
        train_data (dict): Training data
        val_data_raw (dict): Validation data
        test_data_raw (dict): Test data
    """
    all_seqs = set()
    
    for pdb_id, pdb_dict in train_data.items():
        for entity_id in pdb_dict['entity_count']:
            entity_dict = pdb_dict[entity_id]
            all_seqs.add(entity_dict['sequence'])
            
    for pdb_id, pdb_dict in val_data_raw.items():
        for entity_id in pdb_dict['entity_count']:
            entity_dict = pdb_dict[entity_id]
            all_seqs.add(entity_dict['sequence'])
            
    for pdb_id, pdb_dict in test_data_raw.items():
        for entity_id in pdb_dict['entity_count']:
            entity_dict = pdb_dict[entity_id]
            all_seqs.add(entity_dict['sequence'])
    
    # casp 16 benchmark
    if os.path.exists(Config.data.casp16_benchmark_path):
        with open(Config.data.casp16_benchmark_path, 'r') as f:
            casp16_benchmark = json.load(f)
        for pdb_id, pdb_dict in casp16_benchmark.items():
            for entity_id in pdb_dict['entity_count']:
                entity_dict = pdb_dict[entity_id]
                all_seqs.add(entity_dict['sequence'])
    
    with open(os.path.join(Config.data.Dataset, 'all_sequences.fasta'), 'w') as f:
        seqs = sorted(list(all_seqs))
        for i, seq in enumerate(seqs):
            f.write(f'>{i}\n')
            f.write(f'{seq}\n')


def main():
    """Main function to process and filter data."""
    # Load raw data
    raw_data = load_raw_data()
    
    # Apply first round of filtering
    first_filter_data = apply_first_filter(raw_data)
    
    # Create count mappings and save them
    count2label, label2idx = create_count_mappings(first_filter_data)
    save_count_mappings(count2label, label2idx)
    
    # Filter out monomeric data
    filter_data_noMonomer = filter_non_monomeric_data(first_filter_data)
    
    # Split data by date into train, validation, and test sets
    train_data, val_data_raw, test_data_raw = split_data_by_date(filter_data_noMonomer)
    
    # Create cross-validation folds
    fold_data = create_cross_validation_folds(train_data)
    
    # Save the complete dataset
    save_dataset(train_data, val_data_raw, test_data_raw, fold_data)
    
    # Print statistics
    print(f'Number of assemblies in training data for noMonomer: {len(train_data)}')
    print(f'Number of assemblies in valid data for noMonomer: {len(val_data_raw)}')
    print(f'Number of assemblies in test data for noMonomer: {len(test_data_raw)}')

    # Create stoichiometry mapping
    create_stoichiometry_mapping(train_data)
    
    # Extract and save all sequences
    extract_and_save_sequences(train_data, val_data_raw, test_data_raw)


if __name__ == '__main__':
    main()