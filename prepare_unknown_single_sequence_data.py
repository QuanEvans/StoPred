import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import sklearn.model_selection

from config import Config
from prepare_data import apply_first_filter, is_after_date, is_exp_defined


METADATA_KEYS = {
    'unique_id',
    'entity_count',
    'stoichiometry',
    'details',
    'experimental_support',
    'method_details',
    'pdb_pubmed_id',
    'release_date',
}


def create_parser():
    unknown_cfg = Config.unknown_single_sequence
    parser = argparse.ArgumentParser(
        description=(
            'Prepare a monomer-aware StoPred dataset for unknown single-sequence '
            'training. The base dataset supplies homomer/heteromer examples; '
            'monomers are added from processed PDBmmcif.json.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--base-dataset-pkl',
        default=os.path.join(Config.data.Dataset, 'StoPredDataset.pkl'),
        help='Existing StoPred dataset pickle, usually the no-monomer multimer dataset.',
    )
    parser.add_argument(
        '--processed-json',
        default=Config.data.processed_PDBmmcif_path,
        help='Processed PDBmmcif.json used as the monomer source.',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        default=unknown_cfg.dataset_dir,
        help='Output directory for dataset.pkl, label maps, summary.json, and all_sequences.fasta.',
    )
    parser.add_argument(
        '--target-scope',
        choices=['all', 'single_entity'],
        default=unknown_cfg.target_scope,
        help='Use all base targets, or only monomer/homomer single-entity targets.',
    )
    parser.add_argument(
        '--monomer-conflict-policy',
        choices=['drop_existing_single_entity', 'drop_existing_any_entity', 'keep'],
        default=unknown_cfg.monomer_conflict_policy,
        help='How to handle monomer sequences already present in the base dataset.',
    )
    parser.add_argument(
        '--merge-monomers-into-main-splits',
        dest='merge_monomers_into_main_splits',
        default=unknown_cfg.merge_monomers_into_main_splits,
        action=argparse.BooleanOptionalAction,
        help='Add monomer examples to train/valid/test_data; disable to only write monomer_train/valid/test_data splits.',
    )
    parser.add_argument('--cut-off-date', default=Config.data.cut_off_date)
    parser.add_argument('--test-cut-off-date', default=Config.data.test_cut_off_date)
    parser.add_argument('--minimum-sample-count', type=int, default=Config.data.minimum_sample_count)
    parser.add_argument('--minimum-sto-count', type=int, default=Config.data.minimum_sto_count)
    return parser


def load_pickle(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def save_pickle(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as output_file:
        pickle.dump(data, output_file)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output_file:
        json.dump(data, output_file, indent=2)


def split_values(split):
    if isinstance(split, dict):
        return split.values()
    return split


def sample_group(sample):
    entity_count = {key: int(value) for key, value in sample['entity_count'].items()}
    total_count = sum(entity_count.values())
    if total_count == 1:
        return 'monomer'
    if len(entity_count) == 1:
        return 'homomer'
    return 'heteromer'


def filter_base_split(split_data, target_scope):
    if target_scope == 'all':
        return dict(split_data)
    return {
        unique_id: sample
        for unique_id, sample in split_data.items()
        if len(sample.get('entity_count', {})) == 1
    }


def sequence_hash(sequence):
    return hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:16]


def entity_sequence(sample, entity_id):
    entity = sample.get(entity_id) or sample.get(str(entity_id)) or {}
    return entity.get('sequence')


def monomer_record(record):
    if len(record.get('entity_count', {})) != 1:
        return None
    entity_id = next(iter(record['entity_count']))
    if int(record['entity_count'][entity_id]) != 1:
        return None
    sequence = entity_sequence(record, entity_id)
    if not sequence:
        return None
    return entity_id, sequence


def relabel_monomer(record, source_ids):
    entity_id = next(iter(record['entity_count']))
    sequence = entity_sequence(record, entity_id)
    relabeled = copy.deepcopy(record)
    relabeled['unique_id'] = f'MONO_{sequence_hash(sequence)}'
    relabeled['source_unique_ids'] = source_ids
    relabeled['monomer_augmentation_source'] = 'processed_PDBmmcif'
    relabeled['entity_count'] = {entity_id: 1}
    relabeled[entity_id]['count'] = 1
    relabeled['stoichiometry'] = 'A1'
    return relabeled


def collapse_monomers_by_sequence(raw_monomers):
    grouped = {}
    for unique_id, record in raw_monomers.items():
        parsed = monomer_record(record)
        if parsed is None:
            continue
        _, sequence = parsed
        grouped.setdefault(sequence, []).append((unique_id, record))

    collapsed = {}
    for sequence, entries in grouped.items():
        entries.sort(key=lambda item: (item[1].get('release_date') or '', item[0]))
        source_ids = [unique_id for unique_id, _ in entries]
        representative = relabel_monomer(entries[0][1], source_ids)
        collapsed[representative['unique_id']] = representative

    return collapsed, {
        'monomer_unique_sequences': len(grouped),
        'monomer_duplicate_sequence_assemblies_collapsed': sum(
            max(len(entries) - 1, 0)
            for entries in grouped.values()
        ),
    }


def collect_base_sequences(dataset):
    any_entity_sequences = set()
    single_entity_sequences = set()
    for split_name in ('train_data', 'valid_data', 'test_data'):
        for sample in split_values(dataset.get(split_name, {})):
            is_single_entity = len(sample.get('entity_count', {})) == 1
            for entity_id in sample.get('entity_count', {}):
                sequence = entity_sequence(sample, entity_id)
                if not sequence:
                    continue
                any_entity_sequences.add(sequence)
                if is_single_entity:
                    single_entity_sequences.add(sequence)
    return any_entity_sequences, single_entity_sequences


def filter_monomer_conflicts(monomers, any_entity_sequences, single_entity_sequences, policy):
    if policy == 'keep':
        return monomers, 0

    conflict_sequences = (
        any_entity_sequences
        if policy == 'drop_existing_any_entity'
        else single_entity_sequences
    )
    kept = {}
    removed = 0
    for unique_id, record in monomers.items():
        entity_id = next(iter(record['entity_count']))
        sequence = entity_sequence(record, entity_id)
        if sequence in conflict_sequences:
            removed += 1
            continue
        kept[unique_id] = record
    return kept, removed


def split_by_date(records, cut_off_date, test_cut_off_date):
    train_data = {}
    valid_data = {}
    test_data = {}
    for unique_id, record in records.items():
        if not is_after_date(record, cut_off_date):
            train_data[unique_id] = record
        elif is_after_date(record, test_cut_off_date):
            if is_exp_defined(record):
                test_data[unique_id] = record
        elif is_exp_defined(record):
            valid_data[unique_id] = record
    return train_data, valid_data, test_data


def merge_split(base, addition):
    collisions = set(base) & set(addition)
    if collisions:
        raise ValueError(f'Duplicate unique IDs while merging splits: {sorted(collisions)[:5]}')
    merged = dict(base)
    merged.update(addition)
    return merged


def create_count_mappings(train_data, minimum_sample_count):
    freq = {}
    for sample in train_data.values():
        for count in sample['entity_count'].values():
            count = int(count)
            freq[count] = freq.get(count, 0) + 1

    count2label = {0: 0}
    for count, n_samples in freq.items():
        count2label[count] = -1 if n_samples < minimum_sample_count else count
    count2label = dict(sorted(count2label.items()))
    label2idx = {label: idx for idx, label in enumerate(sorted(set(count2label.values())))}
    return count2label, label2idx


def sto_tuple(sample):
    return tuple(sorted((int(value) for value in sample['entity_count'].values()), reverse=True))


def create_stoichiometry_mapping(train_data, minimum_sto_count):
    sto_counts = {}
    sto_labels = []
    for sample in train_data.values():
        label = str(sto_tuple(sample))
        sto_counts[label] = sto_counts.get(label, 0) + 1
        sto_labels.append(label)
    mapped = [
        'other' if sto_counts[label] < minimum_sto_count else label
        for label in sto_labels
    ]
    return {label: idx for idx, label in enumerate(sorted(set(mapped)))}


def create_cross_validation_folds(train_data, minimum_sto_count):
    ids = []
    labels = []
    sto_counts = {}
    for unique_id, sample in train_data.items():
        label = str(sto_tuple(sample))
        sto_counts[label] = sto_counts.get(label, 0) + 1
        ids.append(unique_id)
        labels.append(label)

    labels = [
        'other' if sto_counts[label] < minimum_sto_count else label
        for label in labels
    ]
    label_counts = {label: labels.count(label) for label in set(labels)}
    if min(label_counts.values()) < Config.data.num_folds:
        rare_labels = {label for label, count in label_counts.items() if count < Config.data.num_folds}
        labels = ['other' if label in rare_labels else label for label in labels]

    folds = []
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=Config.data.num_folds,
        shuffle=True,
        random_state=Config.global_seed,
    )
    for train_idx, valid_idx in splitter.split(ids, labels):
        folds.append({
            'train': [ids[idx] for idx in train_idx],
            'valid': [ids[idx] for idx in valid_idx],
        })
    return folds


def write_sequences(dataset, path):
    sequences = set()
    for split_name, split in dataset.items():
        if split_name == 'fold_data' or not isinstance(split, dict):
            continue
        for sample in split.values():
            if not isinstance(sample, dict) or 'entity_count' not in sample:
                continue
            for entity_id in sample['entity_count']:
                sequence = entity_sequence(sample, entity_id)
                if sequence:
                    sequences.add(sequence)

    with path.open('w') as output_file:
        for idx, sequence in enumerate(sorted(sequences)):
            output_file.write(f'>{idx}\n{sequence}\n')
    return len(sequences)


def count_groups(split):
    counts = {'total': len(split), 'monomer': 0, 'homomer': 0, 'heteromer': 0}
    for sample in split.values():
        counts[sample_group(sample)] += 1
    return counts


def main():
    parser = create_parser()
    args = parser.parse_args()
    base_dataset_path = Path(args.base_dataset_pkl).resolve()
    processed_json_path = Path(args.processed_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = load_pickle(base_dataset_path)
    prepared = {
        'train_data': filter_base_split(base_dataset['train_data'], args.target_scope),
        'valid_data': filter_base_split(base_dataset['valid_data'], args.target_scope),
        'test_data': filter_base_split(base_dataset['test_data'], args.target_scope),
    }

    with processed_json_path.open() as input_file:
        raw_data = json.load(input_file)
    first_filter_data = apply_first_filter(raw_data)
    raw_monomers = {
        unique_id: record
        for unique_id, record in first_filter_data.items()
        if monomer_record(record) is not None
    }
    collapsed_monomers, collapse_summary = collapse_monomers_by_sequence(raw_monomers)
    any_sequences, single_sequences = collect_base_sequences(prepared)
    filtered_monomers, conflict_removed = filter_monomer_conflicts(
        collapsed_monomers,
        any_sequences,
        single_sequences,
        args.monomer_conflict_policy,
    )
    monomer_train, monomer_valid, monomer_test = split_by_date(
        filtered_monomers,
        args.cut_off_date,
        args.test_cut_off_date,
    )

    if args.merge_monomers_into_main_splits:
        prepared['train_data'] = merge_split(prepared['train_data'], monomer_train)
        prepared['valid_data'] = merge_split(prepared['valid_data'], monomer_valid)
        prepared['test_data'] = merge_split(prepared['test_data'], monomer_test)

    prepared['monomer_train_data'] = monomer_train
    prepared['monomer_valid_data'] = monomer_valid
    prepared['monomer_test_data'] = monomer_test
    if 'casp16_benchmark' in base_dataset:
        prepared['casp16_benchmark'] = base_dataset['casp16_benchmark']

    prepared['fold_data'] = create_cross_validation_folds(
        prepared['train_data'],
        args.minimum_sto_count,
    )
    count2label, label2idx = create_count_mappings(
        prepared['train_data'],
        args.minimum_sample_count,
    )
    sto2idx = create_stoichiometry_mapping(
        prepared['train_data'],
        args.minimum_sto_count,
    )

    save_pickle(prepared, output_dir / 'dataset.pkl')
    save_json(count2label, output_dir / 'count2label.json')
    save_json(label2idx, output_dir / 'label2idx.json')
    save_json(sto2idx, output_dir / 'sto2idx.json')
    n_sequences = write_sequences(prepared, output_dir / 'all_sequences.fasta')

    summary = {
        'dataset_name': output_dir.name,
        'base_dataset_pkl': str(base_dataset_path),
        'processed_json': str(processed_json_path),
        'target_scope': args.target_scope,
        'cut_off_date': args.cut_off_date,
        'test_cut_off_date': args.test_cut_off_date,
        'monomer_conflict_policy': args.monomer_conflict_policy,
        'merge_monomers_into_main_splits': args.merge_monomers_into_main_splits,
        'base_splits': {
            key: count_groups(value)
            for key, value in prepared.items()
            if key in {'train_data', 'valid_data', 'test_data'}
        },
        'monomer_splits': {
            'monomer_train_data': count_groups(monomer_train),
            'monomer_valid_data': count_groups(monomer_valid),
            'monomer_test_data': count_groups(monomer_test),
        },
        'raw_monomer_assemblies_after_filter': len(raw_monomers),
        **collapse_summary,
        'monomers_removed_for_existing_sequence_conflict': conflict_removed,
        'num_count_labels': len(label2idx),
        'num_stoichiometry_labels': len(sto2idx),
        'num_fasta_sequences': n_sequences,
    }
    save_json(summary, output_dir / 'summary.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
