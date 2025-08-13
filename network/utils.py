from torch.utils.data import Dataset
import torch
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Dict
import ml_collections
import numpy as np
from tqdm import tqdm
import json
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from collections import Counter
import pandas as pd
from itertools import permutations

def get_mean_embedding(sequence: str, client=None) -> np.ndarray:
    """
    Get mean embedding for a sequence using ESM3.
    
    Args:
        sequence (str): Protein sequence string
        client: ESM3 model client instance
        
    Returns:
        np.ndarray: Mean embedding of the sequence
    """
    if client is None:
        client = ESM3.from_pretrained('esm3-open')
        if torch.cuda.is_available():
            client = client.to('cuda')
    
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    per_residue_embeddings = output.per_residue_embedding
    
    # Compute mean embedding
    mean_embedding = per_residue_embeddings.mean(dim=0)
    sum_embedding = per_residue_embeddings.sum(dim=0)
    last_tokens = per_residue_embeddings[-1]
    
    mean_embedding = mean_embedding.cpu().numpy()
    sum_embedding = sum_embedding.cpu().numpy()
    last_tokens = last_tokens.cpu().numpy()
    
    return mean_embedding


def merge_features(
    dataset_list: List[Dict],
    sequence_features: dict = None,
    structure_features: dict = None,
    config: ml_collections.ConfigDict = None
) -> List[Dict]:
    """
    Merge features from different sources into a unified dataset format.
    
    Args:
        dataset_list (List[Dict]): List of samples with entity information
        sequence_features (dict): Dictionary mapping sequences to their features
        structure_features (dict): Dictionary mapping sequences to their structure features
        config: Configuration dictionary containing model parameters
        
    Returns:
        List[Dict]: Processed dataset with merged features
    """
    return_dataset = []
    num_subunits = config.model.num_subunits
    sto2idx = json.load(open(config.model.sto2idx))
    count2label = json.load(open(config.model.count2label))
    label2idx = json.load(open(config.model.label2idx))
    num_labels = len(label2idx)

    for sample in tqdm(dataset_list):
        sample_data = dict()
        unique_id = sample['unique_id']
        
        # Sort entity_ids by the order of entity_count, max count first
        entity_ids = list(sample['entity_count'].keys())
        if len(entity_ids) > num_subunits:
            print(f'{unique_id} has more than {num_subunits} subunits')
            continue
            
        mask = np.zeros(config.model.num_subunits)  # Mask for subunits
        sample_sequence_features = np.zeros((config.model.num_subunits, config.model.embedding_dim.sequence))
        sample_structure_features = np.zeros((config.model.num_subunits, config.model.embedding_dim.structure))
        sample_labels = np.ones((config.model.num_subunits, num_labels), dtype=np.int32) * -100
        global_labels = np.ones((config.model.num_subunits, len(sto2idx)), dtype=np.int32) * -100
        
        # One-hot indicator for the number of subunits
        num_subunits_indicator = np.zeros((config.model.num_subunits, 10))
        num_subunits_indicator[len(entity_ids)-1] = 1
        
        sto = str(tuple(sorted([v for v in sample['entity_count'].values()], reverse=True)))
        skip = False
        
        for i, entity_id in enumerate(entity_ids):
            if skip:
                continue
                
            this_entity_id = f'{unique_id}-{entity_id}'
            
            # Get the label
            counts = str(sample['entity_count'][entity_id])
            mask[i] = 1
            
            onhot_count = np.zeros(num_labels)
            
            if counts in count2label:
                onhot_count[label2idx[str(count2label[counts])]] = 1
            else:
                print(f'{unique_id} has no label for {entity_id}')
                onhot_count[label2idx['-1']] = 1
                
            sample_labels[i] = onhot_count
            
            if sto in sto2idx:
                global_labels[i, sto2idx[sto]] = 1
            else:
                global_labels[i, sto2idx['other']] = 1

            # Get sequence feature
            sequence = sample[entity_id]['sequence']
            if sequence_features is not None:
                if sequence not in sequence_features:
                    print(f'{unique_id} has no sequence feature for {entity_id}')
                    skip = True
                    continue
                sequence_feature = sequence_features[sequence]
                sample_sequence_features[i] = sequence_feature

            # Get structure feature
            if structure_features is not None:
                if sequence not in structure_features:
                    # Try to get mean embedding
                    try:
                        structure_feature = get_mean_embedding(sequence)
                        structure_features[sequence] = structure_feature
                    except Exception as e:
                        print(e)
                        print(f'{unique_id} has no structure feature for {entity_id}')
                        skip = True
                        continue
                        
                structure_feature = structure_features[sequence]
                sample_structure_features[i] = structure_feature

        if skip:
            continue
            
        sample_data['entity_count'] = sample['entity_count']
        sample_data['unique_id'] = sample['unique_id']
        
        if sequence_features is not None:
            sample_data['sequenceFeatures'] = sample_sequence_features
        if structure_features is not None:
            sample_data['structureFeatures'] = sample_structure_features
            
        sample_data['labels'] = sample_labels
        sample_data['labels_global'] = global_labels
        sample_data['mask'] = mask
        sample_data['num_subunits_indicator'] = num_subunits_indicator
        
        return_dataset.append(sample_data)
        
    return return_dataset


class StoDataset(Dataset):
    """Dataset class for StoPred training data."""
    
    def __init__(self, data_list: List[Dict]):
        """
        Initialize the dataset.
        
        Args:
            data_list (List[Dict]): List of sample dictionaries
        """
        self.data = data_list
        self.data_tensor = self.preprocessing(data_list)
        print(len(self.data_tensor))
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data_tensor)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict[str, torch.Tensor]: Sample data
        """
        return self.data_tensor[idx]

    def preprocessing(self, data_list: List[Dict]) -> List[Dict]:
        """
        Preprocess data into tensor format.
        
        Args:
            data_list (List[Dict]): List of sample dictionaries
            
        Returns:
            List[Dict]: Processed samples as tensors
        """
        data_tensor = []
        for sample in data_list:
            processed_sample = {
                'labels': torch.tensor(sample['labels'], dtype=torch.long),
                'labels_global': torch.tensor(sample['labels_global'], dtype=torch.float32),
                'mask': torch.tensor(sample['mask'], dtype=torch.float32),
                'unique_id': sample.get('unique_id', ''),
            }
            
            if 'sequenceFeatures' in sample:
                processed_sample['sequence'] = torch.tensor(sample['sequenceFeatures'], dtype=torch.float32)
                
            if 'structureFeatures' in sample:
                processed_sample['structure'] = torch.tensor(sample['structureFeatures'], dtype=torch.float32)
                
            data_tensor.append(processed_sample)
        return data_tensor
    
    def collate_fn(self, batch: List[Dict], max_len_cap: int = 1024) -> Dict[str, any]:
        """
        Collate function for StoDataset DataLoader.
        
        Handles padding for variable-length per-residue features.
        
        Args:
            batch (List[Dict]): Batch of samples
            max_len_cap (int): Maximum length cap for sequences
            
        Returns:
            Dict[str, any]: Collated batch data
        """
        collated_batch = {}
        # Get keys from the first sample, assuming all samples have the same keys
        keys = batch[0].keys()
        batch_size = len(batch)

        for key in keys:
            # Handle unique_id separately
            if key == 'unique_id':
                collated_batch[key] = [sample[key] for sample in batch]
                continue

            # Ensure all items for this key are tensors before stacking
            if isinstance(batch[0][key], torch.Tensor):
                try:
                    collated_batch[key] = torch.stack([sample[key] for sample in batch], dim=0)
                except RuntimeError as e:
                    print(f"Error stacking key '{key}': {e}")
                    # Optionally handle specific keys differently or raise error
                    # Example: check shapes if mismatch error occurs
                    shapes = [sample[key].shape for sample in batch]
                    print(f"Shapes for key '{key}': {shapes}")
                    raise e  # Re-raise the error after printing info
            elif isinstance(batch[0][key], (list, np.ndarray)):
                # If it's a list/numpy array that wasn't converted in preprocessing, try converting now
                # This shouldn't happen if preprocessing is correct, but as a fallback:
                try:
                    tensor_list = [torch.tensor(sample[key]) if not isinstance(sample[key], torch.Tensor) else sample[key] for sample in batch]
                    collated_batch[key] = torch.stack(tensor_list, dim=0)
                except Exception as e:
                    print(f"Error converting/stacking key '{key}': {e}")
                    # Fallback: return as list if stacking fails
                    collated_batch[key] = [sample[key] for sample in batch]
            else:
                # For non-tensor data that isn't unique_id or structureRes (e.g., metadata)
                collated_batch[key] = [sample[key] for sample in batch]
                
        return collated_batch


class StoInferenceDataset(Dataset):
    """Dataset class for StoPred inference data."""
    
    def __init__(self, data_list: List[Dict]):
        """
        Initialize the dataset.
        
        Args:
            data_list (List[Dict]): List of sample dictionaries
        """
        self.data = data_list
        self.data_tensor = self.preprocessing(data_list)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data_tensor)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict[str, torch.Tensor]: Sample data
        """
        return self.data_tensor[idx]

    def preprocessing(self, data_list: List[Dict]) -> List[Dict]:
        """
        Preprocess data into tensor format.
        
        Args:
            data_list (List[Dict]): List of sample dictionaries
            
        Returns:
            List[Dict]: Processed samples as tensors
        """
        data_tensor = []
        for sample in data_list:
            processed_sample = {
                'mask': torch.tensor(sample['mask'], dtype=torch.float32),
                'target_name': sample.get('target_name', ''),
            }
            
            if 'input_sequences_features' in sample:
                processed_sample['sequence'] = torch.tensor(sample['input_sequences_features'], dtype=torch.float32)
                
            if 'input_structures_features' in sample:
                processed_sample['structure'] = torch.tensor(sample['input_structures_features'], dtype=torch.float32)
                
            data_tensor.append(processed_sample)
        return data_tensor
    

# utils for sto prediction
def parse_sto(sto):
    if sto == 'other':
        return 'ohter'
    sto = str(sto).replace('(', '').replace(')', '')
    sto_counts = [int(i) for i in sto.split(',') if i.strip()]
    return sto_counts

def list2tagAlpha(label_list):
    tag = ''
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, v in enumerate(label_list):
        tag += f'{alpha[i]}{v}'
    return tag

def reformate_global_pred(pred_global, num_subunits, idx2sto):
    pred_global_slice = pred_global[:num_subunits]
    pred_global_mean = np.mean(pred_global_slice, axis=0)
    pred_global_dict = {idx2sto[i]: v for i, v in enumerate(pred_global_mean)}
    pred_global_dict = dict(sorted(pred_global_dict.items(), key=lambda x: x[1], reverse=True))
    # remove other
    pred_global_dict = {k: v for k, v in pred_global_dict.items()}
    pred_global_pairs = [(parse_sto(k), v) for k, v in pred_global_dict.items()]
    return pred_global_pairs

def top_k_stoichiometries_combined(prob_matrix, K, idx2label=None, pred_global_pairs=None, alpha=0.7):
    """
    Hybrid strategy for predicting stoichiometries using both global and chain-level predictions.

    Args:
        prob_matrix: (n x m) numpy array of chain-level probabilities, n is number of subunits, m is number of labels
        K: number of top predictions to return
        idx2label: dict mapping column indices to stoichiometry labels
        pred_global_pairs: list of tuples (stoich_list, prob) from global predictions
        alpha: weighting factor for combining global and chain-level scores (default: 0.7)

    Returns:
        List of tuples: (combined_score, stoich_list)
    """
    n, m = prob_matrix.shape
    log_matrix = np.log(prob_matrix + 1e-12)  # add epsilon to prevent log(0)

    if idx2label is None:
        idx2label = {i: i for i in range(m)}
    label2idx = {v: k for k, v in idx2label.items()}

    # ---- Chain-Level Beam Search ----
    row_lists = []
    for i in range(n):
        row = [(log_matrix[i, j], idx2label.get(j, j)) for j in range(m)]
        row.sort(key=lambda x: x[0], reverse=True)
        row_lists.append(row)

    current_top = [(log_prob, [label]) for log_prob, label in row_lists[0][:K]]
    for i in range(1, n):
        new_combos = []
        for score, stoich in current_top:
            for log_prob, label in row_lists[i][:K]:  # keep beam width small
                new_score = score + log_prob
                new_combos.append((new_score, stoich + [label]))
        new_combos.sort(key=lambda x: x[0], reverse=True)
        current_top = new_combos[:K]

    # no need to use global predictions if alpha is 0 or pred_global_pairs is None
    if alpha == 0 or pred_global_pairs is None:
        return current_top[:K]
    # ---- Scoring Helper ----
    def get_chain_log_score(perm):
        try:
            return sum(log_matrix[i, label2idx[label]] for i, label in enumerate(perm))
        except KeyError:
            return float('-inf')

    def combined_score(global_prob, chain_log_prob):
        if global_prob <= 0:
            return float('-inf')
        if alpha > 1:
            # just use multiplication between global and chain
            return np.log(global_prob*np.exp(chain_log_prob)+1e-12)
        combine = alpha * global_prob + np.exp(np.log(1 - alpha + 1e-12) + chain_log_prob)
        combine = max(combine, 1e-12)  # or use np.clip(combine, 1e-12, 1.0)
        return np.log(combine)
        #return alpha * np.log(global_prob) + (1 - alpha) * chain_log_prob

    # ---- Score Global Predictions ----
    all_predictions = []
    used_patterns = set()
    pred_global_dcit = dict()
    min_score = 1e-12
    if pred_global_pairs:
        pred_global_dcit = {tuple(stoich_list): global_prob for stoich_list, global_prob in pred_global_pairs}
        non_zero_global_preds = [pred for pred in pred_global_pairs if pred[1] > 0]
        min_score = min(non_zero_global_preds, key=lambda x: x[1])[1] * 0.99
        # min_score = max(min_score, 1e-12)
        for stoich_list, global_prob in pred_global_pairs[:K]:  # take top-N global predictions
            if stoich_list == None:
                continue
            if len(stoich_list) != n or stoich_list == 'other' or len(stoich_list) == 1:
                continue
            best_perm = None
            best_chain_score = float('-inf')
            for perm in set(permutations(stoich_list)):
                perm_score = get_chain_log_score(perm)
                if perm_score > best_chain_score:
                    best_chain_score = perm_score
                    best_perm = perm
            if best_perm:
                final_score = combined_score(global_prob, best_chain_score)
                all_predictions.append((final_score, list(best_perm)))
                used_patterns.add(tuple(best_perm))

    # ---- Add Chain-Level Only Predictions (Fallbacks) ----
    for chain_log_prob, stoich in current_top:
        if tuple(stoich) not in used_patterns:
            # get the global score,
            tuple_sorted_stoich = tuple(sorted(stoich))
            score = pred_global_dcit.get(tuple_sorted_stoich, min_score)
            # if score is None:
            #     score = pred_global_dcit.get(tuple('other'), 1e-6)
            final_score = combined_score(score, chain_log_prob)  # small prob if no global info
            all_predictions.append((final_score, stoich))

    # ---- Final Top-K Selection ----
    all_predictions.sort(key=lambda x: x[0], reverse=True)
    # move the top1 chain pred to the front
    # if top1_chain_pred_prob > np.log(0.2**n):
    # find the index of the top1 chain pred
    # all_predictions = [pred for pred in all_predictions if pred[1] != top1_chain_pred_sto]
    # all_predictions.insert(0, (top1_chain_pred_prob, top1_chain_pred_sto))
    return all_predictions[:K]

def get_stopred_result_report(pred_dict, test_data, idx2label, idx2sto, alpha=0, min_support=1):
    test_name = []
    test_preds = []
    test_labels = []
    for uniq_id, v in test_data.items():
        result = pred_dict[uniq_id]
        y_pred = result['y_hat']
        y_pred_global = result['global']
        gt = v['entity_count']
        # sort gt by key
        num_subunits = len(gt)
        # if num_subunits == 1:
        #     continue
        # y_pred shape is (num_subunits, num_labels)
        y_pred_slice = y_pred[:num_subunits]
        pred_global_pairs = reformate_global_pred(y_pred_global, num_subunits, idx2sto)
        if num_subunits == 1:
            pred_global_pairs = None
        
        prob, top1_pred = top_k_stoichiometries_combined(y_pred_slice, 2, idx2label, pred_global_pairs, alpha=alpha)[0]
        top1_pred = [int(i) for i in top1_pred]
        gt_list = list(gt.values())
        pairs = list(zip(gt_list, top1_pred))
        # sort pairs by gt_list
        pairs.sort(key=lambda x: x[0], reverse=True)
        gt_list = [i[0] for i in pairs]
        top1_pred = [i[1] for i in pairs]
        # test_preds.append(str(top1_pred))
        # test_labels.append(str(gt_list))
        test_name.append(uniq_id)
        test_preds.append(list2tagAlpha(top1_pred))
        test_labels.append(list2tagAlpha(gt_list))
  
    label_counts = Counter(test_labels)
    rare_labels  = {label for label,count in label_counts.items() if count < min_support}
    mapped_true = []
    mapped_pred = []
    for true, pred in zip(test_labels, test_preds):
        if true in rare_labels:
            mapped_true.append('Other')
            if pred == true:
                mapped_pred.append('Other')
            else:
                mapped_pred.append('WrongOther')
        else:
            mapped_true.append(true)
            if pred in rare_labels or pred not in label_counts:
                mapped_pred.append('Other')
            else:
                mapped_pred.append(pred)

    # Get full label set after mapping
    final_labels = sorted(set(mapped_true + mapped_pred))

    res_csv = pd.DataFrame(
        classification_report(np.array(mapped_true), np.array(mapped_pred), labels=list(set(mapped_true)), zero_division=0, output_dict=True)
    ).T
    accuracy = accuracy_score(np.array(test_labels), np.array(test_preds))
    res_csv.loc['accuracy'] = [accuracy, '', '', 0]
    res_csv = res_csv.sort_values(by='support',ascending=False)

    raw_pred = list(zip(test_name, test_labels, test_preds))
    return res_csv, raw_pred