import os
import argparse
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pickle, json
import torch.nn.functional as F
from config import Config, config_dict
from torch.utils.data import DataLoader
import torch
from network.utils import StoInferenceDataset
from network.sto_net import StoPredNet
from tqdm import tqdm
from typing import Tuple, Dict, Set, List
from itertools import permutations
from utils.ESMC_SequenceFeatureExtraction import load_pretrained_esmc_600m

def read_fasta(fasta_file:str) -> dict:
    """
    Read fasta file and return a dictionary with sequence names as keys and sequences as values

    Args:
        fasta_file (str): path to fasta file

    Returns:
        dict: dictionary with sequence names as keys and sequences as values
    """
    name2seq = {}
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                name = line.strip()[1:].strip()
                name2seq[name] = ""
            else:
                name2seq[name] += line.strip()
    return name2seq

def read_input_dir(input_dir:str) -> Tuple[Dict[str, Dict[str, str]], Set[str]]:
    """
    Read all fasta files in the input directory and return a dictionary with fasta file names as keys and a dictionary with sequence names as keys and sequences as values.
    Also return a set of all sequences.
    """
    prediction_dict = {}
    all_sequences = set()
    for fasta_file in os.listdir(input_dir):
        fasta_path = os.path.join(input_dir, fasta_file)
        target_name = fasta_file.split('.')[0]
        name2seq = read_fasta(fasta_path)
        prediction_dict[target_name] = {
            name: seq for name, seq in name2seq.items()
        }
        all_sequences.update(name2seq.values())
    return prediction_dict, all_sequences

def parse_sto(sto):
    if sto == 'other':
        return 'ohter'
    sto = str(sto).replace('(', '').replace(')', '')
    sto_counts = [int(i) for i in sto.split(',') if i.strip()]
    return sto_counts

def reformate_global_pred(pred_global, num_subunits, idx2sto):
    pred_global_slice = pred_global[:num_subunits]
    pred_global_mean = np.mean(pred_global_slice, axis=0)
    pred_global_dict = {idx2sto[i]: v for i, v in enumerate(pred_global_mean)}
    pred_global_dict = dict(sorted(pred_global_dict.items(), key=lambda x: x[1], reverse=True))
    # remove other
    pred_global_dict = {k: v for k, v in pred_global_dict.items()}
    pred_global_pairs = [(parse_sto(k), v) for k, v in pred_global_dict.items()]
    #pred_global_pairs = [(parse_sto(k), v) for k, v in pred_global_dict.items() if len(parse_sto(k)) == num_subunits or k == 'other']
    return pred_global_pairs

def generate_input_sequences_embeddings(sequences:Set[str], device:str) -> np.ndarray:
    """
    Generate embeddings for the input sequences.
    """
    client = load_pretrained_esmc_600m(device=torch.device(device))
    features = {}
    for seq in tqdm(sequences, desc="Generating embeddings"):
        protein = ESMProtein(sequence=seq[:2048])
        protein_tensor = client.encode(protein)
        logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
        last_3_hidden_states = logits_output.hidden_states[-3:]
        mean_hidden_states = last_3_hidden_states[:, :, 1:-1, :].mean(dim=2)
        flattened_hidden_states = mean_hidden_states.flatten()
        features[seq] = flattened_hidden_states.to(dtype=torch.float32).detach().cpu().numpy()
    # delete the client
    del client
    return features

def prepare_stopred_input(prediction_dict:Dict[str, Dict[str, str]], sequenceFeatures:Dict[str, np.ndarray]) -> Dict[str, Dict[str, str]]:
    orginal_num_subunits = Config.model.num_subunits
    # check the max number of subunits in the prediction dict
    max_num_subunits = max(len(v) for v in prediction_dict.values())
    # override the max number of subunits in the config
    Config.model.num_subunits = max_num_subunits

    inference_datasets = []
    for target_name, name2seq in prediction_dict.items():
        if len(name2seq) == 0:
            print(f"No sequences found for {target_name}")
            continue
        elif len(name2seq) > orginal_num_subunits:
            print(f"Number of sequences for {target_name} is greater than the max subunits number during training: {len(name2seq)} > {orginal_num_subunits}\n Eventhough the model can still make the prediction, it is not recommended.")
            #continue

        input_sequences_features = np.zeros((max_num_subunits, Config.model.embedding_dim.sequence), dtype=np.float32)
        mask = np.zeros(Config.model.num_subunits, dtype=np.float32)

        skip = False
        for i, (name, seq) in enumerate(name2seq.items()):
            if seq not in sequenceFeatures:
                print(f"Sequence {seq} not found in sequenceFeatures")
                skip = True
                break
            mask[i] = 1
            input_sequences_features[i, :] = sequenceFeatures[seq]
        if skip:
            continue
        inference_datasets.append({
            "target_name": target_name,
            "mask": mask,
            "input_sequences_features": input_sequences_features
        })
    
    # prepare the dataset and dataloader
    inference_dataset = StoInferenceDataset(inference_datasets)
    inference_dataloader = DataLoader(inference_dataset, batch_size=Config.data.batch_size, shuffle=False)
    return inference_dataloader

def predict(model_dir:str, inference_dataloader:DataLoader, device:str) -> Dict[str, Dict[str, str]]:
    all_fold_results = dict()
    num_folds = Config.data.num_folds
    label2idx = None
    sto2idx = None
    for fold in range(num_folds):
        model = StoPredNet.load_from_pkl(os.path.join(model_dir, f'model_fold{fold}.pkl'))
        if label2idx is None:
            label2idx = model.config['label2idx']
        if sto2idx is None:
            sto2idx = model.config['sto2idx']
        model = model.to(device)
        model.eval()
        all_fold_results[fold] = dict()
        with torch.no_grad():
            for batch in inference_dataloader:
                # move to device, only for tensor
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                y_hats, y_hats_global = model(batch)
                # apply softmax
                y_hats = F.softmax(y_hats, dim=2)
                y_hats_global = F.softmax(y_hats_global, dim=2)
                y_hats = y_hats.cpu().detach().numpy()
                y_hats_global = y_hats_global.cpu().detach().numpy()
                for unique_id, y_hat, y_hat_global in zip(batch['target_name'], y_hats, y_hats_global):
                    all_fold_results[fold][unique_id] = {
                        "y_hat": y_hat,
                        "y_hat_global": y_hat_global
                    }
    mean_results = dict()
    for fold, fold_results in all_fold_results.items():
        for unique_id, result in fold_results.items():
            if unique_id not in mean_results:
                mean_results[unique_id] = {
                    "y_hat": result['y_hat'],
                    "y_hat_global": result['y_hat_global']
                }
            else:
                mean_results[unique_id]['y_hat'] += result['y_hat']
                mean_results[unique_id]['y_hat_global'] += result['y_hat_global']
    for unique_id, result in mean_results.items():
        mean_results[unique_id]['y_hat'] /= num_folds
        mean_results[unique_id]['y_hat_global'] /= num_folds
    return mean_results, label2idx, sto2idx

def top_k_stoichiometries(prob_matrix, K, idx2label=None):
    """
    Finds the top K most likely stoichiometries (and their log-probabilities)
    for an n x m probability matrix, where:
      - n = number of chains
      - m = number of possible copy-number labels per chain
      - prob_matrix[i][j] = Probability that chain i has copy-number j.
      
    Using log probabilities to avoid underflow/rounding errors, we combine probabilities
    by summing their logarithms.
    
    Returns a list of (log_prob, [label0, label1, ..., label_{n-1}])
    sorted by log_prob in descending order.
    If desired, you can recover the raw probability with np.exp(log_prob).
    """
    n, m = prob_matrix.shape

    # Create a mapping from index to label if not provided
    if idx2label is None:
        idx2label = {i: i for i in range(m)}
    
    # add a small epsilon to the probability matrix to avoid log(0)
    prob_matrix += 1e-10
    
    # Compute the log probabilities for the matrix.
    log_matrix = np.log(prob_matrix)
    
    # Convert each row of log_matrix into a list of (log_prob, label) sorted in descending order.
    row_lists = []
    for i in range(n):
        row_list = [(log_matrix[i, j], idx2label[j]) for j in range(m)]
        row_list.sort(key=lambda x: x[0], reverse=True)  # higher log_prob means higher original prob.
        row_lists.append(row_list)
    
    # Initialize current_top from the first row.
    # Each entry is a tuple (cumulative_log_prob, [label]) for chain 0.
    current_top = [(row_lists[0][j][0], [row_lists[0][j][1]]) for j in range(m)]
    # Keep only the top K from chain 0.
    current_top.sort(key=lambda x: x[0], reverse=True)
    current_top = current_top[:K]
    
    # Iteratively merge with subsequent rows.
    for i in range(1, n):
        new_combos = []
        # For each current combination, add each possibility from the next chain.
        for (old_log_prob, old_stoich) in current_top:
            for (chain_log_prob, label) in row_lists[i]:
                new_log_prob = old_log_prob + chain_log_prob
                new_stoich = old_stoich + [label]
                new_combos.append((new_log_prob, new_stoich))
                
        # Sort and keep only the top K combinations.
        new_combos.sort(key=lambda x: x[0], reverse=True)
        current_top = new_combos[:K]
    
    return current_top

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

def add_seqname(sto:List[int], names:List[str]) -> List[Tuple[str, List[int]]]:
    """
    Add the sequence name to the stoichiometry
    """
    return [(name, sto) for name, sto in zip(names, sto)]

def reformate_prediction(mean_results: Dict[str, Dict[str, str]], prediction_dict:Dict[str, Dict[str, str]], topk:int, alpha:float, label2idx:Dict[int, int], sto2idx:Dict[int, str]) -> Dict[str, Dict[str, str]]:
    """
    Reformat the prediction to the format of the input fasta files
    """
    # load the model's sto2idx, count2label, label2idx
    #sto2idx = json.load(open(Config.model.sto2idx))
    idx2sto = {v: k for k, v in sto2idx.items()}
    #label2idx = json.load(open(Config.model.label2idx))
    idx2label = {int(v): int(k) for k, v in label2idx.items()}

    final_predictions = dict()
    for unique_id, result in mean_results.items():
        final_predictions[unique_id] = {
            'chain_level_predictions': dict(),
            'global_predictions': dict(),
            'topk_predictions': dict()
        }
        y_hat = result['y_hat']
        y_hat_global = result['y_hat_global']

        # chain level predictions and global predictions
        for i, (name, seq) in enumerate(prediction_dict[unique_id].items()):
            final_predictions[unique_id]['chain_level_predictions'][name] = dict()
            i_pred = y_hat[i, :]
            for j, prob in enumerate(i_pred):
                num_copies = idx2label[j]
                final_predictions[unique_id]['chain_level_predictions'][name][num_copies] = float(prob)
            # sort the final_predictions[unique_id][name] by the probability
            final_predictions[unique_id]['chain_level_predictions'][name] = dict(sorted(final_predictions[unique_id]['chain_level_predictions'][name].items(), key=lambda x: x[1], reverse=True))

            # current global prediction
            final_predictions[unique_id]['global_predictions'][name] = dict()
            i_pred_global = y_hat_global[i, :]
            for m, global_prob in enumerate(i_pred_global):
                sto = idx2sto[m]
                final_predictions[unique_id]['global_predictions'][name][sto] = float(global_prob)
            # sort the final_predictions[unique_id]['global_predictions'][name] by the probability
            final_predictions[unique_id]['global_predictions'][name] = dict(sorted(final_predictions[unique_id]['global_predictions'][name].items(), key=lambda x: x[1], reverse=True))
        
        num_subunits = len(prediction_dict[unique_id])
        # do not use the global prediction if is a homo oligomer
        if num_subunits == 1:
            pred_global_pairs = None
            # pop the global prediction
            final_predictions[unique_id].pop('global_predictions')
        pred_global_pairs = reformate_global_pred(y_hat_global, num_subunits, idx2sto)
        # topk predictions, only for yhat
        y_hat_slice = y_hat[:num_subunits]
        topk_predictions = top_k_stoichiometries_combined(y_hat_slice, topk, idx2label, pred_global_pairs, alpha)
        names = list(prediction_dict[unique_id].keys())
        # apply reverse log
        topk_predictions = [ (add_seqname(sto, names), float(np.exp(log_prob))) for log_prob, sto in topk_predictions if np.exp(log_prob) > 0.0001]
        # add to final_predictions
        final_predictions[unique_id]['topk_predictions'] = topk_predictions
    
    return final_predictions


def main(args):
    # read the input fasta files
    prediction_dict, all_sequences = read_input_dir(args.input_dir)
    # generate the embeddings
    sequenceFeatures = generate_input_sequences_embeddings(all_sequences, args.device)
    # prepare the inference dataloader
    inference_dataloader = prepare_stopred_input(prediction_dict, sequenceFeatures)
    # predict
    mean_result, label2idx, sto2idx = predict(args.model_dir, inference_dataloader, args.device)
    # reformat the prediction
    final_predictions = reformate_prediction(mean_result, prediction_dict, args.topk, args.alpha, label2idx, sto2idx)
    # save the prediction
    for target_name, result in final_predictions.items():
        with open(os.path.join(args.output_dir, f'{target_name}.json'), 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(root_dir, "models")
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to the input directory that contains the fasta files")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--topk", type=int, default=10, help="Number of top K stoichiometries to predict")
    parser.add_argument("--model_dir", type=str, default=default_model_dir, help="Path to the model directory")
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for the combined strategy')
    args = parser.parse_args()
    args.input_dir = os.path.realpath(args.input_dir)
    args.output_dir = os.path.realpath(args.output_dir)
    args.model_dir = os.path.realpath(args.model_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
