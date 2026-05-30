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
import re
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

def normalize_release_tag(release_date: str) -> str:
    """
    Convert a release date such as 2026-01-01 or 20260101 to YYYYMMDD.
    """
    release_tag = str(release_date).strip().replace('-', '').replace('_', '').replace('.', '')
    if len(release_tag) != 8 or not release_tag.isdigit():
        raise ValueError(
            f'Invalid --train-release-date: {release_date}. '
            'Use YYYYMMDD or YYYY-MM-DD.'
        )
    return release_tag

def release_tag_to_date(release_tag: str) -> str:
    return f'{release_tag[:4]}-{release_tag[4:6]}-{release_tag[6:]}'

def has_fold_models(model_dir: str, num_folds: int = None) -> bool:
    if num_folds is None:
        num_folds = Config.data.num_folds
    if not os.path.isdir(model_dir):
        return False
    return all(
        os.path.exists(os.path.join(model_dir, f'model_fold{fold}.pkl'))
        for fold in range(num_folds)
    )

def release_tag_from_model_name(name: str, prefixes: List[str]):
    for prefix in prefixes:
        pattern = rf'^{re.escape(prefix)}[_-](\d{{8}})$'
        match = re.match(pattern, name)
        if match:
            return match.group(1)
    return None

def validate_model_dir(model_dir: str, model_role: str) -> str:
    model_dir = os.path.realpath(model_dir)
    missing = [
        f'model_fold{fold}.pkl'
        for fold in range(Config.data.num_folds)
        if not os.path.exists(os.path.join(model_dir, f'model_fold{fold}.pkl'))
    ]
    if missing:
        raise FileNotFoundError(
            f'{model_role} model directory is missing fold checkpoints: {model_dir}\n'
            f'Missing: {", ".join(missing)}'
        )
    return model_dir

def candidate_model_dirs(model_root: str, release_tag: str, model_role: str) -> List[str]:
    model_root = os.path.realpath(model_root)
    if model_role == 'default':
        default_prefix = Config.inference.get('default_model_release_prefix', 'default')
        names = [
            f'{default_prefix}_{release_tag}',
            f'stopred_{release_tag}',
            'default',
        ]
    elif model_role == 'unknown_single':
        unknown_prefix = Config.inference.get('unknown_single_model_release_prefix', 'unk_single')
        names = [
            f'{unknown_prefix}_{release_tag}',
            f'unknown_single_{release_tag}',
            f'unknown_single_sequence_{release_tag}',
            'unknown_single_sequence',
        ]
        if release_tag == normalize_release_tag(Config.data.cut_off_date):
            config_model_dir = os.path.realpath(Config.unknown_single_sequence.model_dir)
            names.append(os.path.relpath(config_model_dir, model_root))
    else:
        raise ValueError(f'Unsupported model role: {model_role}')
    return [os.path.join(model_root, name) for name in dict.fromkeys(names)]

def release_model_prefixes(model_role: str) -> List[str]:
    if model_role == 'default':
        default_prefix = Config.inference.get('default_model_release_prefix', 'default')
        return [default_prefix, 'stopred']
    if model_role == 'unknown_single':
        unknown_prefix = Config.inference.get('unknown_single_model_release_prefix', 'unk_single')
        return [unknown_prefix, 'unknown_single', 'unknown_single_sequence']
    raise ValueError(f'Unsupported model role: {model_role}')

def available_release_models(model_root: str, model_role: str) -> Dict[str, str]:
    model_root = os.path.realpath(model_root)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f'Model root does not exist: {model_root}')

    models = {}
    prefixes = release_model_prefixes(model_role)
    for name in os.listdir(model_root):
        path = os.path.join(model_root, name)
        if not has_fold_models(path):
            continue
        release_tag = release_tag_from_model_name(name, prefixes)
        if release_tag:
            models[release_tag] = os.path.realpath(path)
    return models

def resolve_latest_model_dir(model_root: str, model_role: str):
    models = available_release_models(model_root, model_role)
    if not models:
        return None, None
    release_tag = sorted(models, reverse=True)[0]
    return models[release_tag], release_tag

def resolve_model_dir(
    explicit_model_dir: str,
    model_root: str,
    release_tag: str,
    model_role: str,
) -> Tuple[str, str]:
    if explicit_model_dir:
        model_dir = validate_model_dir(explicit_model_dir, model_role)
        explicit_tag = release_tag
        if explicit_tag is None:
            explicit_tag = release_tag_from_model_name(
                os.path.basename(model_dir),
                release_model_prefixes(model_role),
            )
        return model_dir, explicit_tag

    if release_tag is None:
        latest_model_dir, latest_release_tag = resolve_latest_model_dir(model_root, model_role)
        if latest_model_dir is not None:
            return latest_model_dir, latest_release_tag
        if model_role == 'default':
            fallback = os.path.join(os.path.realpath(model_root), 'default')
            if has_fold_models(fallback):
                return os.path.realpath(fallback), None
        elif model_role == 'unknown_single':
            fallback = os.path.join(os.path.realpath(model_root), 'unknown_single_sequence')
            if has_fold_models(fallback):
                return os.path.realpath(fallback), None
        raise FileNotFoundError(
            f'Could not resolve the latest {model_role} model in {os.path.realpath(model_root)}. '
            'Use --train-release-date or an explicit model directory override.'
        )

    candidates = candidate_model_dirs(model_root, release_tag, model_role)
    for candidate in candidates:
        if has_fold_models(candidate):
            return os.path.realpath(candidate), release_tag

    formatted = '\n  '.join(os.path.realpath(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f'Could not resolve a {model_role} model for release {release_tag_to_date(release_tag)}. '
        f'Tried:\n  {formatted}\n'
        'Use --model_dir for the default/heteromer model or --unk_model_dir for the unknown-single model.'
    )

def is_single_entity_target(name2seq: Dict[str, str]) -> bool:
    return len(name2seq) == 1

def plan_model_selection(args, prediction_dict: Dict[str, Dict[str, str]]):
    if not prediction_dict:
        raise ValueError(f'No FASTA targets found in {args.input_dir}')
    requested_release_tag = normalize_release_tag(args.train_release_date) if args.train_release_date else None
    release_tag = requested_release_tag
    model_root = os.path.realpath(args.model_root)

    target_is_single_entity = {
        target_name: is_single_entity_target(name2seq)
        for target_name, name2seq in prediction_dict.items()
    }
    needs_unknown_model = args.unk_for_homomer and any(target_is_single_entity.values())
    needs_default_model = any(
        (not is_single_entity) or (not args.unk_for_homomer)
        for is_single_entity in target_is_single_entity.values()
    )

    if release_tag is None and needs_default_model and needs_unknown_model:
        default_models = available_release_models(model_root, 'default')
        unknown_models = available_release_models(model_root, 'unknown_single')
        common_release_tags = sorted(set(default_models) & set(unknown_models), reverse=True)
        if not common_release_tags:
            raise FileNotFoundError(
                f'Could not resolve a common latest release with both default and unknown-single models in {model_root}. '
                'Use --train-release-date with a complete release, or pass --model_dir and --unk_model_dir explicitly.'
            )
        release_tag = common_release_tags[0]

    default_model_dir = None
    default_release_tag = None
    if needs_default_model:
        default_model_dir, default_release_tag = resolve_model_dir(
            args.model_dir,
            model_root,
            release_tag,
            'default',
        )
    unknown_model_dir = None
    unknown_release_tag = None
    if needs_unknown_model:
        unknown_model_dir, unknown_release_tag = resolve_model_dir(
            args.unk_model_dir,
            model_root,
            release_tag,
            'unknown_single',
        )

    plan = {}
    grouped_targets = {}
    for target_name, name2seq in prediction_dict.items():
        single_entity = target_is_single_entity[target_name]
        if single_entity and args.unk_for_homomer:
            model_role = 'unknown_single'
            model_dir = unknown_model_dir
        else:
            model_role = 'default'
            model_dir = default_model_dir
        plan[target_name] = {
            'model_role': model_role,
            'model_dir': model_dir,
            'single_entity_target': single_entity,
            'train_release_date': release_tag_to_date(unknown_release_tag if model_role == 'unknown_single' else default_release_tag)
                if (unknown_release_tag if model_role == 'unknown_single' else default_release_tag)
                else None,
        }
        grouped_targets.setdefault(model_dir, []).append(target_name)

    print('Model selection:')
    if requested_release_tag:
        print(f'  requested train release date: {release_tag_to_date(requested_release_tag)}')
    elif release_tag:
        print(f'  selected latest common train release date: {release_tag_to_date(release_tag)}')
    else:
        print('  requested train release date: latest available')
    if needs_default_model:
        default_release_text = release_tag_to_date(default_release_tag) if default_release_tag else 'unversioned'
        print(f'  default model: {default_model_dir} ({default_release_text})')
    else:
        print('  default model: not needed for this input set')
    if needs_unknown_model:
        unknown_release_text = release_tag_to_date(unknown_release_tag) if unknown_release_tag else 'unversioned'
        print(f'  unknown-single model: {unknown_model_dir} ({unknown_release_text})')
    elif args.unk_for_homomer:
        print('  unknown-single model: not needed for this input set')
    else:
        print('  unknown-single model: disabled; single-entity targets use the default model')
    return plan, grouped_targets

def parse_sto(sto):
    if sto == 'other':
        return 'other'
    sto = str(sto).replace('(', '').replace(')', '')
    sto_counts = [int(i) for i in sto.split(',') if i.strip()]
    return sto_counts

def canonical_stoich(stoich):
    return tuple(sorted((int(i) for i in stoich), reverse=True))

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

def prepare_stopred_input(
    prediction_dict:Dict[str, Dict[str, str]],
    sequenceFeatures:Dict[str, np.ndarray],
    trained_num_subunits:int = None,
) -> Dict[str, Dict[str, str]]:
    trained_num_subunits = trained_num_subunits or Config.model.num_subunits
    max_num_subunits = max(len(v) for v in prediction_dict.values())
    inference_datasets = []
    for target_name, name2seq in prediction_dict.items():
        if len(name2seq) == 0:
            print(f"No sequences found for {target_name}")
            continue
        elif len(name2seq) > trained_num_subunits:
            print(f"Number of sequences for {target_name} is greater than the max subunits number during training: {len(name2seq)} > {trained_num_subunits}\n Even though the model can still make the prediction, it is not recommended.")
            #continue

        input_sequences_features = np.zeros((max_num_subunits, Config.model.embedding_dim.sequence), dtype=np.float32)
        mask = np.zeros(max_num_subunits, dtype=np.float32)

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

def model_num_subunits(model_dir: str) -> int:
    model = StoPredNet.load_from_pkl(os.path.join(model_dir, 'model_fold0.pkl'))
    return int(model.num_subunits)

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

def top_k_stoichiometries_combined(prob_matrix, K, idx2label=None, pred_global_pairs=None, alpha=0.5):
    """
    Hybrid strategy for predicting stoichiometries using both global and chain-level predictions.

    Args:
        prob_matrix: (n x m) numpy array of chain-level probabilities, n is number of subunits, m is number of labels
        K: number of top predictions to return
        idx2label: dict mapping column indices to stoichiometry labels
        pred_global_pairs: list of tuples (stoich_list, prob) from global predictions
        alpha: weighting factor for combining global and chain-level scores (default: 0.5)

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

    # no need to use global predictions if alpha is 0 or pred_global_pairs is None
    if alpha == 0 or pred_global_pairs is None:
        return current_top[:K]

    # ---- Score Global Predictions ----
    all_predictions = []
    used_patterns = set()
    pred_global_dcit = dict()
    other_score = 1e-12
    if pred_global_pairs:
        for stoich_list, global_prob in pred_global_pairs:
            global_prob = float(global_prob)
            if stoich_list == 'other':
                other_score = max(global_prob / max(n, 1), 1e-12)
            elif stoich_list is not None and len(stoich_list) == n:
                pred_global_dcit[canonical_stoich(stoich_list)] = global_prob
        for stoich_list, global_prob in pred_global_pairs[:K]:  # take top-N global predictions
            if stoich_list is None or stoich_list == 'other':
                continue
            if len(stoich_list) != n or len(stoich_list) == 1:
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
            tuple_sorted_stoich = canonical_stoich(stoich)
            score = pred_global_dcit.get(tuple_sorted_stoich, other_score)
            final_score = combined_score(score, chain_log_prob)
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
        else:
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
    model_plan, grouped_targets = plan_model_selection(args, prediction_dict)
    # generate the embeddings
    sequenceFeatures = generate_input_sequences_embeddings(all_sequences, args.device)
    final_predictions = {}
    for model_dir, target_names in grouped_targets.items():
        group_prediction_dict = {
            target_name: prediction_dict[target_name]
            for target_name in target_names
        }
        trained_num_subunits = model_num_subunits(model_dir)
        inference_dataloader = prepare_stopred_input(
            group_prediction_dict,
            sequenceFeatures,
            trained_num_subunits=trained_num_subunits,
        )
        mean_result, label2idx, sto2idx = predict(model_dir, inference_dataloader, args.device)
        group_predictions = reformate_prediction(
            mean_result,
            group_prediction_dict,
            args.topk,
            args.alpha,
            label2idx,
            sto2idx,
        )
        for target_name, result in group_predictions.items():
            result['model_selection'] = model_plan[target_name]
            final_predictions[target_name] = result

    # save the prediction
    for target_name, result in final_predictions.items():
        with open(os.path.join(args.output_dir, f'{target_name}.json'), 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_root = Config.inference.get('model_root', os.path.join(root_dir, "models_collection"))
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to the input directory that contains the fasta files")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--topk", type=int, default=10, help="Number of top K stoichiometries to predict")
    parser.add_argument(
        "--train-release-date",
        type=str,
        default=None,
        help="Optional training release cutoff used to resolve release-specific model names. If omitted, use the latest available release-tagged model.",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=default_model_root,
        help="Root directory containing release-specific model directories",
    )
    parser.add_argument(
        "--model_dir",
        "--model-dir",
        type=str,
        default=None,
        help="Override default/heteromer model directory. If omitted, resolve from --model-root and --train-release-date.",
    )
    parser.add_argument(
        "--unk_model_dir",
        "--unk-model-dir",
        type=str,
        default=None,
        help="Override unknown-single model directory used when -unk is enabled.",
    )
    parser.add_argument(
        "-unk",
        "--unk-for-homomer",
        "--use-unk-for-homomer",
        action="store_true",
        dest="unk_for_homomer",
        help="Use the unknown-single model for single-entity targets. Heteromer targets always use the default release model.",
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--alpha', type=float, default=Config.inference.alpha, help='Alpha for the combined strategy')
    args = parser.parse_args()
    args.input_dir = os.path.realpath(args.input_dir)
    args.output_dir = os.path.realpath(args.output_dir)
    args.model_root = os.path.realpath(args.model_root)
    if args.model_dir:
        args.model_dir = os.path.realpath(args.model_dir)
    if args.unk_model_dir:
        args.unk_model_dir = os.path.realpath(args.unk_model_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
