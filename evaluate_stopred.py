import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from config import Config
from network.sto_net import StoPredNet
from network.utils import (
    StoDataset,
    list2tagAlpha,
    merge_features,
    reformate_global_pred,
    top_k_stoichiometries_combined,
)


def create_parser():
    parser = argparse.ArgumentParser(description='Evaluate StoPred top-N stoichiometry predictions')
    parser.add_argument('-model', '--model_path', type=str, required=True, help='Path to a model file or fold model directory')
    parser.add_argument(
        '--dataset-pkl',
        type=str,
        default=os.path.join(Config.data.Dataset, 'StoPredDataset.pkl'),
        help='Path to the dataset pickle containing named splits',
    )
    parser.add_argument('--split', type=str, default='test_data', help='Dataset split key inside --dataset-pkl')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Directory for evaluation output files')
    parser.add_argument(
        '--target-scope',
        choices=['non_monomer', 'single_entity', 'all'],
        default='non_monomer',
        help='Which targets to evaluate. Use single_entity for the monomer-aware homomer model.',
    )
    parser.add_argument('--top-n', type=int, default=10, help='Largest N for top-N accuracy')
    parser.add_argument('-a', '--alpha', type=float, default=Config.inference.alpha, help='Global/local score mixing alpha')
    parser.add_argument(
        '--min-support',
        type=int,
        default=Config.inference.min_support,
        help='Merge true classes with support below this count into Other for classification report',
    )
    parser.add_argument(
        '--allow-extra-subunits',
        action='store_true',
        help='Keep targets with more unique subunits than the model was trained with by padding eval tensors. Use for legacy CASP reproduction.',
    )
    parser.add_argument('--sequence-features', type=str, default=Config.data.sequenceFeaturesPath, help='sequence feature pickle path')
    parser.add_argument('--structure-features', type=str, default=Config.data.structureFeaturesPath, help='structure feature pickle path')
    parser.add_argument('--batch-size', type=int, default=Config.data.batch_size, help='Evaluation batch size')
    parser.add_argument('--num-folds', type=int, default=Config.data.num_folds, help='Expected number of fold models when --model_path is a directory')
    parser.add_argument('--device', type=str, default=None, help='Device, default cuda when available else cpu')
    args = parser.parse_args()
    args.dataset_pkl = os.path.abspath(args.dataset_pkl)
    args.output_dir = os.path.abspath(args.output_dir)
    args.model_path = os.path.abspath(args.model_path)
    args.sequence_features = os.path.abspath(args.sequence_features)
    args.structure_features = os.path.abspath(args.structure_features)
    args.top_n = max(1, args.top_n)
    args.min_support = max(1, args.min_support)
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if Path(args.output_dir).suffix.lower() in {'.pkl', '.csv', '.json', '.tsv'}:
        parser.error('--output-dir must be a directory, not a file path')
    return args


def load_pkl(path: str):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def evaluation_output_paths(output_dir: str) -> dict[str, str]:
    return {
        'mean_predictions': os.path.join(output_dir, 'mean_predictions.pkl'),
        'all_fold_results': os.path.join(output_dir, 'all_fold_results.pkl'),
        'topn_accuracy': os.path.join(output_dir, 'topn_accuracy.csv'),
        'classification_report': os.path.join(output_dir, 'classification_report.csv'),
        'per_target_predictions': os.path.join(output_dir, 'per_target_predictions.csv'),
    }


def validate_output_paths(output_dir: str, output_paths: dict[str, str], input_paths: list[str]) -> None:
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise NotADirectoryError(f'--output-dir points to an existing file: {output_dir}')

    input_realpaths = {
        os.path.realpath(path)
        for path in input_paths
        if path and os.path.exists(path)
    }
    collisions = [
        output_path
        for output_path in output_paths.values()
        if os.path.realpath(output_path) in input_realpaths
    ]
    if collisions:
        raise ValueError(
            'Refusing to overwrite an input file: '
            + ', '.join(collisions)
        )


def resolve_model_paths(model_path: str, num_folds: int) -> list[str]:
    if os.path.isdir(model_path):
        fold_paths = [
            os.path.join(model_path, f'model_fold{fold}.pkl')
            for fold in range(num_folds)
        ]
        existing_fold_paths = [path for path in fold_paths if os.path.exists(path)]
        if existing_fold_paths:
            missing_fold_paths = [path for path in fold_paths if not os.path.exists(path)]
            if missing_fold_paths:
                raise FileNotFoundError(
                    f'Found only {len(existing_fold_paths)} of {num_folds} expected fold models in {model_path}. '
                    f'Missing: {", ".join(os.path.basename(path) for path in missing_fold_paths)}'
                )
            return fold_paths
        all_pkls = sorted(
            os.path.join(model_path, filename)
            for filename in os.listdir(model_path)
            if filename.endswith('.pkl')
        )
        if all_pkls:
            return all_pkls
        raise FileNotFoundError(f'No .pkl models found in {model_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return [model_path]


def merge_config_for_model(model: StoPredNet):
    merge_config = copy.deepcopy(Config)
    merge_config.model.num_subunits = int(model.num_subunits)
    with merge_config.model.ignore_type():
        merge_config.model.sto2idx = model.config['sto2idx']
        merge_config.model.count2label = model.config['count2label']
        merge_config.model.label2idx = model.config['label2idx']
    return merge_config


def load_features_for_model(model: StoPredNet, args):
    feature_names = set(model.features)
    sequence_features = None
    structure_features = None
    if 'sequence' in feature_names:
        sequence_features = load_pkl(args.sequence_features)
    if 'structure' in feature_names:
        structure_features = load_pkl(args.structure_features)
        structure_features = {
            key.split('-')[1] if isinstance(key, str) and key.startswith('AF') else key: value
            for key, value in structure_features.items()
        }
    return sequence_features, structure_features


def load_split(args):
    all_dataset = load_pkl(args.dataset_pkl)
    split_name = args.split
    if split_name not in all_dataset:
        raise KeyError(f'{split_name} not found in {args.dataset_pkl}')

    split_data = all_dataset[split_name]
    if isinstance(split_data, dict):
        return list(split_data.values())
    return list(split_data)


def keep_sample_for_scope(sample: dict, target_scope: str) -> bool:
    entity_count = {key: int(value) for key, value in sample['entity_count'].items()}
    total_count = sum(entity_count.values())
    if target_scope == 'all':
        return True
    if target_scope == 'single_entity':
        return len(entity_count) == 1
    return total_count != 1


def prepare_dataset(raw_samples: list[dict], sequence_features: dict, structure_features: dict, merge_config, args):
    raw_samples = [
        sample for sample in raw_samples
        if keep_sample_for_scope(sample, args.target_scope)
    ]
    if raw_samples and args.allow_extra_subunits:
        max_entities = max(len(sample['entity_count']) for sample in raw_samples)
        merge_config.model.num_subunits = max(
            int(merge_config.model.num_subunits),
            int(max_entities),
        )
    merged_samples = merge_features(
        raw_samples,
        sequence_features=sequence_features,
        structure_features=structure_features,
        config=merge_config,
    )
    prepared_samples = []
    for sample in merged_samples:
        if not keep_sample_for_scope(sample, args.target_scope):
            continue
        prepared_samples.append(sample)
    return prepared_samples


def predict_folds(model_paths: list[str], dataloader: DataLoader, device: str):
    all_fold_results = {}
    label2idx = None
    sto2idx = None
    for fold, model_path in enumerate(model_paths):
        model = StoPredNet.load_from_pkl(model_path)
        if label2idx is None:
            label2idx = model.config['label2idx']
        if sto2idx is None:
            sto2idx = model.config['sto2idx']
        model = model.to(device)
        model.eval()
        all_fold_results[fold] = {}
        with torch.no_grad():
            for batch in dataloader:
                batch_device = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                y_hats, y_hats_global = model(batch_device)
                y_hats = F.softmax(y_hats, dim=2).cpu().detach().numpy()
                y_hats_global = F.softmax(y_hats_global, dim=2).cpu().detach().numpy()
                ys = batch_device['labels'].cpu().detach().numpy()
                for unique_id, y, y_hat, y_global in zip(
                    batch_device['unique_id'],
                    ys,
                    y_hats,
                    y_hats_global,
                ):
                    all_fold_results[fold][unique_id] = {
                        'y': y,
                        'y_hat': y_hat,
                        'global': y_global,
                    }

    mean_results = {}
    for fold_results in all_fold_results.values():
        for unique_id, result in fold_results.items():
            if unique_id not in mean_results:
                mean_results[unique_id] = {
                    'y': result['y'],
                    'y_hat': result['y_hat'].copy(),
                    'global': result['global'].copy(),
                }
            else:
                mean_results[unique_id]['y_hat'] += result['y_hat']
                mean_results[unique_id]['global'] += result['global']

    num_models = len(all_fold_results)
    for result in mean_results.values():
        result['y_hat'] /= num_models
        result['global'] /= num_models
    return all_fold_results, mean_results, label2idx, sto2idx


def ordered_labels_for_report(gt_counts: list[int], pred_counts: list[int]):
    pairs = list(zip(gt_counts, pred_counts))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [item[0] for item in pairs], [item[1] for item in pairs]


def sample_group(entity_count: dict) -> str:
    entity_count = {key: int(value) for key, value in entity_count.items()}
    if len(entity_count) == 1 and sum(entity_count.values()) == 1:
        return 'monomer'
    if len(entity_count) == 1:
        return 'homomer'
    return 'heteromer'


def ranked_predictions_for_sample(result: dict, sample: dict, idx2label: dict, idx2sto: dict, top_n: int, alpha: float):
    entity_count = sample['entity_count']
    num_subunits = len(entity_count)
    y_pred_slice = result['y_hat'][:num_subunits]
    if num_subunits == 1:
        pred_global_pairs = None
    else:
        pred_global_pairs = reformate_global_pred(result['global'], num_subunits, idx2sto)
    return top_k_stoichiometries_combined(
        y_pred_slice,
        top_n,
        idx2label,
        pred_global_pairs,
        alpha=alpha,
    )


def map_rare_classes(test_labels: list[str], test_preds: list[str], min_support: int):
    label_counts = pd.Series(test_labels).value_counts().to_dict()
    rare_labels = {label for label, count in label_counts.items() if count < min_support}
    mapped_true = []
    mapped_pred = []
    for true_label, pred_label in zip(test_labels, test_preds):
        if true_label in rare_labels:
            mapped_true.append('Other')
            mapped_pred.append('Other' if pred_label == true_label else 'WrongOther')
        else:
            mapped_true.append(true_label)
            mapped_pred.append('Other' if pred_label in rare_labels or pred_label not in label_counts else pred_label)
    return mapped_true, mapped_pred


def evaluate_rankings(mean_results: dict, sample_by_id: dict, idx2label: dict, idx2sto: dict, args):
    groups = ['all', 'monomer', 'homomer', 'heteromer']
    counts = {group: 0 for group in groups}
    hits = {group: np.zeros(args.top_n, dtype=np.int64) for group in groups}
    test_labels = []
    test_preds = []
    prediction_rows = []

    for unique_id, result in mean_results.items():
        sample = sample_by_id[unique_id]
        gt_counts = [int(count) for count in sample['entity_count'].values()]
        group = sample_group(sample['entity_count'])
        ranked_predictions = ranked_predictions_for_sample(
            result,
            sample,
            idx2label,
            idx2sto,
            args.top_n,
            args.alpha,
        )

        true_ordered = None
        pred_labels = []
        is_hit_by_rank = []
        for _, pred_counts_raw in ranked_predictions[:args.top_n]:
            pred_counts = [int(count) for count in pred_counts_raw]
            true_ordered, pred_ordered = ordered_labels_for_report(gt_counts, pred_counts)
            pred_labels.append(list2tagAlpha(pred_ordered))
            is_hit_by_rank.append(pred_ordered == true_ordered)

        if true_ordered is None:
            continue

        cumulative_hit = False
        for rank in range(args.top_n):
            if rank < len(is_hit_by_rank) and is_hit_by_rank[rank]:
                cumulative_hit = True
            if cumulative_hit:
                hits['all'][rank] += 1
                hits[group][rank] += 1

        counts['all'] += 1
        counts[group] += 1
        true_label = list2tagAlpha(true_ordered)
        top1_label = pred_labels[0] if pred_labels else ''
        test_labels.append(true_label)
        test_preds.append(top1_label)
        prediction_rows.append({
            'unique_id': unique_id,
            'group': group,
            'true_label': true_label,
            'top1_label': top1_label,
            'top1_correct': bool(is_hit_by_rank[0]) if is_hit_by_rank else False,
            f'top{args.top_n}_correct': any(is_hit_by_rank),
            'top_predictions': ';'.join(pred_labels),
        })

    topn_rows = []
    for group in groups:
        row = {
            'group': group,
            'n_samples': counts[group],
        }
        for rank in range(args.top_n):
            column = f'top{rank + 1}_acc'
            row[column] = float(hits[group][rank] / counts[group]) if counts[group] else np.nan
        topn_rows.append(row)

    mapped_true, mapped_pred = map_rare_classes(test_labels, test_preds, args.min_support)
    report_labels = sorted(set(mapped_true))
    report_df = pd.DataFrame(
        classification_report(
            np.array(mapped_true),
            np.array(mapped_pred),
            labels=report_labels,
            zero_division=0,
            output_dict=True,
        )
    ).T
    report_df.loc['accuracy'] = [
        accuracy_score(np.array(test_labels), np.array(test_preds)),
        '',
        '',
        len(test_labels),
    ]
    report_df = report_df.sort_values(by='support', ascending=False)

    return pd.DataFrame(topn_rows), report_df, pd.DataFrame(prediction_rows)


def main(args):
    model_paths = resolve_model_paths(args.model_path, args.num_folds)
    first_model = StoPredNet.load_from_pkl(model_paths[0])
    Config.model.num_subunits = int(first_model.num_subunits)
    sequence_features, structure_features = load_features_for_model(first_model, args)
    merge_config = merge_config_for_model(first_model)

    output_paths = evaluation_output_paths(args.output_dir)
    input_paths = [
        args.dataset_pkl,
        args.sequence_features if sequence_features is not None else None,
        args.structure_features if structure_features is not None else None,
        *model_paths,
    ]
    validate_output_paths(args.output_dir, output_paths, input_paths)

    print(f'Dataset pickle: {args.dataset_pkl}')
    print(f'Split: {args.split}')
    print(f'Target scope: {args.target_scope}')
    print(f'Models: {len(model_paths)}')
    print(f'Device: {args.device}')
    print(f'Alpha: {args.alpha}')

    raw_samples = load_split(args)
    prepared_samples = prepare_dataset(raw_samples, sequence_features, structure_features, merge_config, args)
    sample_by_id = {sample['unique_id']: sample for sample in prepared_samples}
    print(f'Evaluation samples: {len(prepared_samples)}')

    test_dataset = StoDataset(prepared_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_fold_results, mean_results, label2idx, sto2idx = predict_folds(
        model_paths,
        test_loader,
        args.device,
    )
    idx2label = {int(value): int(key) for key, value in label2idx.items()}
    idx2sto = {value: key for key, value in sto2idx.items()}

    topn_df, report_df, predictions_df = evaluate_rankings(
        mean_results,
        sample_by_id,
        idx2label,
        idx2sto,
        args,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_paths['mean_predictions'], 'wb') as output_file:
        pickle.dump(mean_results, output_file)
    with open(output_paths['all_fold_results'], 'wb') as output_file:
        pickle.dump(all_fold_results, output_file)

    topn_df.to_csv(output_paths['topn_accuracy'], index=False)
    report_df.to_csv(output_paths['classification_report'])
    predictions_df.to_csv(output_paths['per_target_predictions'], index=False)

    print(topn_df)
    print(report_df)
    print(f"Saved mean predictions to: {output_paths['mean_predictions']}")
    print(f"Saved top-N accuracy to: {output_paths['topn_accuracy']}")
    print(f"Saved classification report to: {output_paths['classification_report']}")
    print(f"Saved per-target predictions to: {output_paths['per_target_predictions']}")


if __name__ == '__main__':
    main(create_parser())
