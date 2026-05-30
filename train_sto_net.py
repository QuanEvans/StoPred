import os
import pickle
import argparse
import copy
from config import Config, config_dict
from network.utils import merge_features, StoDataset
from network.sto_net import StoPredNet
import ml_collections
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
from tqdm import tqdm

from network.utils import get_stopred_result_report

project_root = os.path.dirname(os.path.abspath(__file__))

def create_parser():
    parser = argparse.ArgumentParser(description='Train StoNet')
    parser.add_argument('-features', '--features', type=str, nargs='+', default=['sequence'], help='Features to use', choices=['sequence', 'structure'])
    parser.add_argument('-data', '--data', type=str, help='dataset pkl path', default=os.path.join(project_root, 'Dataset', 'StoPredDataset.pkl'))
    parser.add_argument('--sto2idx', type=str, default=Config.model.sto2idx, help='stoichiometry vocabulary JSON path')
    parser.add_argument('--count2label', type=str, default=Config.model.count2label, help='copy-count-to-label JSON path')
    parser.add_argument('--label2idx', type=str, default=Config.model.label2idx, help='copy-count label vocabulary JSON path')
    parser.add_argument('--sequence-features', type=str, default=Config.data.sequenceFeaturesPath, help='sequence feature pickle path')
    parser.add_argument('--structure-features', type=str, default=Config.data.structureFeaturesPath, help='structure feature pickle path')
    parser.add_argument('--num-subunits', type=int, default=None, help='override Config.model.num_subunits for specialized models')
    parser.add_argument(
        '--local-class-weighting',
        choices=['none', 'inverse', 'inverse_sqrt'],
        default='none',
        help='Apply copy-count class weights to the local cross-entropy loss.',
    )
    parser.add_argument(
        '--local-class-weight-max',
        type=float,
        default=5.0,
        help='Maximum local class weight after normalization. Use <=0 to disable clipping.',
    )
    parser.add_argument(
        '--local-loss',
        choices=['ce', 'focal'],
        default='ce',
        help='Local copy-count loss. Focal loss uses local class weights as alpha_t.',
    )
    parser.add_argument(
        '--local-focal-gamma',
        type=float,
        default=2.0,
        help='Gamma for local focal loss.',
    )
    parser.add_argument(
        '--local-soft-f1-weight',
        type=float,
        default=0.0,
        help='Weight for the local soft-F1 loss term over classes present in the batch.',
    )
    parser.add_argument(
        '--local-soft-f1-mode',
        choices=['multiply', 'add'],
        default='multiply',
        help='Combine local CE and soft-F1 loss as CE * (1 + weight * softF1Loss) or CE + weight * softF1Loss.',
    )
    parser.add_argument('-seed', '--seed', type=int, default=None, help='overall seed, will override default seed in config')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory that will store model and test results', required=True)
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training, default is cuda')
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.data = os.path.abspath(args.data)
    args.sto2idx = os.path.abspath(args.sto2idx)
    args.count2label = os.path.abspath(args.count2label)
    args.label2idx = os.path.abspath(args.label2idx)
    args.sequence_features = os.path.abspath(args.sequence_features)
    args.structure_features = os.path.abspath(args.structure_features)
    return args

def load_pkl(path:str):
    return pickle.load(open(path, 'rb'))

def load_json(path: str):
    with open(path) as input_file:
        return json.load(input_file)

def make_training_config(args):
    cfg = copy.deepcopy(Config)
    if args.num_subunits is not None:
        cfg.model.num_subunits = args.num_subunits
    cfg.model.sto2idx = args.sto2idx
    cfg.model.count2label = args.count2label
    cfg.model.label2idx = args.label2idx
    cfg.data.sequenceFeaturesPath = args.sequence_features
    cfg.data.structureFeaturesPath = args.structure_features
    return cfg

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out

def local_label_index_for_count(count, count2label, label2idx):
    count_key = str(count)
    if count_key in count2label:
        label = str(count2label[count_key])
    else:
        label = '-1'
    return int(label2idx[label])

def compute_local_class_weights(raw_samples, count2label, label2idx, scheme='none', max_weight=5.0):
    if scheme == 'none':
        return None
    counts = np.zeros(len(label2idx), dtype=np.float64)
    for sample in raw_samples:
        entity_count = sample.get('entity_count', {})
        for count in entity_count.values():
            idx = local_label_index_for_count(count, count2label, label2idx)
            counts[idx] += 1.0
    observed = counts > 0
    weights = np.ones(len(label2idx), dtype=np.float64)
    if not np.any(observed):
        return weights.tolist()
    total = counts[observed].sum()
    n_observed = observed.sum()
    balanced = total / (n_observed * counts[observed])
    if scheme == 'inverse_sqrt':
        balanced = np.sqrt(balanced)
    elif scheme != 'inverse':
        raise ValueError(f'Unsupported local class weighting: {scheme}')
    weights[observed] = balanced

    # Keep the average loss scale close to the unweighted loss.
    weighted_mean = float((counts[observed] * weights[observed]).sum() / total)
    if weighted_mean > 0:
        weights[observed] /= weighted_mean
    if max_weight and max_weight > 0:
        weights[observed] = np.minimum(weights[observed], float(max_weight))
    return weights.astype(float).tolist()

def main_crossval(args):
    # create output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    # check if cuda is available
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, please check your setup.')
    
    
    # set seed
    if args.seed is not None:
        Config.global_seed = args.seed
    train_config = make_training_config(args)
    np.random.seed(Config.global_seed)
    torch.manual_seed(Config.global_seed)
    torch.cuda.manual_seed_all(Config.global_seed)
    # first load feature
    sequenceFeatures = None
    structureFeatures = None
    if 'sequence' in args.features:
        sequenceFeatures = load_pkl(train_config.data.sequenceFeaturesPath)
    if 'structure' in args.features:
        structureFeatures = load_pkl(train_config.data.structureFeaturesPath)
    All_dataset = load_pkl(args.data)

    trainDatasetsRaw = list(All_dataset['train_data'].values())
    trainDatasetsRaw += list(All_dataset['valid_data'].values())
    valDatasetsRaw = list(All_dataset['valid_data'].values())
    val_ids = set(All_dataset['valid_data'].keys())
    testDatasetsRaw = list(All_dataset['test_data'].values())
    test_ids = set(All_dataset['test_data'].keys())

    # Merge features
    trainDatasetsAll = merge_features(trainDatasetsRaw, sequenceFeatures, structureFeatures, train_config)
    valDatasetsAll = merge_features(valDatasetsRaw, sequenceFeatures, structureFeatures, train_config)
    testDatasets = merge_features(testDatasetsRaw, sequenceFeatures, structureFeatures, train_config)
    allDatasets = trainDatasetsAll + valDatasetsAll + testDatasets

    test_dataset = StoDataset(testDatasets)
    test_loader = DataLoader(test_dataset, batch_size=Config.data.batch_size, shuffle=False)

    final_val_dataset = StoDataset(valDatasetsAll)
    final_val_loader = DataLoader(final_val_dataset, batch_size=Config.data.batch_size, shuffle=False)

    # 5-fold cross-validation
    fold_datasets = []
    fold_data = All_dataset['fold_data']  # List of dict containing train and val IDs

    # Reference by IDs, avoid copying data
    for fold_dict in fold_data:
        train_ids = set(fold_dict['train'])
        fold_val_ids = set(fold_dict['valid'])

        fold_train_indices = [idx for idx, data in enumerate(allDatasets) if data['unique_id'] in train_ids]
        fold_val_indices = [idx for idx, data in enumerate(allDatasets) if data['unique_id'] in fold_val_ids]

        fold_train_dataset = Subset(allDatasets, fold_train_indices)
        fold_val_dataset = Subset(allDatasets, fold_val_indices)

        fold_datasets.append((fold_train_dataset, fold_val_dataset))

    # Loop through the folds and train
    for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f'Fold {fold}')
        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Val dataset size: {len(val_dataset)}')
        base_ds = val_dataset.dataset  
        idxs    = val_dataset.indices     
        val_ids = { base_ds[i]['unique_id'] for i in idxs }
        # Create datasets for the current fold
        train_dataset = StoDataset(train_dataset)
        val_dataset = StoDataset(val_dataset)
        val_dataset_raw = {
            k: v for k, v in All_dataset['train_data'].items() if k in val_ids
        }

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=Config.data.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.data.batch_size, shuffle=False)

        # Check sample batch shapes
        for batch in val_loader:
            print(f'Label shape: {batch["labels"].shape}')
            break
        
        feature_config_dict = dict()
        for feature in args.features:
            feature_config_dict[feature] = {'dim': train_config.model.embedding_dim[feature]}
        print(feature_config_dict)

        model_config = ml_collections.ConfigDict({
            'num_subunits': train_config.model.num_subunits,
            'dropout': train_config.model.dropout,
            'num_feature_layers': train_config.model.num_feature_layers,
            'num_gnn_layers': train_config.model.num_gnn_layers,
            'features': feature_config_dict,
            'hidden_dim': train_config.model.hidden_dim,
            'sto2idx': load_json(train_config.model.sto2idx),
            'count2label': load_json(train_config.model.count2label),
            'label2idx': load_json(train_config.model.label2idx),
            'agg_methods': train_config.model.agg_methods,
            'use_moe': train_config.model.use_moe,
            'use_global_state': train_config.model.use_global_state,
            'learning_rate': train_config.model.learning_rate,
            'weight_local': train_config.model.weight_local,
            'weight_global': train_config.model.weight_global,
            'local_loss_type': args.local_loss,
            'local_focal_gamma': args.local_focal_gamma,
            'local_soft_f1_weight': args.local_soft_f1_weight,
            'local_soft_f1_mode': args.local_soft_f1_mode,
        })
        local_class_weights = compute_local_class_weights(
            trainDatasetsRaw,
            model_config['count2label'],
            model_config['label2idx'],
            scheme=args.local_class_weighting,
            max_weight=args.local_class_weight_max,
        )
        if local_class_weights is not None:
            model_config['local_class_weights'] = local_class_weights
            idx2label = {int(value): key for key, value in model_config['label2idx'].items()}
            class_weight_rows = [
                {
                    'label': idx2label[idx],
                    'weight': float(weight),
                }
                for idx, weight in enumerate(local_class_weights)
            ]
            class_weight_path = os.path.join(args.output_dir, 'local_class_weights.csv')
            pd.DataFrame(class_weight_rows).to_csv(class_weight_path, index=False)
            print(f'Local class weights saved to {class_weight_path}')
        model = StoPredNet(model_config)
        model = model.to(device)
        monitor = 'val_loss'
        mode = 'min'
        early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=0.0001,
            patience=5,
            verbose=True,
            mode=mode
        )
        trainer = Trainer(
            min_epochs=Config.train.min_epochs,
            max_epochs=Config.train.max_epochs,
            callbacks=[early_stop_callback],# checkpoint_callback],
            logger=False,
            enable_checkpointing=False,
        )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        model_pkl_path = os.path.join(args.output_dir, f'model_fold{fold}.pkl')
        if not os.path.exists(model_pkl_path):
            trainer.fit(model, train_loader, val_loader)
            state_dict = model.state_dict()
            model_pkl = {
                'state_dict': state_dict,
                'config': model_config,
            }
            model_pkl_path = os.path.join(args.output_dir, f'model_fold{fold}.pkl')
            torch.save(model_pkl, model_pkl_path)
            print(f'Model saved to {model_pkl_path}')
        else:
            del train_dataset
            model_pkl = torch.load(model_pkl_path, weights_only=False)
            model.load_state_dict(model_pkl['state_dict'])
        # evaluate the model
        model.to(device)
        model.eval()
        results = dict()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch_device = to_device(batch, device)
                ys = batch_device['labels']
                unique_ids = batch_device['unique_id']
                y_hats, y_hats_global = model(batch_device)
                # softmax
                y_hats = F.softmax(y_hats, dim=2)
                y_hats_global = F.softmax(y_hats_global, dim=2)
                y_hats = y_hats.cpu().detach().numpy()
                y_hats_global = y_hats_global.cpu().detach().numpy()
                ys = ys.cpu().detach().numpy()

                for unique_id, y, y_hat, y_global in zip(unique_ids, ys, y_hats, y_hats_global):
                    results[unique_id] = {
                        'y': y,
                        'y_hat': y_hat,
                        'global': y_global,
                    }
        # get the result report
        idx2label = {v: k for k, v in model_config['label2idx'].items()}
        idx2sto = {v: k for k, v in model_config['sto2idx'].items()}
        result_report, _ = get_stopred_result_report(results, val_dataset_raw, idx2label, idx2sto, alpha=Config.inference.alpha, min_support=Config.inference.min_support)
        print(result_report)
        # save the result report
        result_report_path = os.path.join(args.output_dir, f'result_report_fold{fold}.csv')
        result_report.to_csv(result_report_path)

if __name__ == '__main__':
    args = create_parser()
    main_crossval(args)
    
