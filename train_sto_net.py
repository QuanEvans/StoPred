import os
import pickle
import argparse
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
    parser.add_argument('-seed', '--seed', type=int, default=None, help='overall seed, will override default seed in config')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory that will store model and test results', required=True)
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training, default is cuda')
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    return args

def load_pkl(path:str):
    return pickle.load(open(path, 'rb'))

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
    np.random.seed(Config.global_seed)
    torch.manual_seed(Config.global_seed)
    torch.cuda.manual_seed_all(Config.global_seed)
    # first load feature
    sequenceFeatures = None
    structureFeatures = None
    if 'sequence' in args.features:
        sequenceFeatures = load_pkl(Config.data.sequenceFeaturesPath)
    if 'structure' in args.features:
        structureFeatures = load_pkl(Config.data.structureFeaturesPath)
    All_dataset = load_pkl(args.data)

    trainDatasetsRaw = list(All_dataset['train_data'].values())
    trainDatasetsRaw += list(All_dataset['valid_data'].values())
    valDatasetsRaw = list(All_dataset['valid_data'].values())
    val_ids = set(All_dataset['valid_data'].keys())
    testDatasetsRaw = list(All_dataset['test_data'].values())
    test_ids = set(All_dataset['test_data'].keys())

    # Merge features
    trainDatasetsAll = merge_features(trainDatasetsRaw, sequenceFeatures, structureFeatures, Config)
    valDatasetsAll = merge_features(valDatasetsRaw, sequenceFeatures, structureFeatures, Config)
    testDatasets = merge_features(testDatasetsRaw, sequenceFeatures, structureFeatures, Config)
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
            feature_config_dict[feature] = {'dim': Config.model.embedding_dim[feature]}
        print(feature_config_dict)

        model_config = ml_collections.ConfigDict({
            'num_subunits': Config.model.num_subunits,
            'dropout': Config.model.dropout,
            'num_feature_layers': Config.model.num_feature_layers,
            'num_gnn_layers': Config.model.num_gnn_layers,
            'features': feature_config_dict,
            'hidden_dim': Config.model.hidden_dim,
            'sto2idx': json.load(open(Config.model.sto2idx)),
            'count2label': json.load(open(Config.model.count2label)),
            'label2idx': json.load(open(Config.model.label2idx)),
            'agg_methods': Config.model.agg_methods,
            'use_moe': Config.model.use_moe,
            'use_global_state': Config.model.use_global_state,
            'learning_rate': Config.model.learning_rate,
            'weight_local': Config.model.weight_local,
            'weight_global': Config.model.weight_global,
        })
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
    

