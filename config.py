import os
from os.path import join
import ml_collections
import json
import pickle

root_dir = os.path.dirname(os.path.abspath(__file__))
config_dict = {
    "global_seed": 1354,  # 4227
    "data": {
        "cut_off_date": "2023-10-01",
        "test_cut_off_date": "2024-05-01",
        "Dataset": join(root_dir, "Dataset"),
        "processed_PDBmmcif_path": join(root_dir, "raw_data", "PDBmmcif.json"),
        "casp16_benchmark_path": join(root_dir, "raw_data", "casp16_benchmark.json"),
        "minimum_seq_len": 20,
        "minimum_sample_count": 50,  # number of proteins with count n in the dataset, should be greater than this number default 50
        "minimum_sto_count": 10,  # number of proteins with count n in the dataset, should be greater than this number default 50
        "num_folds": 5,
        "batch_size": 1024,
        "sequenceFeaturesPath": join(root_dir, "Dataset", "features", "sequenceFeatures.pkl"),
        "structureFeaturesPath": join(root_dir, "Dataset", "features", "structureFeatures.pkl"),
        "structureFeaturesResPath": join(root_dir, "Dataset", "features", "structureFeaturesRes.pkl"),
    },
    "model": {
        "num_subunits": 6,
        "embedding_dim": {
            "sequence": 1152 * 3, # last 3 hidden layers from ESMC
            "structure": 1536, # ESM3-650m
            "structureRes": 384,
        },
        "num_feature_layers": 3,
        "num_gnn_layers": 3,
        "num_heads": 6,
        "dropout": 0.2,
        "hidden_dim": 258*2,
        "agg_methods": "gat",
        "use_moe": True,
        "use_global_state": True,
        "sto2idx": join(root_dir, "Dataset", "sto2idx.json"),
        "count2label": join(root_dir, "Dataset", "count2label.json"),
        "label2idx": join(root_dir, "Dataset", "label2idx.json"),

        "learning_rate": 0.0005,
        "weight_local": 0.5,
        "weight_global": 0.5,
    },
    "train": {
        "min_epochs": 10,
        "max_epochs": 100,
    },
    "inference": {
        "alpha": 0.7,
        "min_support": 1,
    }
}
Config = ml_collections.ConfigDict(config_dict)