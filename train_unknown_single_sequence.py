import argparse
import os
import pickle
from types import SimpleNamespace

from config import Config
from train_sto_net import main_crossval


def get_unknown_single_config():
    return Config.unknown_single_sequence


def create_parser():
    unknown_cfg = get_unknown_single_config()
    parser = argparse.ArgumentParser(
        description=(
            'Train a monomer-aware StoPred model for unknown single-sequence '
            'inputs. The dataset should already contain monomer examples and '
            'matching label-map JSON files.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset-pkl',
        default=os.path.join(unknown_cfg.dataset_dir, 'dataset.pkl'),
        help='Prepared dataset pickle with train_data, valid_data, test_data, and fold_data.',
    )
    parser.add_argument(
        '--maps-dir',
        default=unknown_cfg.dataset_dir,
        help='Directory containing sto2idx.json, count2label.json, and label2idx.json for this dataset.',
    )
    parser.add_argument(
        '--sequence-features',
        default=Config.data.sequenceFeaturesPath,
        help='Sequence feature pickle covering all dataset sequences.',
    )
    parser.add_argument(
        '--structure-features',
        default=Config.data.structureFeaturesPath,
        help='Structure feature pickle, only needed when --features includes structure.',
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=['sequence'],
        choices=['sequence', 'structure'],
        help='Feature types used by StoPred.',
    )
    parser.add_argument(
        '--num-subunits',
        type=int,
        default=None,
        help='Override Config.model.num_subunits when training a specialized model.',
    )
    parser.add_argument(
        '--local-class-weighting',
        choices=['none', 'inverse', 'inverse_sqrt'],
        default=unknown_cfg.local_class_weighting,
        help='Local copy-count class weighting. Default is the unknown-single-sequence setting.',
    )
    parser.add_argument(
        '--local-class-weight-max',
        type=float,
        default=unknown_cfg.local_class_weight_max,
        help='Maximum local class weight after normalization. Use <=0 to disable clipping.',
    )
    parser.add_argument(
        '--local-loss',
        choices=['ce', 'focal'],
        default=unknown_cfg.local_loss,
        help='Local copy-count loss. Weighted CE is the current default.',
    )
    parser.add_argument(
        '--local-focal-gamma',
        type=float,
        default=unknown_cfg.local_focal_gamma,
        help='Gamma used when --local-loss focal is selected.',
    )
    parser.add_argument(
        '--local-soft-f1-weight',
        type=float,
        default=unknown_cfg.local_soft_f1_weight,
        help='Optional local soft-F1 loss weight.',
    )
    parser.add_argument(
        '--local-soft-f1-mode',
        choices=['multiply', 'add'],
        default=unknown_cfg.local_soft_f1_mode,
        help='How to combine CE and soft-F1 when --local-soft-f1-weight > 0.',
    )
    parser.add_argument(
        '--allow-no-monomer',
        action='store_true',
        help='Do not fail if train_data contains no monomer examples.',
    )
    parser.add_argument('-o', '--output-dir', default=unknown_cfg.model_dir, help='Output model directory.')
    parser.add_argument('--device', default='cuda', help='Training device.')
    parser.add_argument('--seed', type=int, default=None, help='Optional training seed.')
    return parser


def load_pickle(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def count_train_groups(dataset):
    counts = {'monomer': 0, 'homomer': 0, 'heteromer': 0}
    for sample in dataset['train_data'].values():
        entity_count = {key: int(value) for key, value in sample['entity_count'].items()}
        total_count = sum(entity_count.values())
        if total_count == 1:
            counts['monomer'] += 1
        elif len(entity_count) == 1:
            counts['homomer'] += 1
        else:
            counts['heteromer'] += 1
    return counts


def validate_inputs(args):
    required_dataset_keys = {'train_data', 'valid_data', 'test_data', 'fold_data'}
    dataset = load_pickle(args.dataset_pkl)
    missing_keys = sorted(required_dataset_keys - set(dataset))
    if missing_keys:
        raise KeyError(f'Missing required dataset keys: {", ".join(missing_keys)}')

    train_groups = count_train_groups(dataset)
    if train_groups['monomer'] == 0 and not args.allow_no_monomer:
        raise ValueError(
            'train_data contains no monomer examples. Unknown-single-sequence '
            'training expects monomer examples; pass --allow-no-monomer to override.'
        )

    required_maps = ['sto2idx.json', 'count2label.json', 'label2idx.json']
    missing_maps = [
        filename for filename in required_maps
        if not os.path.exists(os.path.join(args.maps_dir, filename))
    ]
    if missing_maps:
        raise FileNotFoundError(f'Missing map files in --maps-dir: {", ".join(missing_maps)}')

    print(
        'train_data composition: '
        f'{train_groups["monomer"]} monomer, '
        f'{train_groups["homomer"]} homomer, '
        f'{train_groups["heteromer"]} heteromer'
    )


def to_train_args(args):
    return SimpleNamespace(
        features=args.features,
        data=os.path.abspath(args.dataset_pkl),
        sto2idx=os.path.abspath(os.path.join(args.maps_dir, 'sto2idx.json')),
        count2label=os.path.abspath(os.path.join(args.maps_dir, 'count2label.json')),
        label2idx=os.path.abspath(os.path.join(args.maps_dir, 'label2idx.json')),
        sequence_features=os.path.abspath(args.sequence_features),
        structure_features=os.path.abspath(args.structure_features),
        num_subunits=args.num_subunits,
        local_class_weighting=args.local_class_weighting,
        local_class_weight_max=args.local_class_weight_max,
        local_loss=args.local_loss,
        local_focal_gamma=args.local_focal_gamma,
        local_soft_f1_weight=args.local_soft_f1_weight,
        local_soft_f1_mode=args.local_soft_f1_mode,
        seed=args.seed,
        output_dir=os.path.abspath(args.output_dir),
        device=args.device,
    )


def main():
    parser = create_parser()
    args = parser.parse_args()
    args.dataset_pkl = os.path.abspath(args.dataset_pkl)
    args.maps_dir = os.path.abspath(args.maps_dir)
    args.sequence_features = os.path.abspath(args.sequence_features)
    args.structure_features = os.path.abspath(args.structure_features)
    args.output_dir = os.path.abspath(args.output_dir)
    validate_inputs(args)
    main_crossval(to_train_args(args))


if __name__ == '__main__':
    main()
