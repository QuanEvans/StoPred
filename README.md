# StoPred

StoPred is a deep learning method for predicting the stoichiometry of protein complexes. It combines embeddings from pretrained protein language models with a graph attention neural network to enable subunit-level reasoning for both homo- and hetero-oligomeric assemblies. By integrating local and global predictions, StoPred can infer stoichiometry directly from sequence or structure features without requiring prior knowledge of the complex composition or template structures, and the repository provides the full workflow for data preparation, model training, and inference on new complexes.

## Repository Layout
- `config.py`: Central configuration (data paths, cut-off dates, model hyperparameters, inference knobs).
- `prepare_data.py`: Data curation pipeline that builds dataset pickles and label mappings from processed mmCIF JSON.
- `train_sto_net.py`: Five-fold cross-validation training entry point for StoPredNet.
- `stopred_prediction.py`: Inference CLI that creates embeddings on the fly and writes JSON predictions.
- `utils/`: Feature extraction utilities (`ESMC_SequenceFeatureExtraction.py`), parsing helpers, and alignment scripts.
- `network/`: StoPredNet architecture and dataset wrappers.
- `external/`: Third-party assets, including the `download_pdb_mmcif.sh` helper and local ESM-C weights.
- `casp_example/`: Example FASTA inputs and expected JSON outputs for inference sanity checks.

## Requirements
- Linux environment with CUDA-capable GPUs for training and large-scale inference. CPU-only execution is supported for lightweight experimentation but will be significantly slower.
- Access to the wwPDB mmCIF archive and internet access (or cached copies) for feature model weights.
- Python 3.11 (the repo ships a `pyproject.toml` and `requirements.txt` capturing the tested dependency versions, including CUDA 12.8 builds of PyTorch).

## Installation

### Option 1: Online server or CodeOcean capsule

For users who do not want to install StoPred locally, we provide:

- StoPred web server: https://seq2fun.dcmb.med.umich.edu/StoPred/
- CodeOcean capsule: https://codeocean.com/capsule/7746930/tree

No local installation is required when using the web server or CodeOcean capsule.

### Option 2: Local installation with `uv` recommended

Create a local Python 3.11 environment and install dependencies using `uv`:

```bash
uv sync
source .venv/bin/activate
```

If `uv` is not installed:

```bash
pip install uv
uv sync
source .venv/bin/activate
```

The `pyproject.toml` file specifies the tested Python version and dependencies, including CUDA 12.8 PyTorch wheels.

### Option 3: Local installation with Conda/pip

```bash
conda create -n stopred python=3.11
conda activate stopred
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins GPU-enabled PyTorch packages. Adjust the CUDA wheels if you are targeting a different CUDA driver/toolkit combination.

### Typical installation time

Using the CodeOcean capsule or a prepared Singularity/Apptainer container, setup takes less than 1 minute because the environment is already configured.

For local installation with `uv` on a Linux workstation with CUDA-compatible drivers already available, environment setup is typically about 1 minute when packages are cached. A fresh installation may take longer depending on network speed.

## Data Preparation Workflow
1. **Download mmCIF files** (PDB archive + obsolete list):

   ```bash
   bash external/download_pdb_mmcif.sh /path/to/download
   ```

   The script depends on `aria2c` and `rsync`. It creates `pdb_mmcif/raw`, `pdb_mmcif/mmcif_files`, and stores `obsolete.dat` alongside the downloads.

2. **Parse mmCIF to JSON** that conforms to StoPred's schema:

   ```bash
   python parse_PDBmmcif.py /path/to/download/pdb_mmcif \
       --output_json raw_data/PDBmmcif.json \
       --obsolete
   ```

   The output path should match `Config.data.processed_PDBmmcif_path`.

3. **Tune configuration** in `config.py` if needed:
   - `Config.data.cut_off_date` and `test_cut_off_date` control the temporal split.
   - Update feature paths if you store embeddings outside of `Dataset/features`.

4. **Build training artifacts**:

   ```bash
   python prepare_data.py
   ```

   The script writes to `Dataset/` by default:
   - `StoPredDataset.pkl`, : dataset splits.
   - `all_sequences.fasta`: aggregated sequences for feature extraction.
   - `count2label.json`, `label2idx.json`, `sto2idx.json`: label dictionaries consumed during training/inference.

5. **Generate sequence embeddings** using ESM-C:

   ```bash
   mkdir -p Dataset/features
   python utils/ESMC_SequenceFeatureExtraction.py Dataset/all_sequences.fasta \
       Dataset/features/sequenceFeatures.pkl --gpu
   ```

   Place the ESM-C weights (`esmc_600m_2024_12_v0.pth`) under `external/` or let the script fetch them from Hugging Face on first run. Use `utils/ESMC_SequenceFeatureExtraction_multigpu.py` for distributed extraction.

After these steps, `train_sto_net.py` can locate all artifacts using the default configuration.

## Training StoPredNet
Launch cross-validation training with configurable features and output directory:

```bash
python train_sto_net.py \
    --features sequence \
    --data Dataset/StoPredDataset.pkl \
    --output_dir runs/stopred_default \
    --device cuda
```

- Use `--seed` to override `Config.global_seed` and ensure reproducibility.
- `--output_dir` collects one checkpoint (`model_fold{k}.pkl`) per fold plus CSV reports.

## Inference on New Complexes

`stopred_prediction.py` consumes a directory of FASTA files and exports JSON predictions:

```bash
python stopred_prediction.py \
    casp_example/input_fasta \
    casp_example/output_dir_pred \
    --model_dir models_collection/default \
    --topk 10 \
    --alpha 0.7 \
    --device cuda
```

If using `uv`, the same command can be run as:

```bash
uv run python stopred_prediction.py \
    casp_example/input_fasta \
    casp_example/output_dir_pred \
    --model_dir models_collection/default \
    --topk 10 \
    --alpha 0.7 \
    --device cuda
```

- File names must match target IDs, for example `TargetID.fasta`.
- Sequence headers should use target-chain labels separated by a dash.
- The script creates ESM-C embeddings on demand; ensure the ESM-C weights remain accessible as described in the data section.
- Output JSON files contain:
  - `chain_level_predictions`: per-chain copy-number distributions.
  - `global_predictions`: per-chain stoichiometry hypotheses aggregated across folds.
  - `topk_predictions`: highest-probability stoichiometry assignments with confidence scores.

Compare outputs against `casp_example/output_dir` for the expected output structure.

### Expected demo runtime

The included CASP example demo is expected to finish in less than 1 minute on a CUDA-capable GPU after the model files and ESM-C weights are available. 

## Container Image
- `container/singularity.def` document tested container images (CUDA 12.9). Adapt them to rebuild deterministic environments.

## License
StoPred is released under the MIT License (see `LICENSE`).

## Troubleshooting
- Verify that feature pickle paths in `config.py` align with actual locations before training or inference.
- GPU memory usage scales with the number of subunits; adjust `Config.model.num_subunits` or batch size if you encounter OOM errors.
