# Release Notes

## StoPred 1.0.0

StoPred 1.0.0 is a major workflow update for release-aware inference, updated PDB parsing, and monomer-aware single-sequence prediction.

- PDB mmCIF parsing supports the native wwPDB `divided/` and `obsolete/` layout, compressed `.cif.gz` files, parser cache/resume metadata, and checkpointed parsing through `parse_PDBmmcif_gz.py`.
- Exact duplicate PDB entities are merged during parsing when their sequences and database identifiers are identical, preventing artificial stoichiometry splits from duplicated entity records.
- The legacy flat uncompressed parser `parse_PDBmmcif.py` has been removed.
- `external/download_pdb_mmcif.sh` downloads the compressed native wwPDB layout, and `external/download_esmc_600m.sh` downloads the ESM-C 600M weights expected by StoPred.
- Inference auto-resolves release-specific model directories under `models_collection/`. A training cutoff date can be supplied to pin a release, and `-unk` enables the unknown-single model for single-entry FASTA targets while heteromer targets always use the default release model.
- Global/local score combination uses the global `other` probability divided by the number of input entities as the fallback score for out-of-vocabulary global stoichiometries.
- `evaluate_stopred.py` reports top-N accuracy by target group and writes per-class reports using the manuscript-style minimum-support mapping to `Other`.
- Training supports local copy-count class weights, focal loss, and optional soft-F1 loss for imbalance experiments.
- `prepare_unknown_single_sequence_data.py` and `train_unknown_single_sequence.py` provide the monomer-aware unknown-single-sequence workflow.
