# StoPred Model Collection

This directory is the local runtime location for release checkpoints and
evaluation outputs. Large model and generated evaluation files are not tracked
in git.

Expected local layout:

```text
models_collection/
  default_20231001/
    model_fold0.pkl ... model_fold4.pkl
  default_20260101/
    model_fold0.pkl ... model_fold4.pkl
  unk_single_20260101/
    model_fold0.pkl ... model_fold4.pkl
  evaluations/
    default_20231001/
    default_20260101/
```

`default_YYYYMMDD` is the standard multimer model for the training release
cutoff. `unk_single_YYYYMMDD` is the monomer-aware model used with `-unk` for
single-sequence targets.
