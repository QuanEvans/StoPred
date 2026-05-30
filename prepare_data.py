import json
import pickle
import random
import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Set

import sklearn.model_selection
from tqdm import tqdm

from config import Config
import ml_collections

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

seed = Config.global_seed
random.seed(seed)

def load_raw_data() -> dict:
    """Load raw data from the JSON file defined in the configuration."""
    with open(Config.data.processed_PDBmmcif_path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Filtering utilities
# ---------------------------------------------------------------------------

def filter_assembly(
    assembly: dict,
    cfg: ml_collections.ConfigDict,
    monomer_rate: float = 0.6,
) -> bool:
    """Return ``True`` if *assembly* passes all quality filters.

    The function mirrors the original logic but uses clearer variable names and
    avoids deep nesting where possible.
    """
    entity_count = assembly["entity_count"]
    pdb_id = assembly["unique_id"]
    if not entity_count:
        return False

    # Homomeric case -------------------------------------------------------
    if len(entity_count) == 1:
        # Monomeric handling
        if list(entity_count.values())[0] == 1:
            if monomer_rate == 0.0:
                return False
            if random.random() < 1 - monomer_rate:
                return False
        # Sequence length & unknown‑aa check
        seq = assembly[list(entity_count.keys())[0]]["sequence"]
        seq_len = len(seq)
        if seq_len < cfg.data.minimum_seq_len:
            return False
        if seq.count("X") > seq_len * 0.6:
            return False
    # Heteromeric case -----------------------------------------------------
    else:
        if len(entity_count) > cfg.model.num_subunits:
            return False
        for k, count in entity_count.items():
            seq = assembly[k]["sequence"]
            seq_len = len(seq)
            if seq_len < cfg.data.minimum_seq_len:
                return False
            if seq.count("X") > seq_len * 0.6:
                return False
    return True

def is_homomeric(assembly: dict) -> bool:
    """Return ``True`` when the assembly contains a single entity type."""
    return len(assembly["entity_count"]) == 1

def is_monomeric(assembly: dict) -> bool:
    """Return ``True`` when the assembly consists of exactly one chain."""
    return (
        len(assembly["entity_count"]) == 1
        and list(assembly["entity_count"].values())[0] == 1
    )

def remove_non_protein_assembly(asse_dict: dict, strict: bool = False) -> dict:
    """Strip non‑protein entities from *asse_dict*.

    If ``strict`` is ``True`` the function returns an empty dict when any entity
    is not a protein.
    """
    count_dict = asse_dict.get("entity_count", {})
    if not count_dict:
        return {}

    proteins: List[str] = []
    for k in count_dict.keys():
        entity_info = asse_dict.get(k, {})
        if entity_info.get("type") == "polypeptide(L)":  # could also accept D‑polypeptides
            proteins.append(k)
        elif strict:
            return {}

    if not proteins:
        return {}

    # Keep only protein entries
    asse_dict["entity_count"] = {k: count_dict[k] for k in proteins}
    for k in set(count_dict) - set(proteins):
        del asse_dict[k]
    return asse_dict

def has_unp_ids(assembly: dict, strict: bool = False) -> bool:
    """Check UniProt (UNP) identifiers.

    ``strict`` requires *all* entities to be UNP; otherwise any UNP is sufficient.
    """
    db_names = [
        assembly[entity_id]["db_name"] for entity_id in assembly["entity_count"]
    ]
    if any(name is None for name in db_names):
        return False
    return all(name == "UNP" for name in db_names) if strict else any(
        name == "UNP" for name in db_names
    )

def is_author_defined(assembly: dict) -> bool:
    """Return ``True`` when the author field exists in the assembly details."""
    return "author" in assembly.get("details", {})

def is_exp_defined(assembly: dict) -> bool:
    """Return ``True`` when experimental support is defined and not ``none``."""
    exp = assembly.get("experimental_support")
    return exp != "none" and bool(exp)

def is_after_date(assembly: dict, cut_off_date: str) -> bool:
    """Compare the assembly release date with *cut_off_date* (YYYY‑MM‑DD)."""
    release = datetime.datetime.strptime(assembly["release_date"], "%Y-%m-%d")
    cutoff = datetime.datetime.strptime(cut_off_date, "%Y-%m-%d")
    return release > cutoff

# ---------------------------------------------------------------------------
# Data processing pipeline
# ---------------------------------------------------------------------------

def apply_first_filter(raw_data: dict) -> dict:
    """First‑pass filtering of the raw dataset.

    Removes entries without protein chains, missing author information and those
    that fail ``filter_assembly`` with a monomer rate of 1.0 (i.e., keep all).
    """
    filtered: dict = {}
    for pdb_id, pdb_dict in raw_data.items():
        if not remove_non_protein_assembly(pdb_dict, strict=True):
            continue
        if not is_author_defined(pdb_dict):
            continue
        if not filter_assembly(pdb_dict, Config, monomer_rate=1.0):
            continue
        filtered[pdb_id] = pdb_dict
    return filtered

def create_count_mappings(first_filter_data: dict) -> Tuple[dict, dict]:
    """Map entity counts to labels and then to consecutive indices.

    Counts with fewer than ``minimum_sample_count`` samples are mapped to ``-1``.
    """
    # Frequency of each entity count across the dataset
    freq: Dict[int, int] = {}
    for pdb_dict in first_filter_data.values():
        for cnt in pdb_dict["entity_count"].values():
            cnt_int = int(cnt)
            freq[cnt_int] = freq.get(cnt_int, 0) + 1

    count2label: Dict[int, int] = {0: 0}
    for cnt, f in freq.items():
        count2label[cnt] = -1 if f < Config.data.minimum_sample_count else cnt
    # Ensure deterministic order
    count2label = dict(sorted(count2label.items()))

    # Convert the *set* of label values to a stable index mapping
    unique_labels = sorted(set(count2label.values()))
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return count2label, label2idx

def save_count_mappings(count2label: dict, label2idx: dict) -> None:
    """Persist the count‑to‑label and label‑to‑index dictionaries as JSON."""
    with open(Config.model.count2label, "w") as f:
        json.dump(count2label, f)
    with open(Config.model.label2idx, "w") as f:
        json.dump(label2idx, f)

def filter_non_monomeric_data(first_filter_data: dict) -> dict:
    """Drop assemblies that are monomeric after the first filtering step."""
    out: dict = {}
    for pdb_id, pdb_dict in first_filter_data.items():
        if not filter_assembly(pdb_dict, Config, monomer_rate=1.0):
            continue
        if is_monomeric(pdb_dict):
            continue
        out[pdb_id] = pdb_dict
    print(f"Number of assemblies in filter_data_noMonomer: {len(out)}")
    return out

def create_sequence_stoichiometry_key(pdb_dict: dict) -> Tuple[Tuple[str, int], ...]:
    """Return a deterministic key representing the (sequence, stoichiometry) pairs.

    The returned tuple can be used for duplicate detection across assemblies.
    """
    pairs = [
        (pdb_dict[entity_id]["sequence"], count)
        for entity_id, count in pdb_dict["entity_count"].items()
    ]
    # Sort for stable ordering
    pairs.sort(key=lambda x: (x[0], x[1]))
    return tuple(pairs)

def split_data_by_date(filter_data_noMonomer: dict) -> Tuple[dict, dict, dict]:
    """Split the dataset into train / validation / test based on release dates.

    * ``cut_off_date`` – everything before this date goes to training.
    * ``test_cut_off_date`` – entries after this date become test data.
    * Remaining entries become validation data.
    Duplicate (seq, stoichiometry) combinations are removed from val/test.
    """
    train_data: dict = {}
    val_data_raw: dict = {}
    test_data_raw: dict = {}

    seen_seq_sto: Dict[Tuple, List[str]] = {}

    # Process oldest releases first so exact duplicate groups keep the training
    # representative when one exists. Relying on JSON insertion order can let a
    # newer val/test entry become the representative and hide an older train
    # duplicate from leakage filtering.
    ordered_items = sorted(
        filter_data_noMonomer.items(),
        key=lambda item: (item[1]["release_date"], item[0]),
    )

    for pdb_id, pdb_dict in ordered_items:
        key = create_sequence_stoichiometry_key(pdb_dict)
        is_duplicate = key in seen_seq_sto
        seen_seq_sto.setdefault(key, []).append(pdb_id)

        if not is_after_date(pdb_dict, Config.data.cut_off_date):
            if is_duplicate:
                continue
            train_data[pdb_id] = pdb_dict
        elif is_after_date(pdb_dict, Config.data.test_cut_off_date):
            if is_duplicate:
                continue
            if is_exp_defined(pdb_dict):
                test_data_raw[pdb_id] = pdb_dict
        else:
            if is_duplicate:
                continue
            if is_exp_defined(pdb_dict):
                val_data_raw[pdb_id] = pdb_dict

    # Ensure no overlap with training IDs (defensive)
    train_ids = set(train_data)
    val_data_raw = {k: v for k, v in val_data_raw.items() if k not in train_ids}
    test_data_raw = {k: v for k, v in test_data_raw.items() if k not in train_ids}

    return train_data, val_data_raw, test_data_raw

def create_cross_validation_folds(train_data: dict) -> List[dict]:
    """Generate stratified K‑fold splits based on stoichiometry.

    Rare stoichiometries (fewer than ``minimum_sto_count`` samples) are grouped
    under the label ``'other'`` before folding.
    """
    ids: List[str] = []
    sto_labels: List[str] = []
    sto_counter: Dict[str, int] = {}

    for pdb_id, pdb_dict in train_data.items():
        sto = tuple(sorted(pdb_dict["entity_count"].values(), reverse=True))
        sto_str = str(sto)
        sto_counter[sto_str] = sto_counter.get(sto_str, 0) + 1
        ids.append(pdb_id)
        sto_labels.append(sto_str)

    # Collapse rare classes to "other"
    for i, label in enumerate(sto_labels):
        if sto_counter[label] < Config.data.minimum_sto_count:
            sto_labels[i] = "other"

    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=Config.data.num_folds,
        shuffle=True,
        random_state=seed,
    )
    folds: List[dict] = []
    for train_idx, test_idx in kf.split(ids, sto_labels):
        folds.append({
            "train": [ids[i] for i in train_idx],
            "valid": [ids[i] for i in test_idx],
        })
    return folds

def save_dataset(
    train_data: dict,
    val_data_raw: dict,
    test_data_raw: dict,
    fold_data: List[dict],
) -> None:
    """Serialize the full dataset (including optional CASP16 benchmark)."""
    save_path = Path(Config.data.Dataset) / "StoPredDataset.pkl"
    payload = {
        "train_data": train_data,
        "valid_data": val_data_raw,
        "test_data": test_data_raw,
        "fold_data": fold_data,
    }
    if Path(Config.data.casp16_benchmark_path).exists():
        with open(Config.data.casp16_benchmark_path, "r") as f:
            payload["casp16_benchmark"] = json.load(f)
        print(
            f"Loaded {len(payload['casp16_benchmark'])} CASP16 benchmark targets"
        )
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

def create_stoichiometry_mapping(train_data: dict) -> dict:
    """Map each stoichiometry pattern to a consecutive integer index.

    Rare patterns are collapsed into ``'other'`` before indexing.
    The mapping is saved as JSON for downstream use.
    """
    sto_counter: Dict[str, int] = {}
    sto_list: List[str] = []
    for pdb_dict in train_data.values():
        sto = tuple(sorted(pdb_dict["entity_count"].values(), reverse=True))
        sto_str = str(sto)
        sto_counter[sto_str] = sto_counter.get(sto_str, 0) + 1
        sto_list.append(sto_str)
    # Collapse rare entries
    for i, s in enumerate(sto_list):
        if sto_counter[s] < Config.data.minimum_sto_count:
            sto_list[i] = "other"
    unique_sto = sorted(set(sto_list))
    sto2idx = {s: idx for idx, s in enumerate(unique_sto)}
    with open(Config.model.sto2idx, "w") as f:
        json.dump(sto2idx, f)
    return sto2idx

def extract_and_save_sequences(
    train_data: dict,
    val_data_raw: dict,
    test_data_raw: dict,
) -> None:
    """Collect all unique protein sequences and write them to a FASTA file."""
    seqs: Set[str] = set()
    for dataset in (train_data, val_data_raw, test_data_raw):
        for pdb_dict in dataset.values():
            for entity_id in pdb_dict["entity_count"]:
                seqs.add(pdb_dict[entity_id]["sequence"])
    # Include CASP16 benchmark if present
    if Path(Config.data.casp16_benchmark_path).exists():
        with open(Config.data.casp16_benchmark_path, "r") as f:
            bench = json.load(f)
        for pdb_dict in bench.values():
            for entity_id in pdb_dict["entity_count"]:
                seqs.add(pdb_dict[entity_id]["sequence"])
    fasta_path = Path(Config.data.Dataset) / "all_sequences.fasta"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sorted(seqs)):
            f.write(f">{i}\n{seq}\n")

def main() -> None:
    """Entry point – orchestrates the full data‑preparation pipeline."""
    raw = load_raw_data()
    first = apply_first_filter(raw)
    count2label, label2idx = create_count_mappings(first)
    save_count_mappings(count2label, label2idx)

    no_monomer = filter_non_monomeric_data(first)
    train, val, test = split_data_by_date(no_monomer)
    folds = create_cross_validation_folds(train)
    save_dataset(train, val, test, folds)

    print(f"Training assemblies: {len(train)}")
    print(f"Validation assemblies: {len(val)}")
    print(f"Test assemblies: {len(test)}")

    create_stoichiometry_mapping(train)
    extract_and_save_sequences(train, val, test)

if __name__ == "__main__":
    main()
