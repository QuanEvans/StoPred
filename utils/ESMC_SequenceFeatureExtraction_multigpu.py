import argparse
import os
import pickle
import time
import sys
from typing import Dict, Any
from multiprocessing import Process, Manager, Value, Lock
from tqdm import tqdm

import torch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.parsers import parse_fasta
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def load_pretrained_esmc_600m(
    pth_path: str = os.path.join(project_root, 'external', 'esmc_600m_2024_12_v0.pth'),
    device: torch.device | None = None
) -> ESMC:
    """
    Load pretrained ESMC 600M model from local path (or remote if missing).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMC_600M_202412(device=device, pth_path=pth_path)
    if device.type != "cpu":
        model = model.to(torch.bfloat16)
    assert isinstance(model, ESMC)
    return model


def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:
    return EsmSequenceTokenizer()


def ESMC_600M_202412(
    device: torch.device | str = "cpu",
    pth_path: str = os.path.join(project_root, 'external', 'esmc_600m_2024_12_v0.pth'),
    use_flash_attn: bool = True
) -> ESMC:
    """
    Instantiate and load ESMC 600M on the given device.
    """
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def worker_loop(
    device_id: int,
    task_queue,
    result_dict,
    pth_path: str,
    counter: Value,
    lock: Lock,
    max_seq_len: int = 2048
) -> None:
    """
    Worker on cuda:device_id; pulls (name, seq) from the queue,
    writes features[name], and increments counter.
    """
    dev = torch.device(f"cuda:{device_id}")
    client = load_pretrained_esmc_600m(pth_path=pth_path, device=dev)
    config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
    while True:
        try:
            name, seq = task_queue.get_nowait()
        except Exception:
            break

        protein = ESMProtein(sequence=seq[:max_seq_len])
        tensor = client.encode(protein)
        logits_out = client.logits(tensor, config)

        # get last 3 hidden states
        last_3_hidden_states = logits_out.hidden_states[-3:]
        # mean of the last 3 hidden states, exclude the first and last token
        mean_hidden_states = last_3_hidden_states[:, :, 1:-1, :].mean(dim=2)
        # concat the mean hidden states to a single vector
        flattened_hidden_states = mean_hidden_states.flatten()
        # Convert to float32 before converting to numpy
        flattened_hidden_states_np = flattened_hidden_states.to(dtype=torch.float32).detach().cpu().numpy()
        result_dict[seq] = flattened_hidden_states_np

        # bump the progress counter
        with lock:
            counter.value += 1


def main():
    parser = argparse.ArgumentParser(description='Extract ESMC features')
    parser.add_argument('input_fasta', type=str, help='Input FASTA file')
    parser.add_argument('save_pkl_path', type=str, help='Output pickle path')
    parser.add_argument(
        '-cuda',
        '--cuda-devices',
        type=str,
        default='-1',
        help='Comma-separated GPU IDs (e.g. "0,1"). "-1" ⇒ CPU only.'
    )
    parser.add_argument('--update', action='store_true', help='Only new seqs')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')
    args = parser.parse_args()

    # parse devices
    raw = args.cuda_devices.strip()
    if raw in ('-1', ''):
        gpu_ids = []
    else:
        gpu_ids = [int(x) for x in raw.split(',')]
        avail = torch.cuda.device_count()
        for gid in gpu_ids:
            if not (0 <= gid < avail):
                raise ValueError(f"GPU {gid} requested but only {avail} available")

    # load seqs & existing features
    name2seq = parse_fasta(args.input_fasta)
    if args.update:
        if not os.path.exists(args.save_pkl_path):
            raise FileNotFoundError(f"{args.save_pkl_path} does not exist")
        with open(args.save_pkl_path, 'rb') as f:
            features = pickle.load(f)
    else:
        features = {}

    to_process = {n: s for n, s in name2seq.items() if s not in features}
    if not to_process:
        print("No new sequences to process.")
        return

    # single-CPU or single-GPU paths (unchanged) …
    if len(gpu_ids) <= 1:
        device = torch.device('cpu') if len(gpu_ids) == 0 else torch.device(f"cuda:{gpu_ids[0]}")
        client = load_pretrained_esmc_600m(device=device)
        for name, seq in tqdm(to_process.items(), desc="CPU"):
            protein = ESMProtein(sequence=seq[:args.max_seq_len])
            tensor = client.encode(protein)
            logits_out = client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True))
            last_3_hidden_states = logits_out.hidden_states[-3:]
            # mean of the last 3 hidden states, exclude the first and last token
            mean_hidden_states = last_3_hidden_states[:, :, 1:-1, :].mean(dim=2)
            # concat the mean hidden states to a single vector
            flattened_hidden_states = mean_hidden_states.flatten()
            # Convert to float32 before converting to numpy
            flattened_hidden_states_np = flattened_hidden_states.to(dtype=torch.float32).detach().cpu().numpy()
            features[seq] = flattened_hidden_states_np

    else:
        # multi-GPU with shared queue + shared counter + tqdm in main
        manager = Manager()
        task_q = manager.Queue()
        result_d = manager.dict()
        counter = Value('i', 0)
        lock = Lock()

        for item in to_process.items():
            task_q.put(item)

        procs: list[Process] = []
        for gid in gpu_ids:
            p = Process(
                target=worker_loop,
                args=(
                    gid,
                    task_q,
                    result_d,
                    os.path.join(project_root, 'external', 'esmc_600m_2024_12_v0.pth'),
                    counter,
                    lock
                )
            )
            p.start()
            procs.append(p)

        total = len(to_process)
        with tqdm(total=total, desc="Multi-GPU") as pbar:
            last_count = 0
            while last_count < total:
                time.sleep(0.5)
                with lock:
                    current = counter.value
                if current > last_count:
                    pbar.update(current - last_count)
                    last_count = current

        for p in procs:
            p.join()

        features.update(result_d.copy())

    # write out
    with open(args.save_pkl_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Done: wrote {len(features)} embeddings to {args.save_pkl_path}")

if __name__ == '__main__':
    main()
