import argparse
import os
import sys
from typing import Dict, Any, List
import torch
import pickle
from tqdm import tqdm
from multiprocessing import Process, Manager

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
    Load pretrained ESMC 600M model from local path.
    
    Args:
        pth_path (str): Path to the model weights file. Defaults to 
            'external/esmc_600m_2024_12_v0.pth' in project root.
        device (torch.device | None): Device to load the model on. If None,
            defaults to CUDA if available, otherwise CPU.
            
    Returns:
        ESMC: Loaded ESMC model instance.
        
    Raises:
        FileNotFoundError: If the model weights file doesn't exist at pth_path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ESMC_600M_202412(device=device, pth_path=pth_path)
    if device.type != "cpu":
        model = model.to(torch.bfloat16)
    assert isinstance(model, ESMC)
    return model


def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:
    """
    Get the tokenizer for ESMC models.
    
    Returns:
        EsmSequenceTokenizer: Tokenizer instance for ESMC models.
    """
    return EsmSequenceTokenizer()


def ESMC_600M_202412(
    device: torch.device | str = "cpu",
    pth_path: str = os.path.join(project_root, 'external', 'esmc_600m_2024_12_v0.pth'),
    use_flash_attn: bool = True
) -> ESMC:
    """
    Create and load ESMC 600M model with specified parameters.
    
    Args:
        device (torch.device | str): Device to create the model on. Defaults to "cpu".
        pth_path (str): Path to the model weights file. Defaults to 
            'external/esmc_600m_2024_12_v0.pth' in project root.
        use_flash_attn (bool): Whether to use flash attention. Defaults to True.
            
    Returns:
        ESMC: Created and loaded ESMC model instance.
    """
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    
    state_dict = torch.load(
        pth_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def main(args: argparse.Namespace) -> None:
    """
    Extract ESMC features from a fasta file.
    
    Features from ESM-C are 1280-dimensional embeddings of the protein sequence.
    The pickle file will save a dictionary with sequence as keys and 
    1280-dimensional embeddings as values.

    Args:
        args (argparse.Namespace): Command line arguments containing input fasta file path
            and output pickle file path, along with optional GPU and update flags.
            
    Raises:
        FileNotFoundError: If the model weights file doesn't exist at expected location,
            or if the save path doesn't exist when updating features.
    """
    # Load ESM-C model - first try to load from local
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        client = load_pretrained_esmc_600m(device=device)
    except Exception as e:
        print(f"""
Error: {e}, loading from remote
Please download the model from https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/blob/main/data/weights/esmc_600m_2024_12_v0.pth
and place it in {os.path.join(project_root, 'external', 'esmc_600m_2024_12_v0.pth')}
loading from remote instead...
""")
        client = ESMC.from_pretrained("esmc_600m", device=device)
    
    # Read fasta file
    name2seq: Dict[str, str] = parse_fasta(args.input_fasta)

    # Extract features
    if args.update:
        # Check if the output path exists
        if not os.path.exists(args.save_pkl_path):
            raise FileNotFoundError(f"Output path {args.save_pkl_path} does not exist")
        
        # Load existing features
        with open(args.save_pkl_path, 'rb') as f:
            features: Dict[str, Any] = pickle.load(f)
    else:
        features: Dict[str, Any] = {}
    
    # Skip the sequences that already have features
    name2seq = {name: seq for name, seq in name2seq.items() if seq not in features}
    
    for name, seq in tqdm(name2seq.items()):
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        # get last 3 hidden states
        last_3_hidden_states = logits_output.hidden_states[-3:]
        # mean of the last 3 hidden states, exclude the first and last token
        mean_hidden_states = last_3_hidden_states[:, :, 1:-1, :].mean(dim=2)
        # concat the mean hidden states to a single vector
        flattened_hidden_states = mean_hidden_states.flatten()
        # Convert to float32 before converting to numpy
        flattened_hidden_states_np = flattened_hidden_states.to(dtype=torch.float32).detach().cpu().numpy()
        features[name] = flattened_hidden_states_np

    with open(args.save_pkl_path, 'wb') as f:
        pickle.dump(features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ESMC features')
    parser.add_argument('input_fasta', type=str, help='Input fasta file')
    parser.add_argument('save_pkl_path', type=str, help='Output path')
    parser.add_argument('-cuda','--cuda_devices', type=str, default="-1", help='Comma-separated list of cuda device ids (e.g., "0,1,2") or "-1" for CPU')
    parser.add_argument('--update', action='store_true', help='Update existing features')
    
    args = parser.parse_args()
    main(args)

