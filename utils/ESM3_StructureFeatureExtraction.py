import os
import argparse
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import shutil, pickle

def setup_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"

def get_mean_embedding(pdb_path, client):
    protein = ESMProtein.from_pdb(pdb_path)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    per_residue_embeddings = output.per_residue_embedding
    mean_embedding = per_residue_embeddings[1:-1, :].mean(dim=0)
    return mean_embedding

def truncat_pdb(pdb_path, tmp_pdb):
    # truncate pdb file to 1024 residues
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    with open(tmp_pdb, 'w') as f:
        for line in lines:
            if line.startswith('ATOM'):
                resi_num = line[22:26].strip()
                resi_num = int(resi_num)
                if resi_num > 1024:
                    break
                f.write(line)
        f.write('TER\n')
    
def process_pdbs(pdb_dir, tmp_pdb_dir, client, save_pkl_path, update=False):
    if os.path.exists(tmp_pdb_dir):
        shutil.rmtree(tmp_pdb_dir)
    os.makedirs(tmp_pdb_dir)

    if update and os.path.exists(save_pkl_path):
        features = pickle.load(open(save_pkl_path, 'rb'))
        processed_ids = set(features.keys())
    else:
        features = {}
        processed_ids = set()
    
    pdb_files = [i.path for i in os.scandir(pdb_dir) if i.is_file() and i.path.endswith('.pdb') and os.path.getsize(i.path) > 0]
    pdb_files = sorted(pdb_files, key=os.path.getsize)
    pdb_files = [i for i in pdb_files if os.path.basename(i).split('.')[0] not in processed_ids]

    count = 0
    for pdb_file in tqdm(pdb_files, desc='Processing PDB files'):
        pdb_name = os.path.basename(pdb_file).split('.')[0]
        if pdb_name in processed_ids:
            continue

        try:
            mean_embedding = get_mean_embedding(pdb_file, client)
            features[pdb_name] = mean_embedding.cpu().numpy()
        except Exception as e:
            print(f"Error processing {pdb_name}: {e}")
            # try to use the truncated pdb file
            tmp_pdb = os.path.join(tmp_pdb_dir, pdb_name+'.pdb')
            truncat_pdb(pdb_file, tmp_pdb)
            try:
                mean_embedding = get_mean_embedding(tmp_pdb, client)
                features[pdb_name] = mean_embedding.cpu().numpy()
            except Exception as e:
                print(f"Error processing after truncation {pdb_name}: {e}")
                continue
        
        processed_ids.add(pdb_name)
        count += 1
        # save the features
        if count % 3000 == 0:
            pickle.dump(features, open(save_pkl_path, 'wb'))
    pickle.dump(features, open(save_pkl_path, 'wb'))
        

def create_parser():
    parser = argparse.ArgumentParser(description='Extract ESM3 structure features')
    parser.add_argument('pdb_dirs', type=str, help='Path to PDB directories')
    parser.add_argument('save_pkl_path', type=str, help='Path to save the pickle file')
    parser.add_argument('tmp_pdb_dir', type=str, help='Path to save the temporary PDB files')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--update', action='store_true', help='Update existing features')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    client = ESM3.from_pretrained('esm3-open').to(device)
    process_pdbs(args.pdb_dirs, args.tmp_pdb_dir, client, args.save_pkl_path, args.update)
