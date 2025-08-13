 #!/usr/bin/python3
import os, subprocess
import argparse
from collections import defaultdict
import multiprocessing as mp
import shutil
import re


def get_absolute_path(path:str)->str:
    """
    Returns the absolute path of a file or directory.

    Args:
    path (str): Path to the file or directory.

    Returns:
    str: Absolute path of the file or directory.
    """
    if path is None:
        return None
    if shutil.which(path) is not None:
        return path
    if path.startswith('.'):
        return os.path.abspath(path)
    else:
        return os.path.abspath(os.path.join(os.getcwd(), path))

def get_tmp_dir()->str:
    """
    Returns the path to the temporary directory.

    Returns:
    str: Path to the temporary directory.
    """
    # first get user
    user = os.environ.get('USER')
    if user is None:
        raise ValueError("Could not determine user.")
    unique_id = 1
    tmp_dir = os.path.join("/tmp", user, f"foldseek_{unique_id}")
    while os.path.exists(tmp_dir):
        unique_id += 1
        tmp_dir = os.path.join("/tmp", user, f"foldseek_{unique_id}")
    return tmp_dir
    
def parse_aligned_chain(result_file:str):
    pattern_1 = r'Name of Structure_1:.*?(\S+?)\s*\('
    pattern_2 = r'Name of Structure_2:.*?(\S+)'
    with open(result_file, 'r') as f:
        result_text = f.read()
    chains_1 = re.search(pattern_1, result_text)
    chains_2 = re.search(pattern_2, result_text)
    
    if chains_1 and chains_2:
        chains_1 = chains_1.group(1)
        chains_2 = chains_2.group(1)
        chains_1 = chains_1[chains_1.index(':'):]
        chains_2 = chains_2[chains_2.index(':'):]
        chains_1 = [chain for chain in chains_1.split(':')]
        chains_2 = [chain for chain in chains_2.split(':')]
    else:
        chains_1 = []
        chains_2 = []
    return chains_1, chains_2

def parse_tmscores(result_file:str)->tuple:
    """
    Parses the TM-scores from the output file of USalign/MMalign.

    Args:
    result_file (str): Path to the output file.

    Returns:
    tuple: Tuple containing the TM-scores for the query and target structures.
    """
    # Read the output file
    with open(result_file, 'r') as f:
        result_text = f.read()

    # Regex pattern to find TM-scores
    tm_score_pattern = r"TM-score= ([0-9.]+) \(normalized by length of Structure_(\d):"

    # Find all matches
    matches = re.findall(tm_score_pattern, result_text)

    # Initialize scores
    qtm, ttm = None, None

    for score, structure in matches:
        if structure == '1':
            qtm = float(score)
        elif structure == '2':
            ttm = float(score)

    return qtm, ttm

def run_foldseek(input_dir:str, foldseek_DB:str, output_tsv:str, threads:int=mp.cpu_count(), foldseek_path:str=shutil.which('mmseqs'), evalue:float=10, tmp_dir:str=None):

    if tmp_dir is None:
        tmp_dir = get_tmp_dir()
    
    search_cmd = [
        foldseek_path, 'easy-search', input_dir, foldseek_DB, output_tsv, tmp_dir,
        '--format-output', "query,target,evalue,bits,nident,qlen,tlen",
        '--threads', str(threads),  # Including threads argument in the command
        '-e', str(evalue),
        '-a'
    ]
    try:
        subprocess.run(search_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FoldSeek search failed with error: {e}")
        return False

    return output_tsv



def read_tsv(out_tsv:str):
    """
    Reads the output TSV file from FoldSeek search.

    Args:
    out_tsv (str): Path to the output TSV file.

    Returns:
    dict: Dictionary containing the query and target structures, and their TM-scores.
    """
    # Initialize the dictionary
    result_dict = dict()

    # Read the TSV file
    with open(out_tsv, 'r') as f:
        for line in f:
            split_line = line.strip().split('\t')
            if len(split_line) != 7:
                continue
            query, target, evalue, bits, nident, qlen, tlen = split_line
            # query = query.split('.pdb')[0]
            # target = target.split('.pdb')[0]
            q_id, q_entity = query.split('-')
            t_id, t_entity = target.split('-')
            if q_id == t_id:
                continue
            if q_id not in result_dict:
                result_dict[q_id] = dict()
            if t_id not in result_dict[q_id]:
                result_dict[q_id][t_id] = dict()
            result_dict[q_id][t_id][(q_entity,t_entity)] = {
                'evalue': float(evalue),
                'bits': int(bits),
                'nident': int(nident),
                'qlen': int(qlen),
                'tlen': int(tlen),
            }
    return result_dict

def determine_best_pair(subdict):
    """
    Determines the best pair of query and target structures based on the TM-score.

    Args:
    subdict (dict): Dictionary containing the TM-scores of all query-target structure pairs.

    Returns:
    tuple: Tuple containing the query and target structure IDs of the best pair.
    """
    keys = list(subdict.keys())
    av_t = set([pair[1] for pair in keys])
    av_q = set([pair[0] for pair in keys])
    # sort dict by bitscore
    sorted_items = sorted(subdict.items(), key=lambda x: x[1]['bits'], reverse=True)
    best_pairs = []
    while len(av_q) > 0 and len(av_t) > 0:
        if len(sorted_items) == 0:
            break
        best_pair = sorted_items.pop(0)
        q, t = best_pair[0]
        if q in av_q and t in av_t:
            best_pairs.append(best_pair)
            av_q.remove(q)
            av_t.remove(t)
        else:
            continue
    # convert to dict
    best_pairs_dict = dict()
    for pair in best_pairs:
        best_pairs_dict[pair[0]] = pair[1]
    return best_pairs_dict


def determine_sto(query_dict, target2num, num_entities, cov_mod='3', seqid_mod='1', num_mod='2'):
    query_dict_new = dict()
    for k,v in query_dict.items():
        if len(v) == 1:
            query_dict_new[k] = v
        else:
            query_dict_new[k] = determine_best_pair(v)
    av_count_dict = dict()
    seqid_dict = dict()
    target2unit = dict()
    q_entity_hits_state = dict()
    for k in target2num.keys():
        pdb = k.split('-')[0]
        target2unit[pdb] = target2unit.get(pdb, 0) + 1
    
    for t, pairs in query_dict.items():
        #cov = len(pairs) / num_entities
        for (q_id, t_id ), values in pairs.items():
            if q_id not in q_entity_hits_state:
                q_entity_hits_state[q_id] = {
                    'bits': [],
                    'seqid': [],
                    'evalue': [],
                }

            if num_mod == '1':
                # num of entities should at least be the same
                if num_entities != target2unit[t]:
                    continue
            elif num_mod == '2':
                # num of entities should be less than or equal to the target
                if num_entities > target2unit[t]:
                    continue
            else:
                pass

            cov = 1
            if cov_mod == '1':
                # query coverage
                cov = len(pairs) / num_entities
            elif cov_mod== '2':
                # target coverage
                cov = len(pairs) / target2unit[t]
            elif cov_mod == '3':
                # min coverage
                cov = len(pairs) / max(num_entities, target2unit[t])
            elif cov_mod == '4':
                # max coverage
                cov = len(pairs) / min(num_entities, target2unit[t])
            else:
                # default is no coverage
                cov = 1
            # cov1 = len(pairs) / num_entities
            # cov = 1
            t_unique = f'{t}-{t_id}'
            num = target2num[t_unique]
            if q_id not in av_count_dict:
                av_count_dict[q_id] = dict()
            if num not in av_count_dict[q_id]:
                av_count_dict[q_id][num] = 0
            if 'total' not in av_count_dict[q_id]:
                av_count_dict[q_id]['total'] = 0

            # determine the score
            weight = 1
            if seqid_mod == '1':
                # query seqid
                weight = values['nident'] / values['qlen']
            elif seqid_mod == '2':
                # target seqid
                weight = values['nident'] / values['tlen']
            elif seqid_mod == '3':
                # min seqid
                weight = min(values['nident'] / values['qlen'], values['nident'] / values['tlen'])
            elif seqid_mod == '4':
                # max seqid
                weight = max(values['nident'] / values['qlen'], values['nident'] / values['tlen'])
            else:
                # default is no seqid
                weight = 1

            # q_entity_hits_state[q_id]['bits'] = max(q_entity_hits_state[q_id].get('bits', 0), values['bits'])
            # q_entity_hits_state[q_id]['seqid'] = max(q_entity_hits_state[q_id].get('seqid', 0), weight)
            # q_entity_hits_state[q_id]['evalue'] = min(q_entity_hits_state[q_id].get('evalue', float('inf')), values['evalue'])

            # use mean top 5 values
            q_entity_hits_state[q_id]['bits'].append(values['bits'])
            q_entity_hits_state[q_id]['seqid'].append(weight)
            q_entity_hits_state[q_id]['evalue'].append(values['evalue'])

            score = values['bits'] * cov * weight
            av_count_dict[q_id][num] += score
            av_count_dict[q_id]['total'] += score
    # print(av_count_dict)
    for q_id, num_dict in av_count_dict.items():
        total = num_dict.pop('total')
        for num, score in num_dict.items():
            num_dict[num] = score / total
        # sort by score, hightest first
        av_count_dict[q_id] = dict(sorted(num_dict.items(), key=lambda x: x[1], reverse=True))
    
    # mean top 5 values for each
    q_entity_hits_state_new = dict()
    for q_id, values in q_entity_hits_state.items():
        q_entity_hits_state_new[q_id] = dict()
        # sort idx by seqid
        seqid_list = values['seqid']
        sorted_idx = sorted(range(len(seqid_list)), key=lambda x: seqid_list[x], reverse=True)
        
        # reorder bits, seqid, and evalue according to the sorted idx
        q_entity_hits_state_new[q_id]['bits'] = [values['bits'][i] for i in sorted_idx]
        q_entity_hits_state_new[q_id]['seqid'] = [values['seqid'][i] for i in sorted_idx]
        q_entity_hits_state_new[q_id]['evalue'] = [values['evalue'][i] for i in sorted_idx]
    # print(q_entity_hits_state_new)
    return av_count_dict, q_entity_hits_state_new


def single_case(query, query_entity_counts, hits_dict, target2num, num_entities, cov_mod='3', seqid_mod='1', num_mod='2'):
    result, q_entity_hits_state = determine_sto(hits_dict, target2num, num_entities, cov_mod, seqid_mod, num_mod)
    query_entity = query_entity_counts[query]
    for entity in query_entity:
        if entity not in result:
            # default to 1
            result[entity] = {1: 100}
    pred = dict()
    for entity, num_dict in result.items():
        pred[entity] = list(num_dict.keys())[0]
    pred = dict(sorted(pred.items(), key=lambda x: int(x[0])))
    return query, result, pred, q_entity_hits_state

def read_fastas(fasta_file:str):
    seq_dict = dict()
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                seq_id = line[1:]
                seq_dict[seq_id] = ''
            else:
                seq_dict[seq_id] += line
    return seq_dict

def main(args):
    # the input structure should be pdb file, and with this naming format
    # MainID-EntityID.pdb, eg. 1a2k_1-1.pdb
    # the MainID cannot contain '_', and EntityID can be any number

    seq_dict = read_fastas(args.inputDir)
    query_entity_counts = dict()
    for pdb_file in seq_dict:
        main_id, entity_id = pdb_file.split('-')
        entity_id = entity_id.split('.pdb')[0]
        if main_id not in query_entity_counts:
            query_entity_counts[main_id] = set()
        query_entity_counts[main_id].add(entity_id)


    target2num = dict()
    with open(args.target2num, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            target, num = line.split('\t')
            target2num[target] = int(num)

    # set up the work dir
    if not os.path.exists(args.workDir):
        os.makedirs(args.workDir)
    # run foldseek
    tmpdir = os.path.join(args.workDir, 'tmp')
    filename = 'mmseqs-e'+str(args.evalue)+'.tsv'
    out_tsv = os.path.join(args.workDir, filename)
    out_file = os.path.join(args.workDir, f'pred-{args.num_mod}-{args.cov_mod}-{args.seqid_mod}-{args.evalue}.json')
    # if os.path.exists(out_file):
    #     return
    if not os.path.exists(out_tsv):
        run_foldseek(args.inputDir, args.foldseekDB, out_tsv, args.threads, args.foldseekPath, args.evalue, tmpdir)
    # rm tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    # read tsv
    result_dict = read_tsv(out_tsv)

    query_result_dict = dict()
    max_pred = dict()
    mp_args = []
    query_results_state = dict()
    for query, hits_dict in result_dict.items():
        mp_args.append(
            (query, query_entity_counts, hits_dict, target2num, len(query_entity_counts[query]), args.cov_mod, args.seqid_mod, args.num_mod)
        )
        # result = determine_sto(hits_dict, target2num, len(query_entity_counts[query]))
        # query_entity = query_entity_counts[query]
        # for entity in query_entity:
        #     if entity not in result:
        #         # default to 1
        #         result[entity] = {1: 100}
        # query_result_dict[query] = result
        # pred = dict()
        # for entity, num_dict in result.items():
        #     pred[entity] = list(num_dict.keys())[0]
        # # sort pred by key
        # pred = dict(sorted(pred.items(), key=lambda x: int(x[0])))
        # # print(f"{query} {pred}")
        # # print('-----')
        # max_pred[query] = pred
    with mp.Pool(args.threads) as pool:
        results = pool.starmap(single_case, mp_args)
        for query, result, pred, q_entity_hits_state in results:
            query_result_dict[query] = result
            query_results_state[query] = q_entity_hits_state
            max_pred[query] = pred
    # save the result
    import json
    with open(os.path.join(args.workDir, f'pred-{args.num_mod}-{args.cov_mod}-{args.seqid_mod}-{args.evalue}.json'), 'w') as f:
        json.dump(max_pred, f, indent=2)
    with open(os.path.join(args.workDir, f'result-{args.num_mod}-{args.cov_mod}-{args.seqid_mod}-{args.evalue}.json'), 'w') as f:
        json.dump(query_result_dict, f, indent=2)
    with open(os.path.join(args.workDir, f'state-{args.num_mod}-{args.cov_mod}-{args.seqid_mod}-{args.evalue}.json'), 'w') as f:
        json.dump(query_results_state, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FoldSeek to find the best target structure for each query structure.")
    parser.add_argument("inputDir", type=str, help="Path to the directory containing the input PDB files.")
    parser.add_argument("foldseekDB", type=str, help="Path to the FoldSeek database.")
    parser.add_argument("target2num", type=str, help="Path to the file containing the mapping of target structures to numbers.")
    parser.add_argument("workDir", type=str, help="Path to the working directory.")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads to use for FoldSeek search.")
    parser.add_argument("--foldseekPath", type=str, default=shutil.which('mmseqs'), help="Path to the FoldSeek executable.")
    parser.add_argument('-e',"--evalue", type=float, default=0.1, help="E-value threshold for FoldSeek search.")
    parser.add_argument("--cov_mod", type=str, default='3', help="Coverage mode for determining the best pair.")
    parser.add_argument("--seqid_mod", type=str, default='3', help="Sequence identity mode for determining the best pair.")
    parser.add_argument("--num_mod", type=str, default='1', help="Number of entities mode for determining the best pair.")
    args = parser.parse_args()
    args.inputDir = get_absolute_path(args.inputDir)
    args.foldseekDB = get_absolute_path(args.foldseekDB)
    args.target2num = get_absolute_path(args.target2num)
    args.workDir = get_absolute_path(args.workDir)
    if args.evalue >= 1:
        # get rid of the decimal point
        args.evalue = int(args.evalue)
    main(args)
            