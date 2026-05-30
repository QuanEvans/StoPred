from pdbecif.mmcif_tools import MMCIF2Dict
import os
import argparse
from functools import partial
import multiprocessing as mp
import json
import gzip
import tempfile
from tqdm import tqdm
from utils.PDBcif import PDBCif
from config import Config


ONLY_CATEGORIES = [
    '_citation',
    '_entity',
    '_entity_poly',
    '_struct_ref',
    '_struct',
    '_struct_keywords',
    '_pdbx_struct_assembly',
    '_pdbx_struct_assembly_gen',
    '_pdbx_struct_assembly_auth_evidence',
    '_struct_asym',
    '_pdbx_struct_oper_list',
    '_pdbx_audit_revision_history',
]
PARSER_METADATA_VERSION = 1

mmcif_dict_parser = MMCIF2Dict()


def iter_cif_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.cif') or filename.endswith('.cif.gz'):
                yield os.path.join(dirpath, filename)


def pdb_id_from_filename(file_path: str) -> str:
    basename = os.path.basename(file_path)
    if basename.endswith('.cif.gz'):
        return basename[:-7].upper()
    if basename.endswith('.cif'):
        return basename[:-4].upper()
    return ''


def parse_cif(cif_file_path: str) -> dict:
    file_pdb_id = pdb_id_from_filename(cif_file_path)
    try:
        if cif_file_path.endswith('.gz'):
            with gzip.open(cif_file_path, 'rt') as gz_in:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=True) as tmp_file:
                    tmp_file.write(gz_in.read())
                    tmp_file.flush()
                    mmcif_dict = mmcif_dict_parser.parse(
                        tmp_file.name,
                        onlyCategories=ONLY_CATEGORIES,
                    )
        else:
            mmcif_dict = mmcif_dict_parser.parse(
                cif_file_path,
                onlyCategories=ONLY_CATEGORIES,
            )
    except Exception as e:
        print(f'Error: {e}, failed in parsing {cif_file_path}')
        return {'pdb_id': file_pdb_id, 'ok': False, 'assemblies': {}}

    try:
        pdb_id = list(mmcif_dict.keys())[0]
        cif_parser = PDBCif(mmcif_dict=mmcif_dict[pdb_id])
        return {
            'pdb_id': file_pdb_id or str(cif_parser.pdb_id).upper(),
            'ok': True,
            'assemblies': cif_parser.get_assembly(),
        }
    except Exception as e:
        print(f'Error: {e}, failed in assembly extraction {cif_file_path}')
        return {'pdb_id': file_pdb_id, 'ok': False, 'assemblies': {}}


def collect_obsolete_from_dir(obsolete_dir: str) -> set:
    obsolete_pdb_ids = set()
    if not os.path.exists(obsolete_dir):
        return obsolete_pdb_ids
    for file_path in iter_cif_files(obsolete_dir):
        pdb_id = pdb_id_from_filename(file_path)
        if pdb_id:
            obsolete_pdb_ids.add(pdb_id)
    return obsolete_pdb_ids


def collect_obsolete_from_dat(obsolete_file_path: str) -> set:
    obsolete_pdb_ids = set()
    if not os.path.exists(obsolete_file_path):
        return obsolete_pdb_ids
    with open(obsolete_file_path, 'r') as file:
        for line in file:
            if not line.startswith('OBSLTE'):
                continue
            fields = line.split()
            if len(fields) >= 3:
                obsolete_pdb_ids.add(fields[2].upper())
    return obsolete_pdb_ids


def collect_obsolete_pdb_ids(pdb_mmcif_dir: str) -> set:
    obsolete_pdb_ids = collect_obsolete_from_dir(os.path.join(pdb_mmcif_dir, 'obsolete'))
    obsolete_pdb_ids.update(collect_obsolete_from_dat(os.path.join(pdb_mmcif_dir, 'obsolete.dat')))
    print(f'Found {len(obsolete_pdb_ids)} obsolete PDB IDs')
    return obsolete_pdb_ids


def find_mmcif_data_dir(pdb_mmcif_dir: str) -> str:
    divided_dir = os.path.join(pdb_mmcif_dir, 'divided')
    if os.path.exists(divided_dir):
        return divided_dir
    mmcif_files_dir = os.path.join(pdb_mmcif_dir, 'mmcif_files')
    if os.path.exists(mmcif_files_dir):
        return mmcif_files_dir
    raise FileNotFoundError(
        f'No divided/ or mmcif_files/ directory found in {pdb_mmcif_dir}'
    )


def assembly_pdb_id(unique_id: str) -> str:
    return str(unique_id).split('_', 1)[0].upper()


def collect_existing_pdb_ids(assemblies: dict) -> set:
    return {assembly_pdb_id(unique_id) for unique_id in assemblies}


def metadata_path(output_json: str) -> str:
    return f'{output_json}.meta.json'


def current_processing_config(args: argparse.Namespace) -> dict:
    return {
        'parser': 'parse_PDBmmcif_gz.py',
        'metadata_version': PARSER_METADATA_VERSION,
        'pdb_mmcif_dir': args.pdb_mmcif_dir,
        'exclude_obsolete': bool(args.exclude_obsolete),
    }


def load_existing_output(
    output_json: str,
    processing_config: dict,
    force_full_parse: bool,
    strict_metadata: bool,
) -> tuple[dict, set]:
    if force_full_parse or not os.path.exists(output_json):
        return {}, set()

    print(f'Loading existing parsed assemblies from: {output_json}')
    with open(output_json, 'r') as input_file:
        assemblies = json.load(input_file)

    meta_file = metadata_path(output_json)
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as input_file:
            metadata = json.load(input_file)
        if metadata.get('config') != processing_config:
            print('Existing metadata config differs from this run; doing a full parse')
            return {}, set()
        processed_pdb_ids = set(metadata.get('processed_pdb_ids') or collect_existing_pdb_ids(assemblies))
        return assemblies, processed_pdb_ids

    if strict_metadata:
        print('Existing output has no metadata sidecar; doing a full parse')
        return {}, set()

    print('Existing output has no metadata sidecar; assuming it matches this run')
    return assemblies, collect_existing_pdb_ids(assemblies)


def atomic_json_dump(data, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=output_dir or None,
        prefix=f'.{os.path.basename(output_path)}.',
        suffix='.tmp',
        delete=False,
    ) as output_file:
        temp_path = output_file.name
        json.dump(data, output_file)
    os.replace(temp_path, output_path)


def write_parse_outputs(
    assemblies: dict,
    output_json: str,
    processing_config: dict,
    processed_pdb_ids: set,
    failed_pdb_ids: list,
    candidate_pdb_ids: set,
) -> None:
    processed_pdb_ids = processed_pdb_ids & candidate_pdb_ids
    failed_pdb_ids = sorted(set(failed_pdb_ids))
    atomic_json_dump(assemblies, output_json)
    atomic_json_dump({
        'config': processing_config,
        'processed_pdb_ids': sorted(processed_pdb_ids),
        'failed_pdb_ids': failed_pdb_ids,
        'stats': {
            'candidate_pdb_ids': len(candidate_pdb_ids),
            'processed_pdb_ids': len(processed_pdb_ids),
            'failed_pdb_ids': len(failed_pdb_ids),
            'assemblies': len(assemblies),
        },
    }, metadata_path(output_json))


def main(args: argparse.Namespace) -> None:
    mmcif_data_dir = find_mmcif_data_dir(args.pdb_mmcif_dir)
    obsolete_pdb_ids = collect_obsolete_pdb_ids(args.pdb_mmcif_dir) if args.exclude_obsolete else set()

    cif_file_by_pdb_id = {}
    for file_path in iter_cif_files(mmcif_data_dir):
        pdb_id = pdb_id_from_filename(file_path)
        if pdb_id and pdb_id not in obsolete_pdb_ids:
            cif_file_by_pdb_id[pdb_id] = file_path

    candidate_pdb_ids = set(cif_file_by_pdb_id)
    print(f'Found {len(candidate_pdb_ids)} candidate mmCIF files')

    processing_config = current_processing_config(args)
    merged_assemblies_dict, processed_pdb_ids = load_existing_output(
        args.output_json,
        processing_config,
        force_full_parse=args.force_full_parse,
        strict_metadata=args.strict_metadata,
    )

    before_filter_count = len(merged_assemblies_dict)
    merged_assemblies_dict = {
        unique_id: assembly
        for unique_id, assembly in merged_assemblies_dict.items()
        if assembly_pdb_id(unique_id) in candidate_pdb_ids
    }
    removed_assemblies = before_filter_count - len(merged_assemblies_dict)
    processed_pdb_ids = processed_pdb_ids & candidate_pdb_ids
    if removed_assemblies:
        print(f'Removed {removed_assemblies} assemblies from obsolete or missing PDB IDs')

    missing_pdb_ids = sorted(candidate_pdb_ids - processed_pdb_ids)
    cif_files = [cif_file_by_pdb_id[pdb_id] for pdb_id in missing_pdb_ids]
    print(f'Reusing {len(processed_pdb_ids)} processed PDB IDs')
    print(f'Parsing {len(cif_files)} unfinished PDB IDs')

    failed_pdb_ids = []
    processed_since_checkpoint = 0

    def consume_result(result: dict) -> None:
        nonlocal processed_since_checkpoint
        pdb_id = str(result.get('pdb_id') or '').upper()
        if not pdb_id:
            return
        if result.get('ok'):
            processed_pdb_ids.add(pdb_id)
            merged_assemblies_dict.update(result.get('assemblies') or {})
        else:
            failed_pdb_ids.append(pdb_id)
        processed_since_checkpoint += 1
        if args.checkpoint_interval > 0 and processed_since_checkpoint >= args.checkpoint_interval:
            write_parse_outputs(
                merged_assemblies_dict,
                args.output_json,
                processing_config,
                processed_pdb_ids,
                failed_pdb_ids,
                candidate_pdb_ids,
            )
            print(
                f'Checkpoint saved after {len(processed_pdb_ids)} processed PDB IDs '
                f'and {len(set(failed_pdb_ids))} failed PDB IDs'
            )
            processed_since_checkpoint = 0

    if cif_files:
        parse_func = partial(parse_cif)
        if args.n_cpu == 1:
            for result in tqdm(
                map(parse_func, cif_files),
                total=len(cif_files),
                desc='Parsing mmCIF files',
            ):
                consume_result(result)
        else:
            with mp.Pool(args.n_cpu) as process_pool:
                for result in tqdm(
                    process_pool.imap_unordered(parse_func, cif_files, chunksize=args.chunksize),
                    total=len(cif_files),
                    desc='Parsing mmCIF files',
                ):
                    consume_result(result)

    write_parse_outputs(
        merged_assemblies_dict,
        args.output_json,
        processing_config,
        processed_pdb_ids,
        failed_pdb_ids,
        candidate_pdb_ids,
    )

    print(f'Saved parsed assemblies to: {args.output_json}')
    print(f'Saved parser metadata to: {metadata_path(args.output_json)}')
    if failed_pdb_ids:
        print(f'Failed PDB IDs will be retried on the next run: {len(set(failed_pdb_ids))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract structural assembly data from PDB mmCIF files with resume/cache support',
    )
    parser.add_argument(
        'pdb_mmcif_dir',
        type=str,
        help='Root directory containing divided/ or mmcif_files/ and optionally obsolete/ or obsolete.dat',
    )
    parser.add_argument(
        '-o',
        '--output_json',
        type=str,
        default=Config.data.processed_PDBmmcif_path,
        help='Output JSON path',
    )
    parser.add_argument(
        '--exclude_obsolete',
        '-obs',
        '--obsolete',
        action='store_true',
        help='Exclude obsolete PDB entries',
    )
    parser.add_argument(
        '--force_full_parse',
        action='store_true',
        help='Ignore existing output JSON and parse every candidate file',
    )
    parser.add_argument(
        '--strict_metadata',
        action='store_true',
        help='Require an output JSON metadata sidecar before reusing existing data',
    )
    parser.add_argument(
        '-n',
        '--n_cpu',
        type=int,
        default=mp.cpu_count(),
        help='Number of CPU cores',
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=1,
        help='Multiprocessing chunk size',
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=8000,
        help='Write output JSON and metadata after this many completed PDB IDs. Use 0 to disable.',
    )

    args = parser.parse_args()
    args.chunksize = max(1, args.chunksize)
    args.checkpoint_interval = max(0, args.checkpoint_interval)
    args.pdb_mmcif_dir = os.path.abspath(args.pdb_mmcif_dir)
    args.output_json = os.path.abspath(args.output_json)
    main(args)
