"""
PDB mmCIF Parser Module

This module provides functionality to parse PDB mmCIF files and extract structural assembly information.
It processes multiple mmCIF files in parallel and generates a consolidated JSON output containing
assembly data for protein structures.

Key Features:
- Parallel processing of mmCIF files using multiprocessing
- Filtering of obsolete PDB entries
- Extraction of specific mmCIF categories relevant to structural assemblies
- Generation of consolidated JSON output

Dependencies:
- pdbecif: For parsing mmCIF files
- multiprocessing: For parallel processing
- tqdm: For progress tracking
- utils.PDBcif: Custom PDB CIF parser
- config: Configuration settings
"""

from pdbecif.mmcif_tools import MMCIF2Dict
import os
import argparse
import multiprocessing as mp
import json
from tqdm import tqdm
from utils.PDBcif import PDBCif
from config import Config

# Categories to extract from mmCIF files - focused on structural and assembly information
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
    '_pdbx_audit_revision_history'
]

# Initialize the mmCIF dictionary parser
mmcif_dict_parser = MMCIF2Dict()


def parse_cif(cif_file_path: str) -> dict:
    """
    Parse a single mmCIF file and extract assembly information.
    
    This function reads an mmCIF file, parses it using the pdbecif library,
    and extracts structural assembly data using the custom PDBCif parser.
    
    Args:
        cif_file_path (str): Path to the mmCIF file to parse
        
    Returns:
        dict: Dictionary containing assembly information for the PDB structure.
              Returns empty dict if parsing fails.
              
    Raises:
        Exception: Propagates any exceptions from the parsing process
    """
    try:
        # Parse the mmCIF file with specified categories
        mmcif_dict = mmcif_dict_parser.parse(cif_file_path, onlyCategories=ONLY_CATEGORIES)
    except Exception as e:
        print(f'Error: {e}, failed in parsing {cif_file_path}')
        return {}

    # Extract PDB ID from the parsed dictionary
    pdb_id = list(mmcif_dict.keys())[0]
    
    # Create custom PDB CIF parser instance
    cif_parser = PDBCif(mmcif_dict=mmcif_dict[pdb_id])
    
    # Extract assembly information
    all_assemblies_dict = cif_parser.get_assembly()
    return all_assemblies_dict


def parse_obsolete(obsolete_file_path: str) -> set:
    """
    Parse obsolete PDB entries file and return set of obsolete PDB IDs.

    This function reads the PDB obsolete entries file and extracts all PDB IDs
    that have been marked as obsolete. These entries are typically replaced
    by newer versions of the same structure.

    Args:
        obsolete_file_path (str): Path to the PDB obsolete entries file

    Returns:
        set: Set of obsolete PDB IDs (4-character codes)
        
    Note:
        The obsolete file format follows PDB standard:
        OBSLTE DATE ENTRY SUCCESSORS (optional)
    """
    obsolete_pdb_ids = set()
    
    with open(obsolete_file_path, 'r') as file:
        for line in file:
            # Skip lines that don't start with 'OBSLTE' (header lines, etc.)
            if not line.startswith('OBSLTE'):
                continue
                
            # Split the line into components
            line_components = line.split()
            
            # Validate line format: OBSLTE, DATE, ENTRY, SUCCESSORS (optional)
            if len(line_components) < 3:
                print(f'Error in parsing obsolete file: {line}')
                continue
                
            # Extract PDB ID (third component)
            pdb_id = line_components[2]
            obsolete_pdb_ids.add(pdb_id)
    
    print(f'Found {len(obsolete_pdb_ids)} obsolete PDB IDs')
    return obsolete_pdb_ids


def main(args: argparse.Namespace) -> None:
    """
    Main function to process PDB mmCIF files and generate assembly data.
    
    This function orchestrates the entire parsing process:
    1. Validates input directories and files
    2. Loads obsolete PDB entries if requested
    3. Collects valid mmCIF files for processing
    4. Processes files in parallel using multiprocessing
    5. Merges results and saves to JSON output
    
    Args:
        args (argparse.Namespace): Command line arguments containing:
            - pdb_mmcif_dir: Directory containing PDB mmCIF data
            - output_json: Output JSON file path
            - obsolete: Whether to filter obsolete entries
            - n_cpu: Number of CPU cores to use
            
    Raises:
        FileNotFoundError: If required directories or files are missing
    """
    # Construct path to mmCIF files directory
    mmcif_files_dir = os.path.join(args.pdb_mmcif_dir, 'mmcif_files')
    if not os.path.exists(mmcif_files_dir):
        raise FileNotFoundError(f'mmcif_files directory not found in {args.pdb_mmcif_dir}')

    # Handle obsolete entries filtering
    obsolete_file_path = os.path.join(args.pdb_mmcif_dir, 'obsolete.dat')
    if args.obsolete:
        if not os.path.exists(obsolete_file_path):
            raise FileNotFoundError(f'Obsolete file {obsolete_file_path} not found')
        obsolete_pdb_ids = parse_obsolete(obsolete_file_path)
    else:
        obsolete_pdb_ids = set()
    
    # Collect all valid mmCIF files (not obsolete, with .cif extension)
    cif_files = [
        file_entry.path for file_entry in os.scandir(mmcif_files_dir) 
        if (file_entry.is_file() and 
            file_entry.name.endswith('.cif') and 
            file_entry.name[:4].upper() not in obsolete_pdb_ids)
    ]

    # Process mmCIF files in parallel
    with mp.Pool(args.n_cpu) as process_pool:
        parsing_results = list(tqdm(
            process_pool.imap_unordered(parse_cif, cif_files, chunksize=10), 
            total=len(cif_files),
            desc="Parsing mmCIF files"
        ))
    
    # Merge all parsing results into a single dictionary
    merged_assemblies_dict = {}
    for result in parsing_results:
        merged_assemblies_dict.update(result)
    
    # Save merged results to JSON file
    with open(args.output_json, 'w') as output_file:
        json.dump(merged_assemblies_dict, output_file)


if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Extract structural assembly data from PDB mmCIF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_PDBmmcif.py /path/to/pdb_mmcif_dir
  python parse_PDBmmcif.py /path/to/pdb_mmcif_dir -o output.json -obs -n 8
        """
    )
    
    # Define command line arguments
    parser.add_argument(
        'pdb_mmcif_dir', 
        type=str, 
        help='Directory containing PDB mmCIF data (must contain mmcif_files/ and obsolete.dat)'
    )
    parser.add_argument(
        '-o', '--output_json', 
        type=str, 
        help='Output JSON file path for assembly data',
        default=Config.data.processed_PDBmmcif_path
    )
    parser.add_argument(
        '-obs', '--obsolete', 
        action="store_true", 
        help='Filter out obsolete PDB entries using obsolete.dat file'
    )
    parser.add_argument(
        '-n', '--n_cpu', 
        type=int, 
        help='Number of CPU cores to use for parallel processing',
        default=mp.cpu_count()
    )
    
    # Parse arguments and convert paths to absolute
    args = parser.parse_args()
    args.pdb_mmcif_dir = os.path.abspath(args.pdb_mmcif_dir)
    args.output_json = os.path.abspath(args.output_json)
    
    # Execute main function
    main(args)
