"""
Protein Data Bank (PDB) and mmCIF file parsing utilities.

This module provides comprehensive tools for parsing PDB files, mmCIF files,
FASTA files, and extracting protein structure and assembly information.
"""

from pdbecif.mmcif_tools import MMCIF2Dict
import os
import multiprocessing as mp
import json
from tqdm import tqdm
import glob
from typing import Dict, List, Tuple, Set, Optional, Any


# Constants
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # 62 chains maximum

# Required categories for PDB mmCIF files
REQUIRED_CATEGORIES = [
    '_citation', '_entity', '_entity_poly', '_struct_ref', '_struct',
    '_struct_keywords', '_pdbx_struct_assembly', '_pdbx_struct_assembly_gen',
    '_pdbx_struct_assembly_auth_evidence', '_struct_asym', '_pdbx_struct_oper_list',
    '_pdbx_audit_revision_history',
]

# Amino acid three-letter to one-letter code mapping
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    'UNK': 'X', 
}

# Amino acid one-letter to three-letter code mapping
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


def pdb2fasta(pdb_file: str) -> Tuple[Dict[str, str], Dict[str, Dict[int, List[str]]]]:
    """
    Parse PDB file and extract sequences and atom content for each chain.
    
    This function reads a PDB file and extracts the amino acid sequences
    and corresponding atom records for each protein chain.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        Tuple containing:
            - Dictionary with chain IDs as keys and sequences as values
            - Dictionary with chain IDs as keys and residue content as values
              (residue_id -> list of atom lines)
              
    Raises:
        FileNotFoundError: If the PDB file doesn't exist
        AssertionError: If sequence and content lengths don't match
    """
    chain_sequences = {}
    chain_content = {}
    
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                amino_acid_three = line[17:20]
                chain_id = line[21]
                residue_id = int(line[22:26])
                
                # Initialize dictionaries for new chains
                if chain_id not in chain_sequences:
                    chain_sequences[chain_id] = {}
                    chain_content[chain_id] = {}
                
                # Convert three-letter to one-letter amino acid code
                amino_acid_one = THREE_TO_ONE[amino_acid_three]
                
                # Store sequence and content
                if residue_id not in chain_sequences[chain_id]:
                    chain_sequences[chain_id][residue_id] = amino_acid_one
                chain_content[chain_id][residue_id] = chain_content[chain_id].get(residue_id, []) + [line]

    # Convert residue dictionaries to sequences
    for chain_id, chain_residues in chain_sequences.items():
        # Sort by residue ID to maintain order
        sorted_residues = dict(sorted(chain_residues.items(), key=lambda x: x[0]))
        chain_sequences[chain_id] = ''.join(sorted_residues.values())
        
        # Verify sequence and content lengths match
        assert len(chain_sequences[chain_id]) == len(chain_content[chain_id]), \
            f"Length of sequence and content is not the same for chain {chain_id}"
    
    # Sort content by residue ID
    for chain_id, chain_content_dict in chain_content.items():
        sorted_content = dict(sorted(chain_content_dict.items(), key=lambda x: x[0]))
        chain_content[chain_id] = sorted_content

    return chain_sequences, chain_content


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """
    Parse FASTA file and extract sequences.
    
    Reads a FASTA file and returns a dictionary mapping sequence names
    to their corresponding amino acid sequences.
    
    Args:
        fasta_file (str): Path to the FASTA file
        
    Returns:
        Dictionary with sequence names as keys and sequences as values
        
    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
    """
    name_to_sequence = {}
    
    with open(fasta_file, "r") as f:
        current_name = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_name = line[1:]  # Remove '>' character
                name_to_sequence[current_name] = ""
            elif current_name is not None:
                name_to_sequence[current_name] += line
                
    return name_to_sequence


class PDBCif:
    """
    Parser for mmCIF files to extract protein structure and assembly information.
    
    This class provides methods to parse mmCIF files and extract various
    types of information including sequences, assemblies, stoichiometry,
    and structural references.
    """
    
    def __init__(self, mmcif_dict: Dict[str, Any]):
        """
        Initialize PDBCif parser with mmCIF dictionary.
        
        Args:
            mmcif_dict (Dict[str, Any]): Dictionary containing parsed mmCIF data
        """
        self.mmcif_dict = mmcif_dict
        self.pdb_id = self.mmcif_dict['_struct']['entry_id']
        
        # Extract PubMed ID if available
        self.pubmed_id = (
            self.mmcif_dict['_citation']['pdbx_database_id_PubMed'] 
            if '_citation' in self.mmcif_dict 
            and 'pdbx_database_id_PubMed' in self.mmcif_dict['_citation'] 
            else None
        )
        
        # Extract release date
        release_date = self.mmcif_dict.get('_pdbx_audit_revision_history', 
                                          {'revision_date': None}).get('revision_date', None)
        if isinstance(release_date, str):
            release_date = [release_date]
        self.release_date = release_date[0] if release_date else None
    
    def parse_operation_expression(self, expression: str) -> List[str]:
        """
        Parse operation expression from mmCIF assembly generation.
        
        Handles various operation expression formats including:
        - Single operations: "1"
        - Multiple operations: "1,2,3"
        - Ranges: "1-5"
        - Complex expressions: "(1-5)", "(X0)(1-60)"
        
        Args:
            expression (str): Operation expression string
            
        Returns:
            List of operation identifiers as strings
        """
        operations = []
        stops = [",", "-", ")"]
        
        i = 1
        while i in range(1, len(expression) - 1):
            pos = i
            
            # Read an operation
            while pos < len(expression) - 1 and expression[pos] not in stops:
                pos += 1
            current_op = expression[i:pos]

            # Handle single operations
            if expression[pos] != "-":
                operations.append(current_op)
                i = pos

            # Handle ranges
            if expression[pos] == "-":
                pos += 1
                i = pos
                
                # Read in the range's end value
                while pos < len(expression) and expression[pos] not in stops:
                    pos += 1
                end = int(expression[i:pos])
                
                # Add all the operations in [current_op, end]
                for val in range(int(current_op), end + 1):
                    operations.append(str(val))
                i = pos
            i += 1
        return operations
    
    def parse_sequence(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        Parse sequence information from mmCIF dictionary.
        
        Extracts entity information, sequences, and chain-to-entity mappings
        from the mmCIF data.
        
        Returns:
            Tuple containing:
                - Dictionary mapping entity IDs to sequence information
                - Dictionary mapping chain IDs to entity IDs
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            return {}, {}
            
        try:
            entity_poly = self.mmcif_dict['_entity_poly']
            seq_dict = {}
            chain2entity = self.parse_struct_asym()
            
            # If empty, return empty dict
            if not chain2entity:
                return {}, {}
                
            # Convert to list if only have one entity
            if isinstance(entity_poly['entity_id'], str):
                entity_poly = {k: [v] for k, v in entity_poly.items()}
                
            entity_ids = set(entity_poly['entity_id'])
            chain2entity = {k: v for k, v in chain2entity.items() if v in entity_ids}
            
            for i in range(len(entity_poly['entity_id'])):
                entity_id = entity_poly['entity_id'][i]
                seq_dict[entity_id] = {
                    'id': entity_id,
                    'type': entity_poly['type'][i],
                    'sequence': entity_poly['pdbx_seq_one_letter_code_can'][i].replace('\n', ''),
                    'strand_id': [i for i, v in chain2entity.items() if v == entity_id]
                }
            return seq_dict, chain2entity

        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_sequence()')
            return {}, {}
    
    def parse_struct_asym(self) -> Dict[str, str]:
        """
        Parse struct_asym information from mmCIF dictionary.
        
        Creates mapping between chain IDs (asym_id) and entity IDs.
        
        Returns:
            Dictionary mapping chain IDs to entity IDs
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            return {}
            
        try:
            struct_asym = self.mmcif_dict['_struct_asym']
            
            if isinstance(struct_asym['id'], str):
                struct_asym = {k: [v] for k, v in struct_asym.items()}
                
            chain2entity = {}
            for i in range(len(struct_asym['id'])):
                chain2entity[struct_asym['id'][i]] = struct_asym['entity_id'][i]
                
            return chain2entity
            
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_struct_asym()')
            return {}
    
    def get_assembly_with_operation(self, mmcif_dict: Dict[str, Any], chain2entity: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        """
        Calculate assembly stoichiometry with operations.
        
        Parses assembly generation information and calculates the stoichiometry
        of each assembly based on the operations applied to chains.
        
        Args:
            mmcif_dict (Dict[str, Any]): mmCIF dictionary
            chain2entity (Dict[str, str]): Mapping of chain IDs to entity IDs
            
        Returns:
            Dictionary mapping assembly IDs to entity stoichiometry
        """
        assembly_sto_dict = {}
        
        oper_list = mmcif_dict['_pdbx_struct_oper_list']
        if isinstance(oper_list['id'], str):
            oper_list = {k: [v] for k, v in oper_list.items()}
            
        if isinstance(mmcif_dict['_pdbx_struct_assembly_gen']['assembly_id'], str):
            mmcif_dict['_pdbx_struct_assembly_gen'] = {
                k: [v] for k, v in mmcif_dict['_pdbx_struct_assembly_gen'].items()
            }

        for i, assembly_id in enumerate(mmcif_dict['_pdbx_struct_assembly_gen']['assembly_id']):
            if assembly_id not in assembly_sto_dict:
                assembly_sto_dict[assembly_id] = {}

            oper = []
            oper2 = []

            oper_expression = mmcif_dict['_pdbx_struct_assembly_gen']['oper_expression'][i]
            # Count the number of left parentheses in the operation expression
            paren_count = oper_expression.count("(")

            # Handles one operation assemblies (e.g., "1")
            if paren_count == 0:
                oper.extend(oper_expression.split(","))
            
            # Handles multiple operation assemblies, no Cartesian products (e.g., "(1-5)")
            if paren_count == 1:
                oper.extend(self.parse_operation_expression(oper_expression))
            
            # Handles Cartesian product expressions (e.g., "(X0)(1-60)")
            if paren_count == 2:
                # Break the expression into two parenthesized expressions and parse them
                temp = oper_expression.find(")")
                oper.extend(self.parse_operation_expression(oper_expression[0:temp+1]))
                oper2.extend(self.parse_operation_expression(oper_expression[temp+1:]))

            # Retrieve the asym_id_list, which indicates which atoms to apply the operations to
            asym_id_list = mmcif_dict['_pdbx_struct_assembly_gen']['asym_id_list'][i].split(',')

            temp = 1 if 1 > len(oper2) else len(oper2)
            
            # For every operation in the first parenthesized list
            for op1 in oper:
                # For every operation in the second parenthesized list (if there is one)
                for i in range(temp):
                    for asym_id in asym_id_list:
                        # Must be protein chain
                        if asym_id not in chain2entity:
                            continue
                        entity_id = chain2entity[asym_id]
                        assembly_sto_dict[assembly_id][entity_id] = assembly_sto_dict[assembly_id].get(entity_id, 0) + 1
                        
        return assembly_sto_dict
    
    def chain2count_2_sto(self, chain2count: Dict[str, int]) -> str:
        """
        Convert chain count dictionary to stoichiometry string.
        
        Args:
            chain2count (Dict[str, int]): Mapping of chain IDs to counts
            
        Returns:
            Stoichiometry string in format "Ax1,Bx2,Cx3"
        """
        chain2count_sorted = dict(sorted(chain2count.items(), key=lambda x: x[1], reverse=True))
        stoichiometry = ""
        
        for k, v in chain2count_sorted.items():
            stoichiometry += f'{k}x{v},'
            
        return stoichiometry[:-1]  # Remove trailing comma

    def parse_struct_ref(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse struct_ref information from mmCIF dictionary.
        
        Extracts database reference information for entities.
        
        Returns:
            Dictionary mapping reference IDs to reference information
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            return {}
            
        try:
            struct_ref = self.mmcif_dict['_struct_ref']
            
            if isinstance(struct_ref['id'], str):
                struct_ref = {k: [v] for k, v in struct_ref.items()}
                
            struct_ref_dict = {}
            for i in range(len(struct_ref['id'])):
                struct_ref_dict[struct_ref['id'][i]] = {
                    'id': struct_ref['id'][i],
                    'entity_id': struct_ref['entity_id'][i],
                    'db_name': struct_ref['db_name'][i],
                    'db_code': struct_ref['db_code'][i],
                    'pdbx_db_accession': struct_ref['pdbx_db_accession'][i],
                    'pdbx_seq_one_letter_code': struct_ref['pdbx_seq_one_letter_code'][i].replace('\n', ''),
                }
                
            return struct_ref_dict
            
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_struct_ref()')
            return {}
    
    def parse_assembly(self, chain2entity: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Parse assembly information from mmCIF dictionary.
        
        Extracts assembly details, stoichiometry, and experimental support
        information for each assembly.
        
        Args:
            chain2entity (Optional[Dict[str, str]]): Mapping of chain IDs to entity IDs
            
        Returns:
            Dictionary mapping assembly IDs to assembly information
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            return {}
            
        try:
            if not chain2entity:
                seq_dict, chain2entity = self.parse_sequence()
                
            assembly = self.mmcif_dict['_pdbx_struct_assembly']
            assembly_gen = self.mmcif_dict['_pdbx_struct_assembly_gen']
            assembly_auth_evidence = self.mmcif_dict.get(
                '_pdbx_struct_assembly_auth_evidence', 
                {'experimental_support': []}
            )

            if isinstance(assembly['id'], str):
                assembly = {k: [v] for k, v in assembly.items()}
                
            if isinstance(assembly_gen['asym_id_list'], str):
                assembly_gen = {k: [v] for k, v in assembly_gen.items()}
                
            if isinstance(assembly_auth_evidence['experimental_support'], str):
                assembly_auth_evidence = {k: [v] for k, v in assembly_auth_evidence.items()}

            assembly_sto_dict = self.get_assembly_with_operation(self.mmcif_dict, chain2entity)
            assembly_dict = {}
            
            # Filtering: if the stoichiometry is the same in different assemblies, just keep one
            present_sto = set()
            
            for i in range(len(assembly['id'])):
                # Check if oligomeric_count matches with our parsed stoichiometry
                our_count = sum(assembly_sto_dict[assembly['id'][i]].values())
                if int(assembly['oligomeric_count'][i]) != our_count:
                    print(f"oligomeric_count not match in {self.pdb_id} asse {assembly['id'][i]}: "
                          f"our_count={our_count}, oligomeric_count={assembly['oligomeric_count'][i]}")
                    continue
                
                sto_str = self.chain2count_2_sto(assembly_sto_dict[assembly['id'][i]])
                if sto_str in present_sto:
                    continue
                present_sto.add(sto_str)

                assembly_dict[assembly['id'][i]] = {
                    'id': assembly['id'][i],
                    'details': assembly['details'][i],
                    'method_details': assembly['method_details'][i],
                    'oligomeric_details': assembly['oligomeric_details'][i],
                    'oligomeric_count': assembly['oligomeric_count'][i],
                    'chain2count': assembly_sto_dict[assembly['id'][i]],
                    'experimental_support': (
                        assembly_auth_evidence['experimental_support'][i] 
                        if i < len(assembly_auth_evidence['experimental_support']) 
                        else None
                    )
                }
                
            return assembly_dict
            
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_assembly()')
            return {}
    
    def get_assembly(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete assembly information.
        
        Parses all assembly-related information including sequences,
        stoichiometry, and database references.
        
        Returns:
            Dictionary mapping unique assembly IDs to complete assembly information
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            return {}
            
        seq_dict, chain2entity = self.parse_sequence()
        
        # Error in parsing sequence or struct_asym
        if not seq_dict or not chain2entity:
            return {}
    
        struct_ref_dict = self.parse_struct_ref()
        assembly = self.parse_assembly(chain2entity=chain2entity)
        assembly_dict = {}
        
        for k, v in assembly.items():
            unique_id = f'{self.pdb_id}_{k}'
            assembly_dict[unique_id] = {
                'unique_id': unique_id,
                'pdb_pubmed_id': self.pubmed_id,
                'entity_count': v['chain2count'],
                'details': v['details'],
                'method_details': v['method_details'],
                'experimental_support': v.get('experimental_support'),
                'release_date': self.release_date,
            }
            
            for entity_id in v['chain2count']:
                if entity_id not in assembly_dict[unique_id]:
                    assembly_dict[unique_id][entity_id] = {
                        'entity_id': entity_id,
                        'count': v['chain2count'][entity_id],
                        'sequence': seq_dict[entity_id]['sequence'],
                        'type': seq_dict[entity_id]['type'],
                        'strand_id': seq_dict[entity_id]['strand_id'],
                    }
                    
                    # Add reference information if available
                    if entity_id in struct_ref_dict:
                        assembly_dict[unique_id][entity_id]['db_name'] = struct_ref_dict[entity_id]['db_name']
                        assembly_dict[unique_id][entity_id]['db_code'] = struct_ref_dict[entity_id]['db_code']
                        assembly_dict[unique_id][entity_id]['pdbx_db_accession'] = struct_ref_dict[entity_id]['pdbx_db_accession']
                    else:
                        assembly_dict[unique_id][entity_id]['db_name'] = None
                        assembly_dict[unique_id][entity_id]['db_code'] = None
                        assembly_dict[unique_id][entity_id]['pdbx_db_accession'] = None
                        
            assembly_dict[unique_id]['stoichiometry'] = self.entity_count_2_stoichiometry(
                assembly_dict[unique_id]['entity_count']
            )
            
        return assembly_dict
    
    def entity_count_2_stoichiometry(self, entity_count: Dict[str, int]) -> str:
        """
        Convert entity count to stoichiometry string.
        
        Converts entity count dictionary to a stoichiometry string
        using chain ID letters (A, B, C, etc.).
        
        Args:
            entity_count (Dict[str, int]): Mapping of entity IDs to counts
            
        Returns:
            Stoichiometry string in format "A2B1C1" or "N/A" if too many entities
        """
        stoichiometry = ""
        
        # Sort entity_count, higher values first
        entity_count = dict(sorted(entity_count.items(), key=lambda x: x[1], reverse=True))
        
        for i, (k, v) in enumerate(entity_count.items()):
            if i >= len(PDB_CHAIN_IDS):
                return 'N/A'
            stoichiometry += f'{PDB_CHAIN_IDS[i]}{int(v)}'
            
        return stoichiometry


def parse_pdb_obsolete(pdb_obsolete_file: str) -> Set[str]:
    """
    Parse obsolete PDB file and extract obsolete PDB IDs.
    
    Reads the PDB obsolete file and extracts all obsolete PDB IDs
    that have been superseded by newer entries.
    
    Args:
        pdb_obsolete_file (str): Path to the obsolete file
        
    Returns:
        Set of obsolete PDB IDs
        
    Raises:
        FileNotFoundError: If the obsolete file doesn't exist
    """
    obsolete_set = set()
    
    with open(pdb_obsolete_file, 'r') as f:
        for line in f:
            # Skip header lines
            if not line.startswith('OBSLTE'):
                continue
            split_line = line.split()
            # OBSLTE, DATE, ENTRY, SUCCESSORS (optional)
            if len(split_line) < 3:
                print(f'Error in parsing obsolete file: {line}')
                continue
            obsolete_set.add(split_line[2])
            
    print(f'Found {len(obsolete_set)} obsolete PDB IDs')
    return obsolete_set


# Global mmCIF parser instance
_mmcif_dict_parser = None


def parse_pdb_cif(pdb_cif_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a PDB mmCIF file and extract assembly information.
    
    Args:
        pdb_cif_file (str): Path to the mmCIF file
        
    Returns:
        Dictionary containing parsed assembly information
        
    Raises:
        FileNotFoundError: If the mmCIF file doesn't exist
    """
    global _mmcif_dict_parser
    
    # Create parser instance if not exists
    if _mmcif_dict_parser is None:
        _mmcif_dict_parser = MMCIF2Dict()
        
    mmcif_dict = _mmcif_dict_parser.parse(pdb_cif_file, onlyCategories=REQUIRED_CATEGORIES)
    pdb_id = list(mmcif_dict.keys())[0]
    cif_parser = PDBCif(mmcif_dict=mmcif_dict[pdb_id])
    all_assemblies_dict = cif_parser.get_assembly()
    return all_assemblies_dict


def parse_assemblies_from_mmcif(
    pdb_mmcif_dir: str,
    output_json_path: str,
    ignore_obsolete: bool = True,
    n_cpu: int = mp.cpu_count()
) -> Dict[str, Dict[str, Any]]:
    """
    Parse assembly information from mmCIF files and save to JSON file.
    
    Processes all mmCIF files in a directory, extracts assembly information,
    and saves the results to a JSON file. Supports parallel processing
    for improved performance.
    
    Args:
        pdb_mmcif_dir (str): Directory containing mmCIF files and obsolete.dat
        output_json_path (str): Path to save the parsed assembly information
        ignore_obsolete (bool): Whether to ignore obsolete PDB IDs
        n_cpu (int): Number of CPUs to use for parallel processing
        
    Returns:
        Dictionary with PDB ID as key and assembly information as value
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If no mmCIF files are found
    """
    # Parse obsolete PDB IDs
    if ignore_obsolete:
        obsolete_file_path = os.path.join(pdb_mmcif_dir, 'obsolete.dat')
        if not os.path.exists(obsolete_file_path):
            raise FileNotFoundError(f'obsolete.dat not found in {pdb_mmcif_dir}')
        obsolete_set = parse_pdb_obsolete(obsolete_file_path)
    else:
        obsolete_set = set()
    
    # Get list of mmCIF files to process
    mmcif_files_dir = os.path.join(pdb_mmcif_dir, 'mmcif_files')
    mmcif_files = [
        f.path for f in os.scandir(mmcif_files_dir) 
        if f.is_file() and f.name.endswith('.cif') 
        and f.name[:4].upper() not in obsolete_set
    ]
    
    if not mmcif_files:
        raise FileNotFoundError(f'No mmCIF files found in {mmcif_files_dir}')
    
    # Process files in parallel or sequentially
    if n_cpu > 1:
        with mp.Pool(n_cpu) as pool:
            results = pool.map(parse_pdb_cif, tqdm(mmcif_files))
    else:
        results = [parse_pdb_cif(mmcif_file) for mmcif_file in tqdm(mmcif_files)]       
    
    # Merge all results
    merged_dict = {}
    for result in results:
        merged_dict.update(result)
    
    # Save results to JSON file
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_json_path, 'w') as f:
        json.dump(merged_dict, f)
        
    return merged_dict




