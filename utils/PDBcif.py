import os
from collections import defaultdict

# required categories
onlyCategories=['_citation','_entity','_entity_poly','_struct_ref','_struct', '_struct_keywords','_pdbx_struct_assembly','_pdbx_struct_assembly_gen','_pdbx_struct_assembly_auth_evidence','_struct_asym']


class PDBCif(object):
    def __init__(self,
        mmcif_dict:dict,
    ):
        self.mmcif_dict = mmcif_dict
        self.pdb_id = self.mmcif_dict['_struct']['entry_id']
        self.pubmed_id = self.mmcif_dict['_citation']['pdbx_database_id_PubMed'] if '_citation' in self.mmcif_dict and 'pdbx_database_id_PubMed' in self.mmcif_dict['_citation'] else None
        #release_date = self.mmcif_dict['_pdbx_audit_revision_history']['revision_date']
        release_date = self.mmcif_dict.get('_pdbx_audit_revision_history', {'revision_date': None}).get('revision_date', None)
        if isinstance(release_date, str):
            release_date = [release_date]
        self.release_date = release_date[0] if release_date else None
    
    def parse_operation_expression(self, expression:str):
        """
        Parse operation expressionq

        Args:
            expression (str): operation expression

        Returns:
            list: list of operations
        """
        operations = []
        stops = [ "," , "-" , ")" ]

        currentOp = ""
        i = 1
        
        # Iterate over the operation expression
        while i in range(1, len(expression) - 1):
            pos = i

            # Read an operation
            while expression[pos] not in stops and pos < len(expression) - 1 : 
                pos += 1    
            currentOp = expression[i : pos]

            # Handle single operations
            if expression[pos] != "-" :
                operations.append(currentOp)
                i = pos

            # Handle ranges
            if expression[pos] == "-" :
                pos += 1
                i = pos
                
                # Read in the range's end value
                while expression[pos] not in stops :
                    pos += 1
                end = int(expression[i : pos])
                
                # Add all the operations in [currentOp, end]
                for val in range((int(currentOp)), end + 1) :
                    operations.append(str(val))
                i = pos
            i += 1
        return operations
    
    def parse_sequence(self):
        """
        Parse sequence from mmcif_dict
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
        try:
            entity_poly = self.mmcif_dict['_entity_poly']
            seq_dict = dict()
            chain2entity = self.parse_struct_asym()
            # if empty, return empty dict
            if not chain2entity:
                return dict(), dict()
            # covnert to list if only have one entity
            if isinstance(entity_poly['entity_id'], str):
                entity_poly = {k:[v] for k, v in entity_poly.items()}
            entity_ids = set(entity_poly['entity_id'])
            chain2entity = {k:v for k,v in chain2entity.items() if v in entity_ids}
            for i in range(len(entity_poly['entity_id'])):
                entity_id = entity_poly['entity_id'][i]
                seq_dict[entity_id] = {
                    'id': entity_id,
                    'type': entity_poly['type'][i],
                    'sequence': entity_poly['pdbx_seq_one_letter_code_can'][i].replace('\n', ''),
                    'strand_id': [ i for i,v in chain2entity.items() if v == entity_id]
                }
            return seq_dict, chain2entity

        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_sequence()')
            return dict(), dict()
    
    def parse_struct_asym(self):
        """
        Parse struct_asym from mmcif_dict
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
        try:
            struct_asym = self.mmcif_dict['_struct_asym']
            if isinstance(struct_asym['id'], str):
                struct_asym = {k:[v] for k, v in struct_asym.items()}
            chain2entity = dict()
            for i in range(len(struct_asym['id'])):
                chain2entity[struct_asym['id'][i]] = struct_asym['entity_id'][i]
            return chain2entity
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_struct_asym()')
            return dict()
    
    def get_assembly_with_operation(self, mmcif_dict:dict, chain2entity:dict):

        asse_sto_dict = dict()
        
        oper_list = mmcif_dict['_pdbx_struct_oper_list']
        if isinstance(oper_list['id'], str):
            oper_list = {k:[v] for k, v in oper_list.items()}
        if isinstance(mmcif_dict['_pdbx_struct_assembly_gen']['assembly_id'], str):
            mmcif_dict['_pdbx_struct_assembly_gen'] = {k:[v] for k, v in mmcif_dict['_pdbx_struct_assembly_gen'].items()}

        for i,assembly_id in enumerate(mmcif_dict['_pdbx_struct_assembly_gen']['assembly_id']):

            if assembly_id not in asse_sto_dict:
                asse_sto_dict[assembly_id] = dict()

            oper = []
            oper2 = []

            oper_expression = mmcif_dict['_pdbx_struct_assembly_gen']['oper_expression'][i]
            # Count the number of left parentheses in the operation expression
            parenCount = oper_expression.count("(")

            # Handles one operation assemblies (e.g., "1")
            if parenCount == 0 : oper.extend(oper_expression.split(","))
            
            # Handles multiple operation assemblies, no Cartesian products (e.g., "(1-5)")
            if parenCount == 1 : oper.extend(self.parse_operation_expression(oper_expression))
            
            # Handles Cartesian product expressions (e.g., "(X0)(1-60)")
            if parenCount == 2 :
                # Break the expression into two parenthesized expressions and parse them
                temp = oper_expression.find(")")
                oper.extend(self.parse_operation_expression(oper_expression[0:temp+1]))
                oper2.extend(self.parse_operation_expression(oper_expression[temp+1:]))

            # Retrieve the asym_id_list, which indicates which atoms to apply the operations to
            asym_id_list = mmcif_dict['_pdbx_struct_assembly_gen']['asym_id_list'][i].split(',')

            temp = (1 > len(oper2)) and 1 or len(oper2)
            # For every operation in the first parenthesized list
            for op1 in oper :
                # For every operation in the second parenthesized list (if there is one)
                for i in range(temp) :		

                    for asym_id in asym_id_list:
                        # must be protein chain
                        if asym_id not in chain2entity:
                            continue
                        entity_id = chain2entity[asym_id]
                        asse_sto_dict[assembly_id][entity_id] = asse_sto_dict[assembly_id].get(entity_id, 0) + 1
        return asse_sto_dict
    
    def chain2count_2_sto(self, chain2count:dict):
        """
        Convert chain2count to stoichiometry
        """
        chain2count_sorted = dict(sorted(chain2count.items(), key=lambda x: x[1], reverse=True))
        stoichiometry = ""
        for k, v in chain2count_sorted.items():
            stoichiometry += f'{k}x{v},'
        return stoichiometry[:-1]

    def parse_struct_ref(self):
        """
        Parse struct_ref from mmcif_dict
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
        try:
            struct_ref = self.mmcif_dict['_struct_ref']
            if isinstance(struct_ref['id'], str):
                struct_ref = {k:[v] for k, v in struct_ref.items()}
            struct_ref_dict = dict()
            for i in range(len(struct_ref['id'])):
                entity_id = struct_ref['entity_id'][i]
                struct_ref_dict[entity_id] = {
                    'id': struct_ref['id'][i],
                    'entity_id': entity_id,
                    'db_name': struct_ref['db_name'][i],
                    'db_code': struct_ref['db_code'][i],
                    'pdbx_db_accession': struct_ref['pdbx_db_accession'][i],
                    'pdbx_seq_one_letter_code': struct_ref['pdbx_seq_one_letter_code'][i].replace('\n', ''),
                }
            return struct_ref_dict
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_struct_ref()')
            return dict()

    def _entity_merge_key(self, entity_id: str, seq_dict: dict, struct_ref_dict: dict):
        seq_info = seq_dict.get(entity_id, {})
        ref_info = struct_ref_dict.get(entity_id, {})
        return (
            seq_info.get('type'),
            seq_info.get('sequence'),
            ref_info.get('db_name'),
            ref_info.get('db_code'),
            ref_info.get('pdbx_db_accession'),
        )

    def _merge_exact_duplicate_entities(
        self,
        entity_count: dict,
        seq_dict: dict,
        struct_ref_dict: dict,
    ):
        groups = defaultdict(list)
        for entity_id in entity_count:
            groups[self._entity_merge_key(entity_id, seq_dict, struct_ref_dict)].append(entity_id)

        merged_count = {}
        merge_groups = {}
        for members in groups.values():
            representative = members[0]
            merged_count[representative] = sum(int(entity_count[entity_id]) for entity_id in members)
            if len(members) > 1:
                merge_groups[representative] = members

        return merged_count, merge_groups

    def _entity_count_identity_key(self, entity_count: dict, seq_dict: dict, struct_ref_dict: dict):
        def sortable_item(item):
            merge_key, count = item
            sortable_key = tuple('' if value is None else str(value) for value in merge_key)
            return sortable_key + (int(count),)

        identity_items = [
            (
                self._entity_merge_key(entity_id, seq_dict, struct_ref_dict),
                int(count),
            )
            for entity_id, count in entity_count.items()
        ]
        return tuple(sorted(identity_items, key=sortable_item))

    def _build_entity_record(
        self,
        representative_entity_id: str,
        member_entity_ids: list,
        count: int,
        seq_dict: dict,
        struct_ref_dict: dict,
    ):
        entity_record = {
            'entity_id': representative_entity_id,
            'count': count,
            'sequence': seq_dict[representative_entity_id]['sequence'],
            'type': seq_dict[representative_entity_id]['type'],
            'strand_id': [
                strand_id
                for entity_id in member_entity_ids
                for strand_id in seq_dict[entity_id]['strand_id']
            ],
        }
        if len(member_entity_ids) > 1:
            entity_record['member_entity_ids'] = member_entity_ids

        if representative_entity_id in struct_ref_dict:
            entity_record['db_name'] = struct_ref_dict[representative_entity_id]['db_name']
            entity_record['db_code'] = struct_ref_dict[representative_entity_id]['db_code']
            entity_record['pdbx_db_accession'] = struct_ref_dict[representative_entity_id]['pdbx_db_accession']
        else:
            entity_record['db_name'] = None
            entity_record['db_code'] = None
            entity_record['pdbx_db_accession'] = None
        return entity_record
    
    def parse_assembly(self, chain2entity:dict=None):
        """
        Parse assembly from mmcif_dict
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
        try:
            if not chain2entity:
                seq_dict, chain2entity = self.parse_sequence()
            assembly = self.mmcif_dict['_pdbx_struct_assembly']
            assembly_gen = self.mmcif_dict['_pdbx_struct_assembly_gen']
            assembly_auth_evidence = self.mmcif_dict.get('_pdbx_struct_assembly_auth_evidence', {'experimental_support': []})

            if isinstance(assembly['id'], str):
                assembly = {k:[v] for k, v in assembly.items()}
            if isinstance(assembly_gen['asym_id_list'], str):
                assembly_gen = {k:[v] for k, v in assembly_gen.items()}
            if isinstance(assembly_auth_evidence['experimental_support'], str):
                assembly_auth_evidence = {k:[v] for k, v in assembly_auth_evidence.items()}

            asse_sto_dict = self.get_assembly_with_operation(self.mmcif_dict, chain2entity)
            assembly_dict = dict()
            # doing filtering, if the sto is the same in different assembly, just keep one
            present_sto = set()
            for i in range(len(assembly['id'])):
                # check if oligomeric_count matchs with our parsed stoichiometry
                our_count = sum(asse_sto_dict[assembly['id'][i]].values())
                if int(assembly['oligomeric_count'][i]) != our_count:
                    print(f"oligomeric_count not match in {self.pdb_id} asse {assembly['id'][i]}: our_count={our_count}, oligomeric_count={assembly['oligomeric_count'][i]}")
                    continue
                
                sto_str = self.chain2count_2_sto(asse_sto_dict[assembly['id'][i]])
                if sto_str in present_sto:
                    continue
                present_sto.add(sto_str)

                assembly_dict[assembly['id'][i]] = {
                    'id': assembly['id'][i],
                    'details': assembly['details'][i],
                    'method_details': assembly['method_details'][i],
                    'oligomeric_details': assembly['oligomeric_details'][i],
                    'oligomeric_count': assembly['oligomeric_count'][i],
                    'chain2count': asse_sto_dict[assembly['id'][i]],
                    'experimental_support': assembly_auth_evidence['experimental_support'][i] if i < len(assembly_auth_evidence['experimental_support']) else None
                }
            return assembly_dict
        except Exception as e:
            print(f'Error: {e} - {self.pdb_id} in parse_assembly()')
            return dict()
    
    def get_assembly(self):
        """
        get assembly
        """
        if not hasattr(self, 'mmcif_dict'):
            print('Error: mmcif_dict is not defined')
            
        seq_dict, chain2entity = self.parse_sequence()
        # error in parsing sequence or struct_asym
        if not seq_dict or not chain2entity:
            return dict()
    
        struct_ref_dict = self.parse_struct_ref()
        assembly = self.parse_assembly(chain2entity=chain2entity)
        assembly_dict = dict()
        present_normalized_sto = set()
        for k,v in assembly.items():
            unique_id = f'{self.pdb_id}_{k}'
            raw_entity_count = v['chain2count']
            entity_count, merge_groups = self._merge_exact_duplicate_entities(
                raw_entity_count,
                seq_dict,
                struct_ref_dict,
            )
            normalized_sto = self._entity_count_identity_key(entity_count, seq_dict, struct_ref_dict)
            if normalized_sto in present_normalized_sto:
                continue
            present_normalized_sto.add(normalized_sto)
            assembly_dict[unique_id] = {
                'unique_id': unique_id,
                'pdb_pubmed_id': self.pubmed_id,
                'entity_count': entity_count,
                'details': v['details'],
                'method_details': v['method_details'],
                'experimental_support': v['experimental_support'] if 'experimental_support' in v else None,
                'release_date': self.release_date,
            }
            if merge_groups:
                assembly_dict[unique_id]['raw_entity_count'] = raw_entity_count
                assembly_dict[unique_id]['entity_merge_groups'] = merge_groups
                assembly_dict[unique_id]['entity_merge_strategy'] = 'exact_sequence_and_identifiers'
            for entity_id, count in entity_count.items():
                member_entity_ids = merge_groups.get(entity_id, [entity_id])
                assembly_dict[unique_id][entity_id] = self._build_entity_record(
                    entity_id,
                    member_entity_ids,
                    count,
                    seq_dict,
                    struct_ref_dict,
                )
            assembly_dict[unique_id]['stoichiometry'] = self.entity_count_2_stoichiometry(assembly_dict[unique_id]['entity_count'])
        return assembly_dict
    
    def entity_count_2_stoichiometry(self, entity_count: dict):
        """
        Convert entity_count to stoichiometry
        """
        PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        stoichiometry = ""
        # sort entity_count, higher values first
        entity_count = dict(sorted(entity_count.items(), key=lambda x: x[1], reverse=True))
        for i, (k, v) in enumerate(entity_count.items()):
            if i >= len(PDB_CHAIN_IDS):
                return 'N/A'
            stoichiometry += f'{PDB_CHAIN_IDS[i]}{int(v)}'
        return stoichiometry
