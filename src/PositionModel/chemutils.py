import random

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem,RDConfig
from rdkit.Chem import Draw,QED
from rdkit.Chem import Descriptors
import networkx as nx
from copy import deepcopy
import  itertools as it
import numpy as np 
import torch
import dgl
import sys,os
import pickle
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def check_validity(s,num_bonds):
    # print(s)
    if type(s) == np.ndarray:
        s = s[0]
    # print(s)
    if val_dict[s] >= num_bonds:
        return True
    else:
        return False

def penalized_logp(s):
    mol = Chem.MolFromSmiles(s)
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    #
    log_p = Descriptors.MolLogP(mol)
    # log_p = logp_modifier(log_p)
    SA = -sascorer.calculateScore(mol)
    #
    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p+normalized_SA+normalized_cycle

def get_sascore(smile):
    tmp_mol = Chem.MolFromSmiles(smile)
    score = sascorer.calculateScore(tmp_mol)
    return score

def get_qedscore(smile):
    tmp_mol = Chem.MolFromSmiles(smile)
    score = QED.qed(tmp_mol)
    return score

def getbond_type(bond):
    bond_type_enum = [Chem.BondType.AROMATIC,
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE]
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        return 1
    return bond_type_enum.index(bond.GetBondType())


def adjust_intersectant_ring(ring_lst):
	intersected_parent = {}
	intersected_child = {}
	ring_lst_new = []

	for i in range(len(ring_lst)):
		intersected_parent[i]=i
		intersected_child[i] = []
	for i in range(len(ring_lst)):
		for j in range(i+1,len(ring_lst)):
			if i == j :
				continue
			if len(set(ring_lst[i]).intersection(set(ring_lst[j]))) > 0:
				intersected_parent[j] = intersected_parent[i]
				intersected_child[intersected_parent[i]].append(j)
	# print(intersected_parent)

	for i in range(len(ring_lst)):
		if intersected_parent[i] == i:
			tmpset = set(ring_lst[i])
			for j in intersected_child[i]:
				tmpset = tmpset.union(set(ring_lst[j]))
			ring_lst_new.append(list(tmpset))
	return ring_lst_new


def getcomplete_graph(num_nodes=36,device='cuda'):
    if device=='cuda':
        tmp_graph = dgl.graph(data=[],num_nodes=num_nodes).to(device)
        tmp_graph.ndata['atom_type'] = torch.ones(num_nodes, dtype=torch.int).cuda()*(len(vocabulary))
        tmp_graph.ndata['label'] = torch.zeros(num_nodes,dtype=torch.int).cuda()
    else:
        tmp_graph = dgl.graph(data=[],num_nodes=num_nodes)
        tmp_graph.ndata['atom_type'] = torch.ones(num_nodes, dtype=torch.int) * (len(vocabulary))
        tmp_graph.ndata['label'] = torch.zeros(num_nodes, dtype=torch.int)

    return tmp_graph


def load_vocabulary():
    datafile = "../data/train_set/vocabulary.txt"
    with open(datafile, 'r') as fin:
        lines = fin.readlines()
    vocabulary = [line.split()[0] for line in lines]
    return vocabulary

def load_valenc():
    f_read = open('val_dict.pkl', 'rb')
    tmp_dict = pickle.load(f_read)
    return tmp_dict
vocabulary = load_vocabulary()
bondtype_list = [Chem.BondType.AROMATIC,
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE]

val_dict = load_valenc()



def ith_substructure_is_atom(i):
    substructure = vocabulary[i]
    mol = Chem.MolFromSmiles(substructure)
    return True if len(mol.GetAtoms())==1 else False

def word2idx(word):
    return vocabulary.index(word)

def isinword(word):
    return word in vocabulary


def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=False)
    return np.array(fp)
    ### shape: (1024,)


def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    if amol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(b, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def similarity_mols(a, b):
    # print(b)
    if a is None or b is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(a, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(b, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def canonical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None 
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def smiles2mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None 
    if mol is None: 
        return None
    Chem.Kekulize(mol,clearAromaticFlags=True)
    return mol 

def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []

    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques = adjust_intersectant_ring(cliques)
    cliques_smiles = []
    for clique in cliques:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        cliques_smiles.append(clique_smiles)
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]

    return cliques_smiles + atom_not_in_rings_list

def is_valid(smiles):
    word_lst = smiles2word(smiles)
    word_set = set(word_lst)
    return word_set.issubset(vocabulary)     


def is_valid_mol(mol):
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
    except:
        return False 
    if smiles.strip() == '':
        return False 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False 
    return True



def smiles2feature_train(smiles, device='cuda',flag=0):

    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    clique_lst = adjust_intersectant_ring(clique_lst)
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]

    ood_set = []
    clique_lst_new = []
    nood_idx_set = []
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        if clique_smiles in vocabulary:
            idx_lst.append(word2idx(clique_smiles))
            clique_lst_new.append(clique)
        else:
            for atom in clique:
                ood_set.append(atom)
                atom_idx_not_in_rings_list.append(atom)
                atom_symbol_not_in_rings_list.append(mol.GetAtomWithIdx(atom).GetSymbol())

    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(atom))
    N = len(idx_lst)
    node_degrees = np.zeros(N, dtype=np.int)
    node_bonds = np.zeros(N, dtype=np.int)
    tmp_graph = getcomplete_graph(N, device=device)

    atomidx_2substridx = dict()
    substructure_lst = clique_lst_new + atom_idx_not_in_rings_list

    for idx, substructure in enumerate(substructure_lst):
        if type(substructure) == list:
            for atom in substructure:
                atomidx_2substridx[atom] = idx
                if atom not in ood_set and idx not in nood_idx_set:
                    nood_idx_set.append(idx)
        else:
            atomidx_2substridx[substructure] = idx
            if substructure not in ood_set and idx not in nood_idx_set:
                nood_idx_set.append(idx)

    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = getbond_type(bond)
            idx1 = atomidx_2substridx[a1]
            idx2 = atomidx_2substridx[a2]
            node_degrees[idx1] += 1
            node_degrees[idx2] += 1
            if btype== 0 or btype == 1:
                node_bonds[idx1] += 1
                node_bonds[idx2] += 1
            elif btype== 2:
                node_bonds[idx1] += 2
                node_bonds[idx2] += 2
            elif btype== 3:
                node_bonds[idx1] += 3
                node_bonds[idx2] += 3
            if device == 'cuda':
                tmp_graph.add_edges(idx1,idx2,data={'bond_type':torch.tensor([btype]).cuda()})
                tmp_graph.add_edges(idx2,idx1, data={'bond_type': torch.tensor([btype]).cuda()})
            else:
                tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype])})
                tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype])})
        else:
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = getbond_type(bond)
            idx1 = atomidx_2substridx[a1]
            idx2 = atomidx_2substridx[a2]
            if idx1 not in nood_idx_set or idx2 not in nood_idx_set:
                node_degrees[idx1] += 1
                node_degrees[idx2] += 1
                if btype == 0 or btype == 1:
                    node_bonds[idx1] += 1
                    node_bonds[idx2] += 1
                elif btype == 2:
                    node_bonds[idx1] += 2
                    node_bonds[idx2] += 2
                elif btype == 3:
                    node_bonds[idx1] += 3
                    node_bonds[idx2] += 3
                if device == 'cuda':
                    tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype]).cuda()})
                    tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype]).cuda()})
                else:
                    tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype])})
                    tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype])})

    node_id_lst = [i for i in range(N)]
    for i, v in enumerate(idx_lst):
        tmp_graph.ndata['atom_type'][i] = v

    tmp_graph_alter = tmp_graph
    if flag == 0:
        change_idx_lst = np.random.choice(node_id_lst, 1, replace=False)
        change_idx_bonds = node_bonds[change_idx_lst]
        for change_idx in change_idx_lst:
            selected_type = ''
            for itertime in range(10):
                selected_type = np.random.choice(vocabulary,1,replace=False)
                if check_validity(selected_type,change_idx_bonds):
                    break
            if not check_validity(selected_type, change_idx_bonds):
                return None,None
            selected_type_idx = vocabulary.index(selected_type)
            if selected_type_idx != tmp_graph_alter.ndata['atom_type'][change_idx]:
                tmp_graph_alter.ndata['atom_type'][change_idx]=selected_type_idx
                tmp_graph_alter.ndata['label'][change_idx]=1

    elif flag == 1:
        num_append_nodes = 1
        for i in range(num_append_nodes):
            selected_type = np.random.choice(vocabulary, 1, replace=False)
            selected_node_to_append = None
            for itertime in range(10):
                selected_node_to_append = np.random.choice(node_id_lst, 1)
                selected_smiles = vocabulary[idx_lst[int(selected_node_to_append)]]
                if check_validity(selected_smiles, node_bonds[selected_node_to_append] + 1):
                    break
            if not check_validity(selected_smiles, node_bonds[selected_node_to_append] + 1):
                return None,None

            if device == 'cuda':
                tmp_graph_alter.add_nodes(1, data={
                    'atom_type': torch.tensor([vocabulary.index(selected_type)], dtype=torch.int32).cuda()})
                tmp_graph_alter.add_edges(N, selected_node_to_append, data={'bond_type': torch.tensor([1]).cuda()})
                tmp_graph_alter.add_edges(selected_node_to_append, N, data={'bond_type': torch.tensor([1]).cuda()})
                tmp_graph_alter.ndata['label'][N] = 2
            else:
                tmp_graph_alter.add_nodes(1, data={
                    'atom_type': torch.tensor([vocabulary.index(selected_type)], dtype=torch.int32)})
                tmp_graph_alter.add_edges(N, selected_node_to_append, data={'bond_type': torch.tensor([1])})
                tmp_graph_alter.add_edges(selected_node_to_append, N, data={'bond_type': torch.tensor([1])})
                tmp_graph_alter.ndata['label'][N] = 2
    else:
        leaf_idx_lst = list(np.where(node_degrees == 1)[0])
        # leaf_idx_lst = [i for i in leaf_idx_lst if i not in change_idx_lst]
        extend_pos_lst = []
        if len(leaf_idx_lst) != 0:
            del_leaf_idx_lst = np.random.choice(leaf_idx_lst, 1, replace=False)
            for leaf_id in del_leaf_idx_lst:
                tmpes = tmp_graph_alter.in_edges(leaf_id,form='eid').data.tolist()
                for tmp_edge in tmpes:
                    tmp_graph_alter.ndata['label'][tmp_graph_alter.edges()[0][[tmp_edge]]] = 3
                tmp_graph_alter.remove_nodes(leaf_id)


    return tmp_graph_alter, tmp_graph_alter.ndata['label'].type(torch.LongTensor)

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def add_atom_at_position(editmol, position_idx, new_atom, new_bond):
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    new_atom_idx = rwmol.AddAtom(new_atom)
    rwmol.AddBond(position_idx, new_atom_idx, order = new_bond)
    if not is_valid_mol(rwmol):
        return None  
    try:
        rwmol.UpdatePropertyCache()
    except:
        return None
    smiles = Chem.MolToSmiles(rwmol)
    assert '.' not in smiles
    return canonical(smiles)

def a2a_at_position(editmol, position_idx, new_atom):
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    rwmol.GetAtomWithIdx(position_idx).SetAtomicNum(new_atom.GetAtomicNum())
    try:
        # rwmol =
        rwmol.UpdatePropertyCache()
    except:
        return None
    if not is_valid_mol(rwmol):
        return None

    smiles = Chem.MolToSmiles(rwmol)
    if '.' in smiles:
        return None
    return canonical(smiles)

def s2a_at_position(editmol, position_idx_lst, new_atom):
    new_atom = Chem.rdchem.Atom(new_atom)
    neighbor_set = []
    bond_type_set = []
    for sub_idx in position_idx_lst:
        tmp_atom = editmol.GetAtomWithIdx(sub_idx)
        tmp_bonds = tmp_atom.GetBonds()
        for bond in tmp_bonds:
            tmp_start_idx = bond.GetBeginAtomIdx()
            tmp_end_idx = bond.GetEndAtomIdx()
            if tmp_start_idx not in position_idx_lst:
                neighbor_set.append(tmp_start_idx)
                bond_type_set.append(bond.GetBondType())
            if tmp_end_idx not in position_idx_lst:
                neighbor_set.append(tmp_end_idx)
                bond_type_set.append(bond.GetBondType())
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx not in position_idx_lst:
            tmp_new_atom = copy_atom(atom)
            new_idx = new_mol.AddAtom(tmp_new_atom)
            old_idx2new_idx[old_idx] = new_idx

    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        if i1 not in position_idx_lst and i2 not in position_idx_lst:
            i1_new = old_idx2new_idx[i1]
            i2_new = old_idx2new_idx[i2]
            bt = bond.GetBondType()
            new_mol.AddBond(i1_new, i2_new, bt)

    new_idx = new_mol.AddAtom(new_atom)
    for neib_idx,neib in enumerate(neighbor_set):
        new_mol.AddBond(new_idx,old_idx2new_idx[neib],bond_type_set[neib_idx])
    if not is_valid_mol(new_mol):
        return None
    try:
        new_mol.UpdatePropertyCache()
    except:
        return None
    smiles = Chem.MolToSmiles(new_mol)
    if '.' in smiles:
        return None
    return canonical(smiles)

def a2s_at_position(editmol, position_idx, fragment):
    neighbor_set = []
    bond_type_set = []
    for sub_idx in [position_idx]:
        tmp_atom = editmol.GetAtomWithIdx(sub_idx)
        tmp_bonds = tmp_atom.GetBonds()
        for bond in tmp_bonds:
            tmp_start_idx = bond.GetBeginAtomIdx()
            tmp_end_idx = bond.GetEndAtomIdx()
            if tmp_start_idx != position_idx:
                neighbor_set.append(tmp_start_idx)
                bond_type_set.append(bond.GetBondType())
            if tmp_end_idx != position_idx:
                neighbor_set.append(tmp_end_idx)
                bond_type_set.append(bond.GetBondType())

    new_smiles_set = set()

    fragment_mol = Chem.MolFromSmiles(fragment)
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx != position_idx:
            new_atom = copy_atom(atom)
            new_idx = new_mol.AddAtom(new_atom)
            old_idx2new_idx[old_idx] = new_idx
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        if i1 != position_idx and i2 != position_idx:
            i1_new = old_idx2new_idx[i1]
            i2_new = old_idx2new_idx[i2]
            bt = bond.GetBondType()
            new_mol.AddBond(i1_new, i2_new, bt)

    old_idx2new_idx2 = {}
    new_atom_idx_lst = []
    for atom in fragment_mol.GetAtoms():
        old_atom_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_atom_idx = new_mol.AddAtom(new_atom)
        new_atom_idx_lst.append(new_atom_idx)
        old_idx2new_idx2[old_atom_idx] = new_atom_idx
    for bond in fragment_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx2[a1]
        i2 = old_idx2new_idx2[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt)

    append_site_enu = list(it.permutations(new_atom_idx_lst, len(bond_type_set)))
    for append_strategy in append_site_enu:
        copy_mol = deepcopy(new_mol)
        for bond_idx,append_site in enumerate(append_strategy):
            copy_mol.AddBond(old_idx2new_idx[neighbor_set[bond_idx]],append_site , bond_type_set[bond_idx])
        if is_valid_mol(copy_mol):
            try:
                copy_mol.UpdatePropertyCache()
                new_smiles = Chem.MolToSmiles(copy_mol)
                new_smiles = canonical(new_smiles)
                if new_smiles is not None:
                    if '.' not in new_smiles:
                        new_smiles_set.add(new_smiles)
            except:
                pass

    return new_smiles_set


def s2s_at_position(editmol, position_idx_lst, fragment):
    neighbor_set = []
    bond_type_set = []
    for sub_idx in position_idx_lst:
        tmp_atom = editmol.GetAtomWithIdx(sub_idx)
        tmp_bonds = tmp_atom.GetBonds()
        for bond in tmp_bonds:
            tmp_start_idx = bond.GetBeginAtomIdx()
            tmp_end_idx = bond.GetEndAtomIdx()
            if tmp_start_idx not in position_idx_lst:
                neighbor_set.append(tmp_start_idx)
                bond_type_set.append(bond.GetBondType())
            if tmp_end_idx not in position_idx_lst:
                neighbor_set.append(tmp_end_idx)
                bond_type_set.append(bond.GetBondType())

    new_smiles_set = set()

    fragment_mol = Chem.MolFromSmiles(fragment)
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx not in position_idx_lst:
            new_atom = copy_atom(atom)
            new_idx = new_mol.AddAtom(new_atom)
            old_idx2new_idx[old_idx] = new_idx
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        if i1 not in position_idx_lst and i2 not in position_idx_lst:
            i1_new = old_idx2new_idx[i1]
            i2_new = old_idx2new_idx[i2]
            bt = bond.GetBondType()
            new_mol.AddBond(i1_new, i2_new, bt)

    old_idx2new_idx2 = {}
    new_atom_idx_lst = []
    for atom in fragment_mol.GetAtoms():
        old_atom_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_atom_idx = new_mol.AddAtom(new_atom)
        new_atom_idx_lst.append(new_atom_idx)
        old_idx2new_idx2[old_atom_idx] = new_atom_idx
    for bond in fragment_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx2[a1]
        i2 = old_idx2new_idx2[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt)

    append_site_enu = list(it.permutations(new_atom_idx_lst, len(bond_type_set)))
    for append_strategy in append_site_enu:
        copy_mol = deepcopy(new_mol)
        for bond_idx, append_site in enumerate(append_strategy):
            copy_mol.AddBond(old_idx2new_idx[neighbor_set[bond_idx]], append_site, bond_type_set[bond_idx])
        if is_valid_mol(copy_mol):
            try:
                copy_mol.UpdatePropertyCache()
                new_smiles = Chem.MolToSmiles(copy_mol)
                new_smiles = canonical(new_smiles)
                if new_smiles is not None:
                    if '.' not in new_smiles:
                        new_smiles_set.add(new_smiles)
            except:
                pass

    return new_smiles_set


def add_fragment_at_position(editmol, position_idx, fragment, new_bond):
    new_smiles_set = set()
    fragment_mol = Chem.MolFromSmiles(fragment)
    current_atom = editmol.GetAtomWithIdx(position_idx)
    neighbor_atom_set = set()

    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
        assert old_idx == new_idx
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        i1 = a1.GetIdx()
        i2 = a2.GetIdx()
        i1_new = old_idx2new_idx[i1]
        i2_new = old_idx2new_idx[i2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1_new, i2_new, bt)
        if (i1==position_idx or i2==position_idx) and (a1.IsInRing() and a2.IsInRing()):
            neighbor_atom_set.add(i1_new)
            neighbor_atom_set.add(i2_new)
    if neighbor_atom_set != set():
        neighbor_atom_set.remove(old_idx2new_idx[position_idx])

    new_atom_idx_lst = []
    old_idx2new_idx2 = dict()
    for atom in fragment_mol.GetAtoms():
        old_atom_idx = atom.GetIdx()
        new_atom = copy_atom(atom)
        new_atom_idx = new_mol.AddAtom(new_atom)
        new_atom_idx_lst.append(new_atom_idx)
        old_idx2new_idx2[old_atom_idx] = new_atom_idx 
    for bond in fragment_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        i1 = old_idx2new_idx2[a1]
        i2 = old_idx2new_idx2[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(i1, i2, bt)

    for i in new_atom_idx_lst:
        copy_mol = deepcopy(new_mol)
        copy_mol.AddBond(old_idx2new_idx[position_idx], i, new_bond)
        if is_valid_mol(copy_mol):
            try:
                copy_mol.UpdatePropertyCache()
                new_smiles = Chem.MolToSmiles(copy_mol)
                new_smiles = canonical(new_smiles)
                if new_smiles is not None:
                    if '.' not in new_smiles:
                        new_smiles_set.add(new_smiles)
            except:
                pass  

    return new_smiles_set

def delete_substructure_at_idx(editmol, atom_idx_lst):
    edit_smiles = Chem.MolToSmiles(editmol)
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    old_idx2new_idx = dict()
    for atom in editmol.GetAtoms():
        old_idx = atom.GetIdx()
        if old_idx in atom_idx_lst: 
            continue 
        new_atom = copy_atom(atom)
        new_idx = new_mol.AddAtom(new_atom)
        old_idx2new_idx[old_idx] = new_idx 
    for bond in editmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if a1 in atom_idx_lst or a2 in atom_idx_lst:
            continue 
        a1_new = old_idx2new_idx[a1]
        a2_new = old_idx2new_idx[a2]
        bt = bond.GetBondType()
        new_mol.AddBond(a1_new, a2_new, bt) 

    if not is_valid_mol(new_mol):
        return None
    try:
        new_mol.UpdatePropertyCache()
    except:
        return None
    smi = Chem.MolToSmiles(new_mol)
    return canonical(smi)

