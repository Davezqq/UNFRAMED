
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem,RDConfig
from rdkit.Chem import Draw,QED
from rdkit.Chem import Descriptors
from copy import deepcopy
import  itertools as it
import numpy as np 
import torch
import dgl
import sys,os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import pickle

f = open('PositionModel/val_dict.pkl', 'rb')
val_dict = pickle.load(f)
# print(pkl_file)

def select_valid_atom(graph,node_degree):
    node_types = graph.ndata['atom_type'].numpy().tolist()
    # print(node_types)
    ab_append = []
    for idx, node_type in enumerate(node_types):
        fragment_smi = vocabulary1[node_type]
        if node_degree[idx]+1>val_dict[fragment_smi]:
            continue
        else:
            ab_append.append(idx)
    return ab_append


def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    # cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

    cycle_list = [list(x) for x in Chem.GetSymmSSSR(mol)]
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

    return normalized_log_p + normalized_SA + normalized_cycle

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
    # if bond.GetBondType() == Chem.BondType.AROMATIC:
    #     return 1
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


def get_graph(num_nodes=36,device='cuda',vocabulary=None):
    if device=='cuda':
        tmp_graph = dgl.graph(data=[],num_nodes=num_nodes).to(device)
        tmp_graph.ndata['atom_type'] = torch.ones(num_nodes, dtype=torch.int).cuda()*(len(vocabulary))
        tmp_graph.ndata['label'] = torch.zeros(num_nodes,dtype=torch.int).cuda()
    else:
        tmp_graph = dgl.graph(data=[],num_nodes=num_nodes)
        tmp_graph.ndata['atom_type'] = torch.ones(num_nodes, dtype=torch.int) * (len(vocabulary))
        tmp_graph.ndata['label'] = torch.zeros(num_nodes, dtype=torch.int)

    return tmp_graph

def load_vocabulary1():
    datafile = "data/train_set/vocabulary.txt"
    with open(datafile, 'r') as fin:
        lines = fin.readlines()
    vocabulary = [line.split()[0] for line in lines]
    return vocabulary

def load_vocabulary_weight():
    datafile = "data/train_set/vocabulary.txt"
    with open(datafile, 'r') as fin:
        lines = fin.readlines()
    counts = np.array([float(line.split()[1]) for line in lines])
    weight = np.zeros_like(counts)
    weight[counts>=50000]=1.0
    weight[counts< 50000] = 10.0
    # print(weight)
    return weight

def load_vocabulary2():
    datafile = "data/train_set/vocabulary.txt"
    with open(datafile, 'r') as fin:
        lines = fin.readlines()
    vocabulary = [line.split()[0] for line in lines]
    return vocabulary

vocabulary1 = load_vocabulary1()
vocabulary2 = load_vocabulary2()
vocabulary_weight = load_vocabulary_weight()
bondtype_list = [Chem.BondType.AROMATIC,
                            Chem.BondType.SINGLE,
                            Chem.BondType.DOUBLE,
                            Chem.BondType.TRIPLE]


def ith_substructure_is_atom(vocabulary,i):
    substructure = vocabulary[i]
    mol = Chem.MolFromSmiles(substructure)
    return True if len(mol.GetAtoms())==1 else False

def word2idx(vocabulary,word):
    # print(word)
    return vocabulary.index(word)



def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=False)
    return np.array(fp)


## similarity of two SMILES 
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
    try:
        Chem.Kekulize(mol,clearAromaticFlags=True)
    except:
        return None
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

def is_valid(smiles,vocabulary):
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

def smiles2expandfeature_new(smiles,num_nodes = 36 , empty_edge = 5, mask_idx_lst=[],act='append'):
    # node_degrees = np.zeros(num_nodes, dtype=np.int)

    ### 0. smiles -> mol
    # if not is_valid(smiles,vocabulary1):
    #     return None
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    clique_lst = adjust_intersectant_ring(clique_lst)
    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    clique_lst_new = []
    ood_set = []
    nood_idx_set = []
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1
        if clique_smiles in vocabulary1:
            idx_lst.append(word2idx(vocabulary1, clique_smiles))
            clique_lst_new.append(clique)
        else:
            for atom in clique:
                ood_set.append(atom)
                atom_idx_not_in_rings_list.append(atom)
                atom_symbol_not_in_rings_list.append(mol.GetAtomWithIdx(atom).GetSymbol())
    # print()
    # print(atom_idx_not_in_rings_list)  ## [0, 1, 2, 3, 11, 12, 13, 14, 21]  nonring atom's index in molecule
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(vocabulary1,atom))
    # print(idx_lst)
    # print(idx_lst) ## [3, 68, 3, 0, 0, 0, 0, 0, 0, 1, 2, 4]
    d = len(vocabulary1)
    N = len(idx_lst)

    tmp_graph = get_graph(N, 'cpu',vocabulary=vocabulary1)
    node_degrees = np.zeros(N, dtype=np.int)
    node_bonds = np.zeros(N, dtype=np.int)
    # tmp_graph = getcomplete_graph(num_nodes, empty_edge, device='cpu')
    # print('idx_lst in gnn:', idx_lst)

    ### 2. substructure_lst & atomidx_2substridx
    ###    map from atom index to substructure index
    atomidx_2substridx = dict()
    substructure_lst = clique_lst_new + atom_idx_not_in_rings_list
    # print(substructure_lst)
    ### [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15], 0, 1, 2, 3, 11, 12, 13, 14, 21]
    ### 4:0  23:0, 22:0, ...   8:1, 7:1, 22:1, ... 16:2, 17:2, 18:2, ... 0:3, 1:4,
    for idx, substructure in enumerate(substructure_lst):
        if type(substructure)==list:
            for atom in substructure:
                atomidx_2substridx[atom] = idx
                if atom not in ood_set and idx not in nood_idx_set:
                    nood_idx_set.append(idx)
        else:
            atomidx_2substridx[substructure] = idx
            if substructure not in ood_set and idx not in nood_idx_set:
                nood_idx_set.append(idx)

    ### 3. adjacency_matrix
    ####### 3.1 atom-atom bonds and atom-ring bonds
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = getbond_type(bond)
            idx1 = atomidx_2substridx[a1]
            idx2 = atomidx_2substridx[a2]
            node_degrees[idx1]+=1
            node_degrees[idx2]+=1
            if btype== 0 or btype == 1:
                node_bonds[idx1] += 1
                node_bonds[idx2] += 1
            elif btype== 2:
                node_bonds[idx1] += 2
                node_bonds[idx2] += 2
            elif btype== 3:
                node_bonds[idx1] += 3
                node_bonds[idx2] += 3
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
                tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype])})
                tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype])})


    # assert np.sum(adjacency_matrix)>=2*(N-1)

    # print(adjacency_matrix, smiles


    for i,v in enumerate(idx_lst):
        tmp_graph.ndata['atom_type'][i] = v

    # mask_idx = np.random.choice([i for i in range(N)], 5, replace=False)
    feature_lst = []
    for mask_idx in mask_idx_lst:
        if act=='alter':
            tmp_graph_copy = tmp_graph.clone()
            origin_word_id = tmp_graph_copy.ndata['atom_type'][mask_idx].item()
            tmp_graph_copy.ndata['atom_type'][mask_idx] = d
            feature_lst.append((tmp_graph_copy,mask_idx,node_bonds[mask_idx],origin_word_id))
        elif act=='append':
            bond_type = 1
            # bond_type = np.random.choice([i for i in range(empty_edge-1)], 1, replace=False)[0]
            # print(tmp_graph)
            tmp_graph_copy = tmp_graph.clone()
            tmp_graph_copy.add_nodes(num=1)
            tmp_graph_copy.ndata['atom_type'][N] = d
            # print(torch.tensor([mask_idx, N]))
            tmp_graph_copy.add_edges(u=torch.tensor([mask_idx, N]),v=torch.tensor([N,mask_idx]),data={'bond_type': torch.tensor([bond_type,bond_type])})
            # new_adj_mat[idx,N] = 1
            # new_adj_mat[N,idx] = 1
            feature_lst.append((tmp_graph_copy, N, mask_idx, bond_type))
        else:
            feature_lst.append(mask_idx)




    return feature_lst,substructure_lst

def smiles2feature_position(smiles,device='cuda'):
    mol = smiles2mol(smiles)
    if mol is None:
        return None

    ### 1. idx_lst
    idx_lst = []
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    clique_lst = adjust_intersectant_ring(clique_lst)
    atom_idx_not_in_rings_list = [atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing()]
    atom_symbol_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]

    # print(clique_lst)  ## [[4, 23, 22, 7, 6, 5], [8, 7, 22, 10, 9], [16, 17, 18, 19, 20, 15]]
    ood_set = []
    nood_idx = []
    clique_lst_new = []
    for clique in clique_lst:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        # print("clique_smiles", clique_smiles)  ## C1=CC=CC=C1, C1=COCC1, C1=CC=CC=C1
        if clique_smiles in vocabulary2:
            idx_lst.append(word2idx(vocabulary2, clique_smiles))
            clique_lst_new.append(clique)
        else:
            for atom in clique:
                ood_set.append(atom)
                atom_idx_not_in_rings_list.append(atom)
                atom_symbol_not_in_rings_list.append(mol.GetAtomWithIdx(atom).GetSymbol())

    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(vocabulary2,atom))
    N = len(idx_lst)
    # print('idx_lst out:',idx_lst)
    node_degrees = np.zeros(N, dtype=np.int)
    node_bonds = np.zeros(N,dtype=np.int)
    tmp_graph = get_graph(N, device=device,vocabulary=vocabulary2)

    atomidx_2substridx = dict()
    substridx_2atomidx = dict()
    substructure_lst = clique_lst_new + atom_idx_not_in_rings_list

    for idx, substructure in enumerate(substructure_lst):
        if type(substructure) == list:
            for atom in substructure:
                atomidx_2substridx[atom] = idx
                if atom not in ood_set and idx not in nood_idx:
                    nood_idx.append(idx)
        else:
            atomidx_2substridx[substructure] = idx
            if substructure not in ood_set and idx not in nood_idx:
                nood_idx.append(idx)


    # print(substructure_lst)


    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = getbond_type(bond)
            idx1 = atomidx_2substridx[a1]
            idx2 = atomidx_2substridx[a2]
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
                tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype],dtype=torch.int32)})
                tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype],dtype=torch.int32)})
        else:
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = getbond_type(bond)
            idx1 = atomidx_2substridx[a1]
            idx2 = atomidx_2substridx[a2]
            if idx1 not in nood_idx or idx2 not in nood_idx:
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
                    tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype], dtype=torch.int32)})
                    tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype], dtype=torch.int32)})


    # leaf_idx_lst = list(np.where(node_degrees == 1)[0])
    for i, v in enumerate(idx_lst):
        tmp_graph.ndata['atom_type'][i] = v

    return [tmp_graph],N,node_degrees,node_bonds

def smiles2feature_train(smiles, device='cpu'):
    mol = smiles2mol(smiles)
    if mol is None:
        return None,None

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
        if clique_smiles in vocabulary1:
            idx_lst.append(word2idx(vocabulary1, clique_smiles))
            clique_lst_new.append(clique)
        else:
            for atom in clique:
                ood_set.append(atom)
                atom_idx_not_in_rings_list.append(atom)
                atom_symbol_not_in_rings_list.append(mol.GetAtomWithIdx(atom).GetSymbol())
    for atom in atom_symbol_not_in_rings_list:
        idx_lst.append(word2idx(vocabulary1,atom))

    d = len(vocabulary1)
    N = len(idx_lst)

    tmp_graph = get_graph(N, device,vocabulary=vocabulary1)
    node_degrees = np.zeros(N, dtype=np.int)

    atomidx_2substridx = dict()
    substructure_lst = clique_lst_new + atom_idx_not_in_rings_list
    ring_idx_lst = []
    not_ring_idx_lst = []
    for idx, substructure in enumerate(substructure_lst):
        if type(substructure) == list:
            ring_idx_lst.append(idx)
            for atom in substructure:
                atomidx_2substridx[atom] = idx
                if atom not in ood_set and idx not in nood_idx_set:
                    nood_idx_set.append(idx)
        else:
            not_ring_idx_lst.append(idx)
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
            if device == 'cuda':
                tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype]).cuda()})
                tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype]).cuda()})
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
                if device == 'cuda':
                    tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype]).cuda()})
                    tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype]).cuda()})
                else:
                    tmp_graph.add_edges(idx1, idx2, data={'bond_type': torch.tensor([btype])})
                    tmp_graph.add_edges(idx2, idx1, data={'bond_type': torch.tensor([btype])})

    atom_idx_lst = list(range(N))
    num_choice_ring = min(max(int(len(atom_idx_lst) * 0.2/2),1),len(ring_idx_lst))
    num_choice_atom = max(min(int(len(atom_idx_lst) * 0.2)-num_choice_ring,len(not_ring_idx_lst)),0)
    mask_ring_idx = []
    if num_choice_ring>0:
        mask_ring_idx = np.random.choice(ring_idx_lst, num_choice_ring, replace=False)
        mask_ring_idx = list(mask_ring_idx)
    mask_atom_idx = np.random.choice(not_ring_idx_lst, num_choice_atom, replace=False)
    mask_atom_idx = list(mask_atom_idx)
    mask_idx = mask_ring_idx+mask_atom_idx
    label = torch.ones(N, dtype=torch.long) * d
    for i, v in enumerate(idx_lst):
        if i in mask_idx:
            tmp_graph.ndata['atom_type'][i] = d
            label[i] = v
        else:
            tmp_graph.ndata['atom_type'][i] = v

    return tmp_graph, label

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def add_atom_at_position(editmol, position_idx, new_atom, new_bond):
    '''
        position_idx:   index of edited atom in editmol
        new_atom: 'C', 'N', 'O', ... 
        new_bond: SINGLE, DOUBLE  
    '''
    ######  1 edit mol 
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    new_atom_idx = rwmol.AddAtom(new_atom)
    rwmol.AddBond(position_idx, new_atom_idx, order = new_bond)
    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() == 'C':
            atom.SetFormalCharge(0)
    ######  2 check valid of new mol 
    if not is_valid_mol(rwmol):
        return None  
    try:
        rwmol.UpdatePropertyCache()
    except:
        return None
    smiles = Chem.MolToSmiles(rwmol)
    assert '.' not in smiles
    return canonical(smiles)

def a2a_at_position(editmol, position_idx, new_atom,node_bonds):
    '''
        position_idx:   index of edited atom in editmol
        new_atom: 'C', 'N', 'O', ...
    '''
    ######  1 edit mol
    new_atom = Chem.rdchem.Atom(new_atom)
    rwmol = deepcopy(editmol)
    rwmol.GetAtomWithIdx(position_idx).SetAtomicNum(new_atom.GetAtomicNum())
    if new_atom.GetSymbol()=='C':
        rwmol.GetAtomWithIdx(position_idx).SetNoImplicit(True)
        if 4-int(node_bonds)<0:
            return None
        rwmol.GetAtomWithIdx(position_idx).SetNumExplicitHs(4-int(node_bonds))
        rwmol.GetAtomWithIdx(position_idx).SetFormalCharge(0)
    elif new_atom.GetSymbol()=='O':
        rwmol.GetAtomWithIdx(position_idx).SetNoImplicit(True)
        if rwmol.GetAtomWithIdx(position_idx).GetFormalCharge() > 0:
            rwmol.GetAtomWithIdx(position_idx).SetFormalCharge(0)
        formal_charge = rwmol.GetAtomWithIdx(position_idx).GetFormalCharge()
        if 2 + formal_charge - int(node_bonds)<0:
            return None
        rwmol.GetAtomWithIdx(position_idx).SetNumExplicitHs(2 + formal_charge - int(node_bonds))
    elif new_atom.GetSymbol()!='N':
        rwmol.GetAtomWithIdx(position_idx).SetNoImplicit(True)
        rwmol.GetAtomWithIdx(position_idx).SetFormalCharge(0)
        rwmol.GetAtomWithIdx(position_idx).SetNumExplicitHs(0)
    try:
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
        for atom in copy_mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                atom.SetFormalCharge(0)
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
        for atom in copy_mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                atom.SetFormalCharge(0)
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


def optimize_single_molecule_one_iterate_drop_new(smiles,featurelst,origin_substructure_lst):
    if smiles == None:
        return set()

    origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(smiles))
    new_smiles_set = set()
    for idx in featurelst:
        leaf_atom_idx_lst= origin_substructure_lst[idx]
        if type(leaf_atom_idx_lst) is not list:
            leaf_atom_idx_lst = [leaf_atom_idx_lst]
        new_smile = delete_substructure_at_idx(origin_mol, leaf_atom_idx_lst)
        new_smiles_set.add(new_smile)

    return new_smiles_set

def add_fragment_at_position(editmol, position_idx, fragment, new_bond):
    '''
        position_idx:  index of edited atom in editmol
        fragment: e.g., "C1=CC=CC=C1", "C1=CC=NC=C1", ... 
        new_bond: {SINGLE, DOUBLE}  

        Return:  
            list of SMILES
    '''  
    new_smiles_set = set()
    fragment_mol = Chem.MolFromSmiles(fragment)
    current_atom = editmol.GetAtomWithIdx(position_idx)
    neighbor_atom_set = set()  ## index of neighbor of current atom in new_mol  


    ## (A) add a bond between atom and ring 
    #### 1. initialize empty new_mol
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
        ### collect the neighbor atoms of current atom, both are in ring. 
        if (i1==position_idx or i2==position_idx) and (a1.IsInRing() and a2.IsInRing()):
            neighbor_atom_set.add(i1_new)
            neighbor_atom_set.add(i2_new)
    if neighbor_atom_set != set():
        neighbor_atom_set.remove(old_idx2new_idx[position_idx])

    #### 3. combine two components 
    #### 3.1 add fragment into new_mol
    new_atom_idx_lst = []
    old_idx2new_idx2 = dict()  ### fragment idx -> new mol idx 
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


    #### 3.2 enumerate possible binding atoms and generate new smiles 
    for i in new_atom_idx_lst:  ### enumeration 
        copy_mol = deepcopy(new_mol)
        copy_mol.AddBond(old_idx2new_idx[position_idx], i, new_bond)
        for atom in copy_mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                atom.SetFormalCharge(0)
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

                    # print(new_smiles)
    # print(new_smiles_set)
    return new_smiles_set

def delete_substructure_at_idx(editmol, atom_idx_lst):
    edit_smiles = Chem.MolToSmiles(editmol)
    #### 1. initialize with empty mol
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))

    #### 2. add editmol into new_mol
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
