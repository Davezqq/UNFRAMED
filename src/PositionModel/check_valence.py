from rdkit import Chem
from tqdm import tqdm
from chemutils_new_ul import smiles2graph,vocabulary,getbond_type
import pickle
from pathos.pools import ParallelPool


def solve_rings(smile,val_dict):
    tmp_mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(tmp_mol, clearAromaticFlags=True)
    # return mol
    maxval = 0
    for atom in tmp_mol.GetAtoms():
        if atom.GetSymbol()=='B':
            maxval+=3
        else:
            maxval+=val_dict[atom.GetSymbol()]
    for bond in tmp_mol.GetBonds():
        btype = getbond_type(bond)
        if btype == 0 or btype == 1:
            maxval-=2
        elif btype == 2:
            maxval-=4
        elif btype == 3:
            maxval-=6
    val_dict[smile]=maxval
    return val_dict

def solve_val(smile_lst,val_dict):
    for smiles in tqdm(smile_lst):
        tmp_graph, node_degree = smiles2graph(smiles, device='cpu')
        node_type_lst = tmp_graph.ndata['atom_type'].numpy().tolist()
        for idx, node_type in enumerate(node_type_lst):
            tmp_smi = vocabulary[node_type]
            if tmp_smi in val_dict:
                val_dict[tmp_smi] = max(val_dict[tmp_smi], node_type_lst[idx])
            else:
                val_dict[tmp_smi] = node_type_lst[idx]
    return val_dict

vocab_path = '../data/train_set/vocabulary.txt'

f_read = open('val_dict.pkl', 'rb')
val_dict = pickle.load(f_read)

with open(vocab_path,'r') as vocab_file:
    for line in vocab_file.readlines():
        line = line.split('	')[0]
        val_dict = solve_rings(line,val_dict)

f_save = open('val_dict.pkl', 'wb')
pickle.dump(val_dict,f_save)










