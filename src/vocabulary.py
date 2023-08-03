# from chemutils import smiles2word

import os
from collections import defaultdict 
from tqdm import tqdm 
from rdkit import Chem, DataStructs

def smiles2mol(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	try:
		Chem.Kekulize(mol, clearAromaticFlags=True)
	except:
		return None
	return mol

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

def smiles2word(smiles):
	mol = smiles2mol(smiles)
	if mol is None:
		return None
	# print(mol)
	word_lst = []

	cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
	cliques = adjust_intersectant_ring(cliques)
	# print(cliques)
	cliques_smiles = []
	try:
		for clique in cliques:
			clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
			cliques_smiles.append(clique_smiles)
	except:
		return None
	atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
	return cliques_smiles + atom_not_in_rings_list






all_vocabulary_file = "data/train_set/Molecule_dataset.txt"
rawdata_file = "data/train_set/all_structures.txt"
select_vocabulary_file = "data/vocabulary.txt"
min_show_time = 1000

if not os.path.exists(all_vocabulary_file):
	with open(rawdata_file) as fin:
		lines = fin.readlines()[1:]
		smiles_lst = [i.strip('\n').strip('\r').split(' ')[0] for i in lines]
	word2cnt = defaultdict(int)
	for smiles in tqdm(smiles_lst):
		word_lst = smiles2word(smiles)
		if word_lst is None:
			continue
		for word in word_lst:
			word2cnt[word] += 1
	word_cnt_lst = [(word,cnt) for word,cnt in word2cnt.items()]
	word_cnt_lst = sorted(word_cnt_lst, key=lambda x:x[1], reverse = True)

	with open(all_vocabulary_file, 'w') as fout:
		for word, cnt in word_cnt_lst:
			fout.write(word + '\t' + str(cnt) + '\n')
else:
	with open(all_vocabulary_file, 'r') as fin:
		lines = fin.readlines()
		word_cnt_lst = [(line.split('\t')[0], int(line.split('\t')[1])) for line in lines]


word_cnt_lst = list(filter(lambda x:x[1]>min_show_time, word_cnt_lst))
print(len(word_cnt_lst))

with open(select_vocabulary_file, 'w') as fout:
	for word, cnt in word_cnt_lst:
		fout.write(word + '\t' + str(cnt) + '\n')



