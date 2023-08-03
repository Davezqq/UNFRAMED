import numpy as np
import torch

from chemutils import *

def lam1(x):
	return x[2]<5

def lam2(x):
	return x[2]<4 and x[1]>x[4]

def lam3(x):
	return x[1]+x[3]

def gnn_prediction_of_single_smiles(smiles, gnn):
	if not is_valid(smiles):
		return 0
	return gnn.smiles2pred(smiles)

def oracle_screening_return_set(smiles_set, oracle,origin_smi,N=20,iteration = 0,T_warm=3,parent_dict={},score_dict={},sim_threshold=0.4):
	# print('start')
	# print(sim_threshold)
	smiles_score_lst = []
	new_set = set()
	# origin_score = oracle(origin_smi) + 1.0
	for smiles in smiles_set:
		if smiles is None:
			continue

		tmp_origin_smi = parent_dict[smiles]
		score = oracle(smiles,tmp_origin_smi)

		tmp_orimol= AllChem.MolFromSmiles(origin_smi)
		sim = similarity(smiles,tmp_orimol)
		# score+=sim
		sascore = get_sascore(smiles)

		smiles_score_lst.append((smiles, score,sascore,sim))
		# score_dict[smiles] = (smiles, score,sascore
	if iteration<T_warm:
		newlist = filter(lambda x:x[2]<5, smiles_score_lst)
	else:
		newlist = filter(lambda x: x[3]>=sim_threshold, smiles_score_lst)
	newscore_lst = []
	for newlist_ele in newlist:
		newscore_lst.append(newlist_ele)
	newscore_lst = sorted(newscore_lst,key=lambda x:x[1],reverse=True)
	newscore_lst = newscore_lst[:N]

	return newscore_lst



def oracle_screening(smiles_set, oracle,baseline):
	smiles_score_lst = []
	# origin_mol = AllChem.MolFromSmiles(origin_smi)
	for smiles in smiles_set:
		if '[NH+]' in smiles or 'C+' in smiles:
			continue
		score = oracle(smiles)
		# sim = similarity(smiles, origin_mol)
		sascore = get_sascore(smiles)
		smiles_score_lst.append((smiles, score,sascore))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	new_list = filter(lambda x: x[1] >= baseline and x[2] <= 4, smiles_score_lst)

	# smiles_score_lst = smiles_score_lst[:min(200,len(smiles_score_lst))]
	# sum = 0
	smiles_score_lst = []
	for ele in new_list:
		smiles_score_lst.append(ele)
	return smiles_score_lst


def cal_original_score(start_smile_list):
	total_qed  = 0
	for smiles in start_smile_list:
		total_qed += get_qedscore(smiles)
	return total_qed/len(start_smile_list)




def gnn_screening(smiles_set, gnn):
	smiles_score_lst = []
	for smiles in smiles_set:
		score = gnn_prediction_of_single_smiles(smiles, gnn)
		smiles_score_lst.append((smiles, score))
	smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return smiles_score_lst
	# smiles_lst = [i[0] for i in smiles_score_lst]
	# return smiles_lst


def optimize_single_molecule_one_iterate_append_new(smiles, cloze_gnn,feature_lst,origin_substructure_lst):
	if smiles == None:
		return set()
	# if not is_valid(smiles,vocabulary1):
	# 	return set()

	origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(smiles))
	new_smiles_set = set()

	# print()
	for tmp_graph,mask_idx,append_pos,new_bond in feature_lst:
		prediction = cloze_gnn.infer(tmp_graph,label=None,mask_idx=mask_idx)
		# print(prediction.shape)

		sorted,indices = torch.sort(prediction, dim=-1, descending=True)
		# top_idxs_p = sorted[:4]
		# top_idxs_p = torch.softmax(top_idxs_p, dim=-1).numpy()

		indices = indices.tolist()
		top_idxs = indices[:3]
		top_words = [vocabulary1[ii] for ii in top_idxs]
		for substru_idx, word in zip(top_idxs, top_words):
			leaf_atom_idx_lst = origin_substructure_lst[append_pos]
			if type(leaf_atom_idx_lst)==int:
				leaf_atom_idx_lst = [leaf_atom_idx_lst]
			for leaf_atom_idx in leaf_atom_idx_lst:
				if ith_substructure_is_atom(vocabulary1,substru_idx):
					# print(leaf_atom_idx,word)
					new_smiles = add_atom_at_position(editmol = origin_mol, position_idx = leaf_atom_idx,
													  new_atom = word, new_bond = bondtype_list[new_bond])
					new_smiles_set.add(new_smiles)
				else:
					new_smiles_batch = add_fragment_at_position(editmol = origin_mol, position_idx = leaf_atom_idx,
																fragment = word , new_bond = bondtype_list[new_bond])
					new_smiles_set = new_smiles_set.union(new_smiles_batch)





	new_smiles_set = set([new_smiles for new_smiles in new_smiles_set if new_smiles != None])
	return new_smiles_set


def optimize_single_molecule_one_iterate_alter_new(smiles, gnn,feature_lst,substructure_lst):
	if smiles == None:
		return set()
	# if not is_valid(smiles,vocabulary1):
	# 	return set()
	origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(smiles))
	new_smiles_set = set()
	for tmp_graph,mask_idx,node_bonds,ori_word_id in feature_lst:
		prediction = gnn.infer(tmp_graph,label=None,mask_idx=mask_idx)
		# print(prediction.shape)

		sorted, indices = torch.sort(prediction,dim=-1,descending=True)
		indices = indices.tolist()

		top_idxs = indices[:4]

		top_words = []

		top_words +=([vocabulary1[ii] if ii < len(vocabulary1) else 'DEL' for ii in top_idxs])

		tmp_topwords = top_words
		tmp_topidxs = top_idxs

		index = -1
		for substru_idx, word in zip(tmp_topidxs, tmp_topwords):
			index +=1
			leaf_atom_idx_lst = substructure_lst[mask_idx]
			if type(leaf_atom_idx_lst) == int:
				leaf_atom_idx_lst = [leaf_atom_idx_lst]
			if word == 'DEL':
				continue
				# if node_degrees[mask_idx]==1:
				# 	new_smile = delete_substructure_at_idx(origin_mol,leaf_atom_idx_lst)
				# 	new_smiles_set.add(new_smile)
			else:
				if ith_substructure_is_atom(vocabulary1,substru_idx):
					if len(leaf_atom_idx_lst) == 1:
					# print(leaf_atom_idx,word)
						new_smiles = a2a_at_position(editmol = origin_mol, position_idx = leaf_atom_idx_lst[0],
														  new_atom = word,node_bonds=node_bonds)
						new_smiles_set.add(new_smiles)
					else:
						new_smiles = s2a_at_position(editmol=origin_mol, position_idx_lst=leaf_atom_idx_lst,
													 new_atom=word)
						new_smiles_set.add(new_smiles)
				else:
					if len(leaf_atom_idx_lst) == 1:
						new_smiles_batch = a2s_at_position(editmol = origin_mol, position_idx = leaf_atom_idx_lst[0],
																fragment = word)

						new_smiles_set = new_smiles_set.union(new_smiles_batch)
					else:
						new_smiles_batch = s2s_at_position(editmol = origin_mol, position_idx_lst = leaf_atom_idx_lst,
																fragment = word)
						new_smiles_set = new_smiles_set.union(new_smiles_batch)

	new_smiles_set = set([new_smiles for new_smiles in new_smiles_set if new_smiles != None])
	return new_smiles_set






