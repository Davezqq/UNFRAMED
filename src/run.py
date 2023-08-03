import os, pickle, torch, random
import numpy as np 
import argparse
from pathos.pools import ThreadPool,ParallelPool
from multiprocessing import cpu_count
from GraphModel_SSFG import Graph_model
import time


import pandas as pd
from tdc import Oracle
from tqdm import tqdm

from generate_utils import *
from generate_utils import oracle_screening_return_set


def sample_positions(pred_pos_prob_alter,pred_pos_prob_append,pred_pos_prob_drop,alter_num=3,append_num=2,drop_num=2,node_degree=[],append_lst=[]):
	alter_prob = pred_pos_prob_alter
	alter_prob = alter_prob.numpy()
	alter_prob[-1] = 1-sum(alter_prob[:-1])
	append_prob = pred_pos_prob_append
	append_prob = append_prob.numpy()
	append_prob[-1] = 1 - sum(append_prob[:-1])
	one_degree_nodes = np.where(node_degree == 1)[0]
	drop_prob = pred_pos_prob_drop
	# print(one_degree_nodes)
	drop_prob = drop_prob[one_degree_nodes]
	# print(drop_prob)
	# drop_prob = drop_prob/drop_prob.sum()
	# print(drop_prob)
	drop_prob = torch.softmax(drop_prob,dim=-1)
	# print(drop_prob)
	drop_prob = drop_prob.numpy()

	alter_lst = [i for i in range(pred_pos_prob_alter.shape[0])]
	if alter_num>len(alter_lst):
		alter_selected = np.random.choice(alter_lst, alter_num, p=alter_prob, replace=True)
	else:
		alter_selected = np.random.choice(alter_lst, alter_num, p=alter_prob,replace=False)
	# print(alter_prob.shape)
	# alter_selected = torch.multinomial(torch.tensor(alter_prob),alter_num)
	# print(alter_selected)
	if append_num>len(append_lst):
		append_selected = np.random.choice(append_lst, append_num, p=append_prob,replace=True)
	else:
		append_selected = np.random.choice(append_lst, append_num, p=append_prob, replace=False)

	# append_selected = torch.multinomial(torch.tensor(append_prob),append_num)
	if len(drop_prob) != 0:
		drop_prob[-1] = 1 - sum(drop_prob[:-1])
		if drop_num > len(one_degree_nodes):
			drop_selected = np.random.choice(one_degree_nodes, drop_num, p=drop_prob,replace=True)
		else:
			drop_selected = np.random.choice(one_degree_nodes, drop_num, p=drop_prob, replace=False)
	else:
		drop_selected = []
	return alter_selected, append_selected, drop_selected


def optimize_one_smile_new(smiles,cloze_gnn,position_gnn_lst,parent_dict,sim_threshold):

	total_set = set()
	for ele_smiles in smiles:
		# if not is_valid(ele_smiles,vocabulary2):
		# 	# print(smiles)
		# 	continue
		graph, N,node_degrees,node_bonds = smiles2feature_position(ele_smiles, device='cpu')

		graph = graph[0]
		pred_pos = None
		# print(position_gnn_lst)
		for position_gnn in position_gnn_lst:
			if pred_pos is None:
				pred_pos= position_gnn.infer(graph)/len(position_gnn_lst)
			else:
				pred_pos+= position_gnn.infer(graph) /len(position_gnn_lst)

		pred_pos = torch.softmax(pred_pos, dim=1)
		pred_pos = pred_pos[:, 1:]
		# print(pred_pos)

		pred_pos_alter = pred_pos[:, 0]
		valid_append = select_valid_atom(graph,node_bonds)
		# valid_append = [i for i in range(pred_pos.shape[0])]
		# print(node_degrees)
		pred_pos_append = pred_pos[valid_append, 2]
		# print(pred_pos_append.shape)
		pred_pos_alter = torch.softmax(pred_pos_alter, dim=0)
		# print(pred_pos_alter)
		# pred_pos_alter = pred_pos_alter/pred_pos_alter.sum()
		pred_pos_append = torch.softmax(pred_pos_append, dim=0)
		# pred_pos_append = pred_pos_append/pred_pos_append.sum()
		pred_pos_drop = pred_pos[:,1]
		# pred_pos_drop =
		# print(pred_pos)
		# pred_pos_nodorp = torch.softmax(pred_pos_nodorp, dim=0)
		# pred_pos_drop = torch.softmax(pred_pos_drop,dim=-1)
		# print(pred_pos)
		alter_selected, append_selected, drop_selected = sample_positions(pred_pos_prob_alter=pred_pos_alter,pred_pos_prob_append=pred_pos_append,pred_pos_prob_drop=pred_pos_drop,node_degree=node_degrees,append_lst=valid_append)
		alter_feature_lst,alter_origin_substructure_lst = smiles2expandfeature_new(ele_smiles, mask_idx_lst=alter_selected, act='alter')
		append_feature_lst,append_origin_substructure_lst = smiles2expandfeature_new(ele_smiles, mask_idx_lst=append_selected,
																			  act='append')

		append_smiles_set = optimize_single_molecule_one_iterate_alter_new(ele_smiles, cloze_gnn,alter_feature_lst,alter_origin_substructure_lst)
		alter_smiles_set = optimize_single_molecule_one_iterate_append_new(ele_smiles, cloze_gnn,append_feature_lst,append_origin_substructure_lst)
		drop_smiles_set = optimize_single_molecule_one_iterate_drop_new(ele_smiles,drop_selected,alter_origin_substructure_lst)
		smiles_set = append_smiles_set.union(alter_smiles_set)
		smiles_set = smiles_set.union(drop_smiles_set)
		# smiles_set = smiles_set.union(drop_smiles_set)
		# print(append_smiles_set)

		total_set = total_set.union(smiles_set)

		# print(len(to))
		for smi in smiles_set:
			# # print(type(parent_dict))
			# if smi in parent_dict:
			# 	total_set.remove(smi)
			# else:
			parent_dict[smi] = parent_dict[ele_smiles]
	orimol = None
	total_set_new = set()
	for smile in total_set:
		if smile is None:
			continue
		tmpmol = AllChem.MolFromSmiles(smile)
		if orimol is None:
			orimol = AllChem.MolFromSmiles(parent_dict[smile])
		sim = similarity_mols(tmpmol,orimol)
		if sim >= sim_threshold:
			total_set_new.add(smile)
	return (total_set_new,parent_dict)



def parallel_screen(next_set,group_num,oracle_new,start_smiles,population_size,iteration,T_warm,parent_dict,score_dict,sim):
	tp = ThreadPool(nthreads=group_num)
	groups = []
	total_result_lst = []
	new_set = set()
	num_per_group = int(len(next_set) / group_num + 1)
	for i in range(group_num):
		if (i + 1) * num_per_group <= len(next_set):
			tmpgroup = next_set[i * num_per_group:(i + 1) * num_per_group]
		else:
			tmpgroup = next_set[i * num_per_group:len(next_set)]
		groups.append(tmpgroup)
	print(len(groups[0]))
	oracle_new_lst = [oracle_new]*group_num
	# print(len(next_set))
	parent_dict_lst = [parent_dict] * group_num
	score_dict_lst = [score_dict] * group_num
	start_smiles_lst = [start_smiles]*group_num
	population_size_lst = [population_size]*group_num
	iteration_lst = [iteration]*group_num
	T_warm_lst = [T_warm]*group_num
	sim_lst = [sim]*group_num
	# print('start')
	result_set = tp.amap(oracle_screening_return_set, groups, oracle_new_lst,start_smiles_lst,population_size_lst,iteration_lst,T_warm_lst,parent_dict_lst,score_dict_lst,sim_lst)
	# oracle_screening_return_set(smiles_set, oracle, baseline, origin_smi, N=20, iteration=0, T_warm=3)
	while not result_set.ready():
		time.sleep(1)
		print(".", end=' ')
	result_set = result_set.get()
	multi_score_lst = []
	for tmp_lst in result_set:
		total_result_lst+=tmp_lst

	newscore_lst = sorted(total_result_lst, key=lambda x: x[1], reverse=True)
	newscore_lst = newscore_lst[:population_size]

	for newlist_ele in newscore_lst:
		new_set.add(newlist_ele[0])
	tp.close()
	tp.join()
	tp.clear()
	return new_set,newscore_lst

def optimization_new(start_smiles_lst, cloze_gnn,position_gnn_list, oracle, oracle_name, generations, population_size, result_pkl,group_num,T_warm,parent_dict,sim):
	p = ParallelPool(ncpus=group_num)
	logfile = open('tmplog.out','w')
	logfile.write(f'{cpu_count()}\n')
	logfile.flush()

	smiles2score = dict()
	def oracle_new(smiles,origin_smiles):
		if smiles not in smiles2score:
			value = oracle(smiles,origin_smiles)
			smiles2score[smiles] = value
		return smiles2score[smiles]
	trace_dict = dict()
	existing_set = set(start_smiles_lst)
	current_set = set(start_smiles_lst)
	score_dict = {}
	idx_2_smiles2f = {}
	smiles2f_new = {smiles:oracle(smiles,smiles) for smiles in start_smiles_lst}
	idx_2_smiles2f[-1] = smiles2f_new, current_set
	for i_gen in tqdm(range(generations)):
		current_list = list(current_set)
		next_set = set()
		num_per_group = int(len(current_set)/group_num+1)
		groups = []
		for i in range(group_num):
			if (i+1)*num_per_group <= len(current_set):
				tmpgroup=current_list[i*num_per_group:(i+1)*num_per_group]
			else:
				tmpgroup = current_list[i * num_per_group:len(current_set)]
			groups.append(tmpgroup)
		cloze_gnn_lst = [cloze_gnn for i in range(group_num)]
		position_gnn_lst = [position_gnn_list for i in range(group_num)]
		sim_lst = [sim for i in range(group_num)]
		parent_lst = []
		for i in range(group_num):
			parent_lst.append(parent_dict)

		smiles_set_lst = p.amap(optimize_one_smile_new,groups,cloze_gnn_lst,position_gnn_lst,parent_lst,sim_lst)
		while not smiles_set_lst.ready():
			time.sleep(1)
			print(".", end=' ')
		# print(len(parent_dict))
		smiles_set_lst = smiles_set_lst.get()

		for sub_set in smiles_set_lst:
			next_set = next_set.union(sub_set[0])
			parent_dict.update(sub_set[1])
		existing_set = existing_set.union(next_set)


		next_set = list(next_set)
		if len(next_set)==0:
			break


		next_set,smiles_score_lst = parallel_screen(next_set,group_num, oracle_new,start_smiles_lst[0],population_size,i_gen,T_warm,parent_dict,score_dict,sim)

		current_set = next_set

		logfile.write(f'{len(current_set)}\n')
		logfile.write(f'{smiles_score_lst[:5]},Oracle num, {len(existing_set)}\n')
		logfile.flush()
	pickle.dump(existing_set, open(result_pkl, 'wb'))
	p.clear()


def evaluate_new(oracle_name='qed',generations=10,population_size=20,start_smiles_lst=[],result_pkl='',group_num=8,T_warm=0,sim=0.4,position_model_ckpt='',cloze_model_ckpt=''):
	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')
	parent_dict = {}
	parser = argparse.ArgumentParser()

	qed = Oracle('qed')
	sa = Oracle('sa')
	drd2 = Oracle('drd2')
	logp = Oracle('logp')
	mu = 2.230044
	sigma = 0.6526308

	def normalize_sa(smiles):
		sa_score = sa(smiles)
		mod_score = np.maximum(sa_score, mu)
		return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.))

	if oracle_name == 'qedsa':
		def oracle(smiles):
			return np.mean((qed(smiles), normalize_sa(smiles)))
	elif oracle_name == 'qed':
		def oracle(smiles,smiles_ori):
			return qed(smiles)-qed(smiles_ori)
	elif oracle_name == 'logp':
		def oracle(smiles):
			return logp(smiles)
	elif oracle_name == 'plogp':
		# print('plogp!')
		def oracle(smiles,smiles_ori):
			return (penalized_logp(smiles)-penalized_logp(smiles_ori))
	elif oracle_name == 'drd':
		def oracle(smiles,smiles_ori):
			return (drd2(smiles)-drd2(smiles_ori))
	elif oracle_name == 'qeddrd':
		def oracle(smiles,smiles_ori):
			return (qed(smiles) - qed(smiles_ori))+(drd2(smiles)-drd2(smiles_ori))

	device = 'cpu'
	# position_model_ckpt = "PositionModel/save_model/GNN_positionsmodel_1_validloss_0.23958.ckpt"
	position_gnn1 = torch.load(position_model_ckpt,map_location=device)
	# position_gnn1.load_state_dict(torch.load(position_model_ckpt1,map_location=device).state_dict())

	# cloze_model_ckpt = "save_model/Graph_2_validloss_1.99502.ckpt"
	gnn = torch.load(cloze_model_ckpt, map_location=device)
	gnn.eval()
	position_gnn1.eval()
	# position_gnn2.eval()
	gnn.switch_device(device)
	for strat_smi in start_smiles_lst:
		parent_dict[strat_smi]=strat_smi

	optimization_new(start_smiles_lst, gnn,[position_gnn1], oracle, oracle_name,
						generations = generations,
						population_size = population_size,
						result_pkl = result_pkl,group_num=group_num,T_warm=T_warm,parent_dict=parent_dict,sim=sim)


	

# if __name__ == "__main__":
# 	main()







