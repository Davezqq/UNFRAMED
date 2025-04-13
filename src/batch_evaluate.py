import os
import argparse
from run import evaluate_new
from chemutils import get_qedscore,penalized_logp,get_sascore
from tqdm import  tqdm
import numpy as np
from load_pkl import load_pkl,load_pkl_for_imp

def batch_evaluate(oracle_name,dir_root,smiles_file,number_to_select=100,group_num=8,Twarm=0,gener=10,population_size=10,sim_cons=0.4,cloze_model_ckpt='',position_model_ckpt=''):
    os.makedirs(dir_root,exist_ok=True)
    smiles_list = []
    logfile = open(os.path.join(dir_root,'logfile.txt'),'w')
    with open(smiles_file,'r') as tmpfile:
        for line in tmpfile.readlines():
            line = line.strip('\n').strip('\r')
            smiles_list.append(line)

    # smiles_list = smiles_list[1:2]
    # selected_smiles_list = np.random.choice(smiles_list,number_to_select,replace=False)
    for idx,smiles in enumerate(smiles_list):
        logfile.write(f'{idx} : {smiles} \n')
        logfile.flush()
        if os.path.exists(os.path.join(dir_root,f'{idx}.pkl')):
            continue
        evaluate_new(start_smiles_lst=[smiles],result_pkl=os.path.join(dir_root,f'{idx}.pkl'),oracle_name=oracle_name,group_num=group_num,population_size=population_size,T_warm=Twarm,generations=gener,sim=sim_cons,cloze_model_ckpt=cloze_model_ckpt,position_model_ckpt=position_model_ckpt)
    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root', '-dir', type=str, default='../result/test')
    parser.add_argument('--smiles_file', '-smi', type=str, default='data/test_set/drd2_25_qed6/test.txt')
    parser.add_argument('--group_num', '-ng', type=int, default=5)
    parser.add_argument('--oracle_name', '-oracle', type=str, default='qed')
    parser.add_argument('--Twarm', '-Twarm', type=int, default=0)
    parser.add_argument('--generation', '-gen', type=int, default=5)
    parser.add_argument('--population_size', '-pop', type=int, default=5)
    parser.add_argument('--sim_cons', '-sim', type=float, default=0.6)
    parser.add_argument('--cloze_model_path', '-clo_path', type=str, default='save_model/Graph_2_validloss_1.99502.ckpt')
    parser.add_argument('--position_model_ckpt', '-pos_path', type=str, default='PositionModel/save_model/GNN_positionsmodel_1_validloss_0.23958.ckpt')

    args = parser.parse_args()
    dir_root = args.dir_root
    smiles_file = args.smiles_file
    group_num = args.group_num
    oracle_name = args.oracle_name
    Twarm = args.Twarm
    generation=args.generation
    population_size = args.population_size
    sim_cons = args.sim_cons
    cloze_model_path = args.cloze_model_path
    position_model_ckpt = args.position_model_ckpt


    batch_evaluate(dir_root=dir_root, smiles_file=smiles_file, group_num=group_num, oracle_name=oracle_name, Twarm=Twarm, gener=generation, population_size=population_size, sim_cons=sim_cons,cloze_model_ckpt=cloze_model_path,position_model_ckpt=position_model_ckpt)





