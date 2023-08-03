import os
import argparse
from run import evaluate_new
from chemutils import get_qedscore,penalized_logp,get_sascore
from tqdm import  tqdm
import numpy as np
from load_pkl import load_pkl,load_pkl_for_imp

def batch_evaluate(oracle_name,dir_root,smiles_file,number_to_select=100,group_num=8,Twarm=0,gener=10,population_size=10,sim_cons=0.4):
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
        evaluate_new(start_smiles_lst=[smiles],result_pkl=os.path.join(dir_root,f'{idx}.pkl'),oracle_name=oracle_name,group_num=group_num,population_size=population_size,T_warm=Twarm,generations=gener,sim=sim_cons)
    logfile.close()

def batch_validation(dir_root,oracle_name):
    log_path = os.path.join(dir_root,'logfile.txt')
    smi_lst = []
    with open(log_path,'r') as logfile:
        for line in logfile.readlines():
            line = line.strip('\n').strip('\r')
            index_num = int(line.split(' : ')[0])
            smiles = line.split(' : ')[1]
            smi_lst.append((index_num,smiles))
    smi_lst = smi_lst[:]
    total_sr = 0
    valid_num = 0
    for idx,pair in tqdm(enumerate(smi_lst)):

        pkl_path = os.path.join(dir_root,f'{pair[0]}.pkl')
        tmp_sr = load_pkl([pair[1]],pkl_path,metric=oracle_name)
        if tmp_sr is None:
            continue
        # print('idx:',idx)
        # print(tmp_sr)
        valid_num+=1
        total_sr += tmp_sr
    print(valid_num)
    return total_sr/valid_num

def impro_evaluate(dir_root,oracle_name):
    log_path = os.path.join(dir_root, 'logfile.txt')
    smi_lst = []
    imp_lst = []
    with open(log_path, 'r') as logfile:
        for line in logfile.readlines():
            line = line.strip('\n').strip('\r')
            index_num = int(line.split(' : ')[0])
            smiles = line.split(' : ')[1]
            # tmpdrd = drd_scorer.get_score(smiles)
            # if tmpdrd >= 0.001:
            #     smi_lst.append((index_num, smiles))
            smi_lst.append((index_num, smiles))

    for pair in tqdm(smi_lst):
        pkl_path = os.path.join(dir_root,f'{pair[0]}.pkl')
        avg_imp= load_pkl_for_imp([pair[1]],pkl_path,metric=oracle_name)
        imp_lst.append(avg_imp)
    mean_imp = np.mean(imp_lst)
    std_imp = np.std(imp_lst)
    # print(mean_imp,std_imp)
    return mean_imp,std_imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root', '-dir', type=str, default='../result/test')
    parser.add_argument('--smiles_file', '-smi', type=str, default='data/test_set/drd2_25_qed6/test.txt')
    parser.add_argument('--group_num', '-ng', type=int, default=5)
    parser.add_argument('--oracle_name', '-oracle', type=str, default='qed')
    parser.add_argument('--Twarm', '-Twarm', type=int, default=0)
    parser.add_argument('--generation', '-gen', type=int, default=5)
    parser.add_argument('--population_size', '-pop', type=int, default=5)
    parser.add_argument('--sim_cons', '-sim', type=int, default=0.6)

    args = parser.parse_args()
    dir_root = args.dir_root
    smiles_file = args.smiles_file
    group_num = args.group_num
    oracle_name = args.oracle_name
    Twarm = args.Twarm
    generation=args.generation
    population_size = args.population_size
    sim_cons = args.sim_cons


    batch_evaluate(dir_root=dir_root, smiles_file=smiles_file, group_num=group_num, oracle_name=oracle_name, Twarm=Twarm, gener=generation, population_size=population_size, sim_cons=sim_cons)





