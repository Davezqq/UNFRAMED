import os
from chemutils import get_qedscore,penalized_logp,get_sascore
from tqdm import  tqdm
from tdc import Oracle
import argparse
import numpy as np
from load_pkl import load_pkl,load_pkl_for_imp

drd2 = Oracle('drd2')

def batch_validation(dir_root,oracle_name):
    log_path = os.path.join(dir_root,'logfile.txt')
    result_path = os.path.join(dir_root,'qed_res.txt')
    total_sr = 0
    valid_num = 0
    resfile = open(result_path, 'a+')
    resfile.seek(0,os.SEEK_SET)
    for line in resfile.readlines():
        line = line.strip('\n')
        # print(total_sr)
        if oracle_name == 'plogp':
            total_sr+=float(line.split(',')[1])
        else:
            if line.split(',')[1] == '1':
                # print(line.split(',')[1])
                total_sr+=1
        valid_num+=1
        # count += 1
        # print(total_sr)
    resfile.seek(0, os.SEEK_END)
    # print(total_sr)
    # return None
    smi_lst = []
    with open(log_path,'r') as logfile:
        for line in logfile.readlines():
            line = line.strip('\n').strip('\r')
            index_num = int(line.split(' : ')[0])
            smiles = line.split(' : ')[1]
            smi_lst.append((index_num,smiles))
    smi_lst = smi_lst[valid_num:-1]
    # print(len(smi_lst))
    for idx,pair in enumerate(smi_lst,start=valid_num):

        pkl_path = os.path.join(dir_root,f'{pair[0]}.pkl')
        tmp_sr = load_pkl([pair[1]],pkl_path,metric=oracle_name)
        # print(tmp_sr)
        if tmp_sr is None:
            # print(pair[1])
            continue
        # if tmp_sr>0:
        #     tmp_sr=1
        if tmp_sr == 0:
            resfile.write(f'{idx},0\n')
        else:
            resfile.write(f'{idx},{tmp_sr}\n')
        resfile.flush()
        # print('idx:',idx)
        # print(tmp_sr)
        valid_num+=1
        total_sr += tmp_sr
    print(valid_num)
    print(total_sr)
    return total_sr/valid_num

def batch_validation_plogp(dir_root,oracle_name):
    log_path = os.path.join(dir_root, 'logfile.txt')
    result_path = os.path.join(dir_root, 'res.txt')
    total_impr = 0
    valid_num = 0
    resfile = open(result_path, 'a+')
    resfile.seek(0, os.SEEK_SET)
    for line in resfile.readlines():
        line = line.strip('\n')
        print()
        # print(total_sr)
        # if line.split(',')[1] == '1':
        total_impr += float(line.split(',')[1])
        valid_num += 1
        # count += 1
        # print(total_sr)
    resfile.seek(0, os.SEEK_END)
    smi_lst = []
    with open(log_path,'r') as logfile:
        for line in logfile.readlines():
            line = line.strip('\n').strip('\r')
            index_num = int(line.split(' : ')[0])
            smiles = line.split(' : ')[1]
            # tmpqed = get_qedscore(smiles)
            # tmplogp = penalized_logp(smiles)
            # tmpsa = get_sascore(smiles)
            # tmpdrd =drd_scorer.get_score(smiles)
            # if tmpdrd>=0.001:
            smi_lst.append((index_num,smiles))
    smi_lst = smi_lst[valid_num:-1]
    for idx,pair in tqdm(enumerate(smi_lst,start=valid_num)):

        pkl_path = os.path.join(dir_root,f'{pair[0]}.pkl')
        tmp_impr = load_pkl([pair[1]],pkl_path,metric=oracle_name)
        print(tmp_impr)
        resfile.write(f'{idx},{tmp_impr}\n')
        if tmp_impr is None:
            continue
        # if tmp_sr>0:
        #     tmp_sr=1


        # print('idx:',idx)
        # print(tmp_sr)
        valid_num+=1
        total_impr += tmp_impr
    print(valid_num)
    return total_impr/valid_num

def impro_evaluate(dir_root,oracle_name):
    log_path = os.path.join(dir_root, 'logfile.txt')
    smi_lst = []
    imp_lst = []
    sim_lst = []
    with open(log_path, 'r') as logfile:
        for line in logfile.readlines():
            line = line.strip('\n').strip('\r')
            index_num = int(line.split(' : ')[0])
            smiles = line.split(' : ')[1]
            # tmpdrd = drd_scorer.get_score(smiles)
            # if tmpdrd >= 0.001:
            #     smi_lst.append((index_num, smiles))
            smi_lst.append((index_num, smiles))
    smi_lst = smi_lst[:-1]
    for pair in tqdm(smi_lst):
        pkl_path = os.path.join(dir_root,f'{pair[0]}.pkl')
        res= load_pkl_for_imp([pair[1]],pkl_path,metric=oracle_name)
        if res is not None:
            imp, sim = res
            imp_lst.append(imp)
            sim_lst.append(sim)
    mean_imp = np.mean(imp_lst)
    std_imp = np.std(imp_lst)
    mean_sim = np.mean(sim_lst)
    std_sim = np.std(sim_lst)
    # print(mean_imp,std_imp)
    return mean_imp,std_imp,mean_sim,std_sim




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_root', '-dir', type=str, default='../result/test')
    parser.add_argument('--oracle_name', '-oracle', type=str, default='qed')

    args = parser.parse_args()
    dir_root = args.dir_root
    oracle_name = args.oracle_name
    print(batch_validation(dir_root, oracle_name=oracle_name))






