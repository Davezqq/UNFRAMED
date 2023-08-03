import pickle

import matplotlib.image
from rdkit import Chem
from rdkit.Chem import QED,Draw
from tqdm import  tqdm
import pandas as pd
from tdc import Oracle
import numpy as np
from chemutils import similarity_mols,penalized_logp
# import drd_scorer
from matplotlib import image
# import sci
drd_scorer = Oracle('drd2')
def cal_mean_impr(data,origin_dict,metric='plogp'):
    count = 0
    score_lst = []
    tmp_origin_logp = None
    impr_lst = []
    for smile in tqdm(data):
        if smile is None:
            continue
        if tmp_origin_logp is None:
            tmp_origin_logp = penalized_logp(origin_dict[smile])

        if metric=='plogp':
            tmpscore = penalized_logp(smile)
            impr_lst.append(tmpscore-tmp_origin_logp)
    impr_lst = sorted(impr_lst,reverse=True)
    # print(impr_lst)
    return impr_lst[0]

def cal_success_rate(data,origin_dict,metric='drd'):
    count = 0
    score_lst = []
    # print(len())
    for smile in tqdm(data):
        if smile is None:
            continue
        tmpmol = Chem.MolFromSmiles(smile)
        tmp_origin_mol = Chem.MolFromSmiles(origin_dict[smile])
        # print([similarity_mols(tmpmol,start_mol) for start_mol in start_mols_lst])
        # min_sim = min([similarity_mols(tmpmol,start_mol) for start_mol in start_mols_lst])
        tmp_sim = similarity_mols(tmpmol,tmp_origin_mol)
        # tmp_qed = QED.qed(tmpmol)
        # tmp_origin_qed = QED.qed(tmp_origin_mol)
        if metric == 'qed':
            tmp_qed = QED.qed(tmpmol)
            tmp_origin_qed = QED.qed(tmp_origin_mol)
            # print(tmp_qed)
            score_lst.append((smile,tmp_qed))
            # print(tmp_origin_qed)
            # if tmp_qed>=0.9:
            if tmp_qed>= 0.6:
                if count < 20:
                    count += 1
                    break

        elif metric=='plogp':
            tmpscore = penalized_logp(smile)
            # print(tmpscore)
            tmp_origin_logp = penalized_logp(origin_dict[smile])
            # print(tmp_origin_logp)
            # tmp_logp = QED.properties(tmpmol).ALOGP
            # print(tmpscore-tmp_origin_logp)
            if (tmpscore-tmp_origin_logp)>=4:
                count+=1

        elif metric == 'drd':
            tmpscore = drd_scorer(smile)

            tmp_origin_drd = drd_scorer(origin_dict[smile])
            # tmp_logp = QED.properties(tmpmol).ALOGP
            # print('sim:',tmp_sim)
            # print('drd:',tmpscore)
            # if  tmpscore >= 0.5:
            if tmpscore >= 0.5:
                # print(tmpscore)
                count += 1
                break

        elif metric == 'qeddrd':
            tmp_qed = QED.qed(tmpmol)
            tmp_drd= drd_scorer(smile)
            score_lst.append((smile, tmp_qed))
            # print(tmp_qed,tmp_drd)
            if tmp_drd >= 0.5 and tmp_qed>=0.6:
                count += 1
                break

    return count/20


def load_pkl(start_smiles_lst,pkl_path,metric):
    with open(pkl_path,'rb') as f:
        origin_dict = {}
        try:
            data = pickle.load(f)
        except EOFError:
            return None
        # data = pickle.load(f)
        # print(len(data))
        if len(data)==1 or len(data)==0:
            f.close()
            return None
        for smile in tqdm(data):
            origin_dict[smile]=start_smiles_lst[0]
        # pickle.dump(origin_dict,open('origin_dict', 'wb'))
        if metric=='plogp':
            return  cal_mean_impr(data,origin_dict,metric=metric)
        else:
            res =  cal_success_rate(data,origin_dict,metric=metric)
            if res > 0:
                res =1
            return res


def load_pkl_for_imp(start_smiles_lst,pkl_path,metric):
    f = open(pkl_path, 'rb')
    origin_dict = {}
    start_mols_lst = [Chem.MolFromSmiles(smile) for smile in start_smiles_lst]
    data = pickle.load(f)
    root_mol = Chem.MolFromSmiles(start_smiles_lst[0])
    if metric=='qed':
        root_score = QED.qed(root_mol)
    elif metric=='plogp':
        root_score = penalized_logp(start_smiles_lst[0])
    elif metric == 'sim':
        root_score = 0
    elif metric == 'drd':
        root_score = drd_scorer(start_smiles_lst[0])
    elif metric == 'qeddrd':
        root_score = QED.qed(root_mol)+drd_scorer(start_smiles_lst[0])
    total = 0
    max_score = 0
    sim = 0
    score_lst = []
    flag = False
    for smile in data:
        if smile is None:
            continue
        tmp_mol = Chem.MolFromSmiles(smile)
        # imp = 0
        if metric == 'qed':
            tmp_score = QED.qed(tmp_mol)
            if tmp_score >= max_score:
                max_score = tmp_score
                if max_score-root_score >= 0.1:
                    sim = similarity_mols(tmp_mol, root_mol)
                    flag = True


        elif metric == 'plogp':
            tmp_score = penalized_logp(smile)
        elif metric == 'sim':
            tmp_score = similarity_mols(tmp_mol,root_mol)
        elif metric == 'drd':
            tmp_score = drd_scorer(smile)
            if tmp_score >= max_score:
                max_score = tmp_score
            # if max_score-root_score>=0.1:
            if max_score >= 0.5:
                sim = similarity_mols(tmp_mol, root_mol)
                flag = True
        elif metric == 'qeddrd':
            tmp_drdscore = drd_scorer(smile)
            tmp_qedscore = QED.qed(tmp_mol)
            if tmp_drdscore >= 0.5 and tmp_qedscore >=0.6:
                if tmp_drdscore+tmp_qedscore >= max_score:
                    max_score = tmp_drdscore+tmp_qedscore
                    sim = similarity_mols(tmp_mol, root_mol)
                    flag=True


    if flag:

        return (max_score-root_score,sim)
    else:
        return None
    # print(avg-root_score)


