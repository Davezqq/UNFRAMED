import torch
from chemutils import smiles2feature_train



class Molecule_Dataset(torch.utils.data.Dataset):
	def __init__(self, smiles_lst):
		self.smiles_lst = smiles_lst

	def __len__(self):
		return len(self.smiles_lst)

	def __getitem__(self, idx):
		tmp_graph1, label1  = smiles2feature_train(self.smiles_lst[idx])

		return [tmp_graph1], [label1]



