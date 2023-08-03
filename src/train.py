import torch
from torch.utils.data import DataLoader
import dgl
from tqdm import tqdm
from random import shuffle
from GraphModel_SSFG import Graph_model
import argparse
from utils import Molecule_Dataset
torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cpu'


def collate_fn(batch_lst):

	return batch_lst


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs','-ep', type=int, default=200)
	parser.add_argument('--batch_size','-bs', type=int, default=128)
	parser.add_argument('--num_workers','-nw', type=int, default=5)
	parser.add_argument('--num_edge_type','-ne', type=int, default=5)
	parser.add_argument('--dim_of_feat','-feat', type=int, default=64)
	parser.add_argument('--num_head','-nh', type=int, default=4)
	parser.add_argument('--num_multi_layers','-nmul', type=int, default=3)
	parser.add_argument('--learning_rate','-lr', type=int, default=1e-2)
	parser.add_argument('--data_file', '-data', type=str, default='')

	args = parser.parse_args()
	# print(args)
	epochs = args.epochs
	batch_size = args.batch_size
	num_workers = args.num_workers
	num_edge_type = args.num_edge_type
	dim_of_feat = args.dim_of_feat
	num_head = args.num_head
	num_multi_layers = args.num_multi_layers
	lr = args.learning_rate
	data_file = args.data_file
	data_params = {'batch_size': batch_size,
	          'shuffle': True,
	          'num_workers': num_workers}

	with open(data_file, 'r') as fin:
		lines = fin.readlines()

	shuffle(lines)
	lines = [line.strip('\n') for line in lines]
	N = int(len(lines) * 0.99)
	train_data = lines[:N]
	valid_data = lines[N:]

	training_set = Molecule_Dataset(train_data)

	valid_set = Molecule_Dataset(valid_data)

	# torch.multiprocessing.set_start_method('spawn')
	train_generator = DataLoader(training_set, collate_fn = collate_fn, **data_params)
	valid_generator = DataLoader(valid_set, collate_fn = collate_fn, **data_params)

	gnn = Graph_model(nfeat=dim_of_feat, nhid=[256,256, 256, 256], num_head=num_head, num_multi_layers=num_multi_layers,num_rel_gcn_layers=4,num_edge_types=num_edge_type,num_linear_layer=4,lr=lr).to(device)
	CosineLR = torch.optim.lr_scheduler.StepLR(gnn.opt, gamma=0.92,step_size=3)

	save_folder = "save_model/Graph_"
	for ep in tqdm(range(0,epochs)):
		gnn.train()
		for i, batch in tqdm(enumerate(train_generator)):
			graph_batch = []
			label_batch = []
			for batch_ele in batch:
				if batch_ele[0] is None:
					continue
				else:
					graph_batch+=batch_ele[0]
					label_batch+=batch_ele[1]
			graph_batch = dgl.batch(graph_batch).to(device)
			label_batch = torch.concat(label_batch).to(device)
			cost = gnn.learn(graph_batch, label_batch)
		gnn.eval()
		valid_loss, valid_num = 0,0
		for batch  in tqdm(valid_generator):
			graph_batch = []
			label_batch = []

			for batch_ele in batch:
				if batch_ele[0] is None:
					continue
				else:
					graph_batch+=batch_ele[0]
					label_batch+=batch_ele[1]
			graph_batch = dgl.batch(graph_batch).to(device)
			label_batch = torch.concat(label_batch).to(device)
			cost, _ = gnn.infer_train(graph_batch, label_batch)
			valid_loss += cost
			valid_num += 1
		valid_loss = valid_loss / valid_num
		file_name = save_folder + str(ep) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
		torch.save(gnn, file_name)
		CosineLR.step()
