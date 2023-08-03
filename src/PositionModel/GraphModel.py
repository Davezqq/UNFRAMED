import dgl
import torch
from torch import nn,autograd
import dgl.function as dglfn
from dgl.nn import pytorch as dglnn
from chemutils import vocabulary
from Multihead_module import multihead_layer

import math

class Graph_model(nn.Module):

    def __init__(self,nfeat, nhid, num_head, num_multi_layers,num_rel_gcn_layers,num_edge_types,num_linear_layer,lr):
        super(Graph_model, self).__init__()
        self.activation = nn.LeakyReLU()
        self.nfeat = nfeat
        self.Softmax = nn.Softmax(dim=-1)
        self.rgc_seq = nn.ModuleList()
        self.nhid = nhid
        self.lr = lr
        self.num_edge_types = num_edge_types
        self.num_head = num_head
        self.num_multi_layers = num_multi_layers
        self.num_rel_gcn_layers = num_rel_gcn_layers
        self.multi_layers_lst = nn.ModuleList()
        self.num_linear_layer = num_linear_layer
        self.vocabulary_size = len(vocabulary)

        self.Atom_embedding = nn.Embedding(num_embeddings=self.vocabulary_size + 1, embedding_dim=nfeat,
                                           padding_idx=self.vocabulary_size)
        self.Bond_embedding = nn.Embedding(num_embeddings=self.num_edge_types + 1, embedding_dim=nfeat,
                                           padding_idx=self.num_edge_types)

        for i in range(self.num_multi_layers):
            self.multi_layers_lst.append(multihead_layer(self.num_head,self.nfeat,self.nfeat))

        for i in range(self.num_rel_gcn_layers):
            if i < self.num_rel_gcn_layers - 1:
                self.rgc_seq.add_module(name=f'rgc_{i}', module=dglnn.RelGraphConv(nhid[i], nhid[i + 1], num_edge_types,
                                                                                   activation=self.activation,
                                                                                   self_loop=True, dropout=0.1,
                                                                                   layer_norm=True))
            else:
                self.rgc_seq.add_module(name=f'rgc_{i}', module=dglnn.RelGraphConv(nhid[i], nhid[i], num_edge_types,
                                                                                   activation=self.activation,
                                                                                   self_loop=True, dropout=0.1,
                                                                                   layer_norm=True))
        self.total_skip_ln = nn.Linear(nfeat*(self.num_multi_layers+1),nhid[0],bias=False)
        self.total_skip_norm = nn.LayerNorm(nhid[i])
        self.linear_seq = nn.ModuleList()
        self.LN_seq = nn.ModuleList()
        linear_nhid = nhid[::-1]
        for i in range(self.num_linear_layer):
            if i == 0:
                self.linear_seq.add_module(name=f'linear_{i}', module=nn.Linear(
                    in_features=linear_nhid[0] * 3 + linear_nhid[1] + linear_nhid[2], out_features=linear_nhid[i]))
            else:
                self.linear_seq.add_module(name=f'linear_{i}', module=nn.Linear(in_features=linear_nhid[i - 1],
                                                                                out_features=linear_nhid[i]))

            self.LN_seq.add_module(name=f'LN_{i}', module=nn.LayerNorm(normalized_shape=linear_nhid[i]))
        self.out_fc = nn.Linear(linear_nhid[-1], 4)
        self.opt = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,graph:dgl.DGLGraph):
        x_node = self.Atom_embedding(graph.ndata['atom_type'])
        # print(x_node.shape)
        x_edge = self.Bond_embedding(graph.edata['bond_type'])
        graph.ndata['nfet'] = x_node
        graph.edata['efet'] = x_edge
        # print(x_node)
        # x_time_step = torch.repeat_interleave()
        multi_head_feature_lst = []
        skip_mulatt = [x_node]
        x_mal = x_node
        for multi_att_layer in self.multi_layers_lst:
            x_mal = multi_att_layer(graph,x_mal,x_edge)
            skip_mulatt.append(x_mal)
        total_skips = torch.concat(skip_mulatt,dim=-1)
        total_skips = self.total_skip_norm(self.activation(self.total_skip_ln(total_skips)))
        # print(total_skips.shape)
        skip_rgc = [total_skips]
        for gc in self.rgc_seq:
            total_skips = gc(graph, total_skips, graph.edata['bond_type'])
            skip_rgc.append(total_skips)
        total_x = skip_rgc
        total_x = torch.concat(total_x, dim=-1)
        for linear, ln in zip(self.linear_seq, self.LN_seq):
            total_x = ln(self.activation((linear(total_x))))
        logits = self.out_fc(total_x)

        return logits

    def switch_device(self, device):
        self.device = device
        self = self.to(device)

    def learn(self, graph, label):
        pred_y = self.forward(graph)
        cost = self.criteria(pred_y, label)
        self.opt.zero_grad()
        cost.backward()
        self.opt.step()
        return cost.data.cpu().numpy(), pred_y.data.cpu().numpy()

    def infer(self, graph):
        pred_y = self.forward(graph)
        return pred_y.data.cpu()

    def infer_train(self, graph, label):
        with torch.no_grad():
            pred_y = self.forward(graph)
            if label is not None:
                cost = self.criteria(pred_y, label).mean()
                return cost.data.cpu().numpy(), pred_y.data.cpu().numpy()

            return pred_y.data.cpu().numpy()







        # multi_head_feature = torch.concat(multi_head_feature_lst,dim=-1)
