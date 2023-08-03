import torch
from torch import nn
import dgl.function as dglfn

class multihead_layer(nn.Module):
    def __init__(self,num_head,nfeat,nhid):
        super(multihead_layer,self).__init__()
        self.activation = nn.ReLU()
        self.num_head = num_head
        self.nfeat = nfeat
        self.dim_multi_head = int(nfeat // self.num_head)
        self.nhid = nhid
        self.multi_head_fc_q_lst = nn.ModuleList()
        self.multi_head_ln_q_lst = nn.ModuleList()
        for i in range(self.num_head):
            self.multi_head_fc_q_lst.append(nn.Linear(self.nfeat,self.dim_multi_head,bias=False))
            self.multi_head_ln_q_lst.append(nn.LayerNorm(self.dim_multi_head))

        self.multi_head_fc_k_lst = nn.ModuleList()
        self.multi_head_ln_k_lst = nn.ModuleList()
        for i in  range(self.num_head):
            self.multi_head_fc_k_lst.append(nn.Linear(self.nfeat, self.dim_multi_head,bias=False))
            self.multi_head_ln_k_lst.append(nn.LayerNorm(self.dim_multi_head))

        self.multi_head_fc_v_lst = nn.ModuleList()
        self.multi_head_ln_v_lst = nn.ModuleList()
        for i in  range(self.num_head):
            self.multi_head_fc_v_lst.append(nn.Linear(self.nfeat, self.dim_multi_head,bias=False))
            self.multi_head_ln_v_lst.append(nn.LayerNorm(self.dim_multi_head))

        self.multi_head_fc_k_edge_lst = nn.ModuleList()
        self.multi_head_ln_k_edge_lst = nn.ModuleList()
        for i in range(self.num_head):
            self.multi_head_fc_k_edge_lst.append(nn.Linear(self.nfeat, self.dim_multi_head, bias=False))
            self.multi_head_ln_k_edge_lst.append(nn.LayerNorm(self.dim_multi_head))

        self.multi_head_fc_v_edge_lst = nn.ModuleList()
        self.multi_head_ln_v_edge_lst = nn.ModuleList()
        for i in range(self.num_head):
            self.multi_head_fc_v_edge_lst.append(nn.Linear(self.nfeat, self.dim_multi_head, bias=False))
            self.multi_head_ln_v_edge_lst.append(nn.LayerNorm(self.dim_multi_head))

        self.rel_layer_fc_k = nn.Linear(self.dim_multi_head * 2, self.dim_multi_head)
        self.rel_layer_ln_k = nn.LayerNorm(self.dim_multi_head)
        self.rel_layer_fc_v = nn.Linear(self.dim_multi_head * 2, self.dim_multi_head)
        self.rel_layer_ln_v = nn.LayerNorm(self.dim_multi_head)

        self.z_fc = nn.Linear(self.dim_multi_head, self.dim_multi_head)
        self.z_ln = nn.LayerNorm(self.dim_multi_head)

    def rel_multihead_atten(self,edges):
        ####for predicting node noise
        feat_concat_k = torch.concat([edges.data['k'], edges.src['k']], dim=-1)
        # feat_concat_v = self.torch.concat([edges.data['v'], edges.dst['v']], dim=-1)

        dst_q = edges.dst['q']
        # print(src_q.shape)

        rel_k = self.rel_layer_ln_k(self.rel_layer_fc_k(feat_concat_k))
        # print(src_q.shape)
        # print(rel_k.T.shape)
        # print(torch.sum(torch.multiply(src_q, rel_k),dim=-1).shape)
        # print(torch.matmul(src_q, rel_k.T).shape)
        return {'q_k_mul': torch.sum(torch.multiply(dst_q, rel_k)/4, dim=-1)}

    def solve_softmax_in_node(self,nodes):
        sum_in_node = torch.logsumexp(nodes.mailbox['q_k_mul'],dim=-1)

        return {'sum_in_node': sum_in_node}

    def att_aggre(self, edges):
        feat_concat_v = torch.concat([edges.data['v'], edges.src['v']], dim=-1)
        rel_v = self.rel_layer_ln_v(self.rel_layer_fc_v(feat_concat_v))
        v = edges.data['q_k_mul'] - edges.dst['sum_in_node']
        v = torch.unsqueeze(v, dim=-1)
        v = v * rel_v
        return {'h': v}

    def reduce_func(self,nodes):
        sum_fet = torch.sum(nodes.mailbox['h'], dim=1)
        z = self.activation(self.z_fc(sum_fet))
        return {'z':z}

    def forward(self,graph,x_node,x_edge):
        multi_head_feature_lst = []
        for i in range(self.num_head):
            k_edge = self.multi_head_ln_k_edge_lst[i](self.multi_head_fc_k_edge_lst[i](x_edge))
            v_edge = self.multi_head_ln_v_edge_lst[i](self.multi_head_fc_v_edge_lst[i](x_edge))
            q = self.multi_head_ln_q_lst[i](self.multi_head_fc_q_lst[i](x_node))
            k = self.multi_head_ln_k_lst[i](self.multi_head_fc_k_lst[i](x_node))
            v = self.multi_head_ln_v_lst[i](self.multi_head_fc_v_lst[i](x_node))
            graph.ndata['q'] = q
            graph.ndata['k'] = k
            graph.ndata['v'] = v
            graph.edata['k'] = k_edge
            graph.edata['v'] = v_edge
            graph.apply_edges(self.rel_multihead_atten)
            graph.update_all(message_func=dglfn.copy_e('q_k_mul', 'q_k_mul'), reduce_func=self.solve_softmax_in_node)
            graph.update_all(message_func=self.att_aggre, reduce_func=self.reduce_func)
            bn_graph_z = self.z_ln(graph.ndata['z'])
            multi_head_feature_lst.append(bn_graph_z)

        total_z = torch.concat(multi_head_feature_lst, dim=-1)
        # print(total_z.shape)
        return total_z




