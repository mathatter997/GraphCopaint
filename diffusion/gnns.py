import torch.nn as nn
import torch
from .trans_layers import *
import torch
import torch.nn.functional as F

from diffusion.layers import DenseGCNConv, MLP
from diffusion.utils import mask_x
import diffusion.layers as layers

class pos_gnn(nn.Module):
    def __init__(self, act, x_ch, pos_ch, out_ch, max_node, graph_layer, n_layers=3, edge_dim=None, heads=4,
                 temb_dim=None, dropout=0.1, attn_clamp=False):
        super().__init__()
        self.out_ch = out_ch
        self.Dropout_0 = nn.Dropout(dropout)
        self.act = act
        self.max_node = max_node
        self.n_layers = n_layers

        if temb_dim is not None:
            self.Dense_node0 = nn.Linear(temb_dim, x_ch)
            self.Dense_node1 = nn.Linear(temb_dim, pos_ch)
            self.Dense_edge0 = nn.Linear(temb_dim, edge_dim)
            self.Dense_edge1 = nn.Linear(temb_dim, edge_dim)

        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.edge_layer = nn.Linear(edge_dim * 2 + self.out_ch, edge_dim)

        for i in range(n_layers):
            if i == 0:
                self.convs.append(eval(graph_layer)(x_ch, pos_ch, self.out_ch//heads, heads, edge_dim=edge_dim*2,
                                                    act=act, attn_clamp=attn_clamp))
            else:
                self.convs.append(eval(graph_layer)
                                  (self.out_ch, pos_ch, self.out_ch//heads, heads, edge_dim=edge_dim*2, act=act,
                                   attn_clamp=attn_clamp))
            self.edge_convs.append(nn.Linear(self.out_ch, edge_dim*2))

    def forward(self, x_degree, x_pos, edge_index, dense_ori, dense_spd, dense_index, temb=None):
        """
        Args:
            x_degree: node degree feature [B*N, x_ch]
            x_pos: node rwpe feature [B*N, pos_ch]
            edge_index: [2, edge_length]
            dense_ori: edge feature [B, N, N, nf//2]
            dense_spd: edge shortest path distance feature [B, N, N, nf//2]
            dense_index
            temb: [B, temb_dim]
        """

        B, N, _, _ = dense_ori.shape

        if temb is not None:
            dense_ori = dense_ori + self.Dense_edge0(self.act(temb))[:, None, None, :]
            dense_spd = dense_spd + self.Dense_edge1(self.act(temb))[:, None, None, :]

            temb = temb.unsqueeze(1).repeat(1, self.max_node, 1)
            temb = temb.reshape(-1, temb.shape[-1])
            x_degree = x_degree + self.Dense_node0(self.act(temb))
            x_pos = x_pos + self.Dense_node1(self.act(temb))

        dense_edge = torch.cat([dense_ori, dense_spd], dim=-1)

        ori_edge_attr = dense_edge
        h = x_degree
        h_pos = x_pos

        for i_layer in range(self.n_layers):
            h_edge = dense_edge[dense_index]
            # update node feature
            h, h_pos = self.convs[i_layer](h, h_pos, edge_index, h_edge)
            h = self.Dropout_0(h)
            h_pos = self.Dropout_0(h_pos)

            # update dense edge feature
            h_dense_node = h.reshape(B, N, -1)
            cur_edge_attr = h_dense_node.unsqueeze(1) + h_dense_node.unsqueeze(2)  # [B, N, N, nf]
            dense_edge = (dense_edge + self.act(self.edge_convs[i_layer](cur_edge_attr))) / math.sqrt(2.)
            dense_edge = self.Dropout_0(dense_edge)

        # Concat edge attribute
        h_dense_edge = torch.cat([ori_edge_attr, dense_edge], dim=-1)
        h_dense_edge = self.edge_layer(h_dense_edge).permute(0, 3, 1, 2)

        return h_dense_edge


class ScoreNetworkA_eigen(torch.nn.Module):

    def __init__(self, max_feat_num, nhid, max_node_num, num_layers, num_linears,
                    c_init, c_hid, c_final, adim, depth=3, num_heads=4, conv='GCN'):

        super(ScoreNetworkA_eigen, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)

        self.final_with_eigen = MLP(num_layers=2, input_dim=self.nfeat + max_node_num, hidden_dim=2 * max_node_num, output_dim=max_node_num,
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags, u, la):

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)
        flag_sum = torch.sum(flags, dim=1).unsqueeze(-1)
        flag_sum[flag_sum < 0.0000001] = 1
        x = torch.sum(x, dim=1)/flag_sum
        x = torch.cat((x, la), dim=-1)
        x = self.final_with_eigen(x)
        return x

class ScoreNetworkX(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid):

        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags, u, la):

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x
    

class ScoreNetworkTest_eigen(torch.nn.Module):

    def __init__(self, max_feat_num, nhid, max_node_num, num_layers, num_linears,
                    c_init, c_hid, c_final, adim, depth=3, num_heads=4, conv='GCN'):

        super(ScoreNetworkTest_eigen, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        self.nf = 256
        modules = []
        modules.append(nn.Linear(self.nf, 2 * self.nf))
        modules.append(nn.Linear(2 * self.nf, 2 * self.nf))
        modules.append(nn.Linear(max_node_num, 2 * self.nf))
        modules.append(nn.Linear(2 * self.nf, 2* self.nf))
        # for _ in range(self.depth):
        #     if _ == 0:
        #         self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
        #     else:
        #         self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        # self.fdim = self.depth * self.nhid
        # self.final = MLP(num_layers=3, input_dim=max_node_num, hidden_dim=2*self.fdim, output_dim=self.nfeat,
        #                     use_bn=False, activate_func=F.elu)

        self.final_with_eigen = MLP(num_layers=3, input_dim= 2 * self.nf, hidden_dim= 4 * self.nf, output_dim=self.nf//2,
                         use_bn=False, activate_func=nn.SiLU())
        
       
        modules.append(nn.Conv1d(1, self.nf // 2, kernel_size=1, 
                  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))
        modules.append(nn.Conv1d(self.nf // 2, self.nf // 2, kernel_size=1, 
                  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))
        
        modules.append(nn.Conv1d(self.nf // 2, self.nf // 2, kernel_size=1, 
                  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))
        modules.append(nn.Conv1d(self.nf // 2, 1, kernel_size=1, 
                  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'))

        modules.append(nn.Linear(self.nf // 2, self.nf // 2))
        modules.append(nn.Linear(self.nf // 2, max_node_num))
        self.emb_modules = nn.ModuleList(modules)


        self.act = nn.SiLU()

    def forward(self, x, adj, flags, u, la, t):

        timesteps = t
        temb = layers.get_timestep_embedding(timesteps, self.nf)
        m_idx=  0
        # time embedding
        temb = self.emb_modules[m_idx](temb)
        m_idx += 1
        temb = self.emb_modules[m_idx](self.act(temb))
        m_idx += 1

        la_emb = self.emb_modules[m_idx](la)
        m_idx += 1
        la_emb = self.emb_modules[m_idx](la_emb)
        m_idx += 1
        emb = la_emb + temb
        emb = emb.unsqueeze(1)
        emb = self.emb_modules[m_idx](emb)
        m_idx += 1
        emb = self.emb_modules[m_idx](emb)
        m_idx += 1

        emb = self.final_with_eigen(emb)
        emb = self.emb_modules[m_idx](emb)
        m_idx += 1
        emb = self.emb_modules[m_idx](emb)
        m_idx += 1
        emb = emb.squeeze(1)
        emb = self.emb_modules[m_idx](emb)
        m_idx += 1
        emb = self.emb_modules[m_idx](emb)

        # x_list = [x]
        # for _ in range(self.depth):
        #     x = self.layers[_](x, adj)
        #     x = self.activation(x)
        #     x_list.append(x)

        # xs = torch.cat(x_list, dim=-1)
        # out_shape = (adj.shape[0], adj.shape[1], -1)
        # x = self.final(adj).view(*out_shape)
        # x = mask_x(x, flags)
        # flag_sum = torch.sum(flags, dim=1).unsqueeze(-1)
        # flag_sum[flag_sum < 0.0000001] = 1
        # x = torch.sum(x, dim=1)/flag_sum
        # x = torch.cat((x, la), dim=-1)
        # x = self.final_with_eigen(emb)
        return emb