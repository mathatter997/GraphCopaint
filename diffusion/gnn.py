import torch
from torch import nn
from torch.nn import Module, SiLU, Linear

# most of this is adapted from egnn code base
class EdgeMLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim, act_fn=SiLU):
        super(EdgeMLP, self).__init__()
        self.edge_mlp = nn.Sequential(
            Linear(input_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, output_dim))

    def forward(self, x_i, x_j, e_ij, t):
        x_1 = x_i + x_j
        x_2 = torch.abs(x_i - x_j)
        v = torch.column_stack([x_1, x_2, e_ij, t])
        esp_e = self.edge_mlp(v)
        return esp_e + e_ij

class EdgeEncoder(Module):
    def __init__(self, input_dim, output_dim, hidden_dim, act_fn=SiLU):
        super(EdgeEncoder, self).__init__()
        self.edge_mlp = nn.Sequential(
            Linear(input_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, output_dim))

    def forward(self, x_i, x_j, e_ij):
        x_1 = x_i + x_j
        x_2 = torch.abs(x_i - x_j)
        v = torch.column_stack([x_1, x_2, e_ij])
        e_enc = self.edge_mlp(v)
        return e_enc

class NodeMLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim, act_fn=SiLU):
        super(NodeMLP, self).__init__()
        self.node_mlp = nn.Sequential(
            Linear(input_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, hidden_dim),
            act_fn,
            Linear(hidden_dim, output_dim))

    def forward(self, x, e, t):
        x = torch.column_stack([x, e, t])
        eps_x = self.node_mlp(x)
        return eps_x


class GCL(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, normalization_factor, aggregation_method,
                 act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        input_dim = 2 * node_dim + edge_dim + 1 # plus time
        self.edge_mlp = EdgeMLP(input_dim=input_dim,
                                output_dim=edge_dim,
                                hidden_dim=hidden_dim,
                                act_fn=act_fn)
        
        input_dim = 2 * node_dim + edge_dim 
        self.edge_enc = EdgeEncoder(input_dim=input_dim,
                                output_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                act_fn=act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
        
        input_dim = node_dim + hidden_dim + 1
        self.node_mlp = NodeMLP(input_dim=input_dim,
                                output_dim=hidden_dim,
                                hidden_dim=node_dim,
                                act_fn=act_fn)
    
    def edge_model(self, source, target, edge_attr, t, edge_mask):
        out = self.edge_mlp(source, target, edge_attr, t)
        if edge_mask is not None:
            out = out * edge_mask
        return out

    def edge_encoder(self, source, target, edge_attr, edge_mask):
        mij = self.edge_enc(source, target, edge_attr)
        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij
        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_emb, t, node_mask=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_emb, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        
        out = self.node_mlp(x, agg, t)
        if node_mask is not None:
            out = out * node_mask
        return out

    # t is a tensor of batch size.
    def forward(self, x, t, edge_index, edge_attr, batch_size, node_mask=None, edge_mask=None):
        row, col = edge_index
        num_nodes = x.size(0) // batch_size  # works because the graphs have the same number of nodes (including masked nodes)
        num_edges = edge_index.size(1) // batch_size # works because the graphs have the same number of edges (including masked edges)
        _t = t.repeat_interleave(num_edges).view(-1, 1)
        edge_attr = self.edge_model(x[row], x[col], edge_attr, _t, edge_mask)
        edge_emb = self.edge_encoder(x[row], x[col], edge_attr, edge_mask)  
        _t = t.repeat_interleave(num_nodes).view(-1, 1)
        x = self.node_model(x, edge_index, edge_emb, _t, node_mask)
        return x, edge_attr


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None, out_edge_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        if out_edge_nf is None:
            out_edge_nf = in_edge_nf

        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.node_embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.node_embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        ### Encoder
        self.edge_embedding = nn.Linear(in_edge_nf, self.hidden_nf)
        self.edge_embedding_out = nn.Linear(self.hidden_nf, out_edge_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                act_fn=act_fn, attention=attention))
        self.to(self.device)

    def forward(self,  x, t, edge_index, edge_attr, batch_size, node_mask=None, edge_mask=None):
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for i in range(0, self.n_layers):
            x, edge_attr = self._modules["gcl_%d" % i](x=x, 
                                                       t=t,
                                                       edge_index=edge_index, 
                                                       edge_attr=edge_attr, 
                                                       batch_size=batch_size,
                                                       node_mask=node_mask, 
                                                       edge_mask=edge_mask)
                   
        x = self.node_embedding_out(x)
        edge_attr = self.edge_embedding_out(edge_attr)
        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            x = x * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return x, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

