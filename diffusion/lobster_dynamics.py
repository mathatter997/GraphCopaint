import torch
import torch.nn as nn
from diffusion.gnn import GNN

# node features: color (independent of other nodes), degree, further node distance, second furthest node distance
# edge features: edge strength [0-1]

class LobsterDynamics(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=100, aggregation_method='sum'):
        super().__init__()

        self.gnn = GNN(
            in_node_nf=in_node_nf, in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf, device=device,
            act_fn=act_fn, n_layers=n_layers, attention=attention,
            normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.device = device

    def forward(self, x, edge_attr, t, edge_index, node_mask=None, edge_mask=None, batch_size=1):
        x, edge_attr = self.gnn(x=x,
                                edge_attr=edge_attr,
                                t=t,
                                edge_index=edge_index,
                                batch_size=batch_size,
                                node_mask=node_mask,
                                edge_mask=edge_mask
                                )
        return x, edge_attr

        
