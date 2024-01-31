import json
import torch
from torch_geometric.data import Data

# filepath = 'dataset/lobsters.json'
def load_data(filepath):

    def keys_to_int(lobster):
        lobster['adj'] = {int(node) : nbrs for node, nbrs in lobster['adj'].items()}
        lobster['colors'] = {int(node) : nbrs for node, nbrs in lobster['colors'].items()}
        return lobster

    with open(filepath, 'r') as f:
        lobsters = json.load(f)   
        for k, lobster in enumerate(lobsters):
            lobsters[k] = keys_to_int(lobster)
        f.close()

    for k, lobster in enumerate(lobsters):
        colors = torch.zeros(size=(lobster['card'], 1), dtype=torch.bool)
        for i in lobster['colors']:
            colors[i] = lobster['colors'][i]
        # lobster is a tree, edges = nodes - 1
        # pyg uses directed graphs, x2
        num_edges = 2 * (lobster['card'] - 1) 
        edge_index = torch.zeros(size=(2, num_edges), dtype=torch.long)
        idx = 0
        for node, nbrs in lobster['adj'].items():
            for nbr in nbrs:
                edge_index[0][idx] = node
                edge_index[1][idx] = nbr
                idx = idx + 1
        lobsters[k] = Data(x=colors, edge_index=edge_index)
    return lobsters                 



