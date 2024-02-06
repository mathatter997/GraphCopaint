import json
import torch
from data.utils import Lobster, encode


# filepath = 'dataset/lobsters.json'
def load_data(filepath):

    def keys_to_int(lobster):
        lobster['adj'] = {int(node) : nbrs for node, nbrs in lobster['adj'].items()}
        return lobster

    with open(filepath, 'r') as f:
        lobster_list = json.load(f)   
        for k, lobster in enumerate(lobster_list):
            lobster = keys_to_int(lobster)
            lobster_list[k] = Lobster(lobster)
        f.close()

    max_nodes = max(lobster.card for lobster in lobster_list)
    max_nbrs = max(max(len(lobster.adj[i]) for i in lobster.adj) for lobster in lobster_list)
    max_md1 = max(max(md1 for md1 in lobster.md1) for lobster in lobster_list)
    max_md2 = max(max(md2 for md2 in lobster.md2) for lobster in lobster_list)
    # encode lobsters as pyg Data objects
    print(max_nodes, max_nbrs, max_md1, max_md2)
    for k, lobster in enumerate(lobster_list):
        lobster_list[k] = encode(lobster, max_nodes, max_nbrs, max_md1, max_md2)
        
    return lobster_list, max_nodes                 



