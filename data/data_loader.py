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
    # encode lobsters as pyg Data objects
    for k, lobster in enumerate(lobster_list):
        lobster_list[k] = encode(lobster, max_nodes)

    return lobster_list, max_nodes                 



