import json
from data.utils import Lobster, encode


def get_lobsters_from_json(filepath):

    def keys_to_int(lobster):
        lobster['adj'] = {int(node) : nbrs for node, nbrs in lobster['adj'].items()}
        return lobster

    with open(filepath, 'r') as f:
        lobster_list = json.load(f)   
        for k, lobster in enumerate(lobster_list):
            lobster = keys_to_int(lobster)
            lobster_list[k] = Lobster(lobster)
        f.close()
    
    return lobster_list

# filepath = 'dataset/lobsters.json'
def get_max_nodes(lobster_list):
    max_nodes = max(lobster.card for lobster in lobster_list)
    return max_nodes


def load_data(filepath):
    lobster_list = get_lobsters_from_json(filepath)
    max_nodes = get_max_nodes(lobster_list)
    for k, lobster in enumerate(lobster_list):
        lobster_list[k] = encode(lobster, max_nodes)

    return lobster_list, max_nodes                 


