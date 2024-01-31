import json
import networkx as nx
from random import random
from random_lobster import random_lobster_


def lobster_list(numsamples, backbonelength, p1, p2, p):
    lobsterlist = []
    for _ in range(numsamples):
        G = random_lobster_(backbonelength, p1, p2)
        colors = {node : {'color' : 0 if random() < p
                                else 1}
                for node in G}
        nx.set_node_attributes(G, colors)
        L = Lobster(G)
        lobsterlist.append(L)
    return lobsterlist

def prepare_json_dataset(lobsterlist, filepath):
    with open(filepath, 'w') as f:
        f.write('[')
        for i, lobster in enumerate(lobsterlist):
            lobsterinfo = {
                'id' : i,
                'card': lobster.card,
                'adj': lobster.adj,
                'colors': lobster.colors
            }   
            jsonobject = json.dumps(lobsterinfo)
            if i < len(lobsterlist) - 1:
                jsonobject += ','
            f.write(jsonobject + "\n")
        f.write(']')
    f.close()         


class Lobster:
    def __init__(self, G : nx.Graph):
        self.card = len(G.nodes)
        self.adj = {node: [nbr for nbr in nbrsdict]
                    for node, nbrsdict in G.adj.items()}
        self.colors = {node: color for node, color in G.nodes('color')}
