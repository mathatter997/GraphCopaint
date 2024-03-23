import json
import torch
import networkx as nx
from typing import Union
from random import random
from torch_geometric.data import Data
from data.random_lobster import random_lobster_


def lobster_list(numsamples, backbonelength, p1, p2, p):
    lobsterlist = []
    for _ in range(numsamples):
        G = random_lobster_(backbonelength, p1, p2)
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
                'adj': lobster.adj
            }   
            jsonobject = json.dumps(lobsterinfo)
            if i < len(lobsterlist) - 1:
                jsonobject += ','
            f.write(jsonobject + "\n")
        f.write(']')
    f.close()         


def encode(lobster, max_nodes):
    edge_index = []
    edge_attr = []
    for i in range(lobster.card):
        for j in range(i + 1, lobster.card):
            if j in lobster.adj[i]:
                edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr).reshape(-1, 1)
    return Data(edge_index=edge_index, edge_attr=edge_attr, card=lobster.card)


class Lobster:
    def __init__(self, G : Union[nx.Graph, dict]):
        if type(G) is nx.Graph:
            self.card = len(G.nodes)
            self.adj = {node: [nbr for nbr in nbrsdict]
                        for node, nbrsdict in G.adj.items()}
        else:
            self.card = G['card']
            self.adj = G['adj']
