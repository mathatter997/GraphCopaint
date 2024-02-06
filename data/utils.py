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
        colors = {node : {'colors' : -1 if random() < p
                                else 1}
                for node in G}
        nx.set_node_attributes(G, colors)
        L = Lobster(G)
        lobsterlist.append(L)
    return lobsterlist

def bfs(lobster, node):
    seen = {node}
    frontier = [(node, 0)]
    idx = 0
    while idx < lobster.card:
        node, dist = frontier[idx]
        for nbr in lobster.adj[node]:
            if nbr not in seen:
                seen.add(nbr)
                frontier.append((nbr, dist + 1))
        idx = idx + 1
    _, dist_max = frontier[-1]
    _, dist_snd_max = frontier[-2]
    return dist_max, dist_snd_max

def prepare_json_dataset(lobsterlist, filepath):
    with open(filepath, 'w') as f:
        f.write('[')
        for i, lobster in enumerate(lobsterlist):
            md1 = [0 for _ in range(lobster.card)]
            md2 = [0 for _ in range(lobster.card)]
            for node in lobster.adj:
                md1[node], md2[node] = bfs(lobster, node)
            lobsterinfo = {
                'id' : i,
                'card': lobster.card,
                'adj': lobster.adj,
                'colors': lobster.colors,
                'md1': md1,
                'md2': md2
            }   
            jsonobject = json.dumps(lobsterinfo)
            if i < len(lobsterlist) - 1:
                jsonobject += ','
            f.write(jsonobject + "\n")
        f.write(']')
    f.close()         

# turn Lobster into Data 
def encode(lobster, max_nodes, max_nbrs, max_md1, max_md2, num_node_feat=4, color_norm_factor=2, edge_attr_norm_factor=2):
    x = torch.zeros(max_nodes, num_node_feat)
    for i in range(lobster.card):
            x[i][0] = lobster.colors[i] / color_norm_factor
            x[i][1] = len(lobster.adj[i]) / (max_nbrs + 1)
            x[i][2] = lobster.md1[i] / (max_md1 + 1)
            x[i][3] = lobster.md2[i] / (max_md2 + 1)
            # x[i][0] = lobster.colors[i] 
            # x[i][1] = len(lobster.adj[i]) 
            # x[i][2] = lobster.md1[i] 
            # x[i][3] = lobster.md2[i]
    edge_index = []
    edge_attr = []
    for i in range(lobster.card):
        for j in range(lobster.card):
            if i != j: # no loops
                edge_index.append([i, j])
                attr = 1 / edge_attr_norm_factor if j in lobster.adj[i] else 0
                edge_attr.append(attr)
    for i in range(max_nodes):
        for j in range(max_nodes):
            if i != j and i < lobster.card and j < lobster.card:
                continue
            edge_index.append([i, j])
            edge_attr.append(0)
    
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr).reshape(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, card=lobster.card)

# convert graph or json dict into Lobster
class Lobster:
    def __init__(self, G : Union[nx.Graph, dict]):
        if type(G) is nx.Graph:
            self.card = len(G.nodes)
            self.adj = {node: [nbr for nbr in nbrsdict]
                        for node, nbrsdict in G.adj.items()}
            self.colors = [0 for _ in range(self.card)]
            for node, color in G.nodes('colors'):
                self.colors[node] = color
        else:
            self.card = G['card']
            self.adj = G['adj']
            self.colors = G['colors']
            self.md1 = G['md1']
            self.md2 = G['md2']
