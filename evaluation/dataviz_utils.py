import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import seaborn as sns

def plot_loss_and_samples(config, adj_0s, size):
    losses = []
    assert len(adj_0s) == 1000

    time = np.arange(0, 1000)[::-1]
    numel = size * size
    for t in time:
        loss = (torch.sum((adj_0s[-1] - adj_0s[999 - t]) ** 2)) ** 0.5 / numel
        losses.append(loss.item())
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which="both", axis="both")  # dashed lines, with width 0.5, and gray color
    plt.xlabel('Timestep')
    plt.ylabel('Average Edge Weight Euclidean Distance')
    plt.plot(time, losses)
    # plt.xticks([1000, 800, 600, 400, 200, 0])  # Specify x-axis grid lines
    # plt.yticks([.07, .06, .05, .04, 0.03, .02, .01, 0])
    plt.savefig('data/sample_losses.png')
    plt.clf()
    times = np.array([1000, 800, 600, 400, 200, 1]) - 1
    for t in times:
        edges = adj_0s[999 - t][0,0,:size,:size].reshape(size, size).detach().cpu().numpy()
        if config.data_format == 'graph' or  config.data_format == 'eigen':
            G = nx.Graph()
            for i in range(size):
                G.add_node(i)
            for i in range(size):
                for j in range(i + 1, size):
                    if edges[i, j] > 0:
                        G.add_edge(i, j, weight = edges[i, j])
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            pos = nx.spring_layout(G, seed=7) 
            nx.draw(G, pos, with_labels=False, node_color='black', node_size=200, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)
            plt.savefig(f'data/images/graph_t={t}.png')
            plt.clf()
        elif config.data_format == 'pixel':
            plt.figure(figsize=(8, 6))
            sns.heatmap(edges, cmap='Greys', annot=False, fmt=".2f",xticklabels=False, yticklabels=False, cbar=False)
            plt.savefig(f'data/images/heatmap_t={t}.png')
            plt.clf()
        
def plot_diffs(config, diffs, size):
    assert len(diffs) == 1000

    time = np.arange(0, 1000)[::-1]
    # numel = size * size
    # for t in time:
    #     diffs[t]
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which="both", axis="both")  # dashed lines, with width 0.5, and gray color
    plt.xlabel('Timestep')
    plt.ylabel('Change in Adjacency per Timestep')
    plt.plot(time, diffs)
    # plt.xticks([1000, 800, 600, 400, 200, 0])  # Specify x-axis grid lines
    # plt.yticks([.07, .06, .05, .04, 0.03, .02, .01, 0])
    plt.savefig('data/sample_diffs.png')
    plt.clf()
    # times = np.array([1000, 800, 600, 400, 200, 1]) - 1
    # for t in times:
    #     edges = adj_0s[999 - t][0,0,:size,:size].reshape(size, size).detach().cpu().numpy()
    #     if config.data_format == 'graph':
    #         G = nx.Graph()
    #         for i in range(size):
    #             G.add_node(i)
    #         for i in range(size):
    #             for j in range(i + 1, size):
    #                 if edges[i, j] > 0:
    #                     G.add_edge(i, j, weight = edges[i, j])
    #         weights = [G[u][v]['weight'] for u, v in G.edges()]
    #         pos = nx.spring_layout(G, seed=7) 
    #         nx.draw(G, pos, with_labels=False, node_color='black', node_size=200, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)
    #         plt.savefig(f'data/images/graph_t={t}.png')
    #         plt.clf()
    #     elif config.data_format == 'pixel':
    #         plt.figure(figsize=(8, 6))
    #         sns.heatmap(edges, cmap='Greys', annot=False, fmt=".2f",xticklabels=False, yticklabels=False, cbar=False)
    #         plt.savefig(f'data/images/heatmap_t={t}.png')
    #         plt.clf()