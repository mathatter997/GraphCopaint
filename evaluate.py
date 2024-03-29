from evaluation.evaluator import get_stats_eval
from evaluation.structure_evaluator import cluster, degree, spectral
from data.data_loader import load_data
from data.dataset import get_dataset
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
import torch

eval_fn = get_stats_eval('rbf', eval_max_subgraph=False)

test_filepath = 'data/dataset/'
test_filename = 'Community_small'
pred_filepath = 'data/dataset/output_160000_t1000_ema_pgsn_v2.json'

# test_graphs, _ = load_data(test_filepath)
# pred_graphs, _ = load_data(pred_filepath)

train_dataset, _, test_dataset, n_node_pmf = get_dataset(test_filepath, test_filename, device='cpu')
pred_graphs, _ = load_data(pred_filepath)

# N = 1000
# sample_node_num = torch.multinomial(torch.Tensor(n_node_pmf), N, replacement=True)
# graphs = [nx.erdos_renyi_graph(n, p=0.7) for n in sample_node_num]
# test_graphs = [from_networkx(g) for g in graphs]

# for i,graph  in enumerate(test_graphs):
#     edge_index = graph.edge_index
#     reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
#     edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
#     temp = Data(edge_index=edge_index)
#     test_graphs[i] = temp

for i,graph  in enumerate(pred_graphs):
    edge_index = graph.edge_index
    if len(edge_index):
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        temp = Data(edge_index=edge_index)
        pred_graphs[i] = temp


    # temp = Data(edge_index=edge_index)

results = eval_fn(test_dataset=train_dataset, pred_graph_list=test_dataset)
print(results)

# half = len(test_graphs) // 2
# results = eval_fn(test_dataset=test_graphs[:half], pred_graph_list=test_graphs[half:])
# print(results)

results = eval_fn(test_dataset=train_dataset, pred_graph_list=pred_graphs)
print(results)