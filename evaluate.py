from evaluation.evaluator import get_stats_eval
from evaluation.structure_evaluator import cluster, degree, spectral
from data.data_loader import load_data
from data.dataset import get_dataset
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import networkx as nx
import torch
import click 


@click.command()
@click.option('--data_path', default='data/dataset')
@click.option('--dataset', default='Community_small')
@click.option('--pred_file', default='data/dataset/output_com_small.json')
def evaluate(data_path, dataset, pred_file):
    eval_fn = get_stats_eval('rbf', eval_max_subgraph=False)
    train_dataset, _, test_dataset, n_node_pmf = get_dataset(data_path, dataset, device='cpu')
    pred_graphs, _ = load_data(pred_file)

    for i,graph  in enumerate(pred_graphs):
        edge_index = graph.edge_index
        if len(edge_index):
            reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
            temp = Data(edge_index=edge_index)
            pred_graphs[i] = temp

    results_data = eval_fn(test_dataset=test_dataset, pred_graph_list=train_dataset)
    results_pred = eval_fn(test_dataset=test_dataset, pred_graph_list=pred_graphs)
    print(f'{dataset} data:', results_data)
    print(f'{dataset} pred:', results_pred)

if __name__ == '__main__':
    evaluate()