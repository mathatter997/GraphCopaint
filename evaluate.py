from evaluation.evaluator import get_stats_eval
from evaluation.structure_evaluator import cluster, degree, spectral
from data.data_loader import load_data
from data.dataset import get_dataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from torch_geometric.data import Data
import networkx as nx
import torch
import click
import numpy as np

@click.command()
@click.option("--data_path", default="data/dataset")
@click.option("--dataset", default="Community_small")
@click.option("--pred_file", default="data/dataset/output_com_small.json")
@click.option("--inpaint_loss", default=False)
@click.option("--mask_path", default=None)
@click.option("--masked_target_path", default=None)
def evaluate(
    data_path, dataset, pred_file, inpaint_loss, mask_path, masked_target_path
):
    if not inpaint_loss:
        eval_fn = get_stats_eval("rbf", eval_max_subgraph=False)
        train_dataset, _, test_dataset, n_node_pmf = get_dataset(
            data_path, dataset, device="cpu"
        )
        pred_graphs, _ = load_data(pred_file)
        for i, graph in enumerate(pred_graphs):
            edge_index = graph.edge_index
            if len(edge_index):
                reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
                temp = Data(edge_index=edge_index, num_nodes=graph.card)
                pred_graphs[i] = temp

        results_data = eval_fn(test_dataset=test_dataset, pred_graph_list=train_dataset)
        results_pred = eval_fn(test_dataset=test_dataset, pred_graph_list=pred_graphs)
        print(f"{dataset} data:", results_data)
        # print(f"{dataset} pred:", results_pred)
        deg = results_pred['degree_rbf']
        clus = results_pred['cluster_rbf']
        spec = results_pred['spectral_rbf']
        avg = (results_pred['degree_rbf'] + results_pred['cluster_rbf'] + results_pred['spectral_rbf'])/3
        ans = f'{deg:.3f} & {clus:.3f} & {spec:.3f} & {avg:.3f}'
        print(ans)
    else:
        pred_graphs, _ = load_data(pred_file)
        for i, graph in enumerate(pred_graphs):
            edge_index = graph.edge_index
            if len(edge_index):
                reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
                pred_graphs[i] = to_dense_adj(edge_index)
        mask_graphs, _ = load_data(mask_path)
        for i, graph in enumerate(mask_graphs):
            edge_index = graph.edge_index
            if len(edge_index):
                reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
                mask_graphs[i] = to_dense_adj(edge_index)
        masked_targets, _ = load_data(masked_target_path)
        for i,graph  in enumerate(masked_targets):
            edge_index = graph.edge_index
            if len(edge_index):
                reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
                masked_targets[i] = to_dense_adj(edge_index)
        assert len(pred_graphs) == len(mask_graphs) == len(masked_targets)

        loss = []
        for i in range(len(pred_graphs)):
            pred = pred_graphs[i].cpu().numpy()[0]
            mask = mask_graphs[i].cpu().numpy()[0]
            target = masked_targets[i].cpu().numpy()[0]
            n = target.shape[0]
            # print(pred.shape, mask.shape, target.shape)
            pred = pred[:n, :n]
            mask = mask[:n, :n]
            loss.append(np.sum(np.abs((pred * mask - target * mask))))
        loss = np.array(loss)
        mean = np.mean(loss)
        print(f"{dataset} mean loss:", mean)

if __name__ == "__main__":
    evaluate()
