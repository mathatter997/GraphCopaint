import os
import time 
import torch
import click
import networkx as nx
from accelerate import Accelerator
from diffusion.pgsn import PGSN
from torch_geometric.utils import to_dense_adj
from diffusers import DDIMScheduler,DDPMScheduler
from vpsde import ScoreSdeVpScheduler
from data.dataset import get_dataset
from data.data_loader import load_data
from diffusion.sample import sample, copaint
from data.utils import Lobster, prepare_json_dataset
from diffusion.ema import ExponentialMovingAverage
from configs.com_small import CommunitySmallConfig
from configs.ego_small import EgoSmallConfig
from configs.ego import EgoConfig
from configs.enzyme import EnzymeConfig

@click.command()
@click.option(
    "--config_type",
    default="community_small",
    type=click.Choice(
        ["community_small", "ego_small", "ego", "enzyme"], case_sensitive=False
    ),
)
@click.option("--checkpoint_path")
@click.option("--scheduler_path")
@click.option("--output_path")
@click.option("--masked_path", default=None)
@click.option("--cpu", default=False)
@click.option("--use_ema", default=True)
@click.option(
    "--sampler",
    default="ddpm",
    type=click.Choice(["ddpm", "ddim", "vpsde"], case_sensitive=False),
)
@click.option("--use_copaint", default = False)
@click.option("--num_samples", default=1000)
@click.option("--num_timesteps", default=1000)
@click.option("--unmask_size", default=8)
def inference(
    config_type,
    checkpoint_path,
    scheduler_path,
    output_path,
    masked_path,
    cpu,
    use_ema,
    sampler,
    use_copaint,
    num_samples,
    num_timesteps,
    unmask_size,
):
    if config_type == "community_small":
        config = CommunitySmallConfig()
    elif config_type == "ego_small":
        config = EgoSmallConfig()
    elif config_type == "ego":
        config = EgoConfig()
    elif config_type == "enzyme":
        config = EnzymeConfig()

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        cpu=cpu,
    )
    targets, _, _, n_node_pmf = get_dataset(
        config.data_filepath, config.data_name, device=accelerator.device
    )
    config.max_n_nodes = max_n_nodes = len(n_node_pmf)
    if use_copaint:
        assert sampler == 'ddim'
        assert masked_path is not None
        all_batches = []
        sizes = []
        for graph in targets:
            n = graph.num_nodes
            adj = torch.zeros(1, 1, max_n_nodes, max_n_nodes)
            adj[0, 0, :n, :n] = to_dense_adj(graph.edge_index)
            all_batches.append(adj)
            sizes.append(n)
        targets = torch.cat(all_batches, dim=0)
        targets = targets.to(device=accelerator.device)
        masks = torch.ones(targets.shape, device=accelerator.device)
        masks = torch.tril(masks, diagonal=-1)
        for k, size in enumerate(sizes):
            masks[k, :, size:] = 0
            unseen = torch.randperm(max_n_nodes)[:size - unmask_size]
            for node in unseen:
                masks[k, :, node, :] = 0
                masks[k, :, :, node] = 0
        masks = masks + masks.transpose(-1, -2)
        masked_targets = (targets * masks).cpu().numpy()[:num_samples]
        pred_adj_list = [nx.from_numpy_array(adj[0]) for adj in masked_targets]
        pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
        prepare_json_dataset(pred_adj_list, masked_path)
    else:
        sizes = torch.multinomial(
            torch.Tensor(n_node_pmf), num_samples, replacement=True
        )
    config.sampler = sampler 
    if sampler == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(
            scheduler_path,
            rescale_betas_zero_snr=False,
            timestep_spacing="trailing",
        )
    elif sampler == "ddpm":
        noise_scheduler = DDPMScheduler.from_pretrained(
            scheduler_path,
            rescale_betas_zero_snr=False,
            timestep_spacing="trailing",
        )
    elif sampler == "vpsde":
        noise_scheduler = ScoreSdeVpScheduler()
    
    model = PGSN(
        max_node=max_n_nodes,
        nf=config.nf,
        num_gnn_layers=config.num_gnn_layers,
        embedding_type=config.embedding_type,
        rw_depth=config.rw_depth,
        graph_layer=config.graph_layer,
        edge_th=config.edge_th,
        heads=config.heads,
        dropout=config.dropout,
        attn_clamp=config.attn_clamp,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(accelerator.device)
    )

    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        # ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = accelerator.prepare(model)
    model.eval()
    pred_adj_list = []
    tstart = time.time()
    for i in range(0, num_samples, config.eval_batch_size):
        batch_sz = min(config.eval_batch_size, num_samples - i + 1)
        if not use_copaint:
            edges = sample(
                config=config,
                model=model,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_timesteps,
                sizes=sizes[i:i+batch_sz],
                accelerator=accelerator,
            )
        else:
            edges = copaint(
                config=config,
                model=model,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_timesteps,
                sizes=sizes[i:i+batch_sz],
                accelerator=accelerator,
                time_travel=True,
                target_mask=masks[i:i+batch_sz],
                target_adj=targets[i:i+batch_sz],
            )
        print(i, batch_sz, edges.shape)
        edges = edges.reshape(batch_sz, max_n_nodes, max_n_nodes)
        for k, size in enumerate(sizes[i:i+batch_sz]):
            edges_k = edges[k, :size, :size]
            edges_k = (edges_k > 0).to(torch.int64)
            edges_k = edges_k.to(device='cpu')
            pred_adj_list.append(edges_k.numpy())
        tnow = time.time()
        print(i + batch_sz, f'{tnow-tstart:.2f} s')

    pred_adj_list = [nx.from_numpy_array(adj) for adj in pred_adj_list]
    pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
    prepare_json_dataset(pred_adj_list, output_path)


if __name__ == "__main__":
    inference()
