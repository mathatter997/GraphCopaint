import os
import torch
import click
import networkx as nx
from accelerate import Accelerator
from diffusion.pgsn import PGSN
from diffusers import DDIMScheduler, DDPMScheduler
from vpsde import ScoreSdeVpScheduler
from data.dataset import get_dataset
from data.data_loader import load_data
from diffusion.ddim_sample import sample
from data.utils import Lobster, prepare_json_dataset
from diffusion.ema import ExponentialMovingAverage
from configs.com_small import CommunitySmallConfig
from configs.ego_small import EgoSmallConfig
from configs.ego import EgoConfig
from configs.enzyme import EnzymeConfig
import time 


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
@click.option("--cpu", default=False)
@click.option("--use_ema", default=True)
@click.option(
    "--sampler",
    default="ddpm",
    type=click.Choice(["ddpm", "ddim", "vpsde"], case_sensitive=False),
)
@click.option("--num_samples", default=1000)
@click.option("--num_timesteps", default=1000)
def inference(
    config_type,
    checkpoint_path,
    scheduler_path,
    output_path,
    cpu,
    use_ema,
    sampler,
    num_samples,
    num_timesteps,
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
    train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
        config.data_filepath, config.data_name, device=accelerator.device
    )
    config.max_n_nodes = max_n_nodes = len(n_node_pmf)
    config.sampler = sampler 
    # https://huggingface.co/papers/2305.08891
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
    sample_node_num = torch.multinomial(
        torch.Tensor(n_node_pmf), num_samples, replacement=True
    )
    model = accelerator.prepare(model)
    pred_adj_list = []
    tstart = time.time()
    for i in range(0, num_samples, config.eval_batch_size):
        sizes = sample_node_num[i:min(i + config.eval_batch_size, num_samples)]
        edges = sample(
            config=config,
            model=model,
            noise_scheduler=noise_scheduler,
            num_inference_steps=num_timesteps,
            sizes=sizes,
            accelerator=accelerator,
        )
        edges = edges.reshape(len(sizes), max_n_nodes, max_n_nodes)
        for k, size in enumerate(sizes):
            edges_k = edges[k, :size, :size]
            edges_k = (edges_k > 0).to(torch.int64)
            # edges = edges + edges.T
            edges_k = edges_k.to(device='cpu')
            pred_adj_list.append(edges_k.numpy())
            tnow = time.time()
            print(i + k, f'{tnow-tstart:.2f} s')

    pred_adj_list = [nx.from_numpy_array(adj) for adj in pred_adj_list]
    pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
    prepare_json_dataset(pred_adj_list, output_path)


if __name__ == "__main__":
    inference()
