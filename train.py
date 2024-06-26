import os
import torch
import torch.nn.functional as F
import click

from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusion.ddpm import train_loop
from data.data_loader import load_data
from torch_geometric.loader import DataLoader
from data.dataset import get_dataset
from diffusion.pgsn import PGSN
from diffusion.eigen_nn import EigenNN
from diffusion.ema import ExponentialMovingAverage
from diffusers.optimization import (
    get_constant_schedule,
)
from configs.com_small import CommunitySmallConfig, CommunitySmallSmoothConfig
from configs.mnist_zeros import MnistZerosConfig
from configs.ego_small import EgoSmallConfig
from configs.ego import EgoConfig
from configs.enzyme import EnzymeConfig
from configs.grid import GridConfig
from diffusion.utils import dense_adj, init_eigen


@click.command()
@click.option("--checkpoint_path", default=None)
@click.option("--config_type",
              default='community_small',
              type=click.Choice(['community_small',
                                 'community_small_smooth',
                                 'mnist_zeros',
                                 'ego_small',
                                 'ego',
                                 'enzyme',
                                 'grid'], case_sensitive=False),
)
@click.option("--cpu", default=False)
def train_ddpm(
    checkpoint_path, cpu, config_type
):
    if config_type == 'community_small':
        config = CommunitySmallConfig()
    elif config_type == 'community_small_smooth':
        config = CommunitySmallSmoothConfig()
    elif config_type == 'mnist_zeros':
        config = MnistZerosConfig()
    elif config_type == 'ego_small':
        config = EgoSmallConfig()
    elif config_type == 'ego':
        config = EgoConfig()
    elif config_type == 'enzyme':
        config = EnzymeConfig()
    elif config_type == 'grid':
        config = GridConfig()
    config.checkpoint_path = checkpoint_path
    config.sampler = 'ddpm'

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        cpu=cpu,
    )
    split = 0.8
    if config.data_format == 'graph':
        train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
            config.data_filepath, config.data_name, device=accelerator.device, split=split
        )
        config.max_n_nodes = max_n_nodes = len(n_node_pmf) - 1
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True
        ) 
    elif config.data_format == 'eigen':
        train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
            config.data_filepath, config.data_name, device=accelerator.device, split=split
        )
        config.max_n_nodes = max_n_nodes = len(n_node_pmf) - 1
        def scale_data(x):
            return 1 * x
        train_dataloader = []
        for i in range(len(train_dataset)):
            adj, _ = dense_adj(train_dataset[i], config.max_n_nodes, scale_data)
            adj = adj.squeeze(0, 1)
            adj, x, la, u, flag = init_eigen(adj, config.max_feat_num, config.max_n_nodes, train_dataset[i].num_nodes)
            train_dataloader.append((adj, x, la, u, flag))
        train_dataloader = DataLoader(train_dataloader, batch_size=config.train_batch_size, shuffle=True)
    elif config.data_format == 'pixel':
        dataset = torch.load(f'{config.data_filepath}raw/{config.data_name}.pth')
        num_train = int(len(dataset) * split)
        train_dataset = dataset[:num_train]
        config.max_n_nodes = max_n_nodes = train_dataset[0][0].shape[-1]
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True
        )
    if config.model == "pgsn":
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
            attn_clamp=config.attn_clamp
        )
    elif config.model == 'eigen':
        model = EigenNN(
            max_feat_num=config.max_feat_num,
            nhid=config.nhid,
            max_node_num=config.max_node_num,
            num_layers=config.num_layers,
            num_linears=config.num_linears,
            c_init=config.c_init,
            c_hid=config.c_hid,
            c_final=config.c_final,
            adim=config.adim,
            depth=config.depth,
            num_heads=config.num_heads,
            conv=config.conv,)
    elif config.model == 'unet':
        model = UNet2DModel(
            sample_size=(max_n_nodes, max_n_nodes),
            in_channels=1,
            out_channels=1,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_constant_schedule(
        optimizer=optimizer,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_rate)
    if checkpoint_path:
        # checkpoint = torch.load(config.output_dir + config.load_model_dir)
        checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        config.start_epoch = checkpoint["epoch"] + 1

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.train_timesteps,
        # beta_schedule="squaredcos_cap_v2",
        beta_schedule="linear",
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )

    # config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        ema=ema,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        label=config.label,
    )


if __name__ == "__main__":
    train_ddpm()
