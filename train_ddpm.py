import os
import torch
import torch.nn.functional as F
import click

from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusion.ddpm import train_loop
from data.data_loader import load_data
from torch_geometric.loader import DataLoader
from data.dataset import get_dataset
from diffusion.pgsn import PGSN
from diffusion.ema import ExponentialMovingAverage
from diffusers.optimization import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
)


@click.command()
@click.option("--checkpoint_path", default=None)
@click.option(
    "--mixed_precision",
    default="no",
    type=click.Choice(["yes", "no"], case_sensitive=False),
)
@click.option("--cpu", default=False)
@click.option("--data_filepath")
@click.option("--data_name")
@click.option("--train_timesteps", default=1000)
def train_ddpm(
    checkpoint_path, mixed_precision, cpu, data_filepath, data_name, train_timesteps
):
    @dataclass
    class TrainingConfig:
        max_n_nodes = None
        data_filepath = "data/dataset/"
        data_name = "Community_small"
        train_batch_size = 32
        eval_batch_size = 2  # how many to sample during evaluation
        num_epochs = 400000
        start_epoch = 0
        gradient_accumulation_steps = 1
        learning_rate = 2e-5
        # lr_warmup_steps = 500
        # save_image_epochs = 10000
        save_model_epochs = 10000
        train_timesteps = 1000
        mixed_precision = "no"
        start = 0
        checkpoint_path = None

        output_dir = "diffusion/models/"  # the model name locally and on the HF Hub
        output_dir_gnn = "gnn/checkpoint_epoch_{}.pth"
        label = f"_t{train_timesteps}_psgn"

        push_to_hub = False  # whether to upload the saved model to the HF Hub
        hub_private_repo = False
        overwrite_output_dir = (
            True  # overwrite the old model when re-running the notebook
        )
        seed = 0

        # lobster dynamics
        # in_node_nf = 4
        # in_edge_nf = 1
        # hidden_nf=64
        # act_fn=torch.nn.SiLU()
        # n_layers=4
        # attention=False
        # normalization_factor=100
        # aggregation_method='sum'

        ema_rate = 0.9999
        normalization = "GroupNorm"
        nonlinearity = "swish"
        nf = 256
        num_gnn_layers = 4
        size_cond = False
        embedding_type = "positional"
        rw_depth = 16
        graph_layer = "PosTransLayer"
        edge_th = -1
        heads = 8
        dropout=0.1
        attn_clamp = False

    config = TrainingConfig()
    config.mixed_precision = mixed_precision
    config.data_filepath = data_filepath
    config.data_name = data_name
    config.checkpoint_path = checkpoint_path
    config.output_dir = f"models/{data_name}/"
    config.train_timesteps = train_timesteps
    config.label = f"_t{train_timesteps}_psgn"

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        cpu=cpu,
    )

    # lobster_list, max_n_nodes = load_data(config.data_filepath)
    train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
        config.data_filepath, config.data_name, device=accelerator.device
    )
    config.max_n_nodes = max_n_nodes = len(n_node_pmf)

    dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )  # load lobsters
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_rate)
    if checkpoint_path:
        # checkpoint = torch.load(config.output_dir + config.load_model_dir)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        config.start_epoch = checkpoint["epoch"] + 1

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.train_timesteps,
        beta_schedule="squaredcos_cap_v2",
    )

    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=(len(dataloader) * config.num_epochs),
    # )

    lr_scheduler = get_constant_schedule(
        optimizer=optimizer,
    )

    # config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        ema=ema,
        train_dataloader=dataloader,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        label=config.label,
    )


if __name__ == "__main__":
    train_ddpm()
