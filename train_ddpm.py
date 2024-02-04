import os
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusion.ddpm import train_loop
from data.data_loader import load_data
from torch_geometric.loader import DataLoader
from diffusion.lobster_dynamics import LobsterDynamics
from diffusers.optimization import get_cosine_schedule_with_warmup

@dataclass
class TrainingConfig:
    max_n_nodes = None
    data_filepath = 'data/dataset/lobsters.json'
    train_batch_size = 16
    eval_batch_size = 2  # how many to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "diffusion/models/"  # the model name locally and on the HF Hub
    output_dir_gnn = 'gnn/checkpoint_epoch_{}.pth'

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    # lobster dynamics
    in_node_nf = 4
    in_edge_nf = 1 
    hidden_nf=64
    act_fn=torch.nn.SiLU()
    n_layers=4
    attention=False
    normalization_factor=100
    aggregation_method='sum'

config = TrainingConfig()

lobster_list, max_n_nodes = load_data(config.data_filepath)
config.max_n_nodes = max_n_nodes

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)

dataloader = DataLoader(lobster_list, batch_size=config.train_batch_size, shuffle=True) # load lobsters
model = LobsterDynamics(in_node_nf=config.in_node_nf, 
                        in_edge_nf=config.in_edge_nf, 
                        hidden_nf=config.hidden_nf, 
                        device=accelerator.device,
                        act_fn=config.act_fn, 
                        n_layers=config.n_layers, 
                        attention=config.attention,
                        normalization_factor=config.normalization_factor, 
                        aggregation_method=config.aggregation_method)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, 
                                beta_schedule='squaredcos_cap_v2')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

# config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
train_loop(config=config,
           model=model,
           noise_scheduler=noise_scheduler,
           optimizer=optimizer,
           train_dataloader=dataloader,
           lr_scheduler=lr_scheduler,
           accelerator=accelerator)

