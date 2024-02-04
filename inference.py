import os
import torch 
from dataclasses import dataclass
from diffusers import DDIMScheduler
from data.data_loader import load_data
from diffusion.lobster_dynamics import LobsterDynamics
from diffusion.ddim_sample import sample

@dataclass
class InferenceConfig:
    max_n_nodes = None
    data_filepath = 'data/dataset/lobsters.json'
    scheduler_filepath = "diffusion/models/scheduler/scheduler_config.json"  # the model name locally and on the HF Hub
    checkpoint_filepath = 'diffusion/models/gnn/checkpoint_epoch_50.pth'
    eta = 1 # DDPM

    device = 'mps'

    # lobster dynamics
    in_node_nf = 4
    in_edge_nf = 1 
    hidden_nf=64
    act_fn=torch.nn.SiLU()
    n_layers=4
    attention=False
    normalization_factor=100
    aggregation_method='sum'

config = InferenceConfig()
_, max_n_nodes = load_data(config.data_filepath)
config.max_n_nodes = max_n_nodes

# https://huggingface.co/papers/2305.08891 
noise_scheduler = DDIMScheduler.from_pretrained(config.scheduler_filepath, 
                                                rescale_betas_zero_snr=True, 
                                                timestep_spacing="trailing")

model = LobsterDynamics(in_node_nf=config.in_node_nf, 
                        in_edge_nf=config.in_edge_nf, 
                        hidden_nf=config.hidden_nf, 
                        device=config.device,
                        act_fn=config.act_fn, 
                        n_layers=config.n_layers, 
                        attention=config.attention,
                        normalization_factor=config.normalization_factor, 
                        aggregation_method=config.aggregation_method)

model = torch.load(config.checkpoint_filepath)
model.eval()

n = 10
s = 1000 # num inference steps

sample(config=config, 
       model=model, 
       noise_scheduler=noise_scheduler, 
       num_inference_steps=s,
       n=n)





