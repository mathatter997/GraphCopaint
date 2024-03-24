import torch
import networkx as nx 
from diffusion.pgsn import PGSN
from dataclasses import dataclass
from diffusers import DDIMScheduler
from data.dataset import get_dataset
from data.data_loader import load_data
from diffusion.ddim_sample import sample
from data.utils import Lobster, prepare_json_dataset

@dataclass
class InferenceConfig:
    max_n_nodes = None
    data_filepath = 'data/dataset/'
    data_name = 'Community_small'
    # scheduler_filepath = "diffusion/models/scheduler_config.json"  # the model name locally and on the HF Hub
    # checkpoint_filepath = 'diffusion/models/gnn/checkpoint_epoch_25000psgn_no_tanh.pth'
    scheduler_filepath = 'models/Community_small/scheduler_config.json'
    checkpoint_filepath = 'models/Community_small/gnn/checkpoint_epoch_60000psgn.pth'
    output_filepath = 'data/dataset/output_60000_pgsn.json'
    eta = 0 # DDPM

    device = 'cpu'

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
# lobster_list, max_n_nodes = load_data(config.data_filepath)
train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(config.data_filepath, 
                                                                    config.data_name,
                                                                    device=config.device)
config.max_n_nodes = max_n_nodes = len(n_node_pmf)

# https://huggingface.co/papers/2305.08891 
noise_scheduler = DDIMScheduler.from_pretrained(config.scheduler_filepath, 
                                                rescale_betas_zero_snr=False, # keep false
                                                timestep_spacing="trailing")

model = PGSN(max_node=max_n_nodes)
checkpoint = torch.load(config.checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# weights = torch.zeros(max_n_nodes + 1, dtype=torch.float)
# for lobster in lobster_list:
#     weights[lobster.card] += 1

s = 25 # num inference steps
N = 250

sample_node_num = torch.multinomial(torch.Tensor(n_node_pmf), N, replacement=True)

pred_adj_list = []
for i in range(N):
    n = sample_node_num[i]
    edges = sample(config=config, 
                            model=model, 
                            noise_scheduler=noise_scheduler, 
                            num_inference_steps=s,
                            n=n)

    edges = edges.reshape(max_n_nodes, max_n_nodes)
    edges = edges[:n, :n]
    edges = (edges > 0).to(torch.int64)
    edges = edges + edges.T
    pred_adj_list.append(edges.numpy())
    # print(edges)
    # if i % 50 == 0:
    #     print(i)

pred_adj_list = [nx.from_numpy_array(adj) for adj in pred_adj_list]
pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
prepare_json_dataset(pred_adj_list, config.output_filepath)






