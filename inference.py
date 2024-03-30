import torch
import networkx as nx
from diffusion.pgsn import PGSN
from dataclasses import dataclass
from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler
from data.dataset import get_dataset
from data.data_loader import load_data
from diffusion.ddim_sample import sample
from data.utils import Lobster, prepare_json_dataset
from diffusion.ema import ExponentialMovingAverage


@dataclass
class InferenceConfig:
    max_n_nodes = None
    data_filepath = "data/dataset/"
    data_name = "Community_small"
    # scheduler_filepath = "diffusion/models/scheduler_config.json"  # the model name locally and on the HF Hub
    # checkpoint_filepath = 'diffusion/models/gnn/checkpoint_epoch_25000psgn_no_tanh.pth'
    scheduler_filepath = "models/Community_small/scheduler_config.json"
    checkpoint_filepath = (
        "models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn_v2.pth"
    )
    output_filepath = "data/dataset/output_400000_t1000_ema_pgsn_v2.json"
    eta = 0  # DDIM
    # eta = 1
    ema = True

    device = "cpu"

    ema_rate = 0.9999
    normalization = "GroupNorm"
    nonlinearity = "swish"
    nf = 128
    nf= 256
    # nf = 384
    num_gnn_layers = 4
    size_cond = False
    embedding_type = "positional"
    rw_depth = 16
    graph_layer = "PosTransLayer"
    edge_th = -1
    heads = 8
    # heads = 12
    dropout = 0.1
    attn_clamp = False


config = InferenceConfig()
# lobster_list, max_n_nodes = load_data(config.data_filepath)
train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
    config.data_filepath, config.data_name, device=config.device
)
config.max_n_nodes = max_n_nodes = len(n_node_pmf)

# https://huggingface.co/papers/2305.08891
noise_scheduler = DDIMScheduler.from_pretrained(
    config.scheduler_filepath,
    rescale_betas_zero_snr=False,  # keep false
    timestep_spacing="trailing",
)

noise_scheduler = DDPMScheduler.from_pretrained(
    config.scheduler_filepath,
    rescale_betas_zero_snr=False,  # keep false
    timestep_spacing="trailing",
)

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

checkpoint = torch.load(config.checkpoint_filepath, map_location=torch.device("cpu"))
if config.ema:
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ema.load_state_dict(checkpoint["ema_state_dict"])
    ema.copy_to(model.parameters())
else:
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# weights = torch.zeros(max_n_nodes + 1, dtype=torch.float)
# for lobster in lobster_list:
#     weights[lobster.card] += 1

s = 1000  # num inference steps
N = 250

sample_node_num = torch.multinomial(torch.Tensor(n_node_pmf), N, replacement=True)

pred_adj_list = []
for i in range(N):
    n = sample_node_num[i]
    edges = sample(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        num_inference_steps=s,
        n=n,
    )

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
