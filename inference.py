import os
import time 
import torch
import click
import numpy as np
import networkx as nx
from accelerate import Accelerator
from diffusion.pgsn import PGSN
from diffusion.eigen_nn import EigenNN
from diffusion.utils import dense_adj, init_eigen
from torch_geometric.utils import to_dense_adj
from diffusers import DDIMScheduler,DDPMScheduler,UNet2DModel
from vpsde import ScoreSdeVpScheduler
from data.dataset import get_dataset
from diffusion.sample import sample, copaint, repaint
from data.utils import Lobster, prepare_json_dataset
from diffusion.ema import ExponentialMovingAverage
from configs.com_small import CommunitySmallConfig, CommunitySmallSmoothConfig
from configs.mnist_zeros import MnistZerosConfig
from configs.ego_small import EgoSmallConfig
from configs.ego import EgoConfig
from configs.enzyme import EnzymeConfig
import random

@click.command()
@click.option(
    "--config_type",
    default="community_small",
    type=click.Choice(
        ["community_small",
         "community_small_smooth",
         "mnist_zeros",
         "ego_small", 
         "ego", 
         "enzyme"], case_sensitive=False
    ),
)
@click.option("--checkpoint_path")
@click.option("--scheduler_path")
@click.option("--output_path")
@click.option("--mask_path", default=None)
@click.option("--masked_output_path", default=None)
@click.option("--cpu", default=False)
@click.option("--use_ema", default=True)
@click.option(
    "--sampler",
    default="ddpm",
    type=click.Choice(["ddpm", "ddim", "vpsde"], case_sensitive=False),
)
@click.option("--log_x0_predictions", default=False)
@click.option("--num_samples", default=1000)
@click.option("--num_timesteps", default=1000)
@click.option("--inpainter",  default="none",
    type=click.Choice(["none", "copaint", "repaint"], case_sensitive=False))
@click.option("--unmask_size", default=8)
@click.option("--num_intervals", default=1)
@click.option("--optimization_steps", default=2)
@click.option("--time_travel", default=True)
@click.option("--repeat_tt", default=1)
@click.option("--loss_mode", default='inpaint',
              type=click.Choice(["inpaint", "naive_inpaint", "none"], case_sensitive=False))
@click.option("--reg_mode", default='square',
              type=click.Choice(["square", "naive_square", "none"], case_sensitive=False))
@click.option("--tau", default=5)
@click.option("--lr_xt", default=0.0025)
@click.option("--lr_xt_decay", default=1.05)
@click.option("--coef_xt_reg", default=0.01)
@click.option("--coef_xt_reg_decay", default=1.0)
@click.option("--use_adaptive_lr_xt", default=True)
def inference(
    config_type,
    checkpoint_path,
    scheduler_path,
    output_path,
    mask_path,
    masked_output_path,
    cpu,
    use_ema,
    sampler,
    inpainter,
    num_samples,
    num_timesteps,
    unmask_size,
    num_intervals,
    optimization_steps,
    time_travel,
    repeat_tt,
    tau,
    loss_mode,
    reg_mode,
    log_x0_predictions,
    lr_xt,
    lr_xt_decay,
    coef_xt_reg,
    coef_xt_reg_decay,
    use_adaptive_lr_xt,
):
    if config_type == "community_small":
        config = CommunitySmallConfig()
    elif config_type == "community_small_smooth":
        config = CommunitySmallSmoothConfig()
    elif config_type == 'mnist_zeros':
        config = MnistZerosConfig()
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
    split = 0.8
    if config.data_format == 'graph':
        targets, _, _, n_node_pmf = get_dataset(
            config.data_filepath, config.data_name, device=accelerator.device, split=split
        )
        config.max_n_nodes = max_n_nodes = len(n_node_pmf)
    elif config.data_format == 'eigen':
        train_dataset, eval_dataset, test_dataset, n_node_pmf = get_dataset(
            config.data_filepath, config.data_name, device=accelerator.device, split=split
        )
        config.max_n_nodes = max_n_nodes = len(n_node_pmf) - 1
        def scale_data(x):
            return 2 * x - 1
        targets = []
        u_mats = {i : [] for i in range(max_n_nodes + 1)}
        for i in range(len(train_dataset)):
            adj, _ = dense_adj(train_dataset[i], config.max_n_nodes, scale_data)
            adj = adj.squeeze(0, 1)
            adj, x, la, u, flag = init_eigen(adj, config.max_feat_num, config.max_n_nodes, train_dataset[i].num_nodes)
            targets.append(adj)
            u_mats[train_dataset[i].num_nodes].append(u)
        
    elif config.data_format == 'pixel':
        dataset = torch.load(f'{config.data_filepath}raw/{config.data_name}.pth')
        num_train = int(len(dataset) * split)
        targets = dataset[:num_train]
        n_node_pmf = np.zeros(25)
        for i in range(len(targets)):
            _, mask = targets[i]
            mask = mask.reshape(24, 24)
            n = int(torch.where(mask[0] == 1)[0][-1].item()) + 1
            n_node_pmf[n] += 1
        n_node_pmf /= np.sum(n_node_pmf)
        config.max_n_nodes = max_n_nodes = 24

    if inpainter != 'none':
        assert mask_path is not None and masked_output_path is not None
        all_batches = []
        sizes = []
        if config.data_format == 'graph':
            while True:
                for graph in targets:
                    n = graph.num_nodes
                    adj = torch.zeros(1, 1, max_n_nodes, max_n_nodes)
                    adj[0, 0, :n, :n] = to_dense_adj(graph.edge_index)
                    all_batches.append(adj)
                    sizes.append(n)
                    if len(sizes) == num_samples:
                        break
                if len(sizes) == num_samples:
                    break
            targets = torch.cat(all_batches, dim=0)
            targets = targets.to(device=accelerator.device)
            targets = targets * 2 - 1
            masks = torch.ones(targets.shape, device=accelerator.device)
            masks = torch.tril(masks, diagonal=-1)
            for k, size in enumerate(sizes):
                masks[k, :, size:] = 0
                unseen = torch.randperm(size)[:size - unmask_size]
                # unseen = torch.arange(unmask_size,size)
                for node in unseen:
                    masks[k, :, node, :] = 0
                    masks[k, :, :, node] = 0
            masks = masks + masks.transpose(-1, -2)
            masked_targets = (targets * masks).cpu().numpy()[:num_samples]
            masked_targets = masked_targets > 0
            pred_adj_list = [nx.from_numpy_array(adj[0]) for adj in masked_targets]
            pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
            prepare_json_dataset(pred_adj_list, masked_output_path)
            pred_adj_list = [nx.from_numpy_array(adj[0]) for adj in masks.cpu().numpy()[:num_samples]]
            pred_adj_list = [Lobster(adj) for adj in pred_adj_list]
            prepare_json_dataset(pred_adj_list, mask_path)
        elif config.data_format == 'pixel':
            masks = []
            for i in range(len(targets)):
                target, mask = targets[i]
                mask = mask.reshape(max_n_nodes, max_n_nodes)
                size = int(torch.where(mask[0] == 1)[0][-1].item()) + 1
                unseen = torch.randperm(size)[:size - unmask_size]
                for node in unseen:
                    mask[node, :] = 0
                    mask[:, node] = 0
                masks.append(mask.reshape(1, 1, max_n_nodes, max_n_nodes))
                sizes.append(size)
                targets[i] = target.reshape(1, 1, max_n_nodes, max_n_nodes)
            masks = torch.vstack(masks).to(device=accelerator.device)
            targets = torch.vstack(targets).to(device=accelerator.device)
                        
    else:
        u_list = None
        sizes = torch.multinomial(
            torch.Tensor(n_node_pmf), num_samples, replacement=True
        )
        if config.data_format == 'eigen':
            u_list = []
            for size in sizes:
                u_list.append(random.choice(u_mats[size.item()]).unsqueeze(0))
            u_list = torch.vstack(u_list)


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
    if config.data_format == 'graph':
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
    elif config.data_format == 'eigen':
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
    elif config.data_format == 'pixel':
        model = UNet2DModel(
            sample_size=(max_n_nodes, max_n_nodes),
            in_channels=1,
            out_channels=1,
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
        # model.layers[0].load_state_dict(checkpoint["x_state_dict"])
        # model.layers[1].load_state_dict(checkpoint["adj_state_dict"])

    model = accelerator.prepare(model)
    model.eval()
    pred_adj_list = []
    tstart = time.time()
    for i in range(0, num_samples, config.eval_batch_size):
        batch_sz = min(config.eval_batch_size, num_samples - i)
        if u_list is None:
            u_batch = None
        else:
            u_batch = u_list[i:i+batch_sz]
        if inpainter == 'none':
            edges = sample(
                config=config,
                model=model,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_timesteps,
                sizes=sizes[i:i+batch_sz],
                u=u_batch,
                accelerator=accelerator,
                log_x0_predictions=log_x0_predictions,
            )
        elif inpainter == 'copaint':
            edges = copaint(
                config=config,
                model=model,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_timesteps,
                sizes=sizes[i:i+batch_sz],
                accelerator=accelerator,
                target_mask=masks[i:i+batch_sz],
                target_adj=targets[i:i+batch_sz],
                interval_num=num_intervals,
                num_iteration_optimize_xt=optimization_steps,
                repeat_tt=repeat_tt,
                time_travel=time_travel,
                tau=tau,
                log_x0_predictions=log_x0_predictions,
                loss_mode=loss_mode,
                reg_mode=reg_mode,
                lr_xt=lr_xt,
                lr_xt_decay=lr_xt_decay,
                coef_xt_reg=coef_xt_reg,
                coef_xt_reg_decay=coef_xt_reg_decay,
                use_adaptive_lr_xt=use_adaptive_lr_xt,
            )
        elif inpainter == 'repaint':
            edges = repaint(
                config=config,
                model=model,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_timesteps,
                sizes=sizes[i:i+batch_sz],
                accelerator=accelerator,
                target_mask=masks[i:i+batch_sz],
                target_adj=targets[i:i+batch_sz],
                repeat_tt=repeat_tt,
                time_travel=time_travel,
                tau=tau,
                log_x0_predictions=log_x0_predictions,)
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
