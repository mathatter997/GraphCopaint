import os
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from .sample_utils import get_loss_fn, pred_x0, get_reg_fn
import wandb


def sample(
    config,
    model,
    noise_scheduler,
    accelerator,
    num_inference_steps=1000,
    inference_timesteps=None,
    sizes=None,
    log_x0_predictions=False,
):
    if inference_timesteps is None:
        noise_scheduler.set_timesteps(num_inference_steps)
    else:
        noise_scheduler.num_inference_steps = len(inference_timesteps)
        noise_scheduler.timesteps = torch.from_numpy(inference_timesteps).to(
            accelerator.device
        )

    batch_size = len(sizes)
    sqrt_2 = 2**0.5
    adj_shape = (batch_size, 1, config.max_n_nodes, config.max_n_nodes)
    adj_mask = torch.ones(adj_shape, device=accelerator.device)
    adj_mask = torch.tril(adj_mask, diagonal=-1)
    for k, size in enumerate(sizes):
        adj_mask[k, :, size:] = 0
    adj_mask = adj_mask + adj_mask.transpose(-1, -2)

    adj_t = torch.randn(adj_shape, device=accelerator.device)
    adj_t = torch.tril(adj_t, -1)
    adj_t = (adj_t + adj_t.transpose(-1, -2)) / sqrt_2
    adj_t = adj_t * adj_mask
    num_timesteps = len(noise_scheduler.timesteps)
    adj_0s = []
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            # time = t.to(device=accelerator.device).reshape(-1)
            time = torch.full((batch_size,), t.item(), device=accelerator.device)
            if config.sampler == "vpsde":
                edge_noise_res = model(adj_t, time * num_timesteps, mask=adj_mask)
            else:
                edge_noise_res = model(adj_t, time, mask=adj_mask)
        if config.sampler == "vpsde":
            adj_next, _ = noise_scheduler.step_pred(score=edge_noise_res, t=t, x=adj_t)
        else:
            adj_next = noise_scheduler.step(edge_noise_res, t, adj_t).prev_sample
        res = adj_next - adj_t
        res = (res + res.transpose(-1, -2)) / sqrt_2
        res = res * adj_mask
        adj_t = adj_t + res
        if log_x0_predictions:
            adj_0 = pred_x0(
                et=edge_noise_res,
                xt=adj_t,
                t=t,
                mask=adj_mask,
                scheduler=noise_scheduler,
                interval_num=1,
            )
            adj_0s.append(adj_0)
        # adj_t = torch.clip(adj_t, -3, 3)
        # adj_t = torch.clip(adj_t, -3, 3)
        # if t.item() in {999, 749, 499, 249, 159, 49, 29, 19, 9}:
        #     g = (input_e + 1) / 2  * adj_mask
        #     g = (g[:, :, :n, :n]).view(n,n)
        #     g = g + g.T
        #     graph = nx.from_numpy_array(g.numpy(), edge_attr='weight')
        #     print(graph)
        #     pos = nx.circular_layout(graph)
        #     edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
        #     nx.draw(graph, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)
        #     plt.savefig('edges/edges_{}.png'.format(t))
    if log_x0_predictions:
        sz = torch.numel(adj_t)
        for i in range(len(adj_0s)):
            adj_0s[i] = torch.sum((adj_0s[i] - adj_t) ** 2).cpu().numpy() / sz
        adj_0s = np.array(adj_0s)
        time = noise_scheduler.timesteps.cpu().numpy()
        plt.plot(time, adj_0s)
        plt.savefig(
            fname=f"data/dataset/{config.data_name}_maskloss_{config.sampler}.pdf"
        )
    return adj_t


def copaint(
    config,
    model,
    noise_scheduler,
    num_inference_steps,
    accelerator,
    interval_num=1,
    sizes=None,
    # time_travel params
    time_travel=True,
    repeat_tt=1,
    tau=5,
    loss_mode="inpaint",
    reg_mode="square",
    lr_xt=0.0025,
    lr_xt_decay=1.05,
    coef_xt_reg=0.01,
    coef_xt_reg_decay=1.0,
    optimize_before_time_travel=False,
    use_adaptive_lr_xt=True,
    num_iteration_optimize_xt=2,
    target_mask=None,
    target_adj=None,
    log_x0_predictions=False,
):
    model.eval()
    noise_scheduler.set_timesteps(num_inference_steps)

    batch_size = len(sizes)
    loss_norm = batch_size
    sqrt_2 = 2**0.5
    init_lr = lr_xt

    adj_shape = (batch_size, 1, config.max_n_nodes, config.max_n_nodes)
    adj_mask = torch.ones(adj_shape, device=accelerator.device)
    adj_mask = torch.tril(adj_mask, diagonal=-1)
    for k, size in enumerate(sizes):
        adj_mask[k, :, size:] = 0
    adj_mask = adj_mask + adj_mask.transpose(-1, -2)
    adj_t = torch.randn(adj_shape, device=accelerator.device)
    adj_t = torch.tril(adj_t, -1)
    adj_t = (adj_t + adj_t.transpose(-1, -2)) / sqrt_2
    adj_t = adj_t * adj_mask
    if time_travel is False:
        k = 1

    loss_fn = get_loss_fn(loss_mode)
    reg_fn = get_reg_fn(reg_mode)

    T = noise_scheduler.timesteps[0].item()
    time_pairs = list(
        zip(noise_scheduler.timesteps[::tau], noise_scheduler.timesteps[tau::tau])
    )
    end = (time_pairs[-1][-1], torch.tensor(-1, device=accelerator.device))
    time_pairs.append(end)
    
    if log_x0_predictions:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Copaint Ablation",
            # track hyperparameters and run metadata
            config={
                "inpainter": 'copaint',
                "batch_size": batch_size,
                "dataset": config.data_name,
                "time_travel": time_travel,
                "interval_num": interval_num,
                "tau": tau,
                "repeat_tt": repeat_tt,
                "loss_mode": loss_mode,
                "reg_mode": reg_mode,
                "lr_xt": init_lr,
                "lr_xt_decay": lr_xt_decay,
                "coef_xt_reg": coef_xt_reg,
                "coef_xt_reg_decay": coef_xt_reg_decay,
            },
        )
    for prev_t, cur_t in time_pairs:
        for repeat_step in range(repeat_tt):
            # optimize x_t given x_0
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    model.eval()
                    t = torch.tensor(t_)
                    adj_t = adj_t * adj_mask
                    origin_adj = adj_t.clone().detach()
                    adj_t = adj_t.detach().requires_grad_()
                    time = torch.full(
                        (batch_size,), t.item(), device=accelerator.device
                    )
                    for step in range(num_iteration_optimize_xt):
                        adj_noise_res = model(adj_t, time, mask=adj_mask)
                        adj_0 = pred_x0(
                            et=adj_noise_res,
                            xt=adj_t,
                            t=t,
                            mask=adj_mask,
                            scheduler=noise_scheduler,
                            interval_num=interval_num,
                        )
                        # adj_0_ = adj_0
                        loss = loss_fn(
                            target_adj, adj_0, target_mask
                        ) + coef_xt_reg * reg_fn(origin_adj, adj_t)
                        loss = loss / loss_norm
                        adj_t_grad = torch.autograd.grad(
                            loss, adj_t, retain_graph=False, create_graph=False
                        )[0].detach()
                        adj_t_grad = (adj_t_grad + adj_t_grad.transpose(-1, -2)) / 2
                        adj_t_grad = adj_t_grad * adj_mask
                        new_adj_t = adj_t - lr_xt * adj_t_grad * loss_norm
                        # if new_x doesn't improve loss
                        # we start from x and try again with a smaller grad step
                        lr_xt_temp = lr_xt
                        while use_adaptive_lr_xt:
                            with torch.no_grad():
                                adj_noise_res = model(
                                    new_adj_t, time, mask=adj_mask
                                )
                                adj_0 = pred_x0(
                                    et=adj_noise_res,
                                    xt=new_adj_t,
                                    t=t,
                                    mask=adj_mask,
                                    scheduler=noise_scheduler,
                                    interval_num=interval_num,
                                )
                                new_loss = loss_fn(
                                    target_adj, adj_0, target_mask
                                ) + coef_xt_reg * reg_fn(origin_adj, new_adj_t)
                                new_loss = new_loss / loss_norm
                                if not torch.isnan(new_loss) and new_loss <= loss:
                                    # print(f'{torch.norm(adj_0_ - adj_0).item():.4f}', f'{(loss - new_loss).item():.4f}', f'{loss.item():.4f}', lr_xt_temp, lr_xt,t_)
                                    break
                                else:
                                    lr_xt_temp *= 0.8
                                    del new_adj_t, adj_0, new_loss
                                    new_adj_t = adj_t - lr_xt_temp * adj_t_grad * loss_norm
                        # optimized x, pred_x0, and e_t
                        adj_t = new_adj_t.detach().requires_grad_()
                        del loss, adj_t_grad, adj_0, adj_noise_res
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                    if log_x0_predictions and repeat_step == repeat_tt - 1:
                        adj_noise_res = model(
                                    adj_t, time, mask=adj_mask
                                )
                        adj_0 = pred_x0(
                                    et=adj_noise_res,
                                    xt=adj_t,
                                    t=t,
                                    mask=adj_mask,
                                    scheduler=noise_scheduler,
                                    interval_num=interval_num,
                                )
                        sz = torch.numel(adj_t)
                        target_loss = (
                            torch.sum((adj_0 * target_mask - target_adj * target_mask) ** 2)
                            .cpu()
                            .detach()
                            .numpy()
                            / sz
                        )
                        wandb.log({"time_step": t_, "target_loss": target_loss.item()})
                        del adj_noise_res, adj_0
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                    adj_noise_res = model(adj_t, time, mask=adj_mask)
                    adj_tm1 = noise_scheduler.step(adj_noise_res, t, adj_t).prev_sample
                    res = adj_tm1 - adj_t
                    res = (res + res.transpose(-1, -2)) / sqrt_2
                    res = res * adj_mask
                    adj_t = adj_t + res

                # time-travel (forward diffusion)
                if time_travel and (cur_t + 1) <= T - tau:
                    if optimize_before_time_travel:
                        pass
                    noise = torch.randn(adj_t.shape, device=accelerator.device)
                    noise = noise + noise.transpose(-1, -2)
                    noise = noise * adj_mask / sqrt_2
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                        dtype=adj_t.dtype, device=accelerator.device
                    )
                    alpha_prod = alphas_cumprod[prev_t] / alphas_cumprod[cur_t + 1]
                    sqrt_alpha_prod = alpha_prod**0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                            -1
                        )

                    adj_prev = (
                        sqrt_alpha_prod * adj_t + sqrt_one_minus_alpha_prod * noise
                    )
                    adj_t = adj_prev * adj_mask

        lr_xt *= lr_xt_decay
        coef_xt_reg *= coef_xt_reg_decay

    adj_t = adj_t * adj_mask
    return adj_t


def repaint(
    config,
    model,
    noise_scheduler,
    num_inference_steps,
    accelerator,
    sizes=None,
    # time_travel params
    time_travel=True,
    repeat_tt=1,
    tau=5,
    target_mask=None,
    target_adj=None,
    log_x0_predictions=False,
):
    model.eval()
    noise_scheduler.set_timesteps(num_inference_steps)

    batch_size = len(sizes)
    sqrt_2 = 2**0.5

    adj_shape = (batch_size, 1, config.max_n_nodes, config.max_n_nodes)
    adj_mask = torch.ones(adj_shape, device=accelerator.device)
    adj_mask = torch.tril(adj_mask, diagonal=-1)
    for k, size in enumerate(sizes):
        adj_mask[k, :, size:] = 0
    adj_mask = adj_mask + adj_mask.transpose(-1, -2)
    adj_t = torch.randn(adj_shape, device=accelerator.device)
    adj_t = torch.tril(adj_t, -1)
    adj_t = (adj_t + adj_t.transpose(-1, -2)) / sqrt_2
    adj_t = adj_t * adj_mask
    if time_travel is False:
        k = 1

    T = noise_scheduler.timesteps[0].item()
    time_pairs = list(
        zip(noise_scheduler.timesteps[::tau], noise_scheduler.timesteps[tau::tau])
    )
    end = (time_pairs[-1][-1], torch.tensor(-1, device=accelerator.device))
    time_pairs.append(end)

    if log_x0_predictions:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Copaint Ablation",
            # track hyperparameters and run metadata
            config={
                "inpainter": 'repaint',
                "batch_size": batch_size,
                "dataset": config.data_name,
                "time_travel": time_travel,
                "tau": tau,
                "repeat_tt": repeat_tt,
            },
        )
    for prev_t, cur_t in time_pairs:
        for repeat_step in range(repeat_tt):
            # optimize x_t given x_0
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    time = torch.full((batch_size,), t_, device=accelerator.device)
                    with torch.no_grad():
                        # time = t.to(device=accelerator.device).reshape(-1)
                        if config.sampler == "vpsde":
                            edge_noise_res = model(adj_t, time * T, mask=adj_mask)
                        else:
                            edge_noise_res = model(adj_t, time, mask=adj_mask)
                    if config.sampler == "vpsde":
                        adj_next, _ = noise_scheduler.step_pred(
                            score=edge_noise_res, t=t_, x=adj_t
                        )
                    else:
                        adj_next = noise_scheduler.step(
                            edge_noise_res, t_, adj_t
                        ).prev_sample
                    res = adj_next - adj_t
                    res = (res + res.transpose(-1, -2)) / sqrt_2
                    res = res * adj_mask
                    adj_t = adj_t + res

                    edge_noise = torch.randn(
                        target_adj.shape, device=accelerator.device
                    )
                    edge_noise = edge_noise + edge_noise.transpose(-1, -2)
                    edge_noise = edge_noise * adj_mask / sqrt_2
                    adj_target_t = noise_scheduler.add_noise(
                        target_adj, edge_noise, time
                    )
                    adj_t = (1 - target_mask) * adj_t + target_mask * adj_target_t

                    if log_x0_predictions and repeat_step == repeat_tt - 1:
                        adj_0 = pred_x0(
                            et=edge_noise_res,
                            xt=adj_t,
                            t=t_,
                            mask=adj_mask,
                            scheduler=noise_scheduler,
                            interval_num=1,
                        )
                        sz = torch.numel(adj_t)
                        target_loss = (
                            torch.sum((adj_0 * target_mask - target_adj * target_mask) ** 2)
                            .cpu()
                            .detach()
                            .numpy()
                            / sz
                        )
                        wandb.log({"time_step": t_, "target_loss": target_loss.item()})
                        del adj_0
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                # time-travel (forward diffusion)
                if time_travel and (cur_t + 1) <= T - tau:
                    noise = torch.randn(adj_t.shape, device=accelerator.device)
                    noise = noise + noise.transpose(-1, -2)
                    noise = noise * adj_mask / sqrt_2
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                        dtype=adj_t.dtype, device=accelerator.device
                    )
                    alpha_prod = alphas_cumprod[prev_t] / alphas_cumprod[cur_t + 1]
                    sqrt_alpha_prod = alpha_prod**0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                            -1
                        )

                    adj_prev = (
                        sqrt_alpha_prod * adj_t + sqrt_one_minus_alpha_prod * noise
                    )
                    adj_t = adj_prev * adj_mask

    adj_t = adj_t * adj_mask
    return adj_t
