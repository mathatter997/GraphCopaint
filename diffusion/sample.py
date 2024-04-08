import os
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy

def get_loss_fn(mode):
    def inpaint_loss(_x0, _pred_x0, _mask):
        ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
        return ret

    def none(**args):
        return 0

    loss_fns = {"inpaint": inpaint_loss, "none": none}
    return loss_fns[mode]


def get_reg_fn(mode):
    def square_reg(_origin_xt, _xt):
        ret = torch.sum((_origin_xt - _xt) ** 2)
        return ret

    def none(**args):
        return 0

    reg_fns = {"square": square_reg, "none": none}
    return reg_fns[mode]


def get_edge_index(max_nodes, n=1):
    edge_index = []
    for i in range(n):
        for j in range(n):
            if i != j:  # no-loops
                edge_index.append([i, j])
    for i in range(max_nodes):
        for j in range(max_nodes):
            if i != j and i < n and j < n:
                continue
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index)
    return edge_index.t().contiguous()

def pred_x0(et, xt, t, mask, scheduler, interval_num):
    sqrt_2 = 2 ** 0.5
    if interval_num == 1:
        x0 = scheduler.step(
                et, t, xt
            ).pred_original_sample
        res = x0 - xt
        res = (res + res.transpose(-1, -2)) / sqrt_2
        res = res * mask
        x0 = xt + res
    else:
        ts = np.linspace(0, t, interval_num, dtype=int)
        ts = np.unique(ts)
        ts = ts[::-1]
        timepairs = list(zip(ts[:-1], ts[1:]))
        for prev_t, next_t in timepairs:
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t]
            alpha_prod_t_next = scheduler.alphas_cumprod[next_t] if next_t >= 0 else scheduler.final_alpha_cumprod

            beta_prod_t = 1 - alpha_prod_t_prev
            pred_original_sample = (xt - beta_prod_t ** (0.5) * et) / alpha_prod_t_prev ** (0.5)
            pred_epsilon = et

            # 4. Clip or threshold "predicted x_0"
            if scheduler.config.thresholding:
                pred_original_sample = scheduler._threshold_sample(pred_original_sample)
            elif scheduler.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
                )
            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            # variance = scheduler._get_variance(timestep, prev_timestep)
            # std_dev_t = eta * variance ** (0.5)
            std_dev_t = 0
            pred_sample_direction = (1 - alpha_prod_t_next - std_dev_t**2) ** (0.5) * pred_epsilon
            xnext = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction
            res = xnext - xt
            res = (res + res.transpose(-1, -2)) / sqrt_2
            res = res * mask
            xnext = xt + res
        x0 = xnext
    return x0



def sample(
    config,
    model,
    noise_scheduler,
    accelerator,
    num_inference_steps=1000,
    inference_timesteps=None,
    sizes=None,
):

    if inference_timesteps is None:
        noise_scheduler.set_timesteps(num_inference_steps)
    else:
        noise_scheduler.num_inference_steps = len(inference_timesteps)
        noise_scheduler.timesteps = torch.from_numpy(inference_timesteps).to(
            accelerator.device
        )

    batch_size = len(sizes)
    sqrt_2 = 2 ** 0.5
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
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            # time = t.to(device=accelerator.device).reshape(-1)
            time = torch.full((batch_size,), t.item(), device=accelerator.device)
            if config.sampler == 'vpsde':
                edge_noise_res = model(adj_t, time * num_timesteps, mask=adj_mask)
            else:
                 edge_noise_res = model(adj_t, time, mask=adj_mask)
        if config.sampler == 'vpsde':
            adj_next, _ = noise_scheduler.step_pred(score=edge_noise_res, t=t, x=adj_t)
        else:
            adj_next = noise_scheduler.step(
                edge_noise_res, t, adj_t
            ).prev_sample
        res = adj_next - adj_t
        res = (res + res.transpose(-1, -2)) / sqrt_2
        res = res * adj_mask
        adj_t = adj_t + res
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
):
    model.eval()
    noise_scheduler.set_timesteps(num_inference_steps)

    batch_size = len(sizes)
    sqrt_2 = 2 ** 0.5

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
    time_pairs = list(zip(noise_scheduler.timesteps[:-tau], noise_scheduler.timesteps[tau:]))
    for prev_t, cur_t in time_pairs:
        for _ in range(repeat_tt):
            # optimize x_t given x_0
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    model.eval()
                    t = torch.tensor(t_)
                    adj_t = adj_t * adj_mask
                    origin_adj = adj_t.clone().detach()
                    adj_t = adj_t.detach().requires_grad_()

                    time = torch.full((batch_size,), t.item(), device=accelerator.device)
                    adj_noise_res = model(adj_t, time, mask=adj_mask)
                    adj_tm1 = noise_scheduler.step(
                        adj_noise_res, t, adj_t
                    ).prev_sample
                    res = adj_tm1 - adj_t
                    res = (res + res.transpose(-1, -2)) / sqrt_2
                    res = res * adj_mask
                    adj_t = adj_t + res

                    adj_noise_res = model(adj_t, time - 1, mask=adj_mask)
                    adj_0 = pred_x0(et=adj_noise_res,
                                    xt=adj_t,
                                    t=t,
                                    mask=adj_mask,
                                    scheduler=noise_scheduler,
                                    interval_num=interval_num,
                                    )
                    
                    for step in range(num_iteration_optimize_xt):
                        loss = loss_fn(
                            target_adj, adj_0, target_mask
                        ) + coef_xt_reg * reg_fn(origin_adj, adj_t)
                        adj_t_grad = torch.autograd.grad(
                            loss, adj_t, retain_graph=False, create_graph=False
                        )[0].detach()
                        adj_t_grad = adj_t_grad * adj_mask
                        new_adj_t = adj_t - lr_xt * adj_t_grad
                        # if new_x doesn't improve loss
                        # we start from x and try again with a smaller grad step
                        while use_adaptive_lr_xt:
                            with torch.no_grad():
                                adj_noise_res = model(new_adj_t, time - 1, mask=adj_mask)
                                adj_0 = pred_x0(et=adj_noise_res,
                                                xt=new_adj_t,
                                                t=t,
                                                mask=adj_mask,
                                                scheduler=noise_scheduler,
                                                interval_num=interval_num,
                                        )
                                
                                new_loss = loss_fn(
                                    target_adj, adj_0, target_mask
                                ) + coef_xt_reg * reg_fn(origin_adj, new_adj_t)
                                if not torch.isnan(new_loss) and new_loss <= loss:
                                    break
                                else:
                                    lr_xt *= 0.8
                                    del new_adj_t, adj_0, new_loss
                                    new_adj_t = adj_t - lr_xt * adj_t_grad
                        # optimized x, pred_x0, and e_t
                        adj_t = new_adj_t.detach().requires_grad_()
                        del loss, adj_t_grad
                        if accelerator.device.type == 'cuda':
                            torch.cuda.empty_cache()

                # time-travel (forward diffusion)
                if time_travel and t <= T - tau and t % tau == 0:
                    if optimize_before_time_travel:
                        pass
                    noise = torch.randn(adj_t.shape, device=accelerator.device)
                    noise = noise + noise.transpose(-1, -2)
                    noise = noise * adj_mask / sqrt_2
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(dtype=adj_t.dtype, device=accelerator.device)
                    alpha_prod = alphas_cumprod[prev_t] / alphas_cumprod[cur_t]
                    sqrt_alpha_prod = alpha_prod**0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alpha_prod) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(adj_t.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                    adj_prev = sqrt_alpha_prod * adj_t + sqrt_one_minus_alpha_prod * noise
                    adj_t = adj_prev * adj_mask

        lr_xt *= lr_xt_decay
        coef_xt_reg *= coef_xt_reg_decay

    adj_t = adj_t * adj_mask
    return adj_t
