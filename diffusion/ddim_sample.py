import os
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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


def sample(
    config,
    model,
    noise_scheduler,
    num_inference_steps=1000,
    inference_timesteps=None,
    n=1,
):
    assert 1 <= n <= config.max_n_nodes

    if inference_timesteps is None:
        noise_scheduler.set_timesteps(num_inference_steps)
    else:
        noise_scheduler.num_inference_steps = len(inference_timesteps)
        noise_scheduler.timesteps = torch.from_numpy(inference_timesteps).to(
            config.device
        )

    adj_shape = (1, 1, config.max_n_nodes, config.max_n_nodes)
    adj_t = torch.randn(adj_shape, device=config.device)

    adj_mask = torch.ones(adj_shape, device=config.device)
    adj_mask = torch.tril(adj_mask, diagonal=-1)
    adj_mask[:, :, n:] = 0
    for t in noise_scheduler.timesteps:
        adj_t = adj_t * adj_mask
        with torch.no_grad():
            time = t.to(device=config.device).reshape(-1)
            edge_noise_res = model(adj_t, time, mask=adj_mask)

        adj_t = noise_scheduler.step(
            edge_noise_res, t, adj_t, eta=config.eta
        ).prev_sample
        adj_t = torch.clip(adj_t, -3, 3)
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

    adj_t = adj_t * adj_mask
    return adj_t


def copaint(
    config,
    model,
    noise_scheduler,
    num_inference_steps=1000,
    inference_timesteps=None,
    interval_num=1,
    n=1,
    # time_travel params
    time_travel=False,
    k=1,
    # tau=1,
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
    assert 1 <= n <= config.max_n_nodes
    # assert inference_timesteps is None

    if inference_timesteps is None:
        noise_scheduler.set_timesteps(num_inference_steps)
    else:
        noise_scheduler.num_inference_steps = len(inference_timesteps)
        noise_scheduler.timesteps = torch.from_numpy(inference_timesteps).to(
            config.device
        )

    original_timesteps = torch.copy(noise_scheduler.timesteps)

    adj_shape = (1, 1, config.max_n_nodes, config.max_n_nodes)
    adj_t = torch.randn(adj_shape, device=config.device)

    adj_mask = torch.ones(adj_shape, device=config.device)
    adj_mask = torch.tril(adj_mask, diagonal=-1)
    adj_mask[:, :, n:] = 0

    time_pairs = list(zip(original_timesteps[:-1], original_timesteps[1:]))

    if time_travel is False:
        k = 1

    loss_fn = get_loss_fn(loss_mode)
    reg_fn = get_reg_fn(reg_mode)

    for cur_t, prev_t in time_pairs:
        if interval_num > 1:
            timesteps = np.linspace(0, prev_t.item(), interval_num + 1, dtype=int)
            noise_scheduler.num_inference_steps = len(timesteps)
            noise_scheduler.timesteps = torch.from_numpy(timesteps).to(config.device)

        for _ in range(k):
            # optimize x_t given x_0

            with torch.enable_grad():
                t = prev_t.clone()
                adj_t = adj_t * adj_mask
                origin_adj = adj_t.clone().detach()
                adj_t = adj_t.detach().requires_grad_()

                time = t.to(device=config.device).reshape(-1)
                adj_noise_res = model(adj_t, time, mask=adj_mask)

                step = noise_scheduler.step(
                    adj_noise_res, t, adj_t, eta=config.eta
                ).prev_sample

                adj_t, adj_0 = step.sample, step.pred_original_sample
                adj_t = torch.clip(adj_t, -3, 3)
                adj_0 = torch.clip(adj_0, -1, 1)

                adj_t = adj_t * adj_mask
                adj_0 = adj_0 * adj_mask

                # prev_loss = loss_fn(target_adj, adj_0, target_mask).item()
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
                            adj_noise_res = model(new_adj_t, time, mask=adj_mask)
                            step = noise_scheduler.step(
                                adj_noise_res, t, new_adj_t, eta=config.eta
                            )

                            adj_0 = step.pred_original_sample
                            adj_0 = torch.clip(adj_0, -1, 1)
                            adj_0 = adj_0 * adj_mask

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
                    torch.cuda.empty_cache()

            # time-travel (forward diffusion)
            if time_travel:
                if optimize_before_time_travel:
                    pass
                noise = torch.randn(adj_t.shape, device=config.device)
                noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                    device=adj_t.device
                )
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(dtype=adj_t.dtype)
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
