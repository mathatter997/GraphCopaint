import torch
import numpy as np


def get_loss_fn(mode):
    def naive_inpaint_loss(_x0, _pred_x0, _mask):
        T = _x0 * _mask
        S = _pred_x0 * _mask
        loss = torch.sum((T - S) ** 2)
        return loss

    def inpaint_loss(_x0, _pred_x0, _mask):
        T = _x0 * _mask
        S = _pred_x0 * _mask

        b_sz = _x0.shape[0]
        n = _mask.shape[-1]
        L0, V = torch.linalg.eig(T)
        L0 = L0.to(dtype=_x0.dtype).reshape(b_sz, n)
        V = V.to(dtype=_x0.dtype).reshape(b_sz, n, n)
        _, p = torch.sort(L0)
        p_inv = torch.empty_like(p)
        for i in range(b_sz):
            p_inv[i, p[i]] = torch.arange(n, device=p.device)

        Lpred, U = torch.linalg.eig(S)
        Lpred = Lpred.to(dtype=_pred_x0.dtype).reshape(b_sz, n)
        U = U.to(dtype=_pred_x0.dtype).reshape(b_sz, n, n)
        _, q = torch.sort(Lpred)

        loss = 0
        for i in range(b_sz):
            loss += torch.sum((L0[i, p[i]] - Lpred[i, q[i]]) ** 2)
        return loss

    def none(**args):
        return 0

    loss_fns = {
        "naive_inpaint": naive_inpaint_loss,
        "inpaint": inpaint_loss,
        "none": none,
    }
    return loss_fns[mode]


def get_reg_fn(mode):
    def square_reg(_origin_xt, _xt):
        T = _xt
        S = _origin_xt

        b_sz = _xt.shape[0]
        n = _xt.shape[-1]
        Lorig, V = torch.linalg.eig(T)
        Lorig = Lorig.to(dtype=_origin_xt.dtype).reshape(b_sz, n)
        V = V.to(dtype=_origin_xt.dtype).reshape(b_sz, n, n)
        _, p = torch.sort(Lorig)
        p_inv = torch.empty_like(p)
        for i in range(b_sz):
            p_inv[i, p[i]] = torch.arange(n, device=p.device)

        Lt, U = torch.linalg.eig(S)
        Lt = Lt.to(dtype=_xt.dtype).reshape(b_sz, n)
        U = U.to(dtype=_xt.dtype).reshape(b_sz, n, n)
        _, q = torch.sort(Lt)
        loss = 0
        for i in range(b_sz):
            loss += torch.sum((Lorig[i, p[i]] - Lt[i, q[i]]) ** 2)
        return loss

    def naive_square_reg(_origin_xt, _xt):
        ret = torch.sum((_origin_xt - _xt) ** 2)
        return ret

    def none(**args):
        return 0

    reg_fns = {"naive_square": naive_square_reg, "square": square_reg, "none": none}
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
    sqrt_2 = 2**0.5
    if interval_num == 1:
        x0 = scheduler.step(et, t, xt).pred_original_sample
        res = x0 - xt
        res = (res + res.transpose(-1, -2)) / sqrt_2
        res = res * mask
        x0 = xt + res
    else:
        if t == 0:
            return xt
        ts = np.linspace(0, t, interval_num + 1, dtype=int)
        ts = np.unique(ts)
        ts = ts[::-1]
        timepairs = list(zip(ts[:-1], ts[1:]))
        for prev_t, next_t in timepairs:
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t]
            alpha_prod_t_next = (
                scheduler.alphas_cumprod[next_t]
                if next_t >= 0
                else scheduler.final_alpha_cumprod
            )

            beta_prod_t = 1 - alpha_prod_t_prev
            pred_original_sample = (
                xt - beta_prod_t ** (0.5) * et
            ) / alpha_prod_t_prev ** (0.5)
            pred_epsilon = et

            # 4. Clip or threshold "predicted x_0"
            if scheduler.config.thresholding:
                pred_original_sample = scheduler._threshold_sample(pred_original_sample)
            elif scheduler.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    -scheduler.config.clip_sample_range,
                    scheduler.config.clip_sample_range,
                )
            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            # variance = scheduler._get_variance(timestep, prev_timestep)
            # std_dev_t = eta * variance ** (0.5)
            std_dev_t = 0
            pred_sample_direction = (1 - alpha_prod_t_next - std_dev_t**2) ** (
                0.5
            ) * pred_epsilon
            xnext = (
                alpha_prod_t_next ** (0.5) * pred_original_sample
                + pred_sample_direction
            )
            res = xnext - xt
            res = (res + res.transpose(-1, -2)) / sqrt_2
            res = res * mask
            xnext = xt + res
        x0 = xnext
    return x0


def predict_e0(config, model, adj_t, time, num_timesteps, adj_mask):
    if config.sampler == "vpsde":
        if config.data_format == "graph":
            e0 = model(adj_t, time * num_timesteps, mask=adj_mask)
        elif config.data_format == "pixel":
            e0 = model(adj_t, time * num_timesteps).sample * adj_mask
    else:
        if config.data_format == "graph":
            e0 = model(adj_t, time, mask=adj_mask)
        elif config.data_format == "pixel":
            e0 = model(adj_t, time).sample * adj_mask

    return e0


def predict_xnext(config, noise_scheduler, e0, xt, mask, t):
    sqrt_2 = 2 ** 0.5
    if config.sampler == "vpsde":
        xnext, _ = noise_scheduler.step_pred(score=e0, t=t, x=xt)
    else:
        xnext = noise_scheduler.step(e0, t, xt).prev_sample
    res = xnext - xt
    res = (res + res.transpose(-1, -2)) / sqrt_2
    res = res * mask
    xt = xt + res
    return xt


def time_travel_fn(
    adj_t,
    adj_mask,
    noise_scheduler,
    accelerator,
    prev_t,
    cur_t,
    optimize_before_time_travel,
):
    sqrt_2 = 2**0.5

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
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    adj_prev = sqrt_alpha_prod * adj_t + sqrt_one_minus_alpha_prod * noise
    adj_t = adj_prev * adj_mask

    return adj_t


def optimize_xt(
    config,
    model,
    noise_scheduler,
    adj_t,
    t,
    adj_mask,
    target_adj,
    target_mask,
    batch_size,
    accelerator,
    num_iteration_optimize_xt,
    interval_num,
    loss_fn,
    coef_xt_reg,
    reg_fn,
    loss_norm,
    lr_xt,
    use_adaptive_lr_xt,
    num_timesteps,
):
    time = torch.full((batch_size,), t.item(), device=accelerator.device)
    origin_adj = adj_t.clone().detach()
    for step in range(num_iteration_optimize_xt):
        e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
        adj_0 = pred_x0(
            et=e0,
            xt=adj_t,
            t=t,
            mask=adj_mask,
            scheduler=noise_scheduler,
            interval_num=interval_num,
        )
        loss = loss_fn(target_adj, adj_0, target_mask) + coef_xt_reg * reg_fn(
            origin_adj, adj_t
        )
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
                e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
                adj_0 = pred_x0(
                    et=e0,
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
                    break
                else:
                    lr_xt_temp *= 0.8
                    del new_adj_t, new_loss
                    new_adj_t = adj_t - lr_xt_temp * adj_t_grad * loss_norm
        # optimized x, pred_x0, and e_t
        adj_t = new_adj_t.detach().requires_grad_()
        del loss, adj_t_grad, adj_0, e0
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
    return adj_t

def init_xT(config, batch_size, sizes, accelerator):
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

    return adj_t, adj_mask
