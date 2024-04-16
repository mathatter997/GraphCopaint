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

    reg_fns = {"naive_square": naive_square_reg,
               "square": square_reg, 
               "none": none}
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
        ts = np.linspace(0, t, interval_num, dtype=int)
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