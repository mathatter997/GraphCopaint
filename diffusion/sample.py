import json
import torch
from .sample_utils import (
    get_loss_fn,
    pred_x0,
    get_reg_fn,
    predict_e0,
    predict_xnext,
    time_travel_fn,
    optimize_xt,
    init_xT,
)
from .utils import mask_adjs, mask_x
from evaluation.dataviz_utils import plot_loss_and_samples, plot_diffs
import wandb


def sample(
    config,
    model,
    noise_scheduler,
    accelerator,
    num_inference_steps=1000,
    sizes=None,
    u=None,
    log_x0_predictions=False,
    alpha = 1,
):
    model.eval()
    batch_size = len(sizes)
    noise_scheduler.set_timesteps(num_inference_steps)
    num_timesteps = len(noise_scheduler.timesteps)
    reflect = config.reflect
    zero_diagonal = config.zero_diagonal
    if config.data_format == 'eigen':
        x_t = torch.randn((batch_size, config.max_n_nodes, config.max_feat_num), device=accelerator.device)
        la_t =  torch.randn((batch_size, config.max_n_nodes), device=accelerator.device)
        u_T = u.transpose(-1, -2)
        adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
        flags = torch.zeros(batch_size, config.max_n_nodes, device=accelerator.device)
        for i in range(batch_size):
            flags[i,:sizes[i]] = 1
    else:
        adj_t, adj_mask = init_xT(
            config, batch_size, sizes, accelerator, zero_diagonal=zero_diagonal
        )

    if log_x0_predictions:
        adj_0s = []
        diffs = []
    time = torch.full((batch_size,), noise_scheduler.timesteps[0].item(), device=accelerator.device)
    if config.data_format == 'eigen':
        ex0_prev, ela0_prev = model(x_t, adj_t, flags, u, la_t, time)
    else:
        e_prev = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            time = torch.full((batch_size,), t.item(), device=accelerator.device)
            if config.data_format == 'eigen':
                ex0, ela0 = model(x_t, adj_t, flags, u, la_t, time)
                ela0_prev_ = ela0_prev
                ex0 = alpha * ex0 + (1 - alpha) * ex0_prev 
                ela0 = alpha * ela0 + (1 - alpha) * ela0_prev 
                ex0_prev = ex0
                ela0_prev = ela0

                x_t = predict_xnext(config, noise_scheduler, ex0, x_t, mask=None, t=t, reflect=False)
                la_t = predict_xnext(config, noise_scheduler, ela0, la_t, mask=None, t=t, reflect=False)
                adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
                x_t = mask_x(x_t, flags)
                adj_t = mask_adjs(adj_t, flags)
                if log_x0_predictions:
                    diffs.append(torch.norm(ela0 - ela0_prev_))
                    la_0 = pred_x0(
                        et=ela0,
                        xt=la_t,
                        t=t,
                        mask=None,
                        scheduler=noise_scheduler,
                        interval_num=1,
                        reflect=False,
                    )
                    adj_0 = torch.bmm(u, torch.bmm(torch.diag_embed(la_0), u_T))
                    adj_0s.append(adj_0.cpu().reshape(len(sizes), 1, config.max_n_nodes, config.max_n_nodes))
            else:
                e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
                e_prev_ = e_prev
                e0 = alpha * e0 + (1 - alpha) * e_prev
                e_prev = e0
                adj_t = predict_xnext(
                    config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                )

                if log_x0_predictions:
                    diffs.append(torch.norm(e0 - e_prev_))
                    e_prev = e0
                    adj_0 = pred_x0(
                        et=e0,
                        xt=adj_t,
                        t=t,
                        mask=adj_mask,
                        scheduler=noise_scheduler,
                        interval_num=1,
                        reflect=reflect,
                    )
                    adj_0s.append(adj_0.cpu())
    if log_x0_predictions:
        if len(sizes) == 1:
            size = sizes[0]
            plot_diffs(config, diffs, size)
            plot_loss_and_samples(config, adj_0s, size)
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
    lr_xt_decay=1.0,
    coef_xt_reg=0.01,
    coef_xt_reg_decay=1.0,
    optimize_before_time_travel=False,
    use_adaptive_lr_xt=True,
    num_iteration_optimize_xt=2,
    target_mask=None,
    target_adj=None,
    u=None,
    log_x0_predictions=False,
    lr_xt_path=None,
    opt_num_path=None,
    alpha=1,
):
    model.eval()
    batch_size = len(sizes)
    loss_norm = batch_size
    loss_fn = get_loss_fn(loss_mode)
    reg_fn = get_reg_fn(reg_mode)
    noise_scheduler.set_timesteps(num_inference_steps)
    num_timesteps = num_inference_steps

    reflect = config.reflect
    zero_diagonal = config.zero_diagonal
    adj_mask=None
    if config.data_format == 'eigen':
        x_t = torch.randn((batch_size, config.max_n_nodes, config.max_feat_num), device=accelerator.device)
        la_t =  torch.randn((batch_size, config.max_n_nodes), device=accelerator.device)
        u_T = u.transpose(-1, -2)
        adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
        flags = torch.zeros(batch_size, config.max_n_nodes, device=accelerator.device)
        for i in range(batch_size):
            flags[i,:sizes[i]] = 1
    else:
        flags = None
        adj_t, adj_mask = init_xT(
            config, batch_size, sizes, accelerator, zero_diagonal=zero_diagonal
        )

    T = noise_scheduler.timesteps[0].item()
    time_pairs = list(
        zip(noise_scheduler.timesteps[::tau], noise_scheduler.timesteps[tau::tau])
    )
    end = (time_pairs[-1][-1], torch.tensor(-1, device=accelerator.device))
    time_pairs.append(end)

    time = torch.full((batch_size,), noise_scheduler.timesteps[0].item(), device=accelerator.device)
    if config.data_format == 'eigen':
        ex0_prev, ela0_prev = model(x_t, adj_t, flags, u, la_t, time)
    else:
        e_prev = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
    if lr_xt_path is None:
        lr_x = lr_xt * torch.pow(lr_xt_decay, torch.arange(T + 1))
        lr_x = torch.flip(lr_x, [0])
    else:
        with open(lr_xt_path, "r") as f:
            lr_x = json.load(f)
            lr_x = torch.tensor(lr_x)
            lr_x = torch.flip(lr_x, [0])
    if opt_num_path is None:
        num_iteration_optimize_x = torch.full((num_inference_steps,), num_iteration_optimize_xt,dtype=torch.int)
    else:
        with open(opt_num_path, "r") as f:
            num_iteration_optimize_x = json.load(f)
            num_iteration_optimize_x = torch.tensor(num_iteration_optimize_x, dtype=torch.int)
            num_iteration_optimize_x = torch.flip(num_iteration_optimize_x, [0])
    coef_x_reg = coef_xt_reg * torch.pow(coef_xt_reg_decay, torch.arange(T + 1))
    coef_x_reg = torch.flip(coef_x_reg, [0])

    if log_x0_predictions:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Copaint Eigen Ablation",
            # track hyperparameters and run metadata
            config={
                "inpainter": "copaint",
                "batch_size": batch_size,
                "dataset": config.data_name,
                "time_travel": time_travel,
                "interval_num": interval_num,
                "tau": tau,
                "repeat_tt": repeat_tt,
                "loss_mode": loss_mode,
                "reg_mode": reg_mode,
                "lr_xt": lr_xt,
                "lr_xt_decay": lr_xt_decay,
                "coef_xt_reg": coef_xt_reg,
                "coef_xt_reg_decay": coef_xt_reg_decay,
                "sampler": config.sampler,
                "lr_xt_path": lr_xt_path,
                "opt_num_path": opt_num_path,
            },
        )
    if config.data_format == 'eigen':
        data_t = (x_t, la_t, adj_t)
    else:
        data_t = adj_t
    for prev_t, cur_t in time_pairs:
        for repeat_step in range(repeat_tt + 1):
            # optimize x_t given x_0
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    lr_xt = lr_x[t_]
                    coef_xt_reg = coef_x_reg[t_]
                    t = torch.tensor(t_)
                    time = torch.full(
                        (batch_size,), t.item(), device=accelerator.device
                    )
                    if config.data_format == 'eigen':
                        # u = u.detach().requires_grad_()
                        # flags = flags.detach().requires_grad_()
                        x_t = x_t.detach().requires_grad_()
                        la_t = la_t.detach().requires_grad_()
                        x_t = mask_x(x_t, flags)
                        adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
                        adj_t = mask_adjs(adj_t, flags)
                        data_t = (x_t, la_t, adj_t)
                    else:
                        adj_t = adj_t * adj_mask
                        adj_t = adj_t.detach().requires_grad_()
                        data_t = adj_t
                    data_t = optimize_xt(
                        config,
                        model,
                        noise_scheduler,
                        data_t,
                        t,
                        adj_mask,
                        target_adj,
                        target_mask,
                        batch_size,
                        accelerator,
                        num_iteration_optimize_x[t_],
                        interval_num,
                        loss_fn,
                        coef_xt_reg,
                        reg_fn,
                        loss_norm,
                        lr_xt,
                        use_adaptive_lr_xt,
                        num_timesteps,
                        reflect=reflect,
                        u=u,
                        flags=flags,
                    )
                    if config.data_format == 'eigen':
                        x_t, la_t, adj_t = data_t
                    else:
                        adj_t = data_t
                    
                    if log_x0_predictions and repeat_step == repeat_tt:
                        if config.data_format == 'eigen':
                            ex0, ela0 = model(x_t, adj_t, flags, u, la_t, time)
                            la_0 = pred_x0(
                                et=ela0,
                                xt=la_t,
                                t=t,
                                mask=None,
                                scheduler=noise_scheduler,
                                interval_num=1,
                                reflect=False,
                            )
                            adj_0 = torch.bmm(u, torch.bmm(torch.diag_embed(la_0), u_T))
                        else:
                            e0 = predict_e0(
                                config, model, adj_t, time, num_timesteps, adj_mask
                            )
                            adj_0 = pred_x0(
                                et=e0,
                                xt=adj_t,
                                t=t,
                                mask=adj_mask,
                                scheduler=noise_scheduler,
                                interval_num=interval_num,
                                reflect=reflect,
                            )
                        sz = torch.numel(adj_0)
                        target_loss = (
                            torch.sum(
                                (adj_0 * target_mask - target_adj * target_mask) ** 2
                            )
                            .cpu()
                            .detach()
                            .numpy()
                            / sz
                        )
                        wandb.log({"time_step": t_, "target_loss": target_loss.item()})
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                    if config.data_format == 'eigen':
                        ex0, ela0 = model(x_t, adj_t, flags, u, la_t, time)
                        ex0 = alpha * ex0 + (1 - alpha) * ex0_prev
                        ela0 = alpha * ela0 + (1 - alpha) * ela0_prev
                        ex0_prev = ex0.clone().detach().requires_grad_()
                        ela0_prev = ela0.clone().detach().requires_grad_()
                        x_t = predict_xnext(config, noise_scheduler, ex0, x_t, mask=None, t=t, reflect=False)
                        la_t = predict_xnext(config, noise_scheduler, ela0, la_t, mask=None, t=t, reflect=False)
                        adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
                        adj_t = mask_adjs(adj_t, flags)
                    else:
                        e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
                        e0 = alpha * e0 + (1 - alpha) * e_prev
                        e_prev = e0
                        adj_t = predict_xnext(
                            config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                        )
            # time-travel (forward diffusion)
            if time_travel and (cur_t + 1) <= T - tau and repeat_step < repeat_tt:
                print('time_travel from', cur_t, 'to', prev_t)
                if config.data_format == 'eigen':
                    data_mask = flags
                else:
                    data_mask = adj_mask
                data_t = time_travel_fn(
                        config,
                        data_t,
                        data_mask,
                        noise_scheduler,
                        accelerator,
                        prev_t,
                        cur_t,
                        optimize_before_time_travel,
                        u=u,
                    )
                if config.data_format == 'eigen':
                    x_t, la_t, adj_t = data_t
                else:
                    adj_t = data_t
    if config.data_format == 'eigen':
        adj_t = mask_adjs(adj_t, flags)
    else:
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
    alpha=1,
):
    model.eval()
    noise_scheduler.set_timesteps(num_inference_steps)
    num_timesteps = num_inference_steps

    batch_size = len(sizes)
    sqrt_2 = 2**0.5

    reflect = config.reflect
    zero_diagonal = config.zero_diagonal

    adj_t, adj_mask = init_xT(
        config, batch_size, sizes, accelerator, zero_diagonal=zero_diagonal
    )

    T = noise_scheduler.timesteps[0].item()
    if config.sampler == "vpsde":
        time_pairs = list(
            zip(torch.round((noise_scheduler.timesteps[::tau] * num_timesteps)).to(dtype=torch.int), 
                torch.round(noise_scheduler.timesteps[tau::tau] * num_timesteps).to(dtype=torch.int))
        )
        end = (time_pairs[-1][-1], torch.tensor(0, device=accelerator.device))
    else:
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
                "inpainter": "repaint",
                "batch_size": batch_size,
                "dataset": config.data_name,
                "time_travel": time_travel,
                "tau": tau,
                "repeat_tt": repeat_tt,
                "sampler": config.sampler,
            },
        )
    time = torch.full((batch_size,), noise_scheduler.timesteps[0].item(), device=accelerator.device)
    e_prev= predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
    for prev_t, cur_t in time_pairs:
        for repeat_step in range(repeat_tt + 1):
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    if config.sampler == "vpsde":
                        t_ = t_ / num_inference_steps
                    t = torch.tensor(t_)
                    time = torch.full((batch_size,), t_, device=accelerator.device)
                    with torch.no_grad():
                        e0 = predict_e0(
                            config, model, adj_t, time, num_timesteps, adj_mask
                        )
                        e0 = alpha * e0 + (1 - alpha) * e_prev
                        e_prev = e0
                    adj_t = predict_xnext(
                        config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                    )

                    noise = torch.randn(target_adj.shape, device=accelerator.device)
                    noise = noise + noise.transpose(-1, -2)
                    noise = noise * adj_mask / sqrt_2
                    adj_target_t = noise_scheduler.add_noise(target_adj, noise, time)

                    adj_t = (1 - target_mask) * adj_t + target_mask * adj_target_t
                    if log_x0_predictions and repeat_step == repeat_tt:
                        e0 = predict_e0(
                            config, model, adj_t, time, num_timesteps, adj_mask
                        )
                        adj_0 = pred_x0(
                            et=e0,
                            xt=adj_t,
                            t=t,
                            mask=adj_mask,
                            scheduler=noise_scheduler,
                            interval_num=1,
                            reflect=reflect,
                        )
                        sz = torch.numel(adj_t)
                        target_loss = (
                            torch.sum(
                                (adj_0 * target_mask - target_adj * target_mask) ** 2
                            )
                            .cpu()
                            .detach()
                            .numpy()
                            / sz
                        )
                        wandb.log({"time_step": t_, "target_loss": target_loss.item()})
                        del adj_0, e0
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                # time-travel (forward diffusion)
                if time_travel and (cur_t + 1) <= T - tau and repeat_step < repeat_tt:
                    adj_t = time_travel_fn(
                        config,
                        adj_t,
                        adj_mask,
                        noise_scheduler,
                        accelerator,
                        prev_t,
                        cur_t,
                        optimize_before_time_travel=False,
                        u=None,
                    )

    adj_t = adj_t * adj_mask
    return adj_t
