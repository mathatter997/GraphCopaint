import os
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
from evaluation.dataviz_utils import plot_loss_and_samples
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
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            time = torch.full((batch_size,), t.item(), device=accelerator.device)
            if config.data_format == 'eigen':
                ex0, ela0 = model(x_t, adj_t, flags, u, la_t, time)
                x_t = predict_xnext(config, noise_scheduler, ex0, x_t, mask=None, t=t, reflect=False)
                la_t = predict_xnext(config, noise_scheduler, ela0, la_t, mask=None, t=t, reflect=False)
                if t == 0:
                    adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
                    adj_t = mask_adjs(adj_t, flags)
                adj_t = torch.bmm(u, torch.bmm(torch.diag_embed(la_t), u_T))
                x_t = mask_x(x_t, flags)
                adj_t = mask_adjs(adj_t, flags)
                if log_x0_predictions:
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
                    adj_0s.append(adj_0.cpu().reshape(1, 1, config.max_n_nodes, config.max_n_nodes))
            else:
                e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
                adj_t = predict_xnext(
                    config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                )

                if log_x0_predictions:
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
    log_x0_predictions=False,
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

    adj_t, adj_mask = init_xT(
        config, batch_size, sizes, accelerator, zero_diagonal=zero_diagonal
    )

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
                    adj_t = adj_t.detach().requires_grad_()
                    time = torch.full(
                        (batch_size,), t.item(), device=accelerator.device
                    )
                    adj_t = optimize_xt(
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
                        reflect=reflect,
                    )

                    if log_x0_predictions and repeat_step == repeat_tt - 1:
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
                        del e0, adj_0
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                    e0 = predict_e0(config, model, adj_t, time, num_timesteps, adj_mask)
                    adj_t = predict_xnext(
                        config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                    )

                # time-travel (forward diffusion)
                if time_travel and (cur_t + 1) <= T - tau:
                    adj_t = time_travel_fn(
                        adj_t,
                        adj_mask,
                        noise_scheduler,
                        accelerator,
                        prev_t,
                        cur_t,
                        optimize_before_time_travel,
                    )

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
    num_timesteps = num_inference_steps

    batch_size = len(sizes)
    sqrt_2 = 2**0.5

    reflect = config.reflect
    zero_diagonal = config.zero_diagonal

    adj_t, adj_mask = init_xT(
        config, batch_size, sizes, accelerator, zero_diagonal=zero_diagonal
    )

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
                "inpainter": "repaint",
                "batch_size": batch_size,
                "dataset": config.data_name,
                "time_travel": time_travel,
                "tau": tau,
                "repeat_tt": repeat_tt,
                "sampler": config.sampler,
            },
        )
    for prev_t, cur_t in time_pairs:
        for repeat_step in range(repeat_tt):
            with torch.enable_grad():
                for t_ in range(prev_t.item(), cur_t.item(), -1):
                    t = torch.tensor(t_)
                    time = torch.full((batch_size,), t_, device=accelerator.device)
                    with torch.no_grad():
                        e0 = predict_e0(
                            config, model, adj_t, time, num_timesteps, adj_mask
                        )
                    adj_t = predict_xnext(
                        config, noise_scheduler, e0, adj_t, adj_mask, t, reflect=reflect
                    )

                    noise = torch.randn(target_adj.shape, device=accelerator.device)
                    noise = noise + noise.transpose(-1, -2)
                    noise = noise * adj_mask / sqrt_2
                    adj_target_t = noise_scheduler.add_noise(target_adj, noise, time)

                    adj_t = (1 - target_mask) * adj_t + target_mask * adj_target_t
                    if log_x0_predictions and repeat_step == repeat_tt - 1:
                        e0 = predict_e0(
                            config, model, adj_t, time, num_timesteps, adj_mask
                        )
                        adj_0 = pred_x0(
                            et=e0,
                            xt=adj_t,
                            t=t_,
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
                if time_travel and (cur_t + 1) <= T - tau:
                    adj_t = time_travel_fn(
                        adj_t,
                        adj_mask,
                        noise_scheduler,
                        accelerator,
                        prev_t,
                        cur_t,
                        optimize_before_time_travel=False,
                    )

    adj_t = adj_t * adj_mask
    return adj_t
