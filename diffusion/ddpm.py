import os
import torch
import wandb
import torch.nn.functional as F

from .utils import dense_adj
from tqdm.auto import tqdm
from .ema import ExponentialMovingAverage
from .sample_utils import predict_e0

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    ema,
    train_dataloader,
    lr_scheduler,
    accelerator,
    label="",
):
    # Initialize accelerator and tensorboard logging
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Simple Graph Diffusion",
        # track hyperparameters and run metadata
        config={
            "learning_rate": config.learning_rate,
            "architecture": "PGSN",
            "dataset": config.data_name,
            "epochs": config.num_epochs,
        },
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, lr_scheduler
    # )

    max_n_nodes = config.max_n_nodes
    global_step = 0

    # [0,1] -> [-1, 1]
    def scale_data(x):
        return 2.0 * x - 1
    
    sqrt_2 = 2 ** 0.5
    for epoch in range(config.start_epoch, config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if config.data_format == 'graph':
                adj, adj_mask = dense_adj(batch, max_n_nodes, scale_data)
            elif config.data_format == 'pixel':
                adj, adj_mask = batch
                adj = scale_data(adj) * adj_mask
            edge_noise = torch.randn(adj.shape, device=accelerator.device)
            # make symmetric 
            edge_noise = edge_noise + edge_noise.transpose(-1, -2)
            edge_noise = edge_noise * adj_mask / sqrt_2
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (adj.shape[0],),
                device=accelerator.device,
                dtype=torch.int64,
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_edges = noise_scheduler.add_noise(adj, edge_noise, timesteps)
            noisy_edges = noisy_edges * adj_mask
            with accelerator.accumulate(model):
                # Predict the noise residual
                e0 = predict_e0(config, model, noisy_edges, timesteps, -1, adj_mask)
                loss = F.mse_loss(e0, edge_noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                ema.update(model.parameters())
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch,
            }
            wandb.log(logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if (
                    epoch + 1
                ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    noise_scheduler.save_pretrained(config.output_dir)
                    file_path = config.output_dir + config.output_dir_gnn.format(
                        str(epoch + 1 + config.start) + label
                    )
                    directory = os.path.dirname(file_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": accelerator.unwrap_model(
                                model
                            ).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "ema_state_dict": ema.state_dict(),
                        },
                        file_path,
                    )
