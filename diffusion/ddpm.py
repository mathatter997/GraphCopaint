import os
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, accelerator):
    # Initialize accelerator and tensorboard logging
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Sample noise to add to the images
            node_noise = torch.randn(batch.x.shape, device=accelerator.device)
            edge_noise = torch.randn(batch.edge_attr.shape, device=accelerator.device)
            bs = len(batch)
            
            num_nodes = batch[0].x.size(0)
            num_edges = batch[0].edge_index.size(1)

            node_mask = torch.zeros_like(batch.x[:,0:1], device=accelerator.device)
            edge_mask = torch.zeros_like(batch.edge_attr[:,0:1], device=accelerator.device)
            for i in range(bs):
                n = batch[i].card.item()
                node_mask[i * num_nodes: i * num_nodes + n] = 1
                edge_mask[i * num_edges: i * num_edges + n * (n - 1)] = 1 # directed graph

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device, 
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_nodes = noise_scheduler.add_noise(batch.x.reshape(bs, num_nodes, -1), 
                                                    node_noise.reshape(bs, num_nodes, -1), 
                                                    timesteps).reshape(batch.x.size(0), -1)
            noisy_edges = noise_scheduler.add_noise(batch.edge_attr.reshape(bs, num_edges, -1), 
                                                    edge_noise.reshape(bs, num_edges, -1), 
                                                    timesteps).reshape(batch.edge_index.size(1), -1)

            node_noise = node_noise * node_mask
            edge_noise = edge_noise * edge_mask
            noisy_nodes = noisy_nodes * node_mask
            noisy_edges = noisy_edges * edge_mask
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_node_pred, noise_edge_pred = model(x=noisy_nodes, 
                                                        edge_attr=noisy_edges,
                                                        t=timesteps,
                                                        edge_index=batch.edge_index,
                                                        batch_size=bs,
                                                        node_mask=node_mask,
                                                        edge_mask=edge_mask
                                                        )
                loss = F.mse_loss(noise_node_pred, node_noise) + \
                        F.mse_loss(noise_edge_pred, edge_noise)
                
                if type(loss) is torch.nan:
                    print('node:', noise_node_pred)
                    print('edge:', noise_node_pred)
                    break

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    noise_scheduler.save_pretrained(config.output_dir)
                    torch.save(accelerator.unwrap_model(model), config.output_dir + config.output_dir_gnn.format(epoch + 1))