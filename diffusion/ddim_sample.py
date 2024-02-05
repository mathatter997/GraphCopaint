import os
import torch

def get_edge_index(max_nodes, n=1):
    edge_index = []
    for i in range(n):
        for j in range(n):
            if i != j: # no-loops
                edge_index.append([i, j])
    for i in range(max_nodes):
        for j in range(max_nodes):
            if i != j and i < n and j < n:
                continue
            edge_index.append([i, j])    
    edge_index = torch.tensor(edge_index)
    return edge_index.t().contiguous()

def sample(config, model, noise_scheduler, num_inference_steps=1000, n=1):

    assert 1 <= n <= config.max_n_nodes

    noise_scheduler.set_timesteps(num_inference_steps)

    x_shape = (config.max_n_nodes, config.in_node_nf)
    e_shape = (config.max_n_nodes * config.max_n_nodes,
                config.in_edge_nf)

    input_x = torch.randn(x_shape, device=config.device)
    input_e = torch.randn(e_shape, device=config.device)
    node_mask = torch.zeros((x_shape[0], 1), device=config.device)
    edge_mask = torch.zeros((e_shape[0], 1), device=config.device)
    node_mask[:n] = 1
    edge_mask[:n * (n - 1)] = 1 # loopless, directed graph
    
    edge_index = get_edge_index(max_nodes=config.max_n_nodes, n=n)
    edge_index = edge_index.to(device=config.device)

    for t in noise_scheduler.timesteps:
        input_x = input_x * node_mask
        input_e = input_e * edge_mask

        with torch.no_grad():
            node_noise_res, node_edge_res = model(x=input_x, 
                                                edge_attr=input_e,
                                                t=t.to(device=config.device),
                                                edge_index=edge_index,
                                                batch_size=1,
                                                node_mask=node_mask,
                                                edge_mask=edge_mask
                                                )

        input_x = noise_scheduler.step(node_noise_res, t, input_x, eta=config.eta).prev_sample
        input_e = noise_scheduler.step(node_edge_res, t, input_e, eta=config.eta).prev_sample

    input_x = input_x * node_mask
    input_e = input_e * edge_mask

    return input_x, input_e


