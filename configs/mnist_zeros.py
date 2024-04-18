from dataclasses import dataclass

@dataclass
class MnistZerosConfig:
    max_n_nodes = None
    data_filepath = "data/dataset/"
    data_name = "mnist_zeros"
    train_batch_size = 32
    eval_batch_size = 32  # how many to sample during evaluation
    num_epochs = 400000
    start_epoch = 0
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    save_model_epochs = 100
    train_timesteps = 1000
    mixed_precision = "no"
    start = 0
    checkpoint_path = None
    output_dir = f"models/{data_name}/"  # the model name locally and on the HF Hub
    output_dir_gnn = "gnn/checkpoint_epoch_{}.pth"
    label = f"_t{train_timesteps}_psgn"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = (
        True  # overwrite the old model when re-running the notebook
    )
    seed = 0
    ema_rate = 0.9999
    attn_clamp = False
    beta_start = 0.0001
    beta_end = 0.005

    data_format = 'pixel'
