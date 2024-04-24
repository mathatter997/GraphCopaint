from dataclasses import dataclass

@dataclass
class EgoSmallConfig:
    max_n_nodes = None
    data_filepath = "data/dataset/"
    data_name = "Ego_small"
    train_batch_size = 32
    eval_batch_size = 32  # how many to sample during evaluation
    num_epochs = 400000
    start_epoch = 0
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    save_model_epochs = 10000
    train_timesteps = 1000
    mixed_precision = "no"
    start = 0
    checkpoint_path = None

    output_dir = f"models/{data_name}/"  # the model name locally and on the HF Hub
    output_dir_gnn = "gnn/checkpoint_epoch_{}.pth"
    model = 'pgsn'
    # model = 'eigen'
    data_format = 'graph'
    # data_format = 'eigen'
    label = f"_t{train_timesteps}_{model}"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = (
        True  # overwrite the old model when re-running the notebook
    )
    seed = 0
    ema_rate = 0.9999
    normalization = "GroupNorm"
    nonlinearity = "swish"
    nf = 256
    num_gnn_layers = 3
    size_cond = False
    embedding_type = "positional"
    rw_depth = 8
    graph_layer = "PosTransLayer"
    edge_th = -1
    heads = 8
    dropout=0.1
    attn_clamp = False
    beta_start = 0.0001
    beta_end = 0.02

    reflect=True
    zero_diagonal=True

    # eigen params
    max_node_num = 18
    max_feat_num = 17
    conv = "GCN"
    num_heads = 4
    depth = 3
    adim = 32
    nhid = 32
    num_layers = 5
    num_linears = 2
    c_init = 2
    c_hid = 8
    c_final = 4
