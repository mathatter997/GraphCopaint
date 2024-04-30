from dataclasses import dataclass

@dataclass
class GridConfig:
    max_n_nodes = None
    data_filepath = "data/dataset/"
    data_name = "grid"
    train_batch_size = 32
    eval_batch_size = 4  # how many to sample during evaluation
    num_epochs = 400000
    start_epoch = 0
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    save_model_epochs = 1000
    train_timesteps = 1000
    mixed_precision = "no"
    start = 0
    checkpoint_path = None

    model = 'pgsn'
    data_format = 'graph'
    # model = 'eigen'
    # data_format = 'eigen'

    output_dir = f"models/{data_name}/"  # the model name locally and on the HF Hub
    output_dir_gnn = "gnn/checkpoint_epoch_{}.pth"
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
    num_gnn_layers = 4
    size_cond = False
    embedding_type = "positional"
    rw_depth = 32
    graph_layer = "PosTransLayer"
    edge_th = 0.2
    heads = 8
    dropout=0.1
    attn_clamp = False
    beta_start = 0.0001
    beta_end = 0.02

    reflect=True
    zero_diagonal=True

    max_feat_num = 10
    
    # eigen params
    max_node_num = 361
    max_feat_num = 5
    conv = "GCN"
    num_heads = 4
    depth = 5
    adim = 32
    nhid = 32
    num_layers = 7
    num_linears = 2
    c_init = 2
    c_hid = 8
    c_final = 4

