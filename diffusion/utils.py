import torch
import torch.nn.functional as F
from torch_scatter import scatter
_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels, *args, **kwargs):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data (Adjacency matrices).
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
                for different models.
            mask: Mask for adjacency matrices.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels, *args, **kwargs)
        else:
            model.train()
            return model(x, labels, *args, **kwargs)

    return model_fn

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


@torch.no_grad()
def mask_adj2node(adj_mask):
    """Convert batched adjacency mask matrices to batched node mask matrices.

    Args:
        adj_mask: [B, N, N] Batched adjacency mask matrices without self-loop edge.

    Output:
        node_mask: [B, N] Batched node mask matrices indicating the valid nodes.
    """

    batch_size, max_num_nodes, _ = adj_mask.shape

    node_mask = adj_mask[:, 0, :].clone()
    node_mask[:, 0] = 1

    return node_mask


@torch.no_grad()
def get_rw_feat(k_step, dense_adj):
    """Compute k_step Random Walk for given dense adjacency matrix."""

    rw_list = []
    deg = dense_adj.sum(-1, keepdims=True)
    AD = dense_adj / (deg + 1e-8)
    rw_list.append(AD)

    for _ in range(k_step):
        rw = torch.bmm(rw_list[-1], AD)
        rw_list.append(rw)
    rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

    rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
    rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

    # get the shortest path distance indices
    tmp_rw = rw_map.sort(dim=1)[0]
    spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

    spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step+1).to(torch.float)
    spd_onehot = spd_onehot.permute(0, 3, 1, 2)  # [B, kstep, N, N]

    return rw_landing, spd_onehot


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    """Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    Returns:
        adj: [batch_size, max_num_nodes, max_num_nodes] Dense adjacency matrices.
        mask: Mask for dense adjacency matrices.
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    node_idx = torch.arange(batch.size(0), dtype=torch.long, device=edge_index.device)
    node_idx = (node_idx - cum_nodes[batch]) + (batch * max_num_nodes)
    mask = torch.zeros(batch_size * max_num_nodes, dtype=adj.dtype, device=adj.device)
    mask[node_idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    mask = mask[:, None, :] * mask[:, :, None]

    return adj, mask

def dense_adj(graph_data, max_num_nodes, scaler=None, dequantization=False):
    """Convert PyG DataBatch to dense adjacency matrices.

    Args:
        graph_data: DataBatch object.
        max_num_nodes: The size of the output node dimension.
        scaler: Data normalizer.
        dequantization: uniform dequantization.

    Returns:
        adj: Dense adjacency matrices.
        mask: Mask for adjacency matrices.
    """

    adj, adj_mask = to_dense_adj(graph_data.edge_index, graph_data.batch, max_num_nodes=max_num_nodes)  # [B, N, N]
    if dequantization:
        noise = torch.rand_like(adj)
        noise = torch.tril(noise, -1)
        noise = noise + noise.transpose(1, 2)
        adj = (noise + adj) / 2.
    adj = scaler(adj[:, None, :, :])
    # set diag = 0 in adj_mask
    adj_mask = torch.tril(adj_mask, -1)
    # symmetric
    adj_mask = adj_mask + adj_mask.transpose(-1, -2)

    return adj, adj_mask[:, None, :, :]


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    # print("in mask_adjs")
    # print("flags:", flags.shape)
    # print("adjs:", adjs.shape)
    # for i in range(flags.shape[0]):
    #     print("flags[i]:",flags[i])
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

def init_eigen(adj, max_feat_num, max_num_nodes, num_nodes):
    flag = torch.zeros(max_num_nodes, device=adj.device)
    flag[:num_nodes] = 1
    x = torch.sum(adj > 0, dim=-1)
    x = F.one_hot(x, max_feat_num)
    x = x * flag[:, None]
    la, u = torch.linalg.eigh(adj)
    return adj, x, la, u, flag

