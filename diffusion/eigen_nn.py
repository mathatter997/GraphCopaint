import torch
from diffusion.gnns import ScoreNetworkA_eigen, ScoreNetworkX, ScoreNetworkTest_eigen


class EigenNN(torch.nn.Module):
    def __init__(
        self,
        max_feat_num,
        nhid,
        max_node_num,
        num_layers,
        num_linears,
        c_init,
        c_hid,
        c_final,
        adim,
        depth=3,
        num_heads=4,
        conv="GCN",
    ):
        super(EigenNN, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList(
            [   
                ScoreNetworkX(max_feat_num, depth, nhid),
                ScoreNetworkA_eigen(
                    max_feat_num,
                    nhid,
                    max_node_num,
                    num_layers,
                    num_linears,
                    c_init,
                    c_hid,
                    c_final,
                    adim,
                    depth=depth,
                    num_heads=num_heads,
                    conv=conv,
                ),
            ]
        )

    def forward(self, x, adj, flags, u, la, t):
        ex = self.layers[0](x, adj, flags, u, la)
        # ela = self.layers[1](x, adj, flags, u, la, t)
        ela = self.layers[1](x, adj, flags, u, la)
        return ex, ela
