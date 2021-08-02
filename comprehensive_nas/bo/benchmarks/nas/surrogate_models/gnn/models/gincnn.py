import torch
import torch.nn.functional as F

from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool


"""
Code from official repository of "A Fair Comparison of Graph Neural Networks for Graph Classification", ICLR 2020
https://github.com/diningphil/gnn-comparison under GNU General Public License v3.0
"""


class GIN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, model_config):
        super(GIN, self).__init__()

        self.config = model_config
        self.dropout = model_config["dropout_prob"]
        hidden_units = [
            model_config["gnn_hidden_dimensions"]
            for _ in range(model_config["num_gnn_layers"])
        ]
        self.embeddings_dim = [hidden_units[0]] + hidden_units
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = model_config["train_eps"]

        # TOTAL NUMBER OF PARAMETERS #

        # first: dim_features*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target
        # l-th: input_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*out_emb_dim + 4*out_emb_dim + out_emb_dim*target

        # -------------------------- #

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(
                    Sequential(
                        Linear(input_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                        Linear(out_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                    )
                )
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        # self.first_h = torch.nn.ModuleList(self.first_h)
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(
            self.linears
        )  # has got one more for initial input

    def forward(self, graph_batch):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = (
            graph_batch.x.float(),
            graph_batch.edge_index.long(),
            graph_batch.batch,
        )

        out = 0
        pooling = global_mean_pool

        for layer in range(self.no_layers):
            # print(f'Forward: layer {l}')
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout)
        out = torch.sigmoid(out.view(-1))
        return out
