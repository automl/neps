import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter_, softmax

### Node embedding


class GNNLayer(MessagePassing):
    def __init__(self, ndim):
        super(GNNLayer, self).__init__(aggr="add")
        self.msg = nn.Linear(ndim * 2, ndim * 2)
        self.msg_rev = nn.Linear(ndim * 2, ndim * 2)
        self.upd = nn.GRUCell(2 * ndim, ndim)

    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        m, m_reverse = torch.split(m, m.size(0) // 2, 0)
        a = torch.cat([self.msg(m), self.msg_rev(m_reverse)], dim=0)
        return a

    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h


class NodeEmb(nn.Module):
    def __init__(self, ndim, num_layers, node_dropout, dropout):
        super(NodeEmb, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.NodeInit = nn.Embedding(11, ndim)
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])

    def forward(self, edge_index, node_atts):
        h = self.NodeInit(node_atts)
        edge_index = torch.cat(
            [
                edge_index,
                torch.index_select(edge_index, 0, torch.tensor([1, 0]).to(h.device)),
            ],
            1,
        )
        for layer in self.GNNLayers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return h


class HypEmb(nn.Module):
    def __init__(self):
        super(HypEmb, self).__init__()
        self.fc1 = nn.Linear(5, 5)

    def forward(self, hps):
        hps = F.relu(self.fc1(hps))
        return hps


### Graph embedding


class GraphEmb(nn.Module):
    def __init__(self, ndim, sdim, aggr="gsum"):
        super(GraphEmb, self).__init__()
        self.ndim = ndim
        self.sdim = sdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim + 5, sdim)
        if aggr == "gsum":
            self.g_m = nn.Linear(ndim + 5, 1)
            self.sigm = nn.Sigmoid()

    def forward(self, h, batch):
        if self.aggr == "mean":
            h = self.f_m(h)
            return scatter_("mean", h, batch)  # no function exists by this name
        elif self.aggr == "gsum":
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h_G = torch.mul(h_vG, g_vG)
            return scatter_("add", h_G, batch)


class GNNEncoder(nn.Module):
    def __init__(
        self, ndim, sdim, num_gnn_layers=2, node_dropout=0.0, g_aggr="gsum", dropout=0.0
    ):
        super().__init__()
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, node_dropout, dropout)
        self.GraphEmb_mean = GraphEmb(ndim, sdim, g_aggr)
        self.GraphEmb_var = GraphEmb(ndim, sdim, g_aggr)

    def forward(self, edge_index, node_atts, batch):
        h = self.NodeEmb(edge_index, node_atts)
        h_G_mean = self.GraphEmb_mean(h, batch)
        h_G_var = self.GraphEmb_var(h, batch)
        return h_G_mean, h_G_var

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


### Accuracy Prediction


class GetAcc(nn.Module):
    def __init__(self, sdim, dim_target, num_layers, dropout):
        super(GetAcc, self).__init__()
        self.sdim = sdim
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_target = dim_target
        self.lin_layers = nn.ModuleList(
            [
                nn.Linear(sdim // (2 ** num), sdim // (2 ** (num + 1)))
                for num in range(num_layers - 1)
            ]
        )
        self.lin_layers.append(nn.Linear(sdim // (2 ** (num_layers - 1)), dim_target))

    def forward(self, h):
        for layer in self.lin_layers[:-1]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(layer(h))
        h = self.lin_layers[-1](h)
        return h.reshape(-1)

    def __repr__(self):
        return "{}({}x Linear) Dropout(p={})".format(
            self.__class__.__name__, self.num_layers, self.dropout
        )


class GNNpred_classifier(nn.Module):
    def __init__(self, dim_features, dim_target, model_config):
        super().__init__()
        ndim = dim_features
        num_gnn_layers = model_config["gnn_prop_layers"]
        num_classi_layers = model_config["num_classifier_layers"]
        num_acc_layers = model_config["num_regression_layers"]
        dropout = model_config["dropout"]
        sdim = model_config["dim_embedding"]
        g_aggr = "gsum"
        node_dropout = model_config["node_dropout"]
        nbins = model_config["no_bins"]
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, node_dropout, dropout)
        self.HypEmb = HypEmb()
        self.GraphEmb_mean = GraphEmb(ndim, sdim, g_aggr)

        self.class_lin_layers = nn.ModuleList(
            [
                nn.Linear(sdim // (2 ** num), sdim // (2 ** (num + 1)))
                for num in range(num_classi_layers - 1)
            ]
        )
        self.class_lin_layers.append(
            nn.Linear(sdim // (2 ** (num_classi_layers - 1)), nbins)
        )

        self.fc1 = nn.Linear(sdim + nbins, dim_target)

        self.lin_layers = nn.ModuleList(
            [
                nn.Linear(
                    (sdim + nbins) // (2 ** num), (sdim + nbins) // (2 ** (num + 1))
                )
                for num in range(num_acc_layers - 1)
            ]
        )
        self.lin_layers.append(
            nn.Linear((sdim + nbins) // (2 ** (num_acc_layers - 1)), 1)
        )

    def forward(self, graph_batch):
        node_atts, edge_index, batch, hyperparameters = (
            graph_batch.x.long(),
            graph_batch.edge_index.long(),
            graph_batch.batch,
            graph_batch.hyperparameters.float(),
        )

        hyperparameters = torch.reshape(
            hyperparameters, (int(hyperparameters.shape[0] / 5), 5)
        )
        hyperparameters = self.HypEmb(hyperparameters)
        hyperparameters = torch.cat([i.repeat(30, 1) for i in hyperparameters], 0)

        h = self.NodeEmb(edge_index, node_atts)
        h = torch.cat((h, hyperparameters), 1)

        h_G_mean = self.GraphEmb_mean(h, batch)

        for layer in self.class_lin_layers[:-1]:
            h = F.dropout(h_G_mean, p=0.0, training=self.training)
            h = F.relu(layer(h))
        h = F.softmax(self.class_lin_layers[-1](h))

        hc = torch.cat((h, h_G_mean), 1)
        import pdb

        #         pdb.set_trace()

        for layer in self.lin_layers[:-1]:
            hc = F.dropout(hc, p=0.0, training=self.training)
            hc = F.relu(layer(hc))
        hc = self.lin_layers[-1](hc)
        #         pdb.set_trace()

        #         hc=F.relu(self.fc1(hc))
        return h, hc.reshape(-1)


class GNNpred(nn.Module):
    def __init__(self, dim_features, dim_target, model_config):
        super().__init__()
        ndim = dim_features
        num_gnn_layers = model_config["gnn_prop_layers"]
        dropout = model_config["dropout"]
        sdim = model_config["dim_embedding"]
        g_aggr = "gsum"
        node_dropout = model_config["node_dropout"]
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, node_dropout, dropout)
        self.HypEmb = HypEmb()
        self.GraphEmb_mean = GraphEmb(ndim, sdim, g_aggr)
        self.Accuracy = GetAcc(
            sdim,
            dim_target,
            num_layers=model_config["num_regression_layers"],
            dropout=0.0,
        )

    def forward(self, graph_batch):
        node_atts, edge_index, batch, hyperparameters = (
            graph_batch.x.long(),
            graph_batch.edge_index.long(),
            graph_batch.batch,
            graph_batch.hyperparameters.float(),
        )

        hyperparameters = torch.reshape(
            hyperparameters, (int(hyperparameters.shape[0] / 5), 5)
        )
        hyperparameters = self.HypEmb(hyperparameters)
        hyperparameters = torch.cat([i.repeat(30, 1) for i in hyperparameters], 0)

        h = self.NodeEmb(edge_index, node_atts)
        h = torch.cat((h, hyperparameters), 1)

        h_G_mean = self.GraphEmb_mean(h, batch)

        acc = self.Accuracy(h_G_mean)
        return acc

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
