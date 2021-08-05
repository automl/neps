try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import global_add_pool
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

"""
Code from official repository of "A Fair Comparison of Graph Neural Networks for Graph Classification", ICLR 2020
https://github.com/diningphil/gnn-comparison under GNU General Public License v3.0
"""


class DeepMultisets(torch.nn.Module):
    def __init__(self, dim_features, dim_target, model_config):
        super(DeepMultisets, self).__init__()

        hidden_units = model_config["gnn_hidden_dimensions"]

        self.fc_vertex = Linear(dim_features, hidden_units)
        self.fc_global1 = Linear(hidden_units, hidden_units)
        self.fc_global2 = Linear(hidden_units, dim_target)

    def forward(self, graph_batch):
        x, edge_index, batch = (
            graph_batch.x.float(),
            graph_batch.edge_index.long(),
            graph_batch.batch,
        )
        x = F.relu(self.fc_vertex(x))
        x = global_add_pool(
            x, batch
        )  # sums all vertex embeddings belonging to the same graph!
        x = F.relu(self.fc_global1(x))
        x = self.fc_global2(x)
        return torch.sigmoid(x.view(-1))
