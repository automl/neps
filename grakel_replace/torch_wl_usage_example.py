from __future__ import annotations

import networkx as nx
import torch
from torch_wl_kernel import GraphDataset, BoTorchWLKernel

# Create the same graphs as for the Grakel example
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3)])
G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
G3 = nx.Graph()
G3.add_edges_from([(0, 1), (1, 3), (3, 2)])

# Process graphs
graphs: list[nx.Graph] = GraphDataset.from_networkx([G1, G2, G3])

# Initialize and run WL kernel
wl_kernel = BoTorchWLKernel(
    training_graph_list=graphs,
    n_iter=2,
    normalize=True,
    active_dims=(1,),
)
X1 = torch.tensor([[42.4, 43.4, 44.5], [0, 1, 2]]).T
X2 = torch.tensor([[42.4, 43.4, 44.5], [0, 1, 2]]).T

K = wl_kernel(X1, X2)
print(K.to_dense())  # noqa: T201
