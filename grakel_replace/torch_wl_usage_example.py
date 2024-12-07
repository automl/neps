import networkx as nx
from torch_wl_kernel import GraphDataset, TorchWLKernel

# Create the same graphs as for the Grakel example
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3)])
G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
G3 = nx.Graph()
G3.add_edges_from([(0, 1), (1, 3), (3, 2)])

# Process graphs
graphs = GraphDataset.from_networkx([G1, G2, G3])

# Initialize and run WL kernel
wl_kernel = TorchWLKernel(n_iter=2, normalize=False)

K = wl_kernel(graphs)

print("Kernel matrix (pairwise similarities):")
print(K)
