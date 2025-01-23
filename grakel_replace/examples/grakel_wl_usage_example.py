from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from grakel import WeisfeilerLehman, graph_from_networkx


def visualize_graph(G):
    """Visualize the NetworkX graph."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue")
    plt.show()


def add_labels(G):
    """Add labels to the nodes of the graph."""
    for node in G.nodes():
        G.nodes[node]['label'] = str(node)


# Create graphs
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3)])
add_labels(G1)

G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
add_labels(G2)

G3 = nx.Graph()
G3.add_edges_from([(0, 1), (1, 3), (3, 2)])
add_labels(G3)

# Visualize the graphs
visualize_graph(G1)
visualize_graph(G2)
visualize_graph(G3)

# Convert NetworkX graphs to Grakel format using graph_from_networkx
graph_list = list(
    graph_from_networkx([G1, G2, G3], node_labels_tag="label", as_Graph=True)
)

# Initialize the Weisfeiler-Lehman kernel
wl_kernel = WeisfeilerLehman(n_iter=5, normalize=False)

# Compute the kernel matrix
K = wl_kernel.fit_transform(graph_list)

# Display the kernel matrix
print("Fit and Transform on Kernel matrix (pairwise similarities):")
print(K)
