import networkx as nx
import numpy as np

from grakel.utils import graph_from_networkx


def find_node(gr, att, val):
    """Applicable for the first-layer WL feature (i.e. the nodes themselves)"""
    return len([node for node in gr.nodes(data=True) if node[1][att] == val])


def find_2_structure(gr, att, encoding):
    """Applicable for the second-layer WL features (i.e. the nodes + their 1-neighbours).
    This is actually faulty. Do not use this line."""
    if "~" in encoding:
        encoding = encoding.split("~")
        encoding = [(e,) for e in encoding]
    root_node = encoding[0][0]
    leaf_node = [encoding[e][0] for e in range(1, len(encoding))]
    counter = {x: leaf_node.count(x) for x in set(leaf_node)}
    counts = []
    for node in gr.nodes(data=True):
        if node[1][att] == root_node:
            count = {x: 0 for x in set(leaf_node)}
            for neighbor in nx.neighbors(gr, node[0]):
                if gr.nodes[neighbor][att] in leaf_node:
                    count[gr.nodes[neighbor][att]] += 1
            counts.append(count)
    for c in counts:
        if c == counter:
            return True
    return False


def find_wl_feature(
    test,
    feature,
    kernel,
):
    """Return the number of occurrence of --feature-- in --test--, based on a --kernel--."""
    if not isinstance(test, list):
        test = [test]
    test = graph_from_networkx(
        test,
        "op_name",
    )

    feat_map = kernel.feature_map(flatten=False)
    len_feat_map = [len(f) for f in feat_map.values()]
    try:
        idx = list(kernel.feature_map(flatten=True).values()).index(feature[0])
    except KeyError:
        raise KeyError(
            "Feature " + str(feature) + " is not found in the training set of the kernel!"
        )
    embedding = kernel.kern.transform(test, return_embedding_only=True)
    for i, em in enumerate(embedding):
        embedding[i] = em.flatten()[: len_feat_map[i]]
    return np.hstack(embedding)[idx]
