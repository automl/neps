from typing import Tuple

import networkx as nx


def transform_to_undirected(gr: list):
    """Transform a list of directed graphs by undirected graphs."""
    undirected_gr = []
    for g in gr:
        if not isinstance(g, nx.Graph):
            continue
        if isinstance(g, nx.DiGraph):
            undirected_gr.append(g.to_undirected())
        else:
            undirected_gr.append(g)
    return undirected_gr


def extract_configs(configs: list) -> Tuple[list, list]:
    """Extracts graph & HPs from configs objects

    Args:
        configs (list): Object holding graph and/or HPs

    Returns:
        Tuple[list, list]: list of graphs, list of HPs
    """
    N = len(configs)
    if N > 0 and "get_graphs" in dir(configs[0]):
        graphs = [c.get_graphs() for c in configs]
    elif N > 0 and "graph" in dir(configs[0]):
        graphs = [c.graph for c in configs]
    elif N > 0 and isinstance(configs, list):  # assumes that the list is meaningful!
        graphs = configs
    else:
        graphs = [None] * N

    if N > 0 and "get_hps" in dir(configs[0]):
        hps = [c.get_hps() for c in configs]
    elif N > 0 and "hps" in dir(configs[0]):
        hps = [c.hps for c in configs]
    else:
        hps = [None] * N

    return graphs, hps
