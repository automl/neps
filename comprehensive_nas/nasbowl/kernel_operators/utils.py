import networkx as nx
import numpy as np


def get_dataset_attributes(Gn,
                           target=None,
                           attr_names=[],
                           node_label=None,
                           edge_label=None):
    """Returns the structure and property information of the graph dataset Gn.
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of x1_graphs whose information will be returned.
    target : list
        The list of classification targets corresponding to Gn. Only works for
        classification problems.
    attr_names : list
        List of strings which indicate which informations will be returned. The
        possible choices includes:
        'substructures': sub-structures Gn contains, including 'linear', 'non
            linear' and 'cyclic'.
        'node_labeled': whether vertices have symbolic labels.
        'edge_labeled': whether egdes have symbolic labels.
        'is_directed': whether x1_graphs in Gn are directed.
        'dataset_size': number of x1_graphs in Gn.
        'ave_node_num': average number of vertices of x1_graphs in Gn.
        'min_node_num': minimum number of vertices of x1_graphs in Gn.
        'max_node_num': maximum number of vertices of x1_graphs in Gn.
        'ave_edge_num': average number of edges of x1_graphs in Gn.
        'min_edge_num': minimum number of edges of x1_graphs in Gn.
        'max_edge_num': maximum number of edges of x1_graphs in Gn.
        'ave_node_degree': average vertex degree of x1_graphs in Gn.
        'min_node_degree': minimum vertex degree of x1_graphs in Gn.
        'max_node_degree': maximum vertex degree of x1_graphs in Gn.
        'ave_fill_factor': average fill factor (number_of_edges /
            (number_of_nodes ** 2)) of x1_graphs in Gn.
        'min_fill_factor': minimum fill factor of x1_graphs in Gn.
        'max_fill_factor': maximum fill factor of x1_graphs in Gn.
        'node_label_num': number of symbolic vertex labels.
        'edge_label_num': number of symbolic edge labels.
        'node_attr_dim': number of dimensions of non-symbolic vertex labels.
            Extracted from the 'attributes' attribute of graph nodes.
        'edge_attr_dim': number of dimensions of non-symbolic edge labels.
            Extracted from the 'attributes' attribute of graph edges.
        'class_number': number of classes. Only available for classification
            problems.
    node_label : string
        Node attribute used as label. The default node label is atom. Mandatory
        when 'node_labeled' or 'node_label_num' is required.
    edge_label : string
        Edge attribute used as label. The default edge label is bond_type.
        Mandatory when 'edge_labeled' or 'edge_label_num' is required.
    Return
    ------
    attrs : dict
        Value for each property.
    """
    import networkx as nx
    import numpy as np

    attrs = {}

    def get_dataset_size(Gn):
        return len(Gn)

    def get_all_node_num(Gn):
        return [nx.number_of_nodes(G) for G in Gn]

    def get_ave_node_num(all_node_num):
        return np.mean(all_node_num)

    def get_min_node_num(all_node_num):
        return np.amin(all_node_num)

    def get_max_node_num(all_node_num):
        return np.amax(all_node_num)

    def get_all_edge_num(Gn):
        return [nx.number_of_edges(G) for G in Gn]

    def get_ave_edge_num(all_edge_num):
        return np.mean(all_edge_num)

    def get_min_edge_num(all_edge_num):
        return np.amin(all_edge_num)

    def get_max_edge_num(all_edge_num):
        return np.amax(all_edge_num)

    def is_node_labeled(Gn):
        return False if node_label is None else True

    def get_node_label_num(Gn):
        nl = set()
        for G in Gn:
            nl = nl | set(nx.get_node_attributes(G, node_label).values())
        return len(nl)

    def is_edge_labeled(Gn):
        return False if edge_label is None else True

    def get_edge_label_num(Gn):
        el = set()
        for G in Gn:
            el = el | set(nx.get_edge_attributes(G, edge_label).values())
        return len(el)

    def is_directed(Gn):
        return nx.is_directed(Gn[0])

    def get_ave_node_degree(Gn):
        return np.mean([np.mean(list(dict(G.degree()).values())) for G in Gn])

    def get_max_node_degree(Gn):
        return np.amax([np.mean(list(dict(G.degree()).values())) for G in Gn])

    def get_min_node_degree(Gn):
        return np.amin([np.mean(list(dict(G.degree()).values())) for G in Gn])

    # get fill factor, the number of non-zero entries in the adjacency matrix.
    def get_ave_fill_factor(Gn):
        return np.mean([nx.number_of_edges(G) / (nx.number_of_nodes(G)
                                                 * nx.number_of_nodes(G)) for G in Gn])

    def get_max_fill_factor(Gn):
        return np.amax([nx.number_of_edges(G) / (nx.number_of_nodes(G)
                                                 * nx.number_of_nodes(G)) for G in Gn])

    def get_min_fill_factor(Gn):
        return np.amin([nx.number_of_edges(G) / (nx.number_of_nodes(G)
                                                 * nx.number_of_nodes(G)) for G in Gn])

    def get_substructures(Gn):
        subs = set()
        for G in Gn:
            degrees = list(dict(G.degree()).values())
            if any(i == 2 for i in degrees):
                subs.add('linear')
            if np.amax(degrees) >= 3:
                subs.add('non linear')
            if 'linear' in subs and 'non linear' in subs:
                break

        if is_directed(Gn):
            for G in Gn:
                if len(list(nx.find_cycle(G))) > 0:
                    subs.add('cyclic')
                    break

        return subs

    def get_class_num(target):
        return len(set(target))

    def get_node_attr_dim(Gn):
        for G in Gn:
            for n in G.nodes(data=True):
                if 'attributes' in n[1]:
                    return len(n[1]['attributes'])
        return 0

    def get_edge_attr_dim(Gn):
        for G in Gn:
            if nx.number_of_edges(G) > 0:
                for e in G.edges(data=True):
                    if 'attributes' in e[2]:
                        return len(e[2]['attributes'])
        return 0

    if attr_names == []:
        attr_names = [
            'substructures',
            'node_labeled',
            'edge_labeled',
            'is_directed',
            'dataset_size',
            'ave_node_num',
            'min_node_num',
            'max_node_num',
            'ave_edge_num',
            'min_edge_num',
            'max_edge_num',
            'ave_node_degree',
            'min_node_degree',
            'max_node_degree',
            'ave_fill_factor',
            'min_fill_factor',
            'max_fill_factor',
            'node_label_num',
            'edge_label_num',
            'node_attr_dim',
            'edge_attr_dim',
            'class_number',
        ]

    # dataset size
    if 'dataset_size' in attr_names:
        attrs.update({'dataset_size': get_dataset_size(Gn)})

    # graph node number
    if any(i in attr_names
           for i in ['ave_node_num', 'min_node_num', 'max_node_num']):
        all_node_num = get_all_node_num(Gn)

    if 'ave_node_num' in attr_names:
        attrs.update({'ave_node_num': get_ave_node_num(all_node_num)})

    if 'min_node_num' in attr_names:
        attrs.update({'min_node_num': get_min_node_num(all_node_num)})

    if 'max_node_num' in attr_names:
        attrs.update({'max_node_num': get_max_node_num(all_node_num)})

    # graph edge number
    if any(i in attr_names for i in
           ['ave_edge_num', 'min_edge_num', 'max_edge_num']):
        all_edge_num = get_all_edge_num(Gn)

    if 'ave_edge_num' in attr_names:
        attrs.update({'ave_edge_num': get_ave_edge_num(all_edge_num)})

    if 'max_edge_num' in attr_names:
        attrs.update({'max_edge_num': get_max_edge_num(all_edge_num)})

    if 'min_edge_num' in attr_names:
        attrs.update({'min_edge_num': get_min_edge_num(all_edge_num)})

    # label number
    if any(i in attr_names for i in ['node_labeled', 'node_label_num']):
        is_nl = is_node_labeled(Gn)
        node_label_num = get_node_label_num(Gn)

    if 'node_labeled' in attr_names:
        # x1_graphs are considered node unlabeled if all nodes have the same label.
        attrs.update({'node_labeled': is_nl if node_label_num > 1 else False})

    if 'node_label_num' in attr_names:
        attrs.update({'node_label_num': node_label_num})

    if any(i in attr_names for i in ['edge_labeled', 'edge_label_num']):
        is_el = is_edge_labeled(Gn)
        edge_label_num = get_edge_label_num(Gn)

    if 'edge_labeled' in attr_names:
        # x1_graphs are considered edge unlabeled if all edges have the same label.
        attrs.update({'edge_labeled': is_el if edge_label_num > 1 else False})

    if 'edge_label_num' in attr_names:
        attrs.update({'edge_label_num': edge_label_num})

    if 'is_directed' in attr_names:
        attrs.update({'is_directed': is_directed(Gn)})

    if 'ave_node_degree' in attr_names:
        attrs.update({'ave_node_degree': get_ave_node_degree(Gn)})

    if 'max_node_degree' in attr_names:
        attrs.update({'max_node_degree': get_max_node_degree(Gn)})

    if 'min_node_degree' in attr_names:
        attrs.update({'min_node_degree': get_min_node_degree(Gn)})

    if 'ave_fill_factor' in attr_names:
        attrs.update({'ave_fill_factor': get_ave_fill_factor(Gn)})

    if 'max_fill_factor' in attr_names:
        attrs.update({'max_fill_factor': get_max_fill_factor(Gn)})

    if 'min_fill_factor' in attr_names:
        attrs.update({'min_fill_factor': get_min_fill_factor(Gn)})

    if 'substructures' in attr_names:
        attrs.update({'substructures': get_substructures(Gn)})

    if 'class_number' in attr_names:
        attrs.update({'class_number': get_class_num(target)})

    if 'node_attr_dim' in attr_names:
        attrs['node_attr_dim'] = get_node_attr_dim(Gn)

    if 'edge_attr_dim' in attr_names:
        attrs['edge_attr_dim'] = get_edge_attr_dim(Gn)

    from collections import OrderedDict
    return OrderedDict(
        sorted(attrs.items(), key=lambda i: attr_names.index(i[0])))


import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import sys


def parallel_me(func, func_assign, var_to_assign, itr, len_itr=None, init_worker=None,
                glbv=None, method=None, n_jobs=None, chunksize=None, itr_desc='',
                verbose=True):
    '''
    '''
    if method == 'imap_unordered':
        if glbv:  # global varibles required.
            #            def init_worker(v_share):
            #                global G_var
            #                G_var = v_share
            if n_jobs is None:
                n_jobs = multiprocessing.cpu_count()
            with Pool(processes=n_jobs, initializer=init_worker,
                      initargs=glbv) as pool:
                if chunksize is None:
                    if len_itr < 100 * n_jobs:
                        chunksize = int(len_itr / n_jobs) + 1
                    else:
                        chunksize = 100
                for result in (tqdm(pool.imap_unordered(func, itr, chunksize),
                                    desc=itr_desc, file=sys.stdout) if verbose else
                pool.imap_unordered(func, itr, chunksize)):
                    func_assign(result, var_to_assign)
        else:
            if n_jobs is None:
                n_jobs = multiprocessing.cpu_count()
            with Pool(processes=n_jobs) as pool:
                if chunksize is None:
                    if len_itr < 100 * n_jobs:
                        chunksize = int(len_itr / n_jobs) + 1
                    else:
                        chunksize = 100
                for result in (tqdm(pool.imap_unordered(func, itr, chunksize),
                                    desc=itr_desc, file=sys.stdout) if verbose else
                pool.imap_unordered(func, itr, chunksize)):
                    func_assign(result, var_to_assign)


def parallel_gm(func, Kmatrix, Gn, init_worker=None, glbv=None,
                method='imap_unordered', n_jobs=None, chunksize=None,
                verbose=True):
    from itertools import combinations_with_replacement

    def func_assign(result, var_to_assign):
        var_to_assign[result[0]][result[1]] = result[2]
        var_to_assign[result[1]][result[0]] = result[2]

    itr = combinations_with_replacement(range(0, len(Gn)), 2)
    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
    parallel_me(func, func_assign, Kmatrix, itr, len_itr=len_itr,
                init_worker=init_worker, glbv=glbv, method=method, n_jobs=n_jobs,
                chunksize=chunksize, itr_desc='calculating kernel_operators', verbose=verbose)


def getSPGraph(G, edge_weight=None):
    """Transform graph G to its corresponding shortest-paths graph.
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight.
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
    Notes
    ------
    For an input graph G, its corresponding shortest-paths graph S contains the same set of nodes as G, while there
    exists an edge between all nodes in S which are connected by a walk in G. Every edge in S between two nodes is
    labeled by the shortest distance between these two nodes.
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernel_operators on x1_graphs. InData Mining, Fifth IEEE International Conference
    on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    return floydTransformation(G, edge_weight=edge_weight)


def floydTransformation(G, edge_weight=None):
    """Transform graph G to its corresponding shortest-paths graph using Floyd-transformation.
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernel_operators on x1_graphs. InData Mining, Fifth IEEE International Conference
    on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    spMatrix = nx.floyd_warshall_numpy(G, weight=edge_weight)
    S = nx.Graph()
    S.add_nodes_from(G.nodes(data=True))
    ns = list(G.nodes())
    for i in range(0, G.number_of_nodes()):
        for j in range(i + 1, G.number_of_nodes()):
            if spMatrix[i, j] != np.inf:
                S.add_edge(ns[i], ns[j], cost=spMatrix[i, j])
    return S


# Augmented x1_graphs for GNTK
class S2VGraph(object):
    def __init__(self, g: nx.Graph, label: int, node_tags: list = None, max_node_tag: int = 5):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags. if this is not supplied, we use the degree of each node as the node
            tag of that degree.
            We then apply one-hot encoding to transform node_tags to node_features
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.neighbors, self.max_neighbor = None, None
        if node_tags:
            assert len(node_tags) == len(g.nodes), 'mistmatch between the node_tag list and the number of nodes'
        self.node_tags = node_tags
        self.max_node_tag = max_node_tag
        self.node_features = self._get_node_features()
        self._get_neighbours()

    def _get_neighbours(self):
        self.neighbors = [[] for _ in range(len(self.g))]
        for i, j in self.g.edges():
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
        degree_list = []
        for i in range(len(self.g)):
            degree_list.append(len(self.neighbors[i]))
        self.max_neighbor = np.max(degree_list)

    def _get_node_features(self):
        if not self.node_tags:  # Node tag is not supplied - use degree of each node as the node tag
            try:
                node_tags = list(nx.get_node_attributes(self.g, 'op_name').values())
            except:
                node_tags = [self.g.degree[i] for i in self.g.nodes]
        else:
            node_tags = np.array(self.node_tags).flatten()
        node_tags = np.array(node_tags).astype(int)
        try:
            b = np.zeros((len(self.g.nodes), max(int(np.max(node_tags) + 1), self.max_node_tag))).astype(int)
        except:
            print('hold')
        b[np.arange(node_tags.size), node_tags] = 1
        return b


# Some networkx functions

def add_node_with_attributes(g: nx.Graph, node_id: int, attr: dict) -> nx.Graph:
    if node_id not in g.nodes():
        g.add_node(node_id)
    attrs = {node_id: attr}
    nx.set_node_attributes(g, attrs)
    return g


import torch


# Compute kernel distance
def unscaled_dist(X, X2=None, sqrt=True):
    """Unscaled distance corresponds to the Euclidean distance"""
    if X2 is None: X2 = X.clone()
    if X.ndimension() != 2: X = X.reshape(-1, 1)
    if X2.ndimension() != 2: X2 = X2.reshape(-1, 1)
    X1sq = torch.sum(X ** 2, 1)
    X2sq = torch.sum(X2 ** 2, 1)
    r2 = -2 * X @ X2.t() + (X1sq[:, None] + X2sq[None, :])
    r2.clamp_min_(0.)
    # elif type == 'oa':
    #     # optimal assignment - histogram intersection
    #     r2 = torch.sum(torch.min(X, X2))
    # else:
    #     raise ValueError(type + " is not understood. Possible types: ['euclidean', 'dirac', 'oa']")
    return torch.sqrt(r2) if sqrt else r2


def scaled_dist(ard_lengthscales, X, X2=None, sqrt=True):
    """Scaled distance in the case of ARD lengthscales correspond to the Mahalanobis distance"""
    if X2 is None: X2 = X.clone()
    return unscaled_dist(X / ard_lengthscales, X2 / ard_lengthscales, sqrt=sqrt)


def unscaled_dist_oa():
    pass


def histogram_dict_to_tensor(label2freq: dict, ndim: int, boolean=False):
    """Convert the histogram dictionary to a corresponding tensor
    label2freq: a dict in the form of the e.g. of {0:1, 3:1, 1:1}. Key is the index of the active dimension and the
    value is the histogram frequency. Inactive dimension is omitted in this representation.
    ndim: the resulting dimensionality of the vector
    result:
    e.g.
    given dict of {0:1, 3:1, 1:2}, the resulting tensor is [1, 2, 0, 1]
    """
    # print(label2freq)
    vector = [0] * ndim
    for k, v in label2freq.items():
        if boolean:
            vector[int(k)-1] = 1 if v else 0
        else:
            vector[int(k)-1] = v
    return torch.tensor(vector)


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
