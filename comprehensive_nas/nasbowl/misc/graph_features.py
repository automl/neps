import networkx as nx


class FeatureExtractor:
    """
    Extracting some hand-crafted x1_features for the x1_graphs
    - Number of (effective nodes)
    - Average
    """

    def __init__(self, g: nx.Graph, node_attr_name='op_name', s='input', t='output'):
        """
        g: a valid networkx graph
        node_attr_name: the tag of the node attribute. default is 'op_name'
        s, t: the tag of the two special input and output nodes. Note that there can be more than one input node (s), but
        only one output node (t)
        """
        self.g = g
        self.input_index = []
        self.output_index = None
        for n in range(g.number_of_nodes()):
            assert node_attr_name in list(dict(g.nodes[n]).keys()), node_attr_name + " is not found in " + str(
                g.nodes[n])
            if str(g.nodes[n][node_attr_name]) == str(s):
                self.input_index.append(n)
            elif str(g.nodes[n][node_attr_name]) == str(t):
                self.output_index = n
            self.node_attr_name = node_attr_name
        if len(self.input_index) == 0:
            raise ValueError("Unknown input node!")
        elif self.output_index is None:
            raise ValueError("Unknown output node!")
        # Specify the special nodes (i.e. the input and output, source and sink)
        if isinstance(self.g, nx.DiGraph):
            self.undirected_g = self.g.to_undirected()
        else:
            self.undirected_g = self.g

    def __getattr__(self, item):
        """Identify the feature already implemented in the graph class"""
        try:
            res = getattr(self.g, item)
        except AttributeError:
            raise AttributeError("Item" + str(item) + ' is not found either in the feature extractor nor the graph'
                                                      'instance!')
        if callable(res):
            return res()
        return res

    def _paths(self) -> list:
        """Enumerate all paths from input to output. Return a list of lists with each sub-list the node indices from
        the input to output

        Data shape:
        (N_input x2 N_path x2 length of each path)
        for SISO graph, the data shape is (1 x2 N_path x2 length of each path)

        """
        if not isinstance(self.g, nx.DiGraph):
            raise TypeError("Longest path is only applicable for directed graph!")
        result = []
        for i in self.input_index:
            result.append(list(nx.all_simple_paths(self.g, i, self.output_index)))
        return result

    @property
    def number_of_paths(self):
        paths = self._paths()
        if len(paths) == 1:
            return len(paths[0])
        return [len(i) for i in paths]

    @property
    def longest_path(self):
        """Return the longest path from input to output. the return type is a list in case when there is more than one
        input node."""
        paths = self._paths()
        if len(paths) == 1:  # if the list is a singlet (i.e. the S-T style graph), then return a scalar output only
            return len(max(paths[0], key=lambda x: len(x)))
        return [len(max(i, key=lambda x: len(x))) for i in paths]

    @property
    def degree_distribution(self, normalize=False):
        """
        return the degree distribution of the *undirected* counterpart of the graph, if the graph is directed.
        return a dictionary in the form of ((D1, N1), (D2, N2)... ) where Di is the degree and Ni is the frequency
        """
        from collections import Counter
        degree_seq = sorted([d for d, n in dict(self.undirected_g.degree)], reverse=True)
        degree_count = Counter(degree_seq)
        deg, cnt = zip(*degree_count.items())
        if normalize:
            n = self.undirected_g.number_of_nodes()
            cnt //= n
        return deg, cnt

    @property
    def laplacian_spectrum(self, ):
        return nx.normalized_laplacian_spectrum(self.undirected_g)

    @property
    def average_undirected_degree(self):
        return sum(dict(self.undirected_g.degree).values()) / (self.undirected_g.number_of_nodes() + 0.0)

    @property
    def number_of_conv3x3(self):
        i = 0
        for node, attr in self.g.nodes(data=True):
            if attr['op_name'] == 'conv3x3-bn-relu':
                i += 1
        return i
