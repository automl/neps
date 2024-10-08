from __future__ import annotations

import copy
import inspect
import logging
import random
import types
from pathlib import Path
from more_itertools import collapse

import networkx as nx
import torch
from networkx.algorithms.dag import lexicographical_topological_sort
from torch import nn

from .primitives import AbstractPrimitive, Identity

logger = logging.getLogger(__name__)


class Graph(torch.nn.Module, nx.DiGraph):
    """Base class for defining a search space. Add nodes and edges
    as for a directed acyclic graph in `networkx`. Nodes can contain
    graphs as children, also edges can contain graphs as operations.

    Note, if a graph is copied, the shared attributes of its edges are
    shallow copies whereas the private attributes are deep copies.

    To differentiate copies of the same graph you can define a `scope`
    with `set_scope()`.

    **Graph at nodes:**
    >>> graph = Graph()
    >>> graph.add_node(1, subgraph=Graph())

    If the node has more than one input use `set_input()` to define the
    routing to the input nodes of the subgraph.

    **Graph at edges:**
    >>> graph = Graph()
    >>> graph.add_nodes_from([1, 2])
    >>> graph.add_edge(1, 2, EdgeData({'op': Graph()}))

    **Modify the graph after definition**

    If you want to modify the graph e.g. in an optimizer once
    it has been defined already use the function `update_edges()`
    or `update_nodes()`.

    **Use as pytorch module**
    If you want to learn the weights of the operations or any
    other parameters of the graph you have to parse it first.
    >>> graph = getFancySearchSpace()
    >>> graph.parse()
    >>> logits = graph(data)
    >>> optimizer.min(loss(logits, target))

    To update the pytorch module representation (e.g. after removing or adding
    some new edges), you have to unparse. Beware that this is not fast, so it should
    not be done on each batch or epoch, rather once after discretizising. If you
    want to change the representation of the graph use rather some shared operation
    indexing at the edges.
    >>> graph.update(remove_random_edges)
    >>> graph.unparse()
    >>> graph.parse()
    >>> logits = graph(data)

    """

    """
    Usually the optimizer does not operate on the whole graph, e.g. preprocessing
    and post-processing are excluded. Scope can be used to define that or to
    differentate instances of the "same" graph.
    """
    OPTIMIZER_SCOPE = "all"

    """
    Whether the search space has an interface to one of the tabular benchmarks which
    can then be used to query architecture performances.

    If this is set to true then `query()` should be implemented.
    """
    QUERYABLE = False

    def __init__(self, name: str | None = None, scope: str | None = None):
        """Initialise a graph. The edges are automatically filled with an EdgeData object
        which defines the default operation as Identity. The default combination operation
        is set as sum.

        Note:
            When inheriting form `Graph` note that `__init__()` cannot take any
            parameters. This is due to the way how networkx is implemented, i.e. graphs
            are reconstructed internally and no parameters for init are considered.

            Our recommended solution is to create static attributes before initialization
            and then load them dynamically in `__init__()`.

            >>> def __init__(self):
            >>>     num_classes = self.NUM_CLASSES
            >>> MyGraph.NUM_CLASSES = 42
            >>> my_graph_42_classes = MyGraph()

        """
        # super().__init__()
        nx.DiGraph.__init__(self)
        torch.nn.Module.__init__(self)

        # Make DiGraph a member and not inherit. This is because when inheriting from
        # `Graph` note that `__init__()` cannot take any parameters. This is due to
        # the way how networkx is implemented, i.e. graphs are reconstructed internally
        # and no parameters for init are considered.
        # Therefore __getattr__ and __iter__ forward the DiGraph methods for straight-forward
        # usage as if we would inherit.

        # self._nxgraph = nx.DiGraph()

        # Replace the default dicts at the edges with `EdgeData` objects
        # `EdgeData` can be easily customized and allow shared parameters
        # across different Graph instances.

        # self._nxgraph.edge_attr_dict_factory = lambda: EdgeData()
        self.edge_attr_dict_factory = lambda: EdgeData()

        # Replace the default dicts at the nodes to include `input` from the beginning.
        # `input` is required for storing the results of incoming edges.

        # self._nxgraph.node_attr_dict_factory = lambda: dict({'input': {}, 'comb_op': sum})
        self.node_attr_dict_factory = lambda: {"input": {}, "comb_op": sum}

        # remember to add all members also in `unparse()`
        self.name = name
        self.scope = scope
        self.input_node_idxs = None
        self.is_parsed = False
        self._id = random.random()  # pytorch expects unique modules in `add_module()`

    def __eq__(self, other):
        return self.name == other.name and self.scope == other.scope

    def __hash__(self):
        """As it is very complicated to compare graphs (i.e. check all edge
        attributes, do the have shared attributes, ...) use just the name
        for comparison.

        This is used when determining whether two instances are copies.
        """
        h = 0
        h += hash(self.name)
        h += hash(self.scope) if self.scope else 0
        h += hash(self._id)
        return h

    def __repr__(self):
        return f"Graph {self.name}-{self._id:.07f}, scope {self.scope}, {self.number_of_nodes()} nodes"

    def set_scope(self, scope: str, recursively=True):
        """Sets the scope of this instance of the graph.

        The function should be used in a builder-like pattern
        `'subgraph'=Graph().set_scope("scope")`.

        Args:
            scope (str): the scope
            recursively (bool): Also set the scope for all child graphs.
                default True

        Returns:
            Graph: self with the setted scope.
        """
        self.scope = scope
        if recursively:
            for g in self._get_child_graphs(single_instances=False):
                g.scope = scope
        return self

    def add_node(self, node_index, **attr):
        """Adds a node to the graph.

        Note that adding a node using an index that has been used already
        will override its attributes.

        Args:
            node_index (int): The index for the node. Expect to be >= 1.
            **attr: The attributes which can be added in a dict like form.
        """
        assert node_index >= 1, "Expecting the node index to be greater or equal 1"
        nx.DiGraph.add_node(self, node_index, **attr)

    def copy(self):
        """Copy as defined in networkx, i.e. a shallow copy.

        Just handling recursively nested graphs seperately.
        """

        def copy_dict(d):
            copied_dict = d.copy()
            for k, v in d.items():
                if isinstance(v, Graph):
                    copied_dict[k] = v.copy()
                elif isinstance(v, list):
                    copied_dict[k] = [i.copy() if isinstance(i, Graph) else i for i in v]
                elif isinstance(v, (AbstractPrimitive, torch.nn.Module)):
                    copied_dict[k] = copy.deepcopy(v)
            return copied_dict

        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, copy_dict(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
        )
        G.scope = self.scope
        G.name = self.name
        return G

    def to_pytorch(self, **kwargs) -> nn.Module:
        return self._to_pytorch(**kwargs)

    def _to_pytorch(self, write_out: bool = False) -> nn.Module:
        def _import_code(code: str, name: str):
            module = types.ModuleType(name)
            exec(code, module.__dict__)
            return module

        if not self.is_parsed:
            self.parse()

        input_node = next(n for n in self.nodes if self.in_degree(n) == 0)
        input_name = "x0"
        self.nodes[input_node]["input"] = {0: input_name}

        forward_f = []
        used_input_names = [int(input_name[1:])]
        submodule_list = []
        for node_idx in lexicographical_topological_sort(self):
            node = self.nodes[node_idx]
            if "subgraph" in node:
                # TODO implementation not checked yet!
                max_xidx = max(used_input_names)
                submodule = node["subgraph"].to_pytorch(write_out=write_out)
                submodule_list.append(submodule)
                _forward_f = f"x{max_xidx + 1}=self.module_list[{len(submodule_list) - 1}]({node['input']})"
                input_name = f"x{max_xidx + 1}"
                used_input_names.append(max_xidx + 1)
                forward_f.append(_forward_f)
                x = f"x{max_xidx + 1}"
            else:
                if len(node["input"].values()) == 1:
                    x = next(iter(node["input"].values()))
                else:
                    max_xidx = max(used_input_names)
                    if (
                        "__name__" in dir(node["comb_op"])
                        and node["comb_op"].__name__ == "sum"
                    ):
                        _forward_f = f"x{max_xidx + 1}=sum(["
                    elif isinstance(node["comb_op"], torch.nn.Module):
                        submodule_list.append(node["comb_op"])
                        _forward_f = f"x{max_xidx + 1}=self.module_list[{len(submodule_list) - 1}](["
                    else:
                        raise NotImplementedError

                    for inp in node["input"].values():
                        _forward_f += inp + ","
                    _forward_f = _forward_f[:-1] + "])"
                    forward_f.append(_forward_f)
                    x = f"x{max_xidx + 1}"
                if int(x[1:]) not in used_input_names:
                    used_input_names.append(int(x[1:]))
            node["input"] = {}  # clear the input as we have processed it
            if (
                len(list(self.neighbors(node_idx))) == 0
                and node_idx < list(lexicographical_topological_sort(self))[-1]
            ):
                # We have more than one output node. This is e.g. the case for
                # auxillary losses. Attach them to the graph, handling must done
                # by the user.
                raise NotImplementedError
            else:
                # outgoing edges: process all outgoing edges
                for neigbor_idx in self.neighbors(node_idx):
                    max_xidx = max(used_input_names)
                    edge_data = self.get_edge_data(node_idx, neigbor_idx)
                    # inject edge data only for AbstractPrimitive, not Graphs
                    if isinstance(edge_data.op, Graph):
                        submodule = edge_data.op.to_pytorch(write_out=write_out)
                        submodule_list.append(submodule)
                        _forward_f = f"x{max_xidx + 1}=self.module_list[{len(submodule_list) - 1}]({x})"
                        input_name = f"x{max_xidx + 1}"
                        used_input_names.append(max_xidx + 1)
                        forward_f.append(_forward_f)
                    elif isinstance(edge_data.op, AbstractPrimitive):
                        # edge_data.op.forward = partial(  # type: ignore[assignment]
                        #     edge_data.op.forward, edge_data=edge_data
                        # )
                        submodule_list.append(edge_data.op)
                        _forward_f = f"x{max_xidx + 1}=self.module_list[{len(submodule_list) - 1}]({x})"
                        input_name = f"x{max_xidx + 1}"
                        used_input_names.append(max_xidx + 1)
                        forward_f.append(_forward_f)
                    else:
                        raise ValueError(
                            f"Unknown class as op: {edge_data.op}. Expected either Graph or AbstactPrimitive"
                        )
                    self.nodes[neigbor_idx]["input"].update({node_idx: input_name})

        forward_f.append(f"return {x}")

        model_file = "# Auto generated\nimport torch\nimport torch.nn\n\nclass Model(torch.nn.Module):\n\tdef __init__(self):\n"
        model_file += "\t\tsuper().__init__()\n"
        model_file += "\t\tself.module_list=torch.nn.ModuleList()\n"
        model_file += "\n\tdef set_module_list(self,module_list):\n"
        model_file += "\t\tself.module_list=torch.nn.ModuleList(module_list)\n"
        model_file += "\n\tdef forward(self,x0,*args):\n"
        for forward_lines in forward_f:
            for forward_line in (
                [forward_lines] if isinstance(forward_lines, str) else forward_lines
            ):
                model_file += f"\t\t{forward_line}\n"

        try:
            module_model = _import_code(model_file, "model")
            model = module_model.Model()
        except Exception as e:
            raise Exception(e) from e

        model.set_module_list(submodule_list)

        if write_out:
            tmp_path = Path(__file__).parent.resolve() / "model.py"
            with open(tmp_path, "w", encoding="utf-8") as outfile:
                outfile.write(model_file)

        return model

    def parse(self):
        """Convert the graph into a neural network which can then
        be optimized by pytorch.
        """
        for node_idx in lexicographical_topological_sort(self):
            if "subgraph" in self.nodes[node_idx]:
                self.nodes[node_idx]["subgraph"].parse()
                self.add_module(
                    f"{self.name}-subgraph_at({node_idx})",
                    self.nodes[node_idx]["subgraph"],
                )
            elif isinstance(self.nodes[node_idx]["comb_op"], torch.nn.Module):
                self.add_module(
                    f"{self.name}-comb_op_at({node_idx})",
                    self.nodes[node_idx]["comb_op"],
                )

            for neigbor_idx in self.neighbors(node_idx):
                edge_data = self.get_edge_data(node_idx, neigbor_idx)
                if isinstance(edge_data.op, Graph):
                    edge_data.op.parse()
                elif edge_data.op.get_embedded_ops():
                    for primitive in edge_data.op.get_embedded_ops():
                        if isinstance(primitive, Graph):
                            primitive.parse()

                self.add_module(
                    f"{self.name}-edge({node_idx},{neigbor_idx})",
                    edge_data.op,
                )
        self.is_parsed = True

    def unparse(self):
        """Undo the pytorch parsing by reconstructing the graph uusing the
        networkx data structures.

        This is done recursively also for child graphs.

        Returns:
            Graph: An unparsed shallow copy of the graph.
        """
        g = self.__class__()
        g.clear()

        graph_nodes = self.nodes
        graph_edges = self.edges

        # unparse possible child graphs
        # be careful with copying/deepcopying here cause of shared edge data
        for _, data in graph_nodes.data():
            if "subgraph" in data:
                data["subgraph"] = data["subgraph"].unparse()
        for _, _, data in graph_edges.data():
            if isinstance(data.op, Graph):
                data.set("op", data.op.unparse())

        # create the new graph
        # Remember to add all members here to update. I know it is ugly but don't know better
        g.add_nodes_from(graph_nodes.data())
        g.add_edges_from(graph_edges.data())
        g.graph.update(self.graph)
        g.name = self.name
        g.input_node_idxs = self.input_node_idxs
        g.scope = self.scope
        g.is_parsed = False
        g._id = self._id
        g.OPTIMIZER_SCOPE = self.OPTIMIZER_SCOPE
        g.QUERYABLE = self.QUERYABLE

        return g

    def _get_child_graphs(self, single_instances: bool = False) -> list:
        """Get all child graphs of the current graph.

        Args:
            single_instances (bool): Whether to return multiple instances
                (i.e. copies) of the same graph. When changing shared data
                this should be set to True.

        Returns:
            list: A list of all child graphs (can be empty)
        """
        graphs = []
        for node_idx in lexicographical_topological_sort(self):
            node_data = self.nodes[node_idx]
            if "subgraph" in node_data:
                graphs.append(node_data["subgraph"])
                graphs.append(node_data["subgraph"]._get_child_graphs())

        for _, _, edge_data in self.edges.data():
            if isinstance(edge_data.op, Graph):
                graphs.append(edge_data.op)
                graphs.append(edge_data.op._get_child_graphs())
            elif isinstance(edge_data.op, list):
                for op in edge_data.op:
                    if isinstance(op, Graph):
                        graphs.append(op)
                        graphs.append(op._get_child_graphs())
            elif isinstance(edge_data.op, AbstractPrimitive):
                # maybe it is an embedded op?
                embedded_ops = edge_data.op.get_embedded_ops()
                if embedded_ops is not None:
                    if isinstance(embedded_ops, Graph):
                        graphs.append(embedded_ops)
                        graphs.append(embedded_ops._get_child_graphs())
                    elif isinstance(embedded_ops, list):
                        for child_op in edge_data.op.get_embedded_ops():
                            if isinstance(child_op, Graph):
                                graphs.append(child_op)
                                graphs.append(child_op._get_child_graphs())
                    else:
                        logger.debug(
                            f"Got embedded op, but is neither a graph nor a list: {embedded_ops}"
                        )
            elif inspect.isclass(edge_data.op):
                assert not issubclass(
                    edge_data.op, Graph
                ), "Found non-initialized graph. Abort."
                # we look at an uncomiled op
            elif callable(edge_data.op):
                pass
            else:
                raise ValueError(f"Unknown format of op: {edge_data.op}")

        graphs = list(collapse(graphs))

        if single_instances:
            single: list = []
            for g in graphs:
                if g.name not in [sg.name for sg in single]:
                    single.append(g)
            return sorted(single, key=lambda g: g.name)
        else:
            return sorted(graphs, key=lambda g: g.name)

    def compile(self):
        """Instanciates the ops at the edges using the arguments specified at the edges."""
        for graph in [*self._get_child_graphs(single_instances=False), self]:
            logger.debug(f"Compiling graph {graph.name}")
            for _, v, edge_data in graph.edges.data():
                if not edge_data.is_final():
                    attr = edge_data.to_dict()
                    op = attr.pop("op")

                    if isinstance(op, list):
                        compiled_ops = []
                        for i, o in enumerate(op):
                            if inspect.isclass(o):
                                # get the relevant parameter if there are more.
                                a = {
                                    k: v[i] if isinstance(v, list) else v
                                    for k, v in attr.items()
                                }
                                compiled_ops.append(o(**a))
                            else:
                                logger.debug(f"op {o} already compiled. Skipping")
                        edge_data.set("op", compiled_ops)
                    elif isinstance(op, AbstractPrimitive):
                        logger.debug(f"op {op} already compiled. Skipping")
                    elif inspect.isclass(op) and issubclass(op, AbstractPrimitive):
                        # Init the class
                        if "op_name" in attr:
                            del attr["op_name"]
                        edge_data.set("op", op(**attr))
                    elif isinstance(op, Graph):
                        pass  # This is already covered by _get_child_graphs
                    else:
                        raise ValueError(f"Unkown format of op: {op}")

    def clone(self):
        """Deep copy of the current graph.

        Returns:
            Graph: Deep copy of the graph.
        """
        return copy.deepcopy(self)


class EdgeData:
    """Class that holds data for each edge.
    Data can be shared between instances of the graph
    where the edges lives in.

    Also defines the default key 'op', which is `Identity()`. It must
    be private always.

    Items can be accessed directly as attributes with `.key` or
    in a dict-like fashion with `[key]`. To set a new item use `.set()`.
    """

    def __init__(self, data: dict | None = None):
        """Initializes a new EdgeData object.
        'op' is set as Identity() and private by default.

        Args:
            data (dict): Inject some initial data. Will be always private.
        """
        if data is None:
            data = {}
        self._private = {}
        self._shared = {}

        # set internal attributes
        self._shared["_deleted"] = False
        self._private["_final"] = False

        # set defaults and potential input
        self.set("op", Identity(), shared=False)
        for k, v in data.items():
            self.set(k, v, shared=False)

    def __getitem__(self, key: str):
        assert not str(key).startswith("_"), "Access to private keys not allowed!"
        return self.__getattr__(str(key))

    def get(self, key: str, default):
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def __getattr__(self, key: str):
        if key.startswith("__"):  # Required for deepcopy, not sure why
            raise AttributeError(key)
        assert not key.startswith("_"), "Access to private keys not allowed!"
        if key in self._private:
            return self._private[key]
        elif key in self._shared:
            return self._shared[key]
        else:
            raise AttributeError(f"Cannot find field '{key}' in the given EdgeData!")

    def __setattr__(self, name: str, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise ValueError("not allowed. use set().")

    def __str__(self):
        return f"private: <{self._private!s}>, shared: <{self._shared!s}>"

    def __repr__(self):
        return self.__str__()

    def update(self, data):
        """Update the data in here. If the data is added as dict,
        then all variables will be handled as private.

        Args:
            data (EdgeData or dict): If dict, then values will be set as
                private. If EdgeData then all entries will be replaced.
        """
        if isinstance(data, dict):
            for k, v in data.items():
                self.set(k, v)
        elif isinstance(data, EdgeData):
            # TODO: do update and not replace!
            self.__dict__.update(data.__dict__)
        else:
            raise ValueError(f"Unsupported type {data}")

    def remove(self, key: str):
        """Removes an item from the EdgeData.

        Args:
            key (str): The key for the item to be removed.
        """
        if key in self._private:
            del self._private[key]
        elif key in self._shared:
            del self._shared[key]
        else:
            raise KeyError(f"Tried to delete unkown key {key}")

    def copy(self):
        """When a graph is copied to get multiple instances (e.g. when
        reusing subgraphs at more than one location) then
        this function will be called for all edges.

        It will create a deep copy for the private entries but
        only a shallow copy for the shared entries. E.g. architectural
        weights should be shared, but parameters of a 3x3 convolution not.

        Therefore 'op' must be always private.

        Returns:
            EdgeData: A new EdgeData object with independent private
                items, but shallow shared items.
        """
        new_self = EdgeData()
        new_self._private = copy.deepcopy(self._private)
        new_self._shared = self._shared

        # we need to handle copy of graphs seperately
        for k, v in self._private.items():
            if isinstance(v, Graph):
                new_self._private[k] = v.copy()
            elif isinstance(v, list):
                new_self._private[k] = [
                    i.copy() if isinstance(i, Graph) else i for i in v
                ]

        return new_self

    def set(self, key: str, value, shared=False):
        """Used to assign a new item to the EdgeData object.

        Args:
            key (str): The key.
            value (object): The value to store
            shared (bool): Default: False. Whether the item should
                be a shallow copy between different instances of EdgeData
                (and consequently between different instances of Graph).
        """
        assert isinstance(key, str), f"Accepting only string keys, got {type(key)}"
        assert not key.startswith("_"), "Access to private keys not allowed!"
        assert not self.is_final(), "Trying to change finalized edge!"
        if shared:
            if key in self._private:
                raise ValueError("Key {} alredy defined as non-shared")
            else:
                self._shared[key] = value
        elif key in self._shared:
            raise ValueError(f"Key {key} alredy defined as shared")
        else:
            self._private[key] = value

    def clone(self):
        """Return a true deep copy of EdgeData. Even shared
        items are not shared anymore.

        Returns:
            EdgeData: New independent instance.
        """
        return copy.deepcopy(self)

    def is_final(self):
        """Returns:
        bool: True if the edge was finalized, False else.
        """
        return self._private["_final"]

    def to_dict(self, subset="all"):
        if subset == "shared":
            return {k: v for k, v in self._shared.items() if not k.startswith("_")}
        elif subset == "private":
            return {k: v for k, v in self._private.items() if not k.startswith("_")}
        elif subset == "all":
            d = self.to_dict("private")
            d.update(self.to_dict("shared"))
            return d
        else:
            raise ValueError(f"Unknown subset {subset}")
