import inspect
import queue
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable

from .graph import Graph


class AbstractTopology(Graph, metaclass=ABCMeta):
    edge_list: list = []

    def __init__(self, name: str = None, scope: str = None, merge_fn: Callable = sum):
        super().__init__(name=name, scope=scope)

        self.merge_fn = merge_fn

    def mutate(self):
        pass

    def sample(self):
        pass

    def create_graph(self, vals: dict):
        def get_args_and_defaults(func):
            signature = inspect.signature(func)
            return list(signature.parameters.keys()), {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        def get_op_name_from_dict(val: dict):
            # currently assumes that missing args are ints!
            op = val["op"]
            args: dict = {}
            arg_names, default_args = get_args_and_defaults(op)
            for arg_name in arg_names:
                if arg_name == "self" or arg_name == "kwargs" or arg_name in args.keys():
                    continue
                if arg_name in val.keys():
                    args[arg_name] = val[arg_name]
                elif arg_name in default_args.keys():
                    args[arg_name] = default_args[arg_name]
                else:
                    args[arg_name] = 42

            if "groups" in args and args["groups"] != 1:
                args["C_in"] = args["groups"]
                args["C_out"] = args["groups"]

            return op(**args).get_op_name

        assert isinstance(vals, dict)
        for (u, v), val in vals.items():
            self.add_edge(u, v)
            if isinstance(val, dict):
                _val = val
                _val["op_name"] = get_op_name_from_dict(val)
            else:
                if isinstance(val, int):  # for synthetic benchmarks
                    _val = {"op": val, "op_name": val}
                elif hasattr(val, "get_op_name"):
                    _val = {"op": val, "op_name": val.get_op_name}
                elif callable(val):
                    _val = {"op": val, "op_name": val.__name__}
                else:
                    raise Exception(f"Cannot extract op name from {val}")

            self.edges[u, v].update(_val)

    @property
    def get_op_name(self):
        return type(self).__name__

    def __call__(self, x):
        cur_node_idx = [node for node in self.nodes if self.in_degree(node) == 0][0]
        predecessor_inputs = {cur_node_idx: [x]}
        next_successors = queue.Queue()
        next_successors.put(cur_node_idx)
        cur_successors = queue.Queue()
        inputs = None
        while not cur_successors.empty() or not next_successors.empty():
            if not cur_successors.empty():
                next_node_idx = cur_successors.get(block=False)
                if next_node_idx not in next_successors.queue:
                    next_successors.put(next_node_idx)
                if next_node_idx not in predecessor_inputs:
                    predecessor_inputs[next_node_idx] = []
                predecessor_inputs[next_node_idx].append(
                    self.edges[(cur_node_idx, next_node_idx)].op(inputs)
                )
            else:
                cur_node_idx = next_successors.get(block=False)
                if self.out_degree(cur_node_idx) > 0:
                    for successor in self.successors(cur_node_idx):
                        cur_successors.put(successor)

                if len(predecessor_inputs[cur_node_idx]) == 1:
                    inputs = predecessor_inputs[cur_node_idx][0]
                else:
                    inputs = self.merge_fn(predecessor_inputs[cur_node_idx])
        return inputs


class AbstractVariableTopology(AbstractTopology):
    def __init__(self, name: str = None, scope: str = None, **kwargs):
        super().__init__(name, scope, **kwargs)

    @staticmethod
    @abstractmethod
    def get_edge_list(**kwargs):
        raise NotImplementedError


class _SequentialNEdge(AbstractTopology):
    edge_list: list = []

    def __init__(self, *edge_vals, number_of_edges: int, **kwargs):
        super().__init__(**kwargs)

        self.name = f"sequential_{number_of_edges}_edges"
        self.edge_list = self.get_edge_list(number_of_edges=number_of_edges)
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)

    @staticmethod
    def get_edge_list(number_of_edges: int):
        return [(i + 1, i + 2) for i in range(number_of_edges)]


LinearNEdge = _SequentialNEdge


def get_sequential_n_edge(number_of_edges: int):
    return partial(_SequentialNEdge, number_of_edges=number_of_edges)


class Residual(AbstractTopology):
    edge_list = [
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    def __init__(self, *edge_vals, **kwargs):
        super().__init__(**kwargs)

        self.name = "residual"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Diamond(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals, **kwargs):
        super().__init__(**kwargs)

        self.name = "diamond"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class DiamondMid(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals, **kwargs):
        super().__init__(**kwargs)

        self.name = "diamond_mid"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class _DenseNNodeDAG(AbstractTopology):
    edge_list: list = []

    def __init__(self, *edge_vals, number_of_nodes: int, **kwargs):
        super().__init__(**kwargs)

        self.edge_list = self.get_edge_list(number_of_nodes=number_of_nodes)

        self.name = f"dense_{number_of_nodes}_node_dag"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)

    @staticmethod
    def get_edge_list(number_of_nodes: int):
        return [(i + 1, j + 1) for j in range(number_of_nodes) for i in range(j)]


def get_dense_n_node_dag(number_of_nodes: int):
    return partial(_DenseNNodeDAG, number_of_nodes=number_of_nodes)
