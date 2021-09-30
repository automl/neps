import inspect
from abc import ABCMeta

from .graph import Graph


class AbstractTopology(Graph, metaclass=ABCMeta):
    edge_list = None

    def __init__(self, name: str = None, scope: str = None):
        super().__init__(name=name, scope=scope)

    def mutate(self):
        pass

    def sample_random_architecture(self):
        pass

    def create_graph(self, vals: dict):
        def get_default_args(func):
            signature = inspect.signature(func)
            return {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        def get_op_name_from_dict(val: dict):
            # currently assumes that missing args are ints!
            op = val["op"]
            args = {}
            default_args = get_default_args(op)
            for varname in op.__init__.__code__.co_varnames:
                if varname == "self" or varname == "kwargs" or varname in args.keys():
                    continue
                if varname in val.keys():
                    args[varname] = val[varname]
                elif varname in default_args.keys():
                    args[varname] = default_args[varname]
                else:
                    args[varname] = 42
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
                else:
                    _val = {"op": val, "op_name": val.get_op_name}
            self.edges[u, v].update(_val)

    @property
    def get_op_name(self):
        return type(self).__name__


class Residual(AbstractTopology):
    edge_list = [
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "residual"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Diamond(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "diamond"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)
