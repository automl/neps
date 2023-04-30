# type: ignore
import inspect
from collections import OrderedDict
from typing import Callable

import gpytorch
import torch

from ...search_spaces.architecture.graph import Graph
from ...search_spaces.hyperparameters.float import FloatParameter
from ...search_spaces.parameter import HpTensorShape
from ...search_spaces.search_space import SearchSpace
from .default_consts import LENGTHSCALE_SAFE_MARGIN
from .kernels.utils import extract_configs_hierarchy

DUMMY_VAL = -1
N_GRAPH_FEATURES = 2

# Patch and fixes for gpytorch


class SafeInterval(gpytorch.constraints.Interval):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.safe_lower_bound = self.lower_bound + torch.tensor(LENGTHSCALE_SAFE_MARGIN)
        self.safe_upper_bound = self.upper_bound - torch.tensor(LENGTHSCALE_SAFE_MARGIN)

    def inverse_transform(self, transformed_tensor):
        transformed_tensor = torch.minimum(transformed_tensor, self.safe_upper_bound)
        transformed_tensor = torch.maximum(transformed_tensor, self.safe_lower_bound)
        return super().inverse_transform(transformed_tensor)


def get_default_args(func: Callable) -> dict:
    signature = inspect.signature(func)
    out_dict = OrderedDict()
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            out_dict[k] = v.default
    return out_dict


class GpAuxData:
    """
    Store the auxiliary data and pre-/post-processing steps in a single class,
    """

    def __init__(self, pipeline_space, hierarchy_consider, d_graph_features):
        self.pipeline_space = pipeline_space
        self.hierarchy_consider = hierarchy_consider
        self.d_graph_features = d_graph_features
        self.d_graph_feature_hp_names = None
        self.extended_pipeline_space = None
        self.features = None

        self.hierarchical_hps = []
        self.hp_hierarchical_levels = {}

        self.all_hp_shapes = {}
        self.graph_structures = []
        self.tensor_size = 0
        self.idx_offset = 0

    def reset(self):
        self.tensor_size = 0
        self.all_hp_shapes = {}
        self.graph_structures = []
        self.idx_offset = 0

    def extend_hierarchical_space(self):
        """
        Extend pipeline space by hierarchical levels and graph feature

        This is used because the kernels are initialized based on the pipeline space
        """
        new_pipeline_space = dict()
        self.d_graph_feature_hp_names = []
        for hp_name, hp in dict(self.pipeline_space).items():
            new_pipeline_space.update({hp_name: hp})
            if isinstance(hp, Graph):
                if self.hierarchy_consider is not None:
                    self.hierarchical_hps.append(hp_name)
                    hierarchical_levels = self.hierarchical_levels(hp_name)
                    update_dict = dict(
                        zip(hierarchical_levels, [hp] * len(hierarchical_levels))
                    )
                    new_pipeline_space.update(update_dict)
                    self.hp_hierarchical_levels.update({hp_name: hierarchical_levels})
                    self.hierarchical_hps += hierarchical_levels

                if self.d_graph_features > 0:
                    # for feature in range(N_GRAPH_FEATURES):
                    d_graph_feature_hp_name = f"d_graph_feature_{0}"
                    self.d_graph_feature_hp_names.append(d_graph_feature_hp_name)
                    new_pipeline_space.update(
                        {d_graph_feature_hp_name: FloatParameter(lower=0, upper=1)}
                    )

        self.extended_pipeline_space = new_pipeline_space

    def build_input_tensor(
        self, x_configs: list[SearchSpace]
    ) -> tuple[torch.tensor, list[list[Graph]]]:
        """
        Build input tensor and extract graph data from configurations
        """
        x_tensor = (
            torch.ones(
                (len(x_configs), self.tensor_size), dtype=torch.get_default_dtype()
            )
            * DUMMY_VAL
        )
        if self.graph_structures is not None:
            x_graphs = [[] for _ in range(len(self.graph_structures))]
        else:
            x_graphs = None

        for i_sample, sample in enumerate(x_configs):
            graph_structure_idx = 0
            for hp_idx, (hp_name, hp_shape) in enumerate(self.all_hp_shapes.items()):
                # hp_shape = self.all_hp_shapes[hp_name]
                hp = sample.get(hp_name, None)
                if hp is not None:
                    if hp_idx in self.graph_structures:
                        x_graphs[graph_structure_idx].append(hp.get_tensor_value())
                        graph_structure_idx += 1
                    else:
                        x_tensor[
                            i_sample, hp_shape.begin : hp_shape.end
                        ] = hp.get_tensor_value(hp_shape)
                else:
                    pass

        if self.hierarchy_consider:
            x_graphs, self.features = extract_configs_hierarchy(
                x_configs, hierarchy_consider=self.hierarchy_consider, d_graph_features=0
            )
        # Disregard hierarchy information if there are any
        elif isinstance(x_graphs[0][0], list):
            x_graphs = [[value_list[0] for value_list in hps] for hps in x_graphs]

        return x_tensor, x_graphs

    def hierarchical_levels(self, hp_name):
        if hp_name in self.hp_hierarchical_levels:
            return self.hp_hierarchical_levels.get(hp_name)

        hierarchical_levels = []
        for h_level in self.hierarchy_consider:
            hierarchical_hp = hp_name + "_h_" + str(h_level)
            assert hierarchical_hp not in dict(self.pipeline_space).keys(), (
                f"Can't use both {hp_name} and {hierarchical_hp} as "
                f"parameter names, please change either one"
            )
            hierarchical_levels.append(hierarchical_hp)
        return hierarchical_levels

    def add_graph_hp_shape(self, hp_idx, hp_name, hp_instances):
        self.graph_structures.append(hp_idx)
        hp_shape = HpTensorShape(length=1, hp_instances=hp_instances)
        hp_shape.set_bounds(self.tensor_size)
        self.tensor_size = hp_shape.end
        self.all_hp_shapes[hp_name] = hp_shape
        self.idx_offset += 1

    def add_features_hp_shape(self, hp_name, hp_instances):
        hp_shape = HpTensorShape(length=N_GRAPH_FEATURES, hp_instances=hp_instances)
        hp_shape.set_bounds(self.tensor_size)
        self.tensor_size = hp_shape.end
        self.all_hp_shapes[hp_name] = hp_shape
        self.idx_offset += 1

    def add_hp(self, train_x, hp_idx, hp_name):
        """
        Add hyperparameter shapes if it's hierarchical add all hierarchical levels as well
        """
        hp_instances = [sample[hp_name] for sample in train_x]
        hp_shape = hp_instances[0].get_tensor_shape(hp_instances)
        if hp_shape is None:
            self.add_graph_hp_shape(hp_idx + self.idx_offset, hp_name, hp_instances)
            if self.hierarchy_consider:
                for hierarchical_level in self.hierarchical_levels(hp_name):
                    self.add_graph_hp_shape(
                        hp_idx + self.idx_offset, hierarchical_level, hp_instances
                    )
            if self.d_graph_features > 0:
                self.add_features_hp_shape(self.d_graph_feature_hp_names[0], None)

        else:
            hp_shape.set_bounds(self.tensor_size)
            self.tensor_size = hp_shape.end
            self.all_hp_shapes[hp_name] = hp_shape

    def insert_graph_data(self, x_graphs):
        """
        For each graph type data in the all_hp_shapes replace placeholders with the graphs
        """
        if self.graph_structures is not None:
            for hp_shape in self.all_hp_shapes.values():
                for active_dim in hp_shape.active_dims:
                    if active_dim in self.graph_structures:
                        idx = self.graph_structures.index(active_dim)
                        hp_shape.hp_instances = x_graphs[idx]


def update_default_args(func, **kwargs):
    _func_args_with_defaults = get_default_args(func)
    arg_names = list(kwargs.keys())
    for name in arg_names:
        assert (
            name in _func_args_with_defaults
        ), f"Argument '{name}' is not an argument in the given function"
    indices = [list(_func_args_with_defaults.keys()).index(name) for name in arg_names]

    new_values = []
    index = 0

    for idx, default in enumerate(func.__defaults__):
        if idx in indices:
            new_values.append(kwargs[arg_names[index]])
            index += 1
        else:
            new_values.append(default)
    func.__defaults__ = tuple(new_values)


update_default_args(
    gpytorch.settings.fast_computations.__init__,
    covar_root_decomposition=False,
    log_prob=False,
    solves=False,
)

# gpytorch.settings.verbose_linalg._set_state(True)


# Change the value of max_tries in cholesky
update_default_args(gpytorch.utils.cholesky.psd_safe_cholesky, max_tries=6)
