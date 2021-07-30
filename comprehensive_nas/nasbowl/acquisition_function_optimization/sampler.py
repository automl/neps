import ConfigSpace
import numpy as np

from comprehensive_nas.nasbowl.benchmarks.hpo.branin2 import *
from comprehensive_nas.nasbowl.benchmarks.hpo.counting_ones import *
from comprehensive_nas.nasbowl.benchmarks.hpo.hartmann3 import *
from comprehensive_nas.nasbowl.benchmarks.hpo.hartmann6 import *
from comprehensive_nas.nasbowl.benchmarks.nas.nasbench201 import *
from comprehensive_nas.nasbowl.benchmarks.nas.nasbench301 import *


def create_nas301_graph(adjacency_matrix, edge_attributes):
    rand_arch = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    nx.set_edge_attributes(rand_arch, edge_attributes)
    for i in rand_arch.nodes:
        rand_arch.nodes[i]["op_name"] = "1"
    rand_arch.graph_type = "edge_attr"
    return rand_arch


class Sampler:
    def __init__(self, args, objective):
        self.dataset = args.dataset
        self.optimize_arch = args.optimize_arch
        self.optimize_hps = args.optimize_hps
        self.pool_strategy = args.pool_strategy

        self.dataset = args.dataset
        if self.dataset in [
            "nasbench201",
            "nasbench301",
            "branin2",
            "hartmann3",
            "hartmann6",
            "counting_ones",
        ]:
            self.default_config = (
                objective.get_config_space().get_default_configuration().get_dictionary()
            )
        else:
            raise NotImplementedError

    def sample(self, pool_size):

        pool_graphs = []
        pool_hps = []
        nasbench201_op_label_list = []
        while len(pool_graphs) < pool_size:

            if self.dataset == "nasbench301":

                # Sample configuration
                nas301_cs = NASBench301.get_config_space()
                config = nas301_cs.sample_configuration()

                config_dict = config.get_dictionary()

                # Store graphs
                # matrix of vertices x vertices (6x6)
                normal_adjacency_matrix = np.zeros(
                    (VERTICES_301, VERTICES_301), dtype=np.int8
                )
                reduce_adjacency_matrix = np.zeros(
                    (VERTICES_301, VERTICES_301), dtype=np.int8
                )
                # Store hps
                hyperparameters = [np.nan for _ in range(HPS_301)]
                i = 0

                normal_edge_attributes = {}
                reduce_edge_attributes = {}
                for key, item in config_dict.items():
                    if "edge" in key:
                        x, y = edge_to_coord_mapping[int(key.split("_")[-1])]
                        if "normal" in key:
                            normal_edge_attributes[(x, y)] = {"op_name": config_dict[key]}
                            normal_adjacency_matrix[x][y] = OPS_301.index(item) + 1
                        else:
                            reduce_edge_attributes[(x, y)] = {"op_name": config_dict[key]}
                            reduce_adjacency_matrix[x][y] = OPS_301.index(item) + 1
                    # For now hps should be defined using prefix 'hyperparam' 'OptSel'
                    # Translate to vector while keeping categorical as str for the hamming kernel
                    if "hyperparam" in key:
                        hp = nas301_cs.get_hyperparameter(key)
                        if type(hp) == ConfigSpace.OrdinalHyperparameter:
                            nlevels = len(hp.sequence)
                            hyperparameters[i] = hp.sequence.index(config[key]) / nlevels
                        elif type(hp) == ConfigSpace.CategoricalHyperparameter:
                            nlevels = len(hp.choices)
                            hyperparameters[i] = str(
                                hp.choices.index(config[key]) / nlevels
                            )
                        else:
                            val = config[key]
                            bounds = (hp.lower, hp.upper)
                            if hp.log:
                                hyperparameters[i] = np.log(val / bounds[0]) / np.log(
                                    bounds[1] / bounds[0]
                                )
                            else:
                                hyperparameters[i] = (config[key] - bounds[0]) / (
                                    bounds[1] - bounds[0]
                                )
                        i += 1

                normal_rand_arch = create_nas301_graph(
                    normal_adjacency_matrix, normal_edge_attributes
                )
                reduce_rand_arch = create_nas301_graph(
                    reduce_adjacency_matrix, reduce_edge_attributes
                )

                rand_arch = [normal_rand_arch, reduce_rand_arch]
                rand_hps = hyperparameters

            elif self.dataset == "nasbench201":
                # generate random architecture for nasbench201
                nas201_cs = NASBench201.get_config_space()
                config = nas201_cs.sample_configuration()

                rand_hps = None
                rand_arch = None

                if self.optimize_arch:
                    op_labeling = [
                        config["edge_%d" % i] for i in range(len(config.keys()))
                    ]
                    # skip only duplicating architecture
                    if op_labeling in nasbench201_op_label_list:
                        continue

                    nasbench201_op_label_list.append(op_labeling)
                    rand_arch = create_nasbench201_graph(op_labeling)

                    # IN Nasbench201, it is possible that invalid graphs consisting entirely from None and skip-line are
                    # generated; remove these invalid architectures.

                    # Also remove if the number of edges is zero. This is is possible, one example in NAS-Bench-201:
                    # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|none~2|'
                    if len(rand_arch) == 0 or rand_arch.number_of_edges() == 0:
                        continue

                if self.optimize_hps:
                    rand_hps = []
                    config_dict = config.get_dictionary()
                    for key, item in config_dict.items():
                        hp = nas201_cs.get_hyperparameter(key)
                        nlevels = len(hp.choices)
                        rand_hps.append(str(hp.choices.index(config[key]) / nlevels))

            elif self.dataset == "branin2":
                rand_arch = None
                b2_cs = Branin2.get_config_space()
                config = b2_cs.sample_configuration()
                rand_hps = list(config.get_dictionary().values())

            elif self.dataset == "hartmann3":
                rand_arch = None
                h3_cs = Hartmann3.get_config_space()
                config = h3_cs.sample_configuration()
                rand_hps = list(config.get_dictionary().values())

            elif self.dataset == "hartmann6":
                rand_arch = None
                h6_cs = Hartmann6.get_config_space()
                config = h6_cs.sample_configuration()
                rand_hps = list(config.get_dictionary().values())

            elif self.dataset == "counting_ones":
                rand_arch = None
                co_cs = CountingOnes.get_config_space()
                config = co_cs.sample_configuration()
                rand_hps = list(map(str, config.get_dictionary().values()))[
                    :N_CATEGORICAL
                ]
                rand_hps += list(config.get_dictionary().values())[N_CATEGORICAL:]

            pool_graphs.append(rand_arch)
            pool_hps.append(rand_hps)

        return (
            pool_graphs if self.optimize_arch else [None] * pool_size,
            pool_hps if self.optimize_hps else [None] * pool_size,
        )

    def set_defaults(self, config):
        raise NotImplementedError
        # if not self.optimize_arch:
        #     for param, value in self.default_config.items():
        #         # Override the architectural parameters from the normal and reduction cell
        #         if 'normal' in param or 'reduce' in param:
        #             config[param] = value
        # if not self.optimize_hp:
        #     for param, value in self.default_config.items():
        #         # Override hyperparameters
        #         if not ('normal' in param or 'reduce' in param):
        #             config[param] = value
        #
        # return config
