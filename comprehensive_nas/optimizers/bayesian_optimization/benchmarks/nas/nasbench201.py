# For use of the NAS-Bench-201 dataset version NAS-Bench-201-v1_0-e61699.pth

import random

import ConfigSpace
import networkx as nx
import numpy as np

from ..abstract_benchmark import AbstractBenchmark

MAX_EDGES_201 = None
VERTICES_201 = None
OPS_201 = ["nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3", "skip_connect", "none"]


class NASBench201(AbstractBenchmark):
    def __init__(
        self,
        task="cifar10-valid",
        log_scale=True,
        negative=True,
        hp="12",
        seed=None,
        optimize_arch=False,
        optimize_hps=True,
    ):
        """
        data_dir: data directory that contains NAS-Bench-201-v1_0-e61699.pth file
        task: the target image tasks. Options: cifar10-valid, cifar100, ImageNet16-120
        log_scale: whether output the objective in log scale
        negative: whether output the objective in negative form
        use_12_epochs_result: whether use the statistics at the end of training of the 12th epoch instead of all the
                              way till the end.
        seed: set the random seed to access trained model performance: Options: 0, 1, 2
              seed=None will select the seed randomly
        """

        # self.api = API(os.path.join(data_dir, 'NAS-Bench-201-v1_1-096897.pth', verbose=False)
        if isinstance(task, list):
            task = task[0]
        self.task = task
        self.hp = hp

        if task == "cifar10-valid":
            best_val_arch_index = 6111  # pylint: disable=unused-variable
            best_val_acc = 91.60666665039064 / 100
            best_test_arch_index = 1459  # pylint: disable=unused-variable
            best_test_acc = 91.52333333333333 / 100
        elif task == "cifar100":
            best_val_arch_index = 9930  # pylint: disable=unused-variable
            best_val_acc = 73.49333323567708 / 100
            best_test_arch_index = 9930  # pylint: disable=unused-variable
            best_test_acc = 73.51333326009114 / 100
        elif task == "ImageNet16-120":
            best_val_arch_index = 10676  # pylint: disable=unused-variable
            best_val_acc = 46.766666727701825 / 100
            best_test_arch_index = 857  # pylint: disable=unused-variable
            best_test_acc = 47.311111097547744 / 100
        else:
            raise NotImplementedError(
                "task" + str(task) + " is not implemented in the dataset."
            )

        if log_scale:
            best_val_acc = np.log(best_val_acc)

        best_val_err = 1.0 - best_val_acc
        best_test_err = 1.0 - best_test_acc
        if log_scale:
            best_val_err = np.log(best_val_err)
            best_test_err = np.log(best_val_err)
        if negative:
            best_val_err = -best_val_err
            best_test_err = -best_test_err

        self.best_val_err = best_val_err
        self.best_test_err = best_test_err
        self.best_val_acc = best_val_acc
        self.best_test_acc = best_test_acc

        super().__init__(seed, negative, log_scale, optimize_arch, optimize_hps)
        self.has_continuous_hp = False
        self.has_categorical_hp = self.optimize_hps

        self.X = []
        self.y_valid_acc = []
        self.y_test_acc = []
        self.costs = []
        # self.optimal_val =   # lowest mean validation error
        # self.y_star_test =   # lowest mean test error

    def _retrieve(self, which="eval", **kwargs):
        #  set random seed for evaluation
        if which == "test":
            seed = 3
        else:
            seed_list = [777, 888, 999]
            if self.seed is None:
                seed = random.choice(seed_list)
            elif self.seed >= 3:
                seed = self.seed
            else:
                seed = seed_list[self.seed]

        if self.graph is None:
            op_labeling = dict()
            for i, hp in enumerate(self.get_config_space().get_hyperparameters()):

                ranges = np.arange(start=0, stop=1, step=1 / len(hp.choices))
                # TODO: Fix this pylint
                # pylint: disable=singleton-comparison
                val = hp.choices[np.where((float(self.hps[i]) < ranges) == False)[0][-1]]
                # pylint: enable=singleton-comparison
                op_labeling[hp.name] = val

            op_node_labeling = [
                op_labeling["edge_%d" % i] for i in range(len(op_labeling.keys()))
            ]
            # skip only duplicating architecture
            graph = create_nasbench201_graph(op_node_labeling)
            arch_str = graph.name
        else:
            arch_str = self.graph.name

        # find architecture index
        # print(arch_str)

        try:
            api = kwargs["dataset_api"]
            arch_index = api.query_index_by_arch(arch_str)
            acc_results = api.query_by_index(
                arch_index,
                self.task,
                hp=self.hp,
            )
            if seed is not None and 3 <= seed < 777:
                # some architectures only contain 1 seed result
                acc_results = api.get_more_info(
                    arch_index, self.task, None, hp=self.hp, is_random=False
                )
                val_acc = acc_results["valid-accuracy"] / 100
                test_acc = acc_results["test-accuracy"] / 100
            else:
                try:
                    acc_results = api.get_more_info(
                        arch_index, self.task, None, hp=self.hp, is_random=seed
                    )
                    val_acc = acc_results["valid-accuracy"] / 100
                    test_acc = acc_results["test-accuracy"] / 100
                    # val_acc = acc_results[seed].get_eval('x-valid')['accuracy'] / 100
                    # if self.task == 'cifar10-valid':
                    #     test_acc = acc_results[seed].get_eval('ori-test')['accuracy'] / 100
                    # else:
                    #     test_acc = acc_results[seed].get_eval('x-test')['accuracy'] / 100
                except:  # pylint: disable=bare-except
                    # some architectures only contain 1 seed result
                    acc_results = api.get_more_info(
                        arch_index, self.task, None, hp=self.hp, is_random=False
                    )
                    val_acc = acc_results["valid-accuracy"] / 100
                    test_acc = acc_results["test-accuracy"] / 100

            auxiliary_info = api.query_meta_info_by_index(arch_index, hp=self.hp)
            cost_info = auxiliary_info.get_compute_costs(self.task)

            # auxiliary cost results such as number of flops and number of parameters
            cost_results = {
                "flops": cost_info["flops"],
                "params": cost_info["params"],
                "latency": cost_info["latency"],
            }

        except FileNotFoundError:
            val_acc = 0.01
            test_acc = 0.01
            print("missing arch info")
            cost_results = {"flops": None, "params": None, "latency": None}

        # store val and test performance + auxiliary cost information
        self.X.append(arch_str)
        self.y_valid_acc.append(val_acc)
        self.y_test_acc.append(test_acc)
        self.costs.append(cost_results)

        if which == "eval":
            err = 1 - val_acc
        elif which == "test":
            err = 1 - test_acc
        else:
            raise ValueError("Unknown query parameter: which = " + str(which))

        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y
        return y, {"train_time": cost_results["flops"]}

    def query(self, mode='eval', n_repeat=1, **kwargs):
        if mode == "test":
                return self.test(n_repeat=n_repeat)[0]
        else:
            if n_repeat == 1:
                return self._retrieve("eval", **kwargs)
            return np.mean(
                np.array([self._retrieve("eval", **kwargs) for _ in range(n_repeat)])
            )

    def sample_random_architecture(self):
        nas201_cs = NASBench201.get_config_space()

        if self.optimize_arch:
            nasbench201_op_label_list = []
            while True:
                # generate random architecture for nasbench201
                config = nas201_cs.sample_configuration()

                op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
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
                break
            self.graph = rand_arch  # pylint: disable=attribute-defined-outside-init
            self.name = str(self.parse())

        if self.optimize_hps:
            rand_hps = []
            config = nas201_cs.sample_configuration()
            config_dict = config.get_dictionary()
            for key, _ in config_dict.items():
                hp = nas201_cs.get_hyperparameter(key)
                nlevels = len(hp.choices)
                rand_hps.append(str(hp.choices.index(config[key]) / nlevels))
            self.hps = rand_hps  # pylint: disable=attribute-defined-outside-init

    def test(self, n_repeat=1):
        return np.mean(np.array([self._retrieve("test") for _ in range(n_repeat)]))

    def get_results(self):

        regret_validation = []
        regret_test = []
        costs = []
        model_graph_specs = []

        inc_valid = 0
        inc_test = 0

        for i in range(len(self.X)):

            if inc_valid < self.y_valid_acc[i]:
                inc_valid = self.y_valid_acc[i]
                inc_test = self.y_test_acc[i]

            regret_validation.append(float(self.best_val_acc - inc_valid))
            regret_test.append(float(self.best_test_acc - inc_test))
            model_graph_specs.append(self.X[i])
            costs.append(self.costs[i])

        res = dict()
        res["regret_validation"] = regret_validation
        res["regret_test"] = regret_test
        res["costs"] = costs
        res["model_graph_specs"] = model_graph_specs

        return res

    @staticmethod
    def get_config_space():
        # for unpruned graph
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = [
            "nor_conv_3x3",
            "nor_conv_1x1",
            "avg_pool_3x3",
            "skip_connect",
            "none",
        ]
        for i in range(6):
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices)
            )
        return cs

    @staticmethod
    def get_meta_information():
        return {
            "name": "NASBench201",
            "capital": 100,
            "optima": [None],  # best_test_arch_index
            "bounds": [None],
            "f_opt": [None],  # best_test_err,
            "noise_variance": 0.05,
        }

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative  # pylint: disable=attribute-defined-outside-init
        self.seed = seed  # pylint: disable=attribute-defined-outside-init


# class NAS201edge(NASBench201):
#     def _retrieve(self, G, budget, which="eval"):
#         #  set random seed for evaluation
#         seed_list = [777, 888, 999]
#         if self.seed is None:
#             seed = random.choice(seed_list)
#         elif self.seed >= 3:
#             seed = self.seed
#         else:
#             seed = seed_list[self.seed]
#
#         # find architecture index
#         arch_str = G.name
#         # print(arch_str)
#
#         try:
#             arch_index = self.api.query_index_by_arch(arch_str)
#             acc_results = self.api.query_by_index(arch_index, self.task)
#             if seed >= 3:
#                 # some architectures only contain 1 seed result
#                 acc_results = self.api.get_more_info(
#                     arch_index, self.task, None, self.use_12_epochs_result, False
#                 )
#                 val_acc = acc_results["valid-accuracy"] / 100
#                 test_acc = acc_results["test-accuracy"] / 100
#             else:
#                 try:
#                     val_acc = acc_results[seed].get_eval("x-valid")["accuracy"] / 100
#                     if self.task == "cifar10-valid":
#                         test_acc = (
#                             acc_results[seed].get_eval("ori-test")["accuracy"] / 100
#                         )
#                     else:
#                         test_acc = acc_results[seed].get_eval("x-test")["accuracy"] / 100
#                 except:
#                     # some architectures only contain 1 seed result
#                     acc_results = self.api.get_more_info(
#                         arch_index, self.task, None, self.use_12_epochs_result, False
#                     )
#                     val_acc = acc_results["valid-accuracy"] / 100
#                     test_acc = acc_results["test-accuracy"] / 100
#
#             auxiliary_info = self.api.query_meta_info_by_index(arch_index)
#             cost_info = auxiliary_info.get_compute_costs(self.task)
#
#             # auxiliary cost results such as number of flops and number of parameters
#             cost_results = {
#                 "flops": cost_info["flops"],
#                 "params": cost_info["params"],
#                 "latency": cost_info["latency"],
#             }
#
#         except FileNotFoundError:
#             val_acc = 0.01
#             test_acc = 0.01
#             print("missing arch info")
#             cost_results = {"flops": None, "params": None, "latency": None}
#
#         # store val and test performance + auxiliary cost information
#         self.X.append(arch_str)
#         self.y_valid_acc.append(val_acc)
#         self.y_test_acc.append(test_acc)
#         self.costs.append(cost_results)
#
#         if which == "eval":
#             err = 1.0 - val_acc
#         elif which == "test":
#             err = 1.0 - test_acc
#         else:
#             raise ValueError("Unknown query parameter: which = " + str(which))
#
#         if self.log_scale:
#             y = np.log(err)
#         else:
#             y = err
#         if self.negative:
#             y = -y
#         return y
#


def create_nasbench201_graph(op_node_labelling, edge_attr=True):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    if edge_attr:
        edge_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        G.add_edges_from(edge_list)
        edge_attribute = {}
        remove_edge_list = []
        for i, edge in enumerate(edge_list):
            edge_attribute[edge] = {"op_name": op_node_labelling[i]}
            if op_node_labelling[i] == "none":
                remove_edge_list.append(edge)
        nx.set_edge_attributes(G, edge_attribute)
        G.remove_edges_from(remove_edge_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 3 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        # Assign dummy variables as node attributes:
        for i in G.nodes:
            G.nodes[i]["op_name"] = "1"
        G.graph_type = "edge_attr"
    else:
        edge_list = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 6),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        G.add_edges_from(edge_list)

        # assign node attributes and collate the information for nodes to be removed
        # (i.e. nodes with 'skip_connect' or 'none' label)
        node_labelling = ["input"] + op_node_labelling + ["output"]
        nodes_to_remove_list = []
        remove_nodes_list = []
        edges_to_add_list = []
        for i, n in enumerate(node_labelling):
            G.nodes[i]["op_name"] = n
            if n == "none" or n == "skip_connect":
                input_nodes = [edge[0] for edge in G.in_edges(i)]
                output_nodes = [edge[1] for edge in G.out_edges(i)]
                nodes_to_remove_info = {
                    "id": i,
                    "input_nodes": input_nodes,
                    "output_nodes": output_nodes,
                }
                nodes_to_remove_list.append(nodes_to_remove_info)
                remove_nodes_list.append(i)

                if n == "skip_connect":
                    for n_i in input_nodes:
                        edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                        edges_to_add_list += edges_to_add

        # reconnect edges for removed nodes with 'skip_connect'
        G.add_edges_from(edges_to_add_list)

        # remove nodes with 'skip_connect' or 'none' label
        G.remove_nodes_from(remove_nodes_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 7 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        G.graph_type = "node_attr"

    # create the arch string for querying nasbench dataset
    arch_query_string = (
        f"|{op_node_labelling[0]}~0|+"
        f"|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+"
        f"|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|"
    )

    G.name = arch_query_string
    return G
