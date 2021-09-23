import networkx as nx
import numpy as np

try:
    import torch
    from torch_geometric import utils
    from torch_geometric.data import Data, InMemoryDataset
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

PRIMITIVES = [
    # 'none',
    "input_0",
    "input_1",
    "inter_node",
    "output",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


def draw_graph_to_adjacency_matrix(adjacency_matrix):
    """
    Draws the graph in circular format for easier debugging
    :param adjacency_matrix:
    :return:
    """
    dag = nx.DiGraph(adjacency_matrix)
    nx.draw_circular(dag, with_labels=True)


def draw_graph_to_edge_index(edge_index):
    """
    Draw graph to an edge_index
    :param edge_index:
    :return:
    """
    row_idx, col_idx = edge_index
    num_nodes_in_cell = np.max(np.c_[row_idx, col_idx]) + 1
    adj = np.zeros([int(num_nodes_in_cell), int(num_nodes_in_cell)])
    adj[row_idx.astype(np.int32), col_idx.astype(np.int32)] = 1
    return draw_graph_to_adjacency_matrix(adj)


class NASBenchDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        result_paths,
        config_loader,
        dataset_statistic,
        model_config,
        transform=None,
        pre_transform=None,
    ):
        super(NASBenchDataset, self).__init__(root, transform, pre_transform)
        self.result_paths = result_paths
        self.config_loader = config_loader
        self.dataset_statistic = dataset_statistic

        self.model_config = model_config

        self.hyp_norm_factor_mu = [
            self.dataset_statistic["epochs"][0],
            self.dataset_statistic["init_channels"][0],
            self.dataset_statistic["num_layers"][0],
            self.dataset_statistic["learning_rate"][0],
            self.dataset_statistic["weight_decay"][0],
            # self.dataset_statistic['momentum'][0],
            # self.dataset_statistic['batch_size'][0],
            # self.dataset_statistic['autoaugment'][0],
            # self.dataset_statistic['cutout_holes'][0],
            # self.dataset_statistic['cutout_length'][0],
            # self.dataset_statistic['fastautoaugment'][0],
        ]

        self.hyp_norm_factor_std = [
            self.dataset_statistic["epochs"][1],
            self.dataset_statistic["init_channels"][1],
            self.dataset_statistic["num_layers"][1],
            self.dataset_statistic["learning_rate"][1],
            self.dataset_statistic["weight_decay"][1],
            # self.dataset_statistic['momentum'][1],
            # self.dataset_statistic['batch_size'][1],
            # self.dataset_statistic['autoaugment'][1],
            # self.dataset_statistic['cutout_holes'][1],
            # self.dataset_statistic['cutout_length'][1],
            # self.dataset_statistic['fastautoaugment'][1],
        ]

    def __len__(self):
        return len(self.result_paths)

    def _download(self):
        # Needs to be overridden
        pass

    def _process(self):
        # Needs to be overridden
        pass

    def create_darts_adjacency_matrix_from_config(self, config):
        """
        Parses the config space instance and extracts the adjacency matrix and operation list for the normal and
        the reduction cell
        :param config:
        :return:
        """

        def _create_adjacency_for_cell_type_from_config(config, cell_type):
            if cell_type not in ["normal", "reduce"]:
                raise ValueError("Invalid cell type")

            # Nodes 0 (INPUT - 1), 1 (INPUT), 2, 3, 4, 5, 6 (OUTPUT)
            darts_adjacency_matrix = np.zeros((7, 7))

            # Set inputs for intermediate node 2
            darts_adjacency_matrix[0, 2] = 1
            darts_adjacency_matrix[1, 2] = 1

            for node in range(3, 6):
                # Retrieve parents of intermediate node 3, 4 and 5
                parent_0, parent_1 = config._values[
                    "NetworkSelectorDatasetInfo:darts:inputs_node_{}_{}".format(
                        cell_type, node
                    )
                ].split("_")
                darts_adjacency_matrix[int(parent_0), node] = 1
                darts_adjacency_matrix[int(parent_1), node] = 1

            return darts_adjacency_matrix

        def _extract_operation_list_from_config(config, cell_type):
            operations = []
            for edge in range(0, 14):
                op = config.get(
                    "NetworkSelectorDatasetInfo:darts:edge_{}_{}".format(cell_type, edge),
                    None,
                )
                if op is not None:
                    operations.append(op)
            return operations

        extract_cell = lambda config, cell_type: (
            _create_adjacency_for_cell_type_from_config(config, cell_type=cell_type),
            _extract_operation_list_from_config(config, cell_type=cell_type),
        )

        normal_cell = extract_cell(config, cell_type="normal")
        reduction_cell = extract_cell(config, cell_type="reduce")
        return normal_cell, reduction_cell

    def convert_to_pytorch_format(self, cell):
        """
        Converts the adjacency matrix and operation list to the COO format used by pytorch geometric.
        :param cell:
        :return:
        """

        def _convert_from_darts_graph_representation_to_nasbench(adj_coo_format):
            """
            This function replaces the darts representation of the search space with an equivalent representation,
            in which edges don't denote operations. This is done by replacing an op edge in the darts representation,
            by an additional operation node and two additional activation flow edges. The reason for not directly
            using edge attributes is that most graph neural networks don't take them into account.
            :param adj_coo_format:
            :return:
            """
            adj_coo_format_mod = [[], []]
            # Global cell structure
            general_nodes = [
                "input_0",
                "input_1",
                "inter_node",
                "inter_node",
                "inter_node",
                "inter_node",
                "output",
            ]
            operation_nodes = []
            for edge_idx in range(adj_coo_format.shape[1]):
                from_node, to_node = adj_coo_format[:, edge_idx]
                adj_coo_format_mod = np.c_[
                    adj_coo_format_mod,
                    [
                        [from_node, len(general_nodes) + edge_idx],
                        [len(general_nodes) + edge_idx, to_node],
                    ],
                ]

                # Append op
                operation_nodes.append(ops[edge_idx])

            # Add output connection
            adj_coo_format_mod = np.c_[adj_coo_format_mod, [[2, 3, 4, 5], [6, 6, 6, 6]]]

            # Add the operation nodes
            general_nodes.extend(operation_nodes)
            return adj_coo_format_mod, general_nodes

        adj, ops = cell
        adj_coo_format = np.array(np.nonzero(adj))
        (
            adj_coo_format_mod,
            general_nodes,
        ) = _convert_from_darts_graph_representation_to_nasbench(adj_coo_format)

        op_indices = [PRIMITIVES.index(op) for op in general_nodes]
        make_one_hot = lambda op_indices: np.eye(len(PRIMITIVES))[
            np.array(op_indices).reshape(-1)
        ]
        op_one_hot = make_one_hot(op_indices)
        return adj_coo_format_mod, op_one_hot, op_indices

    def _convert_cell_to_geometric(
        self, normal_cell, reduction_cell, hyperparameters_no_arch, val_accuracy
    ):
        """
        Combines normal cell and reduction cell into a single pytorch geometric data item with disconnect graphs.
        :param normal_cell:
        :param reduction_cell:
        :param hyperparameters_no_arch:
        :param val_accuracy:
        :return:
        """

        num_nodes_in_cell = normal_cell[1].shape[0]
        assert num_nodes_in_cell == 15, "Number of nodes in cell has changed."
        op_index = torch.from_numpy(np.vstack((normal_cell[1], reduction_cell[1])))
        node_embeddings_concat_list = [op_index]

        edge_index = torch.from_numpy(
            np.c_[normal_cell[0], reduction_cell[0] + num_nodes_in_cell]
        ).to(torch.long)

        if self.model_config["graph_preprocessing:undirected_graph"]:
            edge_index = utils.to_undirected(edge_index)

        # Add hyperparameters directly to node feature vectors.
        hyperparameters_no_arch_norm = self.normalize_hyps(hyperparameters_no_arch)
        hyps_per_vertex = hyperparameters_no_arch_norm.reshape(1, -1).repeat(
            op_index.shape[0], 1
        )

        node_embeddings_concat_list.append(hyps_per_vertex)

        if self.model_config["graph_preprocessing:add_node_degree_one_hot"]:
            # Compute node degrees and represent as one-hot vector
            rows, cols = utils.to_undirected(edge_index)
            node_degree = utils.degree(rows.long(), len(op_index)).view(-1, 1)
            make_one_hot = lambda index: torch.eye(7)[index.view(-1).long()]
            node_degree_one_hot = make_one_hot(node_degree)
            node_embeddings_concat_list.append(node_degree_one_hot.double())

        # Form final hyperparameter vector
        op_index_hyps_included = torch.cat(node_embeddings_concat_list, axis=1)

        if (
            self.model_config["model"] == "gnn_vs_gae"
            or self.model_config["model"] == "gnn_vs_gae_classifier"
            and self.model_config["graph_preprocessing:init_node_emb"]
        ):
            edge_index = torch.from_numpy(
                np.c_[normal_cell[0], reduction_cell[0] + num_nodes_in_cell]
            )
            node_atts = torch.from_numpy(np.hstack((normal_cell[2], reduction_cell[2])))
            op_index_hyps_included = torch.LongTensor(node_atts)

        return Data(
            op_index_hyps_included,
            edge_index=edge_index,
            hyperparameters=hyperparameters_no_arch,
            y=val_accuracy,
        )

    def normalize_hyps(self, hyps):
        """
        Whiten the data for hyperparameters
        :param hyps:
        :return:
        """
        hyps_norm = (
            hyps - torch.from_numpy(np.array(self.hyp_norm_factor_mu))
        ) / torch.from_numpy(np.array(self.hyp_norm_factor_std))
        return hyps_norm

    def config_space_instance_to_pytorch_geometric_instance(self, config_space_instance):
        """
        Convert a config_space_instance into a pytorch data object.
        :param config_space_instance:
        :return:
        """
        normal_cell, reduction_cell = self.create_darts_adjacency_matrix_from_config(
            config_space_instance
        )

        normal_cell_pt, reduction_cell_pt = map(
            self.convert_to_pytorch_format, [normal_cell, reduction_cell]
        )

        hyperparameters_no_arch = torch.from_numpy(
            np.array(
                self.config_loader.get_config_without_architecture(config_space_instance)
            )
        )
        data = self._convert_cell_to_geometric(
            normal_cell_pt, reduction_cell_pt, hyperparameters_no_arch, -1
        )
        return data

    def get(self, idx):
        config_space_instance, val_accuracy, test_accuracy, _ = self.config_loader[
            self.result_paths[idx]
        ]
        normal_cell, reduction_cell = self.create_darts_adjacency_matrix_from_config(
            config_space_instance
        )

        normal_cell_pt, reduction_cell_pt = map(
            self.convert_to_pytorch_format, [normal_cell, reduction_cell]
        )

        hyperparameters_no_arch = torch.from_numpy(
            np.array(
                self.config_loader.get_config_without_architecture(config_space_instance)
            )
        )
        data = self._convert_cell_to_geometric(
            normal_cell_pt, reduction_cell_pt, hyperparameters_no_arch, val_accuracy
        )

        return data


"""
Code from official repository of "A Fair Comparison of Graph Neural Networks for Graph Classification", ICLR 2020
https://github.com/diningphil/gnn-comparison under GNU General Public License v3.0
"""


class EarlyStopper:
    def stop(
        self,
        epoch,
        val_loss,
        val_acc=None,
        test_loss=None,
        test_acc=None,
        train_loss=None,
        train_acc=None,
    ):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
            self.best_epoch,
        )


class Patience(EarlyStopper):
    """
    Implement common "patience" technique
    """

    def __init__(self, patience=20, use_loss=True):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss, self.train_acc = None, None
        self.val_loss, self.val_acc = None, None
        self.test_loss, self.test_acc = None, None

    def stop(
        self,
        epoch,
        val_loss,
        val_acc=None,
        test_loss=None,
        test_acc=None,
        train_loss=None,
        train_acc=None,
    ):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                self.test_loss, self.test_acc = test_loss, test_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                self.test_loss, self.test_acc = test_loss, test_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
