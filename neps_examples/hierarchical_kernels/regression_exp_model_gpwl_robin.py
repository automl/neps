import argparse
import os
import pickle
import warnings

import networkx as nx
import numpy as np
from hierarchical_nas_benchmarks.search_spaces.robin_test.graph import PRODUCTIONS
from scipy import stats
from torch import nn

from neps.optimizers.bayesian_optimization.kernels import (
    GraphKernelMapping,
    StationaryKernelMapping,
)
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import (
    ComprehensiveGPHierarchy,
)
from neps.search_spaces.graph_grammar import primitives as ops
from neps.search_spaces.graph_grammar import topologies as topos
from neps.search_spaces.graph_grammar.api import FunctionParameter
from neps.search_spaces.graph_grammar.primitives import ResNetBasicblock
from neps.search_spaces.search_space import SearchSpace

PRODUCTIONS = """S -> "diamond" D2 D2 D1 D1 | "diamond" D1 D2 D2 D1 | "diamond" D1 D1 D2 D2 | "linear" D2 D1 | "linear" D1 D2 | "diamond_mid" D1 D2 D1 D2 D1 | "diamond_mid" D2 D2 Cell D1 D1
D2 -> "diamond" D1 D1 D1 D1 | "linear" D1 D1 | "diamond_mid" D1 D1 Cell D1 D1
D1 -> "diamond" D1Helper D1Helper Cell Cell | "diamond" Cell Cell D1Helper D1Helper | "diamond" D1Helper Cell Cell D1Helper | "linear" D1Helper Cell | "linear" Cell D1Helper | "diamond_mid" D1Helper D1Helper Cell Cell Cell | "diamond_mid" Cell D1Helper D1Helper D1Helper Cell
D1Helper -> "down1" Cell "downsample"
Cell -> "residual" OPS OPS OPS | "diamond" OPS OPS OPS OPS | "linear" OPS OPS | "diamond_mid" OPS OPS OPS OPS OPS
OPS -> "conv3x3" | "conv1x1" | "avg_pool" | "id"
"""

TERMINAL_2_OP_NAMES = {
    "id": ops.Identity(),
    "conv3x3": {"op": ops.ConvBnReLU, "kernel_size": 3, "stride": 1, "padding": 1},
    "conv1x1": {"op": ops.ConvBnReLU, "kernel_size": 1},
    "avg_pool": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "downsample": {"op": ResNetBasicblock, "stride": 2},
    "residual": topos.Residual,
    "diamond": topos.Diamond,
    "linear": topos.Linear,
    "diamond_mid": topos.DiamondMid,
    "down1": topos.DownsampleBlock,
}


def build(graph: nx.DiGraph):
    in_channels = 3
    n_classes = 20
    base_channels = 64
    out_channels = 512

    # Assign channels
    in_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    for n in nx.topological_sort(graph):
        for pred in graph.predecessors(n):
            e = (pred, n)
            if pred == in_node:
                channels = base_channels
            else:
                pred_pred = list(graph.predecessors(pred))[0]
                channels = graph.edges[(pred_pred, pred)]["C_out"]
            if graph.edges[e]["op_name"] == "ResNetBasicblock":
                graph.edges[e].update({"C_in": channels, "C_out": channels * 2})
            else:
                graph.edges[e].update({"C_in": channels, "C_out": channels})

    in_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    out_node = [n for n in graph.nodes if graph.out_degree(n) == 0][0]
    max_node_label = max(graph.nodes())
    graph.add_nodes_from([max_node_label + 1, max_node_label + 2])
    graph.add_edge(max_node_label + 1, in_node)
    graph.edges[max_node_label + 1, in_node].update(
        {
            "op": ops.Stem(base_channels, C_in=in_channels),
            "op_name": "Stem",
        }
    )
    graph.add_nodes_from([out_node, max_node_label + 2])
    graph.add_edge(out_node, max_node_label + 2)

    graph.edges[out_node, max_node_label + 2].update(
        {
            "op": ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, n_classes),
            ),
            "op_name": "Out",
        }
    )


# pipeline_space = SearchSpace(architecture=GraphSpace(...))
# configs = [pipeline_space.copy().hyperparameter["architecture"].create_from_id(st) for st in data]
# graphs, hps = extract_configs_hierarchy(configs, d_graph_features=0, hierarchiy_consider=True)


def data_loader_graph(
    arch_data_path,
    dataset,
    n_train,
    seed,
    output_transform="log",
    hierarchy_considered=None,  # pylint: disable=unused-argument
):
    pipeline_space = dict(
        architecture=FunctionParameter(
            build_fn=build,
            name="makrograph",
            grammar=PRODUCTIONS,
            terminal_to_op_names=TERMINAL_2_OP_NAMES,
            return_graph_per_hierarchy=True,
        )
    )
    pipeline_space = SearchSpace(**pipeline_space)

    arch_data_path = os.path.join(arch_data_path, f"{dataset}.pickle")
    with open(arch_data_path, "rb") as outfile:
        arch_data_list = pickle.load(outfile)
    x_all = []
    y_all = []

    for arch_info in arch_data_list[:100]:
        # for arch_info in arch_data_list[:5000]:
        # arch = {}
        # arch['graph'] = arch_info['graph']
        # if hierarchy_considered is not None:
        #     arch['hierarchy_graphs'] = {hierarchy: arch_info['hierarchy_graphs'][hierarchy] for hierarchy in hierarchy_considered}
        identifier = arch_info["id"]
        copied_pipeline_space = pipeline_space.copy()
        copied_pipeline_space.hyperparameters["architecture"].create_from_id(identifier)
        x_all.append(copied_pipeline_space)

        valid_err = arch_info["y"]
        y_all.append(valid_err)

    n_arch = len(x_all)
    indices = list(range(n_arch))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    if output_transform == "log":
        y_all = [np.log(y) for y in y_all]
    elif output_transform == "log_scale":
        y_all = [np.log(y) * 100 for y in y_all]

    xtrain = [x_all[i] for i in train_indices]
    ytrain = [y_all[i] for i in train_indices]
    xtest = [x_all[i] for i in test_indices]
    ytest = [y_all[i] for i in test_indices]

    return xtrain, ytrain, xtest, ytest


if __name__ == "__main__":

    warnings.filterwarnings("ignore", message="Regression experiments")
    parser = argparse.ArgumentParser(description="Run Graph Similarity Measures")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ps/Desktop/PhD/Hierarchical_NAS/gpwl_surrogate/data/",
    )
    parser.add_argument(
        "-m", "--method", help="regression model", default="gpwl", type=str
    )
    parser.add_argument(
        "-ntr", "--n_train", help="number of training data", default=50, type=int
    )
    parser.add_argument(
        "-d", "--dataset", help="dataset name", default="cifar10spatial_aug", type=str
    )
    # parser.add_argument('-hl', '--hierarchy', help='hierarchies considered', default='null', type=str)
    parser.add_argument(
        "-hl", "--hierarchy", help="hierarchies considered", default="0_1_2_3", type=str
    )
    parser.add_argument(
        "-ns", "--n_seeds", help="number of training data", default=1, type=int
    )

    args = parser.parse_args()
    print(f"{args}")
    n_train = args.n_train
    dataset = args.dataset
    arch_data_path = args.data_path
    n_seeds = args.n_seeds
    early_hierarchies_considered = args.hierarchy
    # pipeline_space = SearchSpace # TODO check with Simon on how to use SearchSpace?
    optimal_assignment = False
    domain_se_kernel = None
    verbose = False
    # set to 2 to use graph meta features else 0
    d_graph_features = 0
    # set whether to use stationary kernels
    if d_graph_features == 0:
        hp_kernels = []
    else:
        hp_kernels = ["rbf"]
        assert d_graph_features > 0, (
            "number of global features used should be > 0 if we impose "
            "stationary kernel on them"
        )

    if early_hierarchies_considered == "null":
        # only consider the final architecture (highest hierarchy)
        hierarchy_considered = None
        graph_kernels = ["wl"]
        wl_h = [2]
    else:
        hierarchy_considered = [int(hl) for hl in early_hierarchies_considered.split("_")]
        graph_kernels = ["wl"] * (len(hierarchy_considered) + 1)
        wl_h = [1] + [2] * len(hierarchy_considered)

    graph_kernels = [
        GraphKernelMapping[kernel](
            h=wl_h[j],
            oa=optimal_assignment,
            se_kernel=None
            if domain_se_kernel is None
            else StationaryKernelMapping[domain_se_kernel],
        )
        for j, kernel in enumerate(graph_kernels)
    ]
    hp_kernels = [StationaryKernelMapping[kernel]() for kernel in hp_kernels]

    pearson_allseed = []
    spearman_allseed = []
    kendalltau_allseed = []
    nll_allseed = []
    y_pred_allseed = []
    y_test_allseed = []

    # seed_list = [0]
    seed_list = range(n_seeds)
    res_dict = {}
    for seed in seed_list:
        # ====== load data ======
        xtrain, ytrain, xtest, ytest = data_loader_graph(
            arch_data_path,
            dataset,
            n_train,
            seed,
            hierarchy_considered=[0, 1, 2, 3],
            output_transform="log",
        )

        n_test_exact = len(xtest)

        # ====== train the model and predict with it ========
        surrogate_model = ComprehensiveGPHierarchy(
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            verbose=verbose,
            hierarchy_consider=hierarchy_considered,
            d_graph_features=d_graph_features,  # set to 0 if not using additional graph topological features
            vectorial_features=None,
            # pipeline_space.get_vectorial_dim()
            # if hasattr(pipeline_space, "get_vectorial_dim")
            # else None,
        )
        surrogate_model.reset_XY(train_x=xtrain, train_y=ytrain)
        surrogate_model.fit()

        # & evaluate
        ypred, ypred_var = surrogate_model.predict(xtest)
        ypred, ypred_var = ypred.cpu().detach().numpy(), ypred_var.cpu().detach().numpy()
        # ====== evaluate regression performance ======
        pearson = stats.pearsonr(ytest, ypred)[0]
        spearman = stats.spearmanr(ytest, ypred)[0]
        kendalltau = stats.kendalltau(ytest, ypred)[0]
        ypred_std = np.sqrt(ypred_var)
        nll = -np.mean(stats.norm.logpdf(np.array(ytest), loc=ypred, scale=ypred_std))
        # ====== evaluate regression performance for each graph ======
        print(
            f"seed={seed}, n_test={len(ytest)}: pearson={pearson :.3f}, spearman={spearman :.3f}, kendalltau={kendalltau :.3f}, NLL={nll}"
        )
        # ================================================

        pearson_allseed.append(pearson)
        spearman_allseed.append(spearman)
        kendalltau_allseed.append(kendalltau)
        nll_allseed.append(nll)
        y_pred_allseed.append(ypred)
        y_test_allseed.append(ytest)

        # ====== store results ======
        # if seed % 5 == 0:
    res_dict["pearson"] = pearson_allseed
    res_dict["spearman"] = spearman_allseed
    res_dict["kendalltau"] = kendalltau_allseed
    res_dict["nll"] = nll_allseed
    res_dict["ypred"] = y_pred_allseed
    res_dict["ytest"] = y_test_allseed
    res_dict["args"] = args  # type: ignore[assignment]

    # file_name =  f'./results/{args.method}_{args.dataset}_nt{args.n_train}_{args.graph_kernel}_{args.stationary_kernel}' \
    #              f'_{args.hierarchy}_kw{args.kernel_weights}.pickle'
    # with open(file_name, 'wb') as outfile:
    #   pickle.dump(res_dict, outfile)
