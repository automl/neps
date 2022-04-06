import numpy as np
from scipy import stats
import argparse
import warnings
import pickle

from neps.search_spaces import (
    CategoricalParameter,
    FloatParameter,
    GraphGrammar,
    IntegerParameter,
)
from neps.search_spaces.search_space import SearchSpace
from neps.optimizers.bayesian_optimization.kernels import GraphKernelMapping, StationaryKernelMapping
from neps.optimizers.bayesian_optimization.models.gp_hierarchy import ComprehensiveGPHierarchy

def data_loader_graph(arch_data_path, dataset, n_train, seed, output_transform='log', hierarchy_considered=None):

    arch_data_path = f'{arch_data_path}{dataset}.pickle'
    with open(arch_data_path, 'rb') as outfile:
        arch_data_list = pickle.load(outfile)
    x_all = []
    y_all = []

    for arch_info in arch_data_list[:5000]:
        arch = {}
        arch['graph'] = arch_info['graph']
        if hierarchy_considered is not None:
            arch['hierarchy_graphs'] = {hierarchy: arch_info['hierarchy_graphs'][hierarchy] for hierarchy in hierarchy_considered}
        valid_err = arch_info['y']
        x_all.append(arch)
        y_all.append(valid_err)

    n_arch = len(x_all)
    indices = list(range(n_arch))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    if output_transform == 'log':
        y_all = [np.log(y) for y in y_all]
    elif output_transform == 'log_scale':
        y_all = [np.log(y)*100 for y in y_all]

    xtrain = [x_all[i] for i in train_indices]
    ytrain = [y_all[i] for i in train_indices]
    xtest = [x_all[i] for i in test_indices]
    ytest = [y_all[i] for i in test_indices]

    return xtrain, ytrain, xtest, ytest

if __name__ == '__main__':

    warnings.filterwarnings("ignore", message="Regression experiments")
    parser = argparse.ArgumentParser(description="Run Graph Similarity Measures")
    parser.add_argument('-m', '--method', help='regression model', default='gpwl', type=str)
    parser.add_argument('-ntr', '--n_train', help='number of training data', default=50, type=int)
    parser.add_argument('-d', '--dataset', help='dataset name', default='cifar10spatial_aug', type=str)
    # parser.add_argument('-hl', '--hierarchy', help='hierarchies considered', default='null', type=str)
    parser.add_argument('-hl', '--hierarchy', help='hierarchies considered', default='0_1_2_3', type=str)
    parser.add_argument('-ns', '--n_seeds', help='number of training data', default=1, type=int)

    args = parser.parse_args()
    print(f'{args}')
    n_train = args.n_train
    dataset = args.dataset
    arch_data_path = '/home/ps/Desktop/PhD/Hierarchical_NAS/gpwl_surrogate/data/'
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
        hp_kernels = ['rbf']
        assert d_graph_features > 0, "number of global features used should be > 0 if we impose " \
                                     "stationary kernel on them"


    if early_hierarchies_considered == "null":
        # only consider the final architecture (highest hierarchy)
        hierarchy_considered = None
        graph_kernels = ["wl"]
        wl_h = [2]
    else:
        hierarchy_considered = [
            int(hl) for hl in early_hierarchies_considered.split("_")
        ]
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
    res_dict = {'args': args}
    for seed in seed_list:
        # ====== load data ======
        xtrain, ytrain, xtest, ytest = data_loader_graph(arch_data_path, dataset, n_train, seed,
                                                         hierarchy_considered=[0, 1, 2,3], output_transform='log')

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
        nll = - np.mean(stats.norm.logpdf(np.array(ytest), loc=ypred, scale=ypred_std))
        # ====== evaluate regression performance for each graph ======
        print(f'seed={seed}, n_test={len(ytest)}: pearson={pearson :.3f}, spearman={spearman :.3f}, kendalltau={kendalltau :.3f}, NLL={nll}')
        # ================================================

        pearson_allseed.append(pearson)
        spearman_allseed.append(spearman)
        kendalltau_allseed.append(kendalltau)
        nll_allseed.append(nll)
        y_pred_allseed.append(ypred)
        y_test_allseed.append(ytest)

        # ====== store results ======
        # if seed % 5 == 0:
    res_dict['pearson'] = pearson_allseed
    res_dict['spearman'] = spearman_allseed
    res_dict['kendalltau'] = kendalltau_allseed
    res_dict['nll'] = nll_allseed
    res_dict['ypred'] = y_pred_allseed
    res_dict['ytest'] = y_test_allseed

    # file_name =  f'./results/{args.method}_{args.dataset}_nt{args.n_train}_{args.graph_kernel}_{args.stationary_kernel}' \
    #              f'_{args.hierarchy}_kw{args.kernel_weights}.pickle'
    # with open(file_name, 'wb') as outfile:
    #   pickle.dump(res_dict, outfile)
