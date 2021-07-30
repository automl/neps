import collections
import random

from copy import deepcopy

import ConfigSpace
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np

from ..benchmarks.nas.nasbench301 import OPS_301
from ..benchmarks.nas.nasbench301 import NASBench301
from ..benchmarks.nas.nasbench301 import edge_to_coord_mapping


# === For NAS301 ===
VERTICES_301 = 6
MAX_EDGES_301 = 13


def get_nas201_configuration_space():
    # for unpruned graph
    cs = ConfigSpace.ConfigurationSpace()
    ops_choices = ["nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3", "skip_connect", "none"]
    for i in range(6):
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices)
        )
    return cs


def create_nas301_graph(adjacency_matrix, edge_attributes):
    rand_arch = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    nx.set_edge_attributes(rand_arch, edge_attributes)
    for i in rand_arch.nodes:
        rand_arch.nodes[i]["op_name"] = "1"
    rand_arch.graph_type = "edge_attr"
    return rand_arch


# Regularised evolution to generate new graphs
def mutate_arch(parent_arch, benchmark, return_unpruned_arch=True):
    # TODO: adjust mutation for 301
    if benchmark == "nasbench301":
        # get parent_arch node label and adjacency matrix
        for arch in parent_arch:
            child_arch = deepcopy(parent_arch)
            node_labeling = list(nx.get_node_attributes(child_arch, "op_name").values())
            adjacency_matrix = np.array(nx.adjacency_matrix(child_arch).todense())

            parent_node_labeling = deepcopy(node_labeling)
            parent_adjacency_matrix = deepcopy(adjacency_matrix)

            dim_op_labeling = len(node_labeling) - 2
            dim_adjacency_matrix = (
                adjacency_matrix.shape[0] * (adjacency_matrix.shape[0] - 1) // 2
            )

            mutation_failed = True

            while mutation_failed:
                # pick random parameter
                dim = np.random.randint(dim_op_labeling + dim_adjacency_matrix)

                if dim < dim_op_labeling:
                    choices = OPS_301
                    node_number = int(dim + 1)
                    parent_choice = node_labeling[node_number]

                    # drop current values from potential choices
                    choices.remove(parent_choice)

                    # flip parameter
                    choice_idx = np.random.randint(len(choices))
                    node_labeling[node_number] = choices[choice_idx]

                else:
                    choices = [0, 1]
                    # find the corresponding row and colum in adjacency matrix
                    idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
                    edge_i = int(dim - dim_op_labeling)
                    row = idx[0][edge_i]
                    col = idx[1][edge_i]
                    parent_choice = adjacency_matrix[row, col]

                    # drop current values from potential choices
                    choices.remove(parent_choice)

                    # flip parameter
                    choice_idx = np.random.randint(len(choices))
                    adjacency_matrix[row, col] = choices[choice_idx]

                try:
                    pruned_adjacency_matrix, pruned_node_labeling = prune(
                        adjacency_matrix, node_labeling
                    )
                    mutation_failed = False
                except:
                    continue

            child_arch = nx.from_numpy_array(
                pruned_adjacency_matrix, create_using=nx.DiGraph
            )

            for i, n in enumerate(pruned_node_labeling):
                child_arch.nodes[i]["op_name"] = n
            if return_unpruned_arch:
                child_arch_unpruned = nx.from_numpy_array(
                    adjacency_matrix, create_using=nx.DiGraph
                )
                for i, n in enumerate(node_labeling):
                    child_arch_unpruned.nodes[i]["op_name"] = n

    if return_unpruned_arch:
        return child_arch, child_arch_unpruned

    return child_arch, None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def regularized_evolution(
    acquisition_func,
    observed_archs,
    observed_archs_unpruned=None,
    benchmark="nasbench101",
    pool_size=200,
    cycles=40,
    n_mutation=10,
    batch_size=1,
    mutate_unpruned_arch=True,
):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    """
    # Generate some random archs into the evaluation pool
    if mutate_unpruned_arch and observed_archs_unpruned is None:
        raise ValueError(
            "When mutate_unpruned_arch option is toggled on, you need to supplied the list of unpruned "
            "observed architectures."
        )
    if observed_archs_unpruned is not None:
        assert len(observed_archs_unpruned) == len(observed_archs), (
            " unequal length between the pruned/unpruned " "architecture lists"
        )

    n_random_archs = pool_size - len(observed_archs)
    if mutate_unpruned_arch:
        (random_archs, _, random_archs_unpruned) = random_sampling(
            pool_size=n_random_archs, benchmark=benchmark, return_unpruned_archs=True
        )
        population_unpruned = observed_archs_unpruned + random_archs_unpruned
    else:
        (random_archs, _, _) = random_sampling(
            pool_size=n_random_archs,
            benchmark=benchmark,
        )
        population_unpruned = None
    population = observed_archs + random_archs

    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    population_performance = []
    for i, archs in enumerate(population):
        arch_acq = acquisition_func.eval(archs)
        population_performance.append(arch_acq)

    # Carry out evolution in cycles. Each cycle produces a bat model and removes another.
    k_cycle = 0

    while k_cycle < cycles:
        # Sample randomly chosen models from the current population based on the acquisition function values
        pseudo_prob = np.array(population_performance) / (np.sum(population_performance))
        if mutate_unpruned_arch:
            samples = random.choices(population_unpruned, weights=pseudo_prob, k=30)
            sample_indices = [population_unpruned.index(s) for s in samples]
        else:
            samples = random.choices(population, weights=pseudo_prob, k=30)
            sample_indices = [population.index(s) for s in samples]
        sample_performance = [population_performance[idx] for idx in sample_indices]

        # The parents is the best n_mutation model in the sample. skip 2-node archs
        top_n_mutation_archs_indices = np.argsort(sample_performance)[
            -n_mutation:
        ]  # argsort>ascending
        parents_archs = [
            samples[idx]
            for idx in top_n_mutation_archs_indices
            if len(samples[idx].nodes) > 3
        ]

        # Create the child model and store it.
        for parent in parents_archs:
            child, child_unpruned = mutate_arch(parent, benchmark)
            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(nx.to_numpy_array(child)) > MAX_EDGES_301:
                continue
            if iso.is_isomorphic(child, parent):
                continue

            skip = False
            for prev_edit in population:
                if iso.is_isomorphic(
                    child,
                    prev_edit,
                ):
                    skip = True
                    break
            if skip:
                continue
            child_arch_acq = acquisition_func.eval(child)
            population.append(child)
            if mutate_unpruned_arch:
                population_unpruned.append(child_unpruned)
            population_performance.append(child_arch_acq)

        # Remove the worst performing model and move to next evolution cycle
        worst_n_mutation_archs_indices = np.argsort(population_performance)[:n_mutation]
        for bad_idx in sorted(worst_n_mutation_archs_indices, reverse=True):
            population.pop(bad_idx)
            population_performance.pop(bad_idx)
            if mutate_unpruned_arch:
                population_unpruned.pop(bad_idx)
            # print(f'len pop = {len(population)}')
        k_cycle += 1

    # choose batch_size archs with highest acquisition function values to be evaluated next
    best_archs_indices = np.argsort(population_performance)[-batch_size:]
    recommended_pool = [population[best_idx] for best_idx in best_archs_indices]
    if mutate_unpruned_arch:
        recommended_pool_unpruned = [
            population_unpruned[best_idx] for best_idx in best_archs_indices
        ]
        return (recommended_pool, recommended_pool_unpruned), (
            population,
            population_unpruned,
            population_performance,
        )
    return (recommended_pool, None), (population, None, population_performance)


class Model(object):
    """A class representing a model."""

    def __init__(self):
        self.archs = None
        self.unpruned_archs = None
        self.error = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{0:b}".format(self.archs)


# Random to generate new graphs
def random_sampling(
    pool_size=100,
    benchmark="nasbench301",
    save_config=False,
    edge_attr=False,
    return_unpruned_archs=False,
    seed=None,
):
    """
    Return_unpruned_archs: bool: If True, both the list of pruned architectures and unpruned architectures will be
        returned.

    """
    evaluation_pool = []
    nasbench_config_list = []
    while len(evaluation_pool) < pool_size:
        if benchmark == "nasbench301":
            cs = NASBench301.get_config_space()
            config = cs.sample_configuration()
            nasbench_config_list.append(config)
            config = config.get_dictionary()
            normal_adjacency_matrix = np.zeros(
                (VERTICES_301, VERTICES_301), dtype=np.int8
            )
            reduce_adjacency_matrix = np.zeros(
                (VERTICES_301, VERTICES_301), dtype=np.int8
            )
            hyperparameters = dict()

            normal_edge_attributes = {}
            reduce_edge_attributes = {}
            for key, item in config.items():
                if "edge" in key:
                    x, y = edge_to_coord_mapping[int(key.split("_")[-1])]
                    if "normal" in key:
                        normal_edge_attributes[(x, y)] = {"op_name": config[key]}
                        normal_adjacency_matrix[x][y] = OPS_301.index(item) + 1
                    else:
                        reduce_edge_attributes[(x, y)] = {"op_name": config[key]}
                        reduce_adjacency_matrix[x][y] = OPS_301.index(item) + 1
                # if 'hyperparam' in key or 'OptimizerSelector' in key:
                #     hyperparameters[key] = item

            normal_rand_arch = create_nas301_graph(
                normal_adjacency_matrix, normal_edge_attributes
            )
            reduce_rand_arch = create_nas301_graph(
                reduce_adjacency_matrix, reduce_edge_attributes
            )

            rand_arch = [normal_rand_arch, reduce_rand_arch]  # , hyperparameters]

        elif benchmark == "nasbench201":
            # generate random architecture for nasbench201

            nas201_cs = get_nas201_configuration_space()
            config = nas201_cs.sample_configuration()
            op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
            # skip only duplicating architecture
            if op_labeling in nasbench201_op_label_list:
                continue

            nasbench201_op_label_list.append(op_labeling)
            rand_arch = create_nasbench201_graph(op_labeling, edge_attr)
            nasbench_config_list.append(config)

            # IN Nasbench201, it is possible that invalid graphs consisting entirely from None and skip-line are
            # generated; remove these invalid architectures.

            # Also remove if the number of edges is zero. This is is possible, one example in NAS-Bench-201:
            # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|none~2|'
            if len(rand_arch) == 0 or rand_arch.number_of_edges() == 0:
                continue

        evaluation_pool.append(rand_arch)

    if save_config:
        return evaluation_pool, nasbench_config_list
    else:
        return evaluation_pool, None
    # TODO: ADD other HPs


def validate(original_matrix):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]
    # DFS forward from input
    visited_from_input = {0, 1}
    frontier = [0, 1]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if original_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)
    # DFS backward from output
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if original_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)
    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output)
    )
    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 3:
        return False
    else:
        indices = np.nonzero(original_matrix)
        inputs = dict()
        inputs[3] = list()
        inputs[4] = list()
        inputs[5] = list()
        for index in zip(indices[0], indices[1]):
            if index[1] != 2:
                inputs[index[1]].append(index[0])
        return all(len(inputs[i]) == 2 for i in [3, 4, 5])


def mutation(
    observed_archs,
    observed_errors,
    n_best=10,
    n_mutate=None,
    pool_size=250,
    allow_isomorphism=False,
    patience=50,
    benchmark="nasbench101",
    observed_archs_unpruned=None,
):
    """
    BANANAS-style mutation.
    The main difference with the previously implemented evolutionary algorithms is that it allows unlimited number of
    edits based on the parent architecture. For previous implementations, we only allow 1-edit distance mutation.
    (although *in expectation*, there is only one edit.)
    """
    if n_mutate is None:
        n_mutate = int(0.5 * pool_size)
    assert pool_size >= n_mutate, " pool_size must be larger or equal to n_mutate"

    def _banana_mutate(arch, mutation_rate=1.0, benchmark="nasbench301"):
        """Bananas Style mutation of the cell"""
        if benchmark == "nasbench301":
            while True:
                new_ops = nx.get_edge_attributes(arch, "op_name")
                new_matrix = np.array(nx.adjacency_matrix(arch).todense())
                vertice = min(VERTICES_301, new_matrix.shape[0])

                if vertice > 2:
                    edge_mutation_prob = mutation_rate / vertice
                    op_mutation_prob = mutation_rate / (vertice - 2)

                    for src in range(0, vertice - 1):
                        for dst in range(src + 1, vertice):
                            # Cant have any operation between 0 and
                            # 1 as they are the input node or delete
                            # any between (0, 1) and 2
                            if dst in (1, 2):
                                continue
                            if random.random() < edge_mutation_prob:
                                if new_matrix[src][dst] == 0:
                                    choice = random.choice(OPS_301)
                                    new_matrix[src][dst] = OPS_301.index(choice) + 1
                                    new_ops[(src, dst)] = choice
                                else:
                                    new_matrix[src][dst] = 0
                                    del new_ops[(src, dst)]
                            if (src, dst) in new_ops.keys():
                                if random.random() < op_mutation_prob:

                                    available = [
                                        o for o in OPS_301 if o != new_ops[(src, dst)]
                                    ]
                                    choice = random.choice(available)
                                    new_matrix[src, dst] = OPS_301.index(choice) + 1
                                    new_ops[(src, dst)] = choice

                if validate(new_matrix):
                    edge_attributes = {}
                    for key, item in new_ops.items():
                        edge_attributes[key] = {}
                        edge_attributes[key]["op_name"] = item
                    child_arch = create_nas301_graph(new_matrix, edge_attributes)
                    return child_arch, child_arch
                else:
                    continue
        else:
            raise NotImplementedError(
                "Search space " + str(benchmark) + " not implemented!"
            )

    population = collections.deque()
    # Fill the population with the observed archs (a list of labelled graphs) and validation error
    for i, archs in enumerate(observed_archs):
        model = Model()
        model.archs = archs
        model.unpruned_archs = (
            observed_archs_unpruned[i] if observed_archs_unpruned is not None else None
        )
        model.error = observed_errors[i]
        population.append(model)

    best_archs = [arch for arch in sorted(population, key=lambda i: -i.error)][:n_best]
    evaluation_pool, evaluation_pool_unpruned = [], []
    per_arch = n_mutate // n_best
    for arch in best_archs:
        n_child = 0
        patience_ = patience
        while n_child < per_arch and patience_ > 0:
            child = Model()
            child_normal_arch, child_normal_unpruned_arch = _banana_mutate(
                arch.unpruned_archs[0]
                if observed_archs_unpruned is not None
                else arch.archs[0],
                benchmark=benchmark,
            )
            child_reduce_arch, child_reduce_unpruned_arch = _banana_mutate(
                arch.unpruned_archs[1]
                if observed_archs_unpruned is not None
                else arch.archs[1],
                benchmark=benchmark,
            )
            child.archs = [child_normal_arch, child_reduce_arch]
            child.unpruned_archs = [child_reduce_arch, child_reduce_unpruned_arch]

            skip = False
            if not allow_isomorphism:
                # if disallow isomorphism, we enforce that each time, we mutate n distinct graphs. For now we do not
                # check the isomorphism in all of the previous graphs though
                if benchmark == "nasbench301":
                    if iso.is_isomorphic(
                        child.archs[0], arch.archs[0]
                    ) or iso.is_isomorphic(child.archs[1], arch.archs[1]):
                        #  and desirable to simply do Weisfiler-Lehman Isomorphism test, since we are based on
                        #  Weisfeiler-Lehman graph kernel already and the isomorphism test is essentially free.
                        patience_ -= 1
                        continue

                    for prev_edit in evaluation_pool:
                        if iso.is_isomorphic(
                            child.archs[0], prev_edit.archs[0]
                        ) or iso.is_isomorphic(child.archs[1], prev_edit.archs[1]):
                            skip = True
                            break
                    if skip:
                        patience_ -= 1
                        continue

            child.error = 0
            evaluation_pool.append(child.archs)
            evaluation_pool_unpruned.append(child.unpruned_archs)
            n_child += 1

    # Add some random archs into the evaluation pool, if either 1) patience is reached or 2)
    nrandom_archs = max(pool_size - len(evaluation_pool), 0)
    if nrandom_archs:
        random_evaluation_pool, _, random_evaluation_pool_unpruned = random_sampling(
            pool_size=nrandom_archs, benchmark=benchmark, return_unpruned_archs=True
        )
        evaluation_pool += random_evaluation_pool
        evaluation_pool_unpruned += random_evaluation_pool_unpruned
    if observed_archs_unpruned is None:
        return evaluation_pool, [None] * len(evaluation_pool)
    return evaluation_pool, evaluation_pool_unpruned
