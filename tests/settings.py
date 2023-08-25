from pathlib import Path

ITERATIONS = 100
MAX_EVALUATIONS_TOTAL = 150

OPTIMIZERS = [
    "random_search",
    # "mf_bayesian_optimization",
    "bayesian_optimization",
    "regularized_evolution",
]

TASKS = [
    # "cifar10", "fashion_mnist",
    "hartmann3",
    "hartmann6",
]

LOSS_FILE = Path(Path(__file__).parent, "losses.json")
