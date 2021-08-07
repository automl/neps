from .mutation_sampler import MutationSampler
from .random_sampler import RandomSampler

AcquisitionOptimizerMapping = {
    "random": RandomSampler,
    "mutate": MutationSampler,
}
