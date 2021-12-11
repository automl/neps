from .evolution_sampler import EvolutionSampler
from .mutation_sampler import MutationSampler
from .random_sampler import RandomSampler

AcquisitionOptimizerMapping = {
    "random": RandomSampler,
    "mutation": MutationSampler,
    "evolution": EvolutionSampler,
}
