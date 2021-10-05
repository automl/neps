from .random_sampler import RandomSampler
from .mutation_sampler import MutationSampler
from .evolution_sampler import EvolutionSampler

AcquisitionOptimizerMapping = {
    "random": RandomSampler,
    "mutation": MutationSampler,
    "evolution": EvolutionSampler,
}
