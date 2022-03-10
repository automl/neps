from .evolution_sampler import EvolutionSampler
from .mutation_sampler import MutationSampler
from .random_sampler import RandomSampler

AcquisitionSamplerMapping = {
    "random": RandomSampler,
    "mutation": MutationSampler,
    "evolution": EvolutionSampler,
}
