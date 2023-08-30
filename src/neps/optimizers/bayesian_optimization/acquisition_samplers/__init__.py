from .evolution_sampler import EvolutionSampler
from .freeze_thaw_sampler import FreezeThawSampler
from .mutation_sampler import MutationSampler
from .random_sampler import RandomSampler

AcquisitionSamplerMapping = {
    "random": RandomSampler,
    "mutation": MutationSampler,
    "evolution": EvolutionSampler,
    "freeze-thaw": FreezeThawSampler,
}
