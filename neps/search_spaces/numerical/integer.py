from __future__ import annotations

from typing_extensions import Literal

from .float import FloatParameter


class IntegerParameter(FloatParameter):
    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        log: bool = False,
        is_fidelity: bool = False,
        default: None | float | int = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        super().__init__(lower, upper, log, is_fidelity, default, default_confidence)
        # We subtract/add 0.499999 from lower/upper bounds respectively, such that
        # sampling in the float space gives equal probability for all integer values,
        # i.e. [x - 499999, x + 499999]
        self.fp = FloatParameter(
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
            is_fidelity=is_fidelity,
            default=default,
            default_confidence=default_confidence,
        )
        self.value: None | int = None

    def __repr__(self):
        return f"Integer, range: [{self.lower}, {self.upper}], value: {self.value}"

    def sample(self, user_priors: bool = False):
        self.fp.sample(user_priors=user_priors)
        self.value = int(round(self.fp.value))  # type: ignore[arg-type]

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
    ):
        mutant = self.fp.mutate(
            parent=parent,
            mutation_rate=mutation_rate,
            mutation_strategy=mutation_strategy,
        )
        child = float_to_integer(mutant)
        return child

    def crossover(self, parent1, parent2=None):
        pass

    # pylint: disable=protected-access
    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        neighbours = self.fp._get_neighbours(std, num_neighbours)
        for idx, neighbour in enumerate(neighbours):
            neighbours[idx] = float_to_integer(neighbour)
        return neighbours

    def _transform(self):
        self.fp._transform()  # pylint: disable=protected-access
        self.value = self.fp.value

    def _inv_transform(self):
        self.fp._inv_transform()  # pylint: disable=protected-access
        self.value = int(round(self.fp.value))


def float_to_integer(float_hp):
    int_hp = IntegerParameter(
        lower=int(round(float_hp.lower)),
        upper=int(round(float_hp.upper)),
        log=float_hp.log,
    )
    int_hp.value = int(round(float_hp.value))

    return int_hp
