from __future__ import annotations

from collections.abc import Sized

from dataclasses import dataclass
from grakel.utils import graph_from_networkx

from typing import Sequence, Iterable, TypeAlias
from typing_extensions import Self
from more_itertools import split_when
from itertools import chain
import torch

from neps.search_spaces.search_space import SearchSpace

WLInput: TypeAlias = tuple[dict, dict | None, dict | None]


@dataclass
class TensorEncodedConfigs(Sized):
    _tensor_pack: torch.Tensor | None
    """Layout such that _tensor_pack[0] is the first config.

    In the case that there are no numeric/categorical hyperparameters,
    this is None.

    index config_row_id | fidelities... | numericals... | one_hot_categoricals...
           0
           1
           2
          ...

    NOTE: A slight memory innefficiency here is that we store the one-hot encoded
    as a float tensor, rather than a byte tensor. This makes joint numerical/categorical
    kernels more efficient, as well as entire config row access at the cost of memory.
    This should not be a problem if we do not have a large number of categorical
    hyperparameters with a high number of choices.
    """
    _graphs: dict[str, Sequence[WLInput]]
    _col_lookup: dict[str, tuple[int, int]]  # range(inclusive, exclusive)

    def __len__(self) -> int:
        return self._tensor_pack.shape[0] if self._tensor_pack is not None else 0

    def wl_graph_input(self, hp: str) -> Sequence[WLInput]:
        return self._graphs[hp]

    def tensor(self, hps: Iterable[str]) -> torch.Tensor:
        if self._tensor_pack is None:
            raise ValueError("No numerical/categorical hyperparameters were encoded.")

        cols: list[tuple[int, int]] = []
        for hp in hps:
            _cols = self._col_lookup.get(hp)
            if _cols is None:
                raise ValueError(f"Hyperparameter {hp} not found in the lookup table.")
            cols.append(_cols)

        # OPTIM: This code with `split_when` and `chunks` makes sure to grab
        # consecutive chunks of memory where possible. For example,
        # if we want all categoricals, this will just return the entire
        # categorical tensor, rather than subselecting each part and then concatenating.
        # Also works for numericals.
        sorted_indices = sorted(cols)
        non_consecutive_tuple = lambda x, y: x[1] != y[0]
        chunks = list(split_when(sorted_indices, non_consecutive_tuple))
        slices = [slice(chunk[0][0], chunk[-1][1]) for chunk in chunks]
        tensors = [self._tensor_pack[:, s] for s in slices]

        if len(tensors) == 1:
            return tensors[0].clone()

        return torch.cat(tensors, dim=1)

    @classmethod
    def encode(
        cls,
        space: SearchSpace,
        configs: list[SearchSpace],
        *,
        node_label: str = "op_name",
        device: torch.device,
    ) -> Self:
        assert node_label == "op_name", "Only 'op_name' is supported for node_label"

        _graphs: dict[str, Sequence[WLInput]] = {}
        for hp_name in space.graphs.keys():
            gs = [conf.graphs[hp_name].value for conf in configs]
            if (
                len(gs) > 0
                and isinstance(gs[0], list)
                and len(gs[0]) > 0
                and isinstance(gs[0][0], list)
            ):
                gs = [_list for list_of_list in gs for _list in list_of_list]
            _graphs[hp_name] = graph_from_networkx(gs)  # type: ignore

        _lookup: dict[str, tuple[int, int]] = {}

        n_fids = len(space.fidelities)
        n_nums = len(space.numerical)
        n_cats = sum(len(hp.choices) for hp in space.categoricals.values())

        width = n_fids + n_nums + n_cats
        if width == 0:
            return cls(_graphs=_graphs, _tensor_pack=None, _col_lookup={})

        _tensor_pack = torch.empty(size=(len(configs), width), dtype=torch.float64)

        offset = 0
        for hp_name in chain(space.fidelities, space.numerical):
            _lookup[hp_name] = (offset, offset + 1)
            _xs = [config.fidelities[hp_name].normalized_value for config in configs]
            values = torch.tensor(_xs, torch.float64, device=device)

            _tensor_pack[:, offset] = values

            offset += 1

        for hp_name, cat in space.categoricals.items():
            n_choices = len(cat.choices)
            _lookup[hp_name] = (offset, offset + n_choices)

            # .. and insert one-hot encoding (ChatGPT solution, verified locally)
            _xs = [config[hp_name].normalized_value for config in configs]
            cat_tensor = torch.tensor(_xs, torch.float64, device=device).unsqueeze(1)

            _tensor_pack[:, offset : offset + n_choices].scatter_(1, cat_tensor, 1)

            offset += n_choices

        return cls(_graphs=_graphs, _tensor_pack=_tensor_pack, _col_lookup=_lookup)
