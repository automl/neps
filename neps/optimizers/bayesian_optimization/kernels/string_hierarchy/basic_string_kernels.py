from __future__ import annotations

import functools
import logging

import torch

from . import config_string

_logger = logging.getLogger(__name__)


def _normalize_gram(K: torch.Tensor) -> torch.Tensor:
    K_diag = torch.sqrt(torch.diag(K))
    K_diag_outer = torch.ger(K_diag, K_diag)
    return K / K_diag_outer


def _compute_joined_operator_operands(
    part: config_string.UnwrappedConfigStringPart,
) -> str:
    return f"{part.operator} ({part.operands})"


# Neural Architecture String Kernel: NASK

class NASK(torch.nn.Module):
    def __init__(
        self,
        hierarchy_level: int | None = None,
        learnable_weights: bool = True,
    ):
        super().__init__()

        self.hierarchy_level = hierarchy_level

        operator_weight = 1.0
        operands_weight = 1.0
        joined_operator_operands_weight = 1.0

        self.weights = torch.nn.Parameter(
            torch.tensor([
                operator_weight,
                operands_weight,
                joined_operator_operands_weight,
            ]),
            requires_grad=learnable_weights,
        )
        if not self.weights.size() == (3,):
            raise ValueError(
                "Expected `weights` to have size (3,). "
                + f"Received (weights, size): ({self.weights}, ({self.weights.size()})"
            )

        # For each instance, cache the last `_process_configs` result
        # There is no need to cache more calls
        #  and no need to share the cache between instances
        self._process_configs = functools.lru_cache(maxsize=1)(self._process_configs)

    # A cached class method.
    # Computations are the same no matter the seed and instance,
    #  but are unique to the class
    # Other classes can have different ways of computing the result!
    @classmethod
    @functools.lru_cache(maxsize=2000)  # no specific meaning, a reasonable default
    def _get_symbols_from_config(cls, config: config_string.ConfigString) -> set[str]:
        symbols = set()
        for part in config.unwrapped:
            symbols.add(part.operator)
            if part.operands:
                symbols.add(part.operands)
                joined_operator_operands = _compute_joined_operator_operands(part)
                symbols.add(joined_operator_operands)
        return symbols

    def _process_configs(
        self,
        configs: tuple[config_string.ConfigString],
    ) -> torch.Tensor:
        symbols = set()
        symbols.update(*list(
            self.__class__._get_symbols_from_config(config=c) for c in configs
        ))
        symbols = sorted(list(symbols), key=len)

        n_symbols = len(symbols)
        n_configs = len(configs)

        result = torch.zeros(size=(n_configs, n_symbols, 3))

        """
        weights:
            torch.Tensor of size (3,) with weights corresponding to
            [0] - weight of the `operator`
            [1] - weight of the `operands`
            [2] - weight of the joined `operator` and `operands`
        """

        symbol_indices = {}
        for i, s in enumerate(symbols):
            symbol_indices[s] = i

        for conf_idx, c in enumerate(configs):
            conf_values = result[conf_idx]
            for part in c.unwrapped:
                # Increment `operator`
                sym_index = symbol_indices[part.operator]
                conf_values[sym_index][0] += 1.0

                if part.operands:
                    # Increment `operands`
                    sym_index = symbol_indices[part.operands]
                    conf_values[sym_index][1] += 1.0
                    # Increment joined `operator` and `operands`
                    joined_operator_operands = _compute_joined_operator_operands(part)
                    sym_index = symbol_indices[joined_operator_operands]
                    conf_values[sym_index][2] += 1.0

        assert result.size() == (n_configs, n_symbols, 3), \
            f"{result.size()} != {(n_configs, n_symbols, 3)}"

        return result

    def forward(self, configs: tuple[config_string.ConfigString]) -> torch.Tensor:
        if self.hierarchy_level is not None:
            configs = tuple(
                c.at_hierarchy_level(self.hierarchy_level) for c in configs
            )

        if not bool((self.weights > 0).all()):
            raise ValueError(f"Weights !>0: {self.weights}")

        _logger.debug("Called method `forward` of kernel `%s`", self)
        _logger.debug("Count of received config strings: %d", len(configs))
        _logger.debug("Part weights: %s", self.weights)

        # processed_configs have shape: (n_configs, n_symbols, 3)
        processed_configs = self._process_configs(configs=configs).clone().detach()

        n_configs = len(configs)
        n_symbols = processed_configs.size()[1]  # (n_configs, n_symbols, 3)

        # Per config, weigh the counts of parts
        K = self.weights * processed_configs
        assert K.size() == processed_configs.size(), \
            f"{K.size()} != {processed_configs.size()}"

        # Per config, sum the counts of parts
        K = torch.sum(K, dim=2)
        assert K.size() == (n_configs, n_symbols), \
            f"{K.size()} != {(n_configs, n_symbols)}"

        K = K @ K.T
        K = _normalize_gram(K)
        assert K.size() == (n_configs, n_configs), \
            (K.size(), (n_configs, n_configs))

        assert bool((torch.diag(K) != 0.0).all()), f"Found value 0.0 in diagonal: {K}"
        assert bool((torch.max(K, dim=1).values - torch.diag(K) <= 1e-5).all()), (
            f"Max value not in diagonal: {torch.max(K, dim=1).values}, {torch.diag(K)}, "
            f"{torch.max(K, dim=1).values == torch.diag(K)}"
        )

        return K
