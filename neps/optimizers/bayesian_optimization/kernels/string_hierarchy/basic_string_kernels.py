from __future__ import annotations

import functools
import logging

import torch

from . import config_string
from . import string_kernel_abc

_logger = logging.getLogger(__name__)


# StringKernelV1

class StringKernelV1(string_kernel_abc.StringKernel):
    def __init__(self, hierarchy_level: int | None = None):
        super().__init__()
        self._hierarchy_level = hierarchy_level

    def __str__(self) -> str:
        return f"{self.__name__}(hierarchy_level={self._hierarchy_level})"

    @classmethod
    def _get_symbols_from_configs(
        cls,
        configs: tuple[config_string.ConfigString],
    ) -> set[str]:
        symbols = set()
        for config in configs:
            for item in config.unwrapped:
                symbols.add(item.operator)
                if item.sub_config:
                    symbols.add(item.sub_config)
                    symbols.add(f"{item.operator} ({item.sub_config})")
        return symbols

    @classmethod
    def _process_configs(
        cls,
        configs: tuple[config_string.ConfigString],
    ) -> torch.Tensor:
        symbols = cls._get_symbols_from_configs(configs=configs)
        symbols = sorted(list(symbols), key=len)

        n_symbols = len(symbols)
        n_configs = len(configs)
        result = torch.zeros(size=(n_configs, n_symbols))

        for i, c in enumerate(configs):
            vals = {}
            for part in c.unwrapped:
                vals[part.operator] = (
                    vals.get(part.operator, 0)
                    + 1 * (c.max_hierarchy_level / part.level) / 10
                )
                if part.sub_config:
                    vals[part.sub_config] = (
                        vals.get(part.sub_config, 0)
                        + 1 * (c.max_hierarchy_level / part.level) / 20
                    )
                    joined_op_subconfig = f"{part.operator} ({part.sub_config})"
                    vals[joined_op_subconfig] = (
                        vals.get(joined_op_subconfig, 0)
                        + 1 * (c.max_hierarchy_level / part.level)
                    )

            vec = [vals.get(item, 0) for item in symbols]
            assert len(vec) == n_symbols, f"{len(vec)} != {n_symbols}"

            vec = torch.reshape(torch.tensor(vec), (1, n_symbols))
            result[i, :] = vec

        return result

    # A cached class method.
    # Computations are the same no matter the seed and instance.
    # At different hierarchy levels, the config tuple is different,
    #  so there will not be collisions in the cached values
    #  between different instances.
    # Class method so that the lru_cache does not prevent garbage collection
    #  of unused kernel instances (by keeping a reference to `self`).
    @classmethod
    @functools.lru_cache(maxsize=64)  # no specific meaning, a good default
    def _transform(cls, configs: tuple[config_string.ConfigString]):
        processed_configs = cls._process_configs(configs)
        result = processed_configs @ processed_configs.T
        result = cls.normalize_gram(result)

        n_configs = len(configs)
        assert result.shape == (n_configs, n_configs), \
            (result.shape, (n_configs, n_configs))

        return result

    def transform(self, configs: tuple[config_string.ConfigString]):
        if self._hierarchy_level is not None:
            configs = tuple(
                c.at_hierarchy_level(self._hierarchy_level) for c in configs
            )

        _logger.debug(f"Called method `transform` of kernel `%s`", self)
        _logger.debug("Count of received config strings: %d", len(configs))
        K = self.__class__._transform(configs=configs)
        _logger.debug(
            "Cache stats (_transform): %r",
            self.__class__._transform.cache_info(),
        )
        _logger.debug("Returning K of size %s", K.shape)
        return K


# StringKernelV2

class StringKernelV2(string_kernel_abc.StringKernel):
    def __init__(self, hierarchy_level: int | None = None):
        super().__init__()
        self._hierarchy_level = hierarchy_level

    def __str__(self) -> str:
        return f"{self.__name__}(hierarchy_level={self._hierarchy_level})"

    @classmethod
    def _process_configs(
        cls,
        configs: tuple[config_string.ConfigString],
    ) -> torch.Tensor:
        n_configs = len(configs)
        result = torch.zeros(size=(n_configs, n_configs))

        config_data = []
        for c in configs:
            config_level_data = []
            for level in range(1, c.max_hierarchy_level + 1):
                relevant_items = (i for i in c.unwrapped if i.level == level)
                elements_at_level = [i.operator for i in relevant_items]
                config_level_data.append(elements_at_level)
            config_data.append(config_level_data)

        # compute the lower triangle part of the Gram
        for i1, c1_data in enumerate(config_data):
            for i2 in range(i1 + 1):  # include the diagonal
                c2_data = config_data[i2]

                # `+ 1` since list indexing from 0, hierarchy indexing from 1
                max_common_level = min(len(c1_data), len(c2_data)) + 1
                same_count = 0
                for level in range(1, max_common_level):
                    # `- 1` since list indexing from 0, hierarchy indexing from 1
                    c1_at_level = c1_data[level - 1]
                    c2_at_level = c2_data[level - 1]
                    min_level_length = min(len(c1_at_level), len(c2_at_level))
                    for level_item_idx in range(min_level_length):
                        if c1_at_level[level_item_idx] == c2_at_level[level_item_idx]:
                            increment = 1
                            same_count += increment

                result[i1, i2] = same_count

        # copy the lower triangle to the upper
        result = result + result.T - torch.diag(torch.diag(result))

        return result

    # A cached class method.
    # Computations are the same no matter the seed and instance.
    # At different hierarchy levels, the config tuple is different,
    #  so there will not be collisions in the cached values
    #  between different instances.
    # Class method so that the lru_cache does not prevent garbage collection
    #  of unused kernel instances (by keeping a reference to `self`).
    @classmethod
    @functools.lru_cache(maxsize=64)  # no specific meaning, a good default
    def _transform(cls, configs: tuple[config_string.ConfigString]):
        processed_configs = cls._process_configs(configs)
        result = processed_configs
        result = cls.normalize_gram(result)

        n_configs = len(configs)
        assert result.shape == (n_configs, n_configs), \
            (result.shape, (n_configs, n_configs))

        return result

    def transform(self, configs: tuple[config_string.ConfigString]):
        if self._hierarchy_level is not None:
            configs = tuple(
                c.at_hierarchy_level(self._hierarchy_level) for c in configs
            )

        _logger.debug(f"Called method `transform` of kernel `%s`", self)
        _logger.debug("Count of received config strings: %d", len(configs))
        K = self.__class__._transform(configs=configs)
        _logger.debug(
            "Cache stats (_transform): %r",
            self.__class__._transform.cache_info(),
        )
        _logger.debug("Returning K of size %s", K.shape)
        return K
