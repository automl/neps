"""The TabPFN pipeline of the scaling tutorial, on the scan_bench surrogate."""

from __future__ import annotations

import pytest

pytest.importorskip("scan_bench")

from tutorials import tabpfn_scan_bench as tabpfn

CONFIG = {
    "total_cells": 2**24,
    "effective_batch_size": 64,
    "lr": 1e-3,
    "embedding_size": 16,
    "num_layers": 4,
}


def test_estimators_match_the_benchmark() -> None:
    bench = tabpfn.get_benchmark()
    config = tabpfn.build_config(**CONFIG)

    assert tabpfn.parameters_estimator(**CONFIG) == bench.model_params(config)
    assert tabpfn.flops_estimator(**CONFIG) == pytest.approx(bench.flops(config))


def test_reference_optimum_improves_with_compute() -> None:
    small, large = (tabpfn.reference_optimum(c) for c in (1e14, 5e15))

    assert large.loss <= small.loss
    assert large.n_params * large.n_data > 0
    flops = tabpfn.flops_estimator(
        embedding_size=large.embedding_size,
        num_layers=large.num_layers,
        total_cells=large.n_data,
    )
    assert flops <= 5e15 * 1.01
