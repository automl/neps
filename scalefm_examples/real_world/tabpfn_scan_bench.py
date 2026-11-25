import logging

from scan_bench import (
    PerformancePredictorType,
    TabPFNBenchmark,
    TabPFNConfig,
    TabPFNTarget,
)

import scalefm

TOTAL_COST = 1e17  
MAX_COST_PER_RUN = 1e16  
TARGET_FLOPS = [1e17, 1e18]

LOSS = TabPFNBenchmark(
    target=TabPFNTarget.VAL_LOSS, predictor_type=PerformancePredictorType.ENSEMBLE_XGB
)
REAL_DATA_NLL = TabPFNBenchmark(
    target=TabPFNTarget.NLL, predictor_type=PerformancePredictorType.ENSEMBLE_XGB
)


def evaluate_pipeline(
    total_cells,
    embedding_size,
    num_layers,
    max_features,
    num_datapoints_max,
    lr,
    effective_batch_size,
):
    config = TabPFNConfig(
        total_cells=total_cells,
        embedding_size=embedding_size,
        num_layers=num_layers,
        max_features=max_features,
        num_datapoints_max=num_datapoints_max,
        lr=lr,
        effective_batch_size=effective_batch_size,
    )
    loss = LOSS.query(config)["predictions"]["mean"]
    real_data_nll = REAL_DATA_NLL.query(config)["predictions"]["mean"]
    return {
        "objective_to_minimize": loss,
        "cost": LOSS.flops(config),
        "downstream_performances": {"real_data_nll": real_data_nll},
        "n_params": LOSS.model_params(config),
        "n_data": total_cells,
    }


def estimate_flops(
    total_cells,
    embedding_size,
    num_layers,
    max_features,
    num_datapoints_max,
    lr,
    effective_batch_size,
):
    config = TabPFNConfig(
        total_cells=total_cells,
        embedding_size=embedding_size,
        num_layers=num_layers,
        max_features=max_features,
        num_datapoints_max=num_datapoints_max,
        lr=lr,
        effective_batch_size=effective_batch_size,
    )
    return LOSS.flops(config)


class TabPFNSpace(scalefm.PipelineSpace):
    total_cells = scalefm.Integer(2**20, 2**35, log=True, is_compute_scaling=True)
    embedding_size = scalefm.Integer(
        4, 256, log=True, log_base=2, is_compute_scaling=True
    )
    num_layers = scalefm.Integer(2, 32, log=True, log_base=2, is_compute_scaling=True)
    max_features = scalefm.Integer(32, 128, log=True, log_base=2)
    num_datapoints_max = scalefm.Integer(128, 512, log=True, log_base=2)
    lr = scalefm.Float(1e-4, 5e-2, log=True)
    effective_batch_size = scalefm.Integer(16, 256, log=True, log_base=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_directory = "results/tabpfn_scan_bench"
    scalefm.acquire(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=TabPFNSpace(),
        root_directory=root_directory,
        optimizer="random_search",
        cost_estimator=estimate_flops,
        max_cost_per_config=MAX_COST_PER_RUN,
        total_cost_to_spend=TOTAL_COST,
    )
    scalefm.predict(root_directory, TARGET_FLOPS)
