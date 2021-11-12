import comprehensive_nas as cnas
from comprehensive_nas.optimizers.bayesian_optimization.kernels import (
    StationaryKernelMapping,
)
import numpy as np
import time
import logging


def run_pipeline(  # pylint: disable=unused-argument
    config, config_working_directory, previous_working_directory
):
    x = np.array(config.get_hps(), dtype=float)
    start = time.time()
    y = -float(np.sum(x))
    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "config_id": config.id,
            "val_score": y,
            "test_score": y,
            "train_time": end - start,
        },
    }


if __name__ == "__main__":
    pipeline_space = cnas.SearchSpace(
        cnas.HyperparameterMapping['float'](name='x1', lower=0, upper=1, log=False),
        cnas.HyperparameterMapping['float'](name='x2', lower=0, upper=1, log=False),
        cnas.HyperparameterMapping['float'](name='x3', lower=0, upper=1, log=False),
        cnas.HyperparameterMapping['float'](name='x4', lower=0, upper=1, log=False),
        cnas.HyperparameterMapping['float'](name='x5', lower=0, upper=1, log=False),
        cnas.HyperparameterMapping['categorical'](name='x6', choices=[0, 1]),
        cnas.HyperparameterMapping['categorical'](name='x7', choices=[0, 1]),
        cnas.HyperparameterMapping['categorical'](name='x8', choices=[0, 1]),
        cnas.HyperparameterMapping['categorical'](name='x9', choices=[0, 1]),
        cnas.HyperparameterMapping['categorical'](name='x10', choices=[0, 1]),
    )

    logging.basicConfig(level=logging.INFO)
    cnas.run_comprehensive_nas(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        working_directory="results/dummy_example",
        n_iterations=100,
        searcher="bayesian_optimization",
        overwrite_logging=True,
        hp_kernels=['m52', 'hm']
    )
