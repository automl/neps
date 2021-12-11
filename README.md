# Neural Pipeline Search

Neural Pipeline Search helps deep learning experts find the best neural pipeline.

## Installation

Using pip

```bash
pip install git+https://github.com/automl-private/neps.git
```

To install torch you might want to use our utility `python -m neps.utils.install_torch`.

## Usage

```python
import neps


def run_pipeline(config, config_working_directory, previous_working_directory):
    return {"loss": config["x"]}


pipeline_space = neps.SearchSpace(
    x=neps.FloatParameter(lower=0, upper=1, log=False),
)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/usage",
    n_iterations=5,
    searcher="bayesian_optimization",
    overwrite_logging=True,
    hp_kernels=["m52"],
)
```

Please also see our examples in [neps_examples](neps_examples).

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
