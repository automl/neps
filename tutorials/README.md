# NePS Tutorials

Interactive tutorials for learning hyperparameter optimization and neural architecture search with NePS.

## Tutorials

| Tutorial | Description | Run |
|----------|-------------|-----|
| **1. Getting Started with HPO** | Basic HPO workflow, synthetic functions, and deep learning tasks | [Local](1_getting_started_hpo.py) · [Colab](https://colab.research.google.com/github/automl/neps/blob/notebooks/notebooks/neps_examples/tutorials/1_getting_started_hpo.ipynb) |
| **2. Defining Search Spaces** | Parameter types, fidelity parameters, and PipelineSpace classes | [Local](2_search_spaces.py) · [Colab](https://colab.research.google.com/github/automl/neps/blob/notebooks/notebooks/neps_examples/tutorials/2_search_spaces.ipynb) |
| **3. Efficient Optimization** | Multi-fidelity optimization, expert priors, and advanced strategies | [Local](3_efficiency_techniques.py) · [Colab](https://colab.research.google.com/github/automl/neps/blob/notebooks/notebooks/neps_examples/tutorials/3_efficiency_techniques.ipynb) |

## Quick Start

### Run in Google Colab (Recommended)
Click the Colab link in the table above. No setup required.

### Run Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run a tutorial:
```bash
python 1_getting_started_hpo.py
```

Or use Jupyter:
```bash
pip install jupytext
jupytext --to notebook 1_getting_started_hpo.py
jupyter notebook 1_getting_started_hpo.ipynb
```

## Resources

- [Documentation](https://automl.github.io/neps/latest/)
- [GitHub Repository](https://github.com/automl/neps)
- [API Reference](https://automl.github.io/neps/latest/api/neps/api/)
- [More Examples](https://github.com/automl/neps/tree/master/neps_examples)
