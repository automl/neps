# ScaleFM Tutorial

The **ScaleFM Demo** is a basic demo of ScaleFM: the basic pipeline,
visualization, and parallelisation of the runs. It runs on a CPU in about 5 minutes.

[Local](scalefm_demo.py) · [Colab](https://colab.research.google.com/github/automl/scalefm/blob/master/tutorials/scalefm_demo.ipynb)

`tabpfn_scan_bench.py` holds the pieces of the TabPFN benchmark the notebook builds
its `evaluate_pipeline` from: the surrogates of [`scan_bench`](https://pypi.org/project/scan-bench/)
(the validation loss, and the NLL on real datasets as a downstream metric), the
`TabPFNConfig` of a config, the parameter and FLOP estimators, and the surrogate's best
config within a FLOP budget. The pipeline and the search space are defined in the
notebook.

## Quick Start

### Run in Google Colab (Recommended)
Click the Colab link above. No setup required.

### Run Locally

Install dependencies, from the repository root:
```bash
pip install -r tutorials/requirements.txt
```

Run the tutorial as a script:
```bash
python tutorials/scalefm_demo.py
```

Or use Jupyter, started in `tutorials/` so that `tabpfn_scan_bench` can be imported:
```bash
pip install jupyter
cd tutorials && jupyter notebook scalefm_demo.ipynb
```

`scalefm_demo.py` is the [jupytext](https://jupytext.readthedocs.io) source of the
notebook. After editing it, regenerate the notebook:
```bash
jupytext --to notebook tutorials/scalefm_demo.py --output tutorials/scalefm_demo.ipynb --update
```

## Resources

- [Documentation](https://automl.github.io/scalefm/latest/)
- [GitHub Repository](https://github.com/automl/scalefm)
- [API Reference](https://automl.github.io/scalefm/latest/api/scalefm/api/)
- [More Examples](https://github.com/automl/scalefm/tree/master/scalefm_examples)
