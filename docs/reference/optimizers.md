# Optimizer Configuration

## 1 What are Multi-Fidelity and Priors?

Low level-explanation of MF & Priors
MF: Tests on small scale before investing a lot into ONE run
Prior: Given an intuition on what could work, taking that into consideration

## 2 Automatic Optimizer Selection

If you prefer not to specify a particular optimizer for your AutoML task, you can simply pass `"default"` or `None`
for the neps searcher. NePS will automatically choose the best optimizer based on the characteristics of your search
space. This provides a hassle-free way to get started quickly.

The optimizer selection is based on the following characteristics of your `pipeline_space`:

- If it has fidelity: `hyperband`
- If it has both fidelity and a prior: `priorband`
- If it has a prior: `pibo`
- If it has neither: `bayesian_optimization`

For example, running the following format, without specifying a searcher will choose an optimizer depending on
the `pipeline_space` passed.

```python
neps.run(
    run_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # no searcher specified
)
```

## 3 Optimizer classes

![Optimizer classes](../doc_images/venn_dia.png)

### 3.1 Multi-Fidelity optimizers

Detailed explanation of MF
Advantages (Heuristic, less compute per config, possibly exploration)
Disadvantages (More compute i.g., variance, low MF-correlation)
Examples (less iterations, smaller datasets, smaller models)

For a detailed list of optimizers using Multi-Fidelity, please refer [here](../optimizer/multifidelity.md).

### 3.2 Prior optimizers

Detailed explanation of Priors
Intution (Experts recommending ranges, Intution for "Trust" in Prior)
Advantages (Less compute wasted, more exploitation)
Disadvantages (Less exploration, bad priors)

For a detailed list of optimizers using Priors, please refer [here](../optimizer/prior.md).

### 3.3 Multi-Fidelity+Prior optimizers

Link to explanation of MF and Priors

For a detailed list of optimizers using Multi-Fidelity and Priors, please refer [here](../optimizer/multifidelity_prior.md).
