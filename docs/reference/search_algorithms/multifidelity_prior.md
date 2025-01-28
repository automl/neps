# Multi-Fidelity and Prior Optimizers

This section concerns optimizers that use both Multi-Fidelity and Priors. They combine the advantages and disadvantages of both methods to exploit all available information.
For a detailed explanation of Multi-Fidelity and Priors, please refer [here](../../reference/search_algorithms/multifidelity.md) and [here](../../reference/search_algorithms/prior.md).

## Optimizers using Multi-Fidelity and Priors

### 1 `PriorBand`

`PriorBand` is an extension of [`HyperBand`](../../reference/search_algorithms/multifidelity.md#2-hyperband) that utilizes expert Priors to choose the next configuration.

``PriorBand``'s sampling module $\mathcal{E}_\pi$ balances the influence of the Prior, the incumbent configurations and randomness to select configurations.

|![PriorBand's Sampler](../../doc_images/optimizers/priorband_sampler.png)|
|:--:|
|The ``PriorBand`` sampling module balances the influence of the Prior, the $1/\eta$ incumbent configurations and randomness to select configurations. (Image Source: [PriorBand-paper](https://openreview.net/pdf?id=uoiwugtpCH), Jan 27, 2025)|

The Prior sampling $p_\pi$ is most meaningful at full fidelity and when not much data is available yet, while the incumbent sampling $p_{\hat{\lambda}}$, coming from actual data, is most significant but sparse, and random sampling $p_{\mathcal{U}}$ is needed for exploration, especially at lower fidelities. This results in these inital sampling probabilities when there is no incument yet:

$$
p_{\mathcal{U}}=1/(1+\eta^r)\\
p_\pi=1-p_{\mathcal{U}}\\
p_{\hat{\lambda}}=0
$$

where $\eta$ is the promotion-hyperparameter from [`HyperBand`](../../reference/search_algorithms/multifidelity.md#2-hyperband) and $r$ is the current fidelity level (_rung_), showing the decay of the random sampling probability with increasing fidelity.

When there is an incumbent, the probabilities are adjusted to:

$$
p_{\mathcal{U}}=1/(1+\eta^r)\\
p_\pi=p_\pi\cdot\mathcal{S}_{\hat{\lambda}}/(\mathcal{S}_\pi+\mathcal{S}_{\hat{\lambda}})\\
p_{\hat{\lambda}}=p_{\hat{\lambda}}\cdot\mathcal{S}_{\pi}/(\mathcal{S}_\pi+\mathcal{S}_{\hat{\lambda}})
$$

where $\mathcal{S}_\pi$ and $\mathcal{S}_{\hat{\lambda}}$ are the summed probabilities of the top $1/\eta$ configurations under Prior and incumbent sampling, respectively. This way, the balance is shifted towards the distribution that would have yielded the best configurations so far. Crucially, this compensates for potentially bad Priors, as the incumbent sampling will take over when it has proven to be better.

!!! example "Practical Tips"

    - ``PriorBand`` is a good choice when you have a Prior but are wary of its quality and you can utilize Multi-Fidelity.

!!! info

    `PriorBand` is chosen as the [default optimizer](../../reference/optimizers.md#21-automatic-optimizer-selection) in NePS when there is both [Prior](../search_algorithms/prior.md) and [Multi-Fidelity](../search_algorithms/multifidelity.md) information available.

#### _Model-based_ `PriorBand`

`PriorBand` can also be extended with a model, where after $n$ evaluations, a [`BO`](../search_algorithms/bayesian_optimization.md) model is trained to advise the sampling module.
