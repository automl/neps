# Multi-Fidelity Optimizers

## What is Multi-Fidelity Optimization?

Multi-Fidelity (MF) optimization leverages the idea of running an AutoML problem on a small scale, which is cheaper and faster, and then using this information to train full-scale models. The _low-fidelity_ runs could be on a smaller dataset, a smaller model, or for shorter training times. MF-algorithms then infer which configurations are likely to perform well on the full problem, before investing larger compute amounts.

!!! tip "Advantages of Multi-Fidelity"

    - **Parallelization**: MF-algorithms can use the information from many parallel low-fidelity runs to guide the search in the few high-fidelity runs.
    - **Exploration**: By using low-fidelity runs, the optimizer can explore more of the search space.

!!! warning "Disadvantages of Multi-Fidelity"

    - **More compute**: Running multiple iterations on different fidelities is generally more compute-intensive.
    - **Variance**: The performance of a configuration on a low-fidelity run might not correlate well with its performance on a high-fidelity run. This can result in misguided decisions.

## Optimizers using Multi-Fidelity

### 1 `Successive Halfing`

`Successive Halfing`/`SH` (see [paper](https://proceedings.mlr.press/v51/jamieson16.pdf)) is a simple but effective Multi-Fidelity algorithm.

It starts with a large number of random configurations and evaluates them on a low-fidelity. The best-performing $1/\eta$ configurations are then promoted to the next fidelity, where they are evaluated again. This process is repeated until only a few configurations remain, evaluated on the highest fidelity.
The process allows for broad exploration in the beginning and focus on the most promising configurations towards the end.

!!! example "Practical Tips"

    - For the same total compute, `SH` outperforms uninformed search algorithms like random search or grid search.
    - It highly depends on the correlation between lower and higher fidelities. If the correlation is low, `SH` underperforms.
    - `SH` has two parameters: $\eta$ and $n$, where $\eta$ is the promotion factor and $n$ is the number of configurations at the lowest fidelity.
    This results in a total of $\frac{n*r}{\eta^r}$ steps (from one fidelity level to the next), where $r$ is the number of fidelity levels.

#### _Asynchronous_ Successive Halving

`Asynchronous Successive Halving`/`ASHA` (see [paper](https://arxiv.org/pdf/1810.05934)) is an asynchronous version of SH that maximizes parallel evaluations.

Instead of waiting for all $n$ configurations to finish on one fidelity, `ASHA` promotes the best configuration to the next fidelity as soon as there are enough evaluations to make a decision ($1/\eta*n\geq 1$). This allows for quicker promotions and earlier high fidelity-results. When there are no promotable configurations, `ASHA` spawns new configurations at the lowest fidelity, so it always utilizes the available compute and increases exploration compared to ``SH``.

#### _Prior-extended_ Successive Halving

Although not inherently a Prior-optimizer, ``SH`` (and ``ASHA``) can make use of [Priors](../search_algorithms/prior.md). Instead of sampling configurations uniformly, the optimizer can directly sample from the Prior, which results in a more focused search - highly beneficial _if_ the Prior is reliable. Alternatively, the ``SH`` can bias the promotion of configurations towards the Prior, keeping worse-performing, but recommended configurations longer in the optimization process.

### 2 `HyperBand`

`HyperBand`/`HB` (see [paper](https://arxiv.org/pdf/1603.06560)) is an extension of [``Successive Halfing``](../search_algorithms/multifidelity.md#1-successive-halfing) that employs multiple ``Successive Halfing``-rounds in parallel.

Each of these runs has a different resource budget and different number of configurations. This makes ``HyperBand`` more flexible and parallelizable than ``SH``.

!!! example "Practical Tips"

    - ``HyperBand`` is a good choice when you have a limited budget and want to parallelize your search.
    - It is more efficient than ``SH`` when the correlation between lower and higher fidelities is low.
    - ``Hyperband`` has two parameters: $\eta$ (typically 3 or 4) and $R$, where $\eta$ is the promotion factor and $R$ is the maximum budget any single configuration will be trained on. A larger $R$ will result in better, but slower results, while a larger $\eta$ will result in faster, but more noisy, potentially worse results. HB then spawns $\lfloor \log_\eta(R)\rfloor$ ``Successive Halfing``-rounds.

!!! info
    ``HyperBand`` is chosen as the [default optimizer](../../reference/optimizers.md#21-automatic-optimizer-selection) in NePS when there is no [Prior](../search_algorithms/prior.md), only Multi-Fidelity information available.

### 3 `BOHB`

`BOHB` (see [paper](https://arxiv.org/pdf/1807.01774)) is a combination of [Bayesian Optimization (BO)](../search_algorithms/bayesian_optimization.md) and [HyperBand](../search_algorithms/multifidelity.md#2-hyperband).

Contrary to ``HyperBand``, which uses random configurations, ``BOHB`` uses ``BO`` to choose the next configurations for ``HyperBand``. This way, it can leverage the advantages of both algorithms: the flexibility and parallelization of ``HyperBand`` for low budgets and the efficiency of BO for higher budgets.

!!! example "Practical Tips"

    - ``BOHB`` is more efficient than both ``HyperBand`` and BO on their own.
    - The effects of ``BO`` only start showing after some evaluations at full fidelity, as it needs those to build its model.
    - It has the same hyperparameters as ``HyperBand``, plus the choice of surrogate and acquisition functions from ``BO``.

### 4 `A-BOHB`

`A-BOHB`/`Mobster` (see [paper](https://arxiv.org/pdf/2204.11051)) is an asynchronous extension of [BOHB](../search_algorithms/multifidelity.md#3-bohb).

Unlike ``BOHB``, which only models the objective function at the highest fidelity, ``A-BOHB`` uses a ``joint Gaussian Process`` to model the objective function across all fidelities. This way, it can leverage the information from all fidelities to make better decisions.
To make this process asynchronous, i.e. run several configurations in parallel, ``A-BOHB`` has to anticipate the results of configurations that are still running. It does this by _fantasizing_ the results of the running configurations and using those fantasies in the acquisition function $\hat{a}$ to decide for the next configuration. Precisely, ``A-BOHB`` marginalizes out the possible results $y_j$ of a running configuration $x_j$:
$$
\hat{a}(\boldsymbol{x}) = \int a(\boldsymbol{x}, y_j)p(y_j|x_j) dy_j
$$
where $a(\boldsymbol{x}, y_j)$ is the acquisition function and $p(y_j|x_j)$ is the distribution of the possible results of $x_j$.
``A-BOHB`` also uses a promotion mechanism similar to [ASHA](../search_algorithms/multifidelity.md#asynchronous-successive-halving) to decide when to promote configurations to higher fidelities and when to stop them, maximizing parallelization.

!!! example "Practical Tips"

    - ``A-BOHB`` is more efficient than ``BOHB`` when the correlation between lower and higher fidelities is low.
    - The algorithm itself is more computationally expensive than ``BOHB``, as it has to model the objective function across all fidelities.

### 5 `In-Context Freeze-Thaw Bayesian Optimization`

`In-Context Freeze-Thaw Bayesian Optimization`/``IfBO`` (see [paper](https://arxiv.org/pdf/2204.11051)) expands on the idea of [Freeze-Thaw Bayesian Optimization](https://arxiv.org/pdf/1406.3896) (``FT-BO``) by using a `Prior-data fitted network` (PFN) as a surrogate for the ``FT-BO``.

Standard ``FT-BO`` models the performance of a configuration with a Gaussian Process, assuming exponential loss decay. Similar to [A-BOHB](../search_algorithms/multifidelity.md#4-a-bohb), it uses this joint GP to fantasize results and decides for the most informative configurations. The ``Entropy Search``-acquisition function (see [paper](https://jmlr.csail.mit.edu/papers/volume13/hennig12a/hennig12a.pdf)) quantifies this information gain:
$$
a(\boldsymbol{x}) = \int\left(H\left(P^y_{\min}\right)\right) - \left(H\left(P_{\min}\right)\right)P(y|\{(\boldsymbol{x},y_n)\}^N)dy
$$
where $H$ is the entropy, $P_{\min}$ is the distribution of the minimum value, $P^y_{\min}$ is the same distribution, but given a new observation $y$ and $P(y|\{(\boldsymbol{x},y_n)\}^N)$ is the probability of this $y$, from a configuration $\boldsymbol{x}$ (given the observations so far). So the acquisition function maximizes the information gain about the location of the minimum from evaluating any configuration $\boldsymbol{x}$.

|![Fantasizing](../../doc_images/optimizers/freeze_thaw_fantasizing.png)|
|:--:|
|The image shows the fantasizing of exponential loss decay in`FT-BO` . (Image Source: [FT-BO-paper](https://arxiv.org/pdf/1406.3896), Jan 27, 2025)|

``IfBO`` employs the same concept, but instead of a GP, it uses a PFN to model the performance of configurations. PFNs (see [paper](https://arxiv.org/pdf/2112.10510)) are transformer networks, fitted to many (synthetic) runs. They can model the performance of configurations across all fidelities and are used in ``IfBO`` to fantasize the outcomes. The deciding advantage is that PFNs can model complex relationships between configurations and fidelities, not just exponential decay. On top of that, PFNs utilize [in-context learning](https://arxiv.org/pdf/2112.10510) to quickly adapt from their general prior to the current optimization process, resulting in a better overall performance compared to GPs.

Lastly, ``IfBO`` adapts the `FT-BO` idea of _freezing_ (pausing training on) configurations that are not informative anymore and _thawing_ (resuming training on) them when they become interesting again. It therefore chooses automatically between starting new configurations or thawing old ones.

|![Freeze-Thaw](../../doc_images/optimizers/freeze_thawing.png)|
|:--:|
|The image shows the Freeze-Thaw-mechanism, with the colors indicating, at what iteration a configuration has been evaluated at this fidelity. Note for example some yellow configurations being reused much later, ending in red. (Image Source: [FT-BO-paper](https://arxiv.org/pdf/1406.3896), Jan 27, 2025)|

!!! example "Practical Tips"

    TODO Do we even use it?
___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
