# Prior Optimizers

## What are Priors?

Priors are used when there exists some information about the search space, that can be used to guide the optimization process. This information could come from expert domain knowledge or previous experiments. A Prior is provided in the form of a distribution over one dimension of the search space, with a `mean` (the suspected optimum) and a `confidence level`, or `variance`. We discuss how Priors can be included in your NePS-search space [here](../../reference/pipeline_space.md#using-your-knowledge-providing-a-prior).

!!! tip "Advantages of using Priors"

    - **Less compute**: By providing a Prior, the optimizer can focus on the most promising regions of the search space, potentially saving a lot of compute.
    - **More exploitation**: By focusing on these regions, the optimizer might find a better final solution.

!!! warning "Disadvantages of using Priors"

    - **Less exploration**: By focusing on these regions, the optimizer _might_ miss out on other regions that could potentially be better.
    - **Bad priors**: If the Prior is not a good representation of the search space, the optimizer might deliver suboptimal results, compared to a search without Priors.

In the following, we will discuss the NePS-optimizers that use Priors.

## Optimizers using Priors

### 1 `PiBO`

`PiBO` (see [paper](https://arxiv.org/pdf/2204.11051)) is an extension of [Bayesian Optimization (BO)](../search_algorithms/bayesian_optimization.md) that uses a specific `acquisition function` that incorporates Priors, by including a `Prior-factor` that decays over time. This way, the optimizer first relies on the Prior knowledge, before shifting focus to the data acquired during the optimization process.
The altered acquisition function takes this form:

$$\boldsymbol{x}_n\in \underset{\boldsymbol{x}\in\mathcal{X}}{\operatorname{argmax}}\alpha(\boldsymbol{x},\mathcal{D}_n)\pi(\boldsymbol{x})^{\beta/n}$$

where after $n$ evaluations, the Prior-function $\pi(\boldsymbol{x})$ is decayed by the factor $\beta/n$ and multiplied with the acquisition function $\alpha(\boldsymbol{x},\mathcal{D}_n)$. In our `PiBO` implementation, we use [`Expected Improvement`](../search_algorithms/bayesian_optimization.md#the-acquisition-function) as the acquisition function.

The following illustration from the `PiBO`-paper shows the influence of a well-chosen and a bad, decaying Prior on the optimization process:

|![Prior-Acquisition function](../../doc_images/optimizers/pibo_acqus.jpg "This is a delicious bowl of ice cream.")|
|:--:|
|Left: A well-located Prior influences the acquisition function leading to quicker convergence and even more exploration. Right: An off-center Prior slows down, but does not prevent convergence. (Image Source: [PiBO-paper](https://arxiv.org/pdf/2204.11051), Jan 27, 2025)|

In both cases, the optimization process uses the additional information provided by the Prior to arrive at the solution, however, the bad Prior (right) results in a slower convergence to the optimum.

!!! example "Practical Tips"

    TODO Write about what to consider when using `PiBO` in NePS.

!!! info
    PiBO is chosen as the [default optimizer](../../reference/optimizers.md#21-automatic-optimizer-selection) in NePS when there is only Prior, but no [Multi-Fidelity](../search_algorithms/multifidelity.md) information available.
___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
