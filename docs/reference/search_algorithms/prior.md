# Prior Optimizers

## What are Priors?

Priors are used when there exists some information about the search space, that can be used to guide the optimization process. This information could come from expert domain knowledge or previous experiments. A Prior is provided in the form of a distribution over one dimension of the search space, with a `mean` (the suspected optimum) and a `confidence level`, or `variance`. We discuss how Priors can be included in your Neps-search space [here](../../reference/pipeline_space.md#using-your-knowledge-providing-a-prior).

### Advantages of using Priors

- **Less compute**: By providing a Prior, the optimizer can focus on the most promising regions of the search space, potentially saving a lot of compute.
- **More exploitation**: By focusing on these regions, the optimizer might find a better final solution.

### Disadvantages of using Priors

- **Less exploration**: By focusing on these regions, the optimizer _might_ miss out on other regions that could potentially be better.
- **Bad priors**: If the Prior is not a good representation of the search space, the optimizer might deliver suboptimal results, compared to a search without Priors.

In the following, we will discuss the Neps-optimizers that use Priors.

## 1 `PiBO`

Detailed explanation of `pibo`:

Link to BO-explanation (extern), to explain BO and acquisition.
Write about the extended acquisition function with decaying prior-factor.

Then show some example (page 6) of the influence of priors.

Write about what to consider when using pibo in Neps.

___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
