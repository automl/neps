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

Link to BO-explanation (extern), to explain BO and acquisition.

`PiBO` is an extension of Bayesian Optimization (BO) that uses a special `acquisition function` that incorporates Priors. Precisely, PiBO uses a `prior-factor` that decays over time, to give more weight to the Prior in the beginning of the optimization process. This way, the optimizer first relies on the existing knowledge, before shifting focus to the data acquired during the optimization process.
The altered acquisition function takes this form:
$\boldsymbol{x}_n\in \argmax_{\boldsymbol{x}\in\mathcal{X}}\alpha(\boldsymbol{x},\mathcal{D}_n)\pi(\boldsymbol{x})^{\beta/n}$
where after $n$ evaluations, the Prior-function $\pi(\boldsymbol{x})$ is decayed by a factor $\beta/n$ and multiplied with the acquisition function $\alpha(\boldsymbol{x},\mathcal{D}_n)$. In our `PiBO` implementation, we use `Expected Improvement` as the acquisition function.

The following illustration from the `PiBO`-paper shows the influence of a well-chosen and a bad, deacying Prior on the optimization process:

![Prior-Acquisition function](../../doc_images/optimizers/pibo_acqus.png)

### Practical Tips

Write about what to consider when using pibo in Neps.

___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
