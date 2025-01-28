# Bayesian Optimization

## What is Bayesian Optimization?

Bayesian Optimization (BO) is a fundamental optimization technique for finding (local) optima of expensive-to-evaluate functions. The main idea of BO is an interplay of a model (the [`surrogate function`](../search_algorithms/bayesian_optimization.md#the-surrogate-function)) of the objective function, built from the data collected during the optimization process, and an [`acquisition function`](../search_algorithms/bayesian_optimization.md#the-acquisition-function) that guides the search for the next evaluation point.

### The surrogate function

For each dimension of the search space, the surrogate function models the objective function as a `Gaussian Process` (GP). A GP consists of a mean function and a covariance function, which are both learned from the data. The mean function represents the expected value of the objective function, while the covariance function models the uncertainty of the predictions.
The following image shows a GP with its mean function and the 95% confidence interval:

|![GP](../../doc_images/optimizers/bo_surrogate.jpg)|
|:--:|
|The dashed line represents the (hidden) objective function, while the solid line is the surrogate mean function. The shaded area around the mean function is it's confidence interval. Note that the confidence interval collapses where observations have been made and gets large in regions where no data is available yet. (Image Source: [Medium.com](https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083), Jan 27, 2025)|

### The acquisition function

The acquisition function is the guiding force in BO. From the information contained in the surrogate function, the acquisition function suggests the next evaluation point. It balances the trade-off between exploration (sampling points where the surrogate function is uncertain) and exploitation (sampling points where the surrogate function is promising).

|![Acquisition function](../../doc_images/optimizers/bo_acqu.jpg)|
|:--:|
|The image shows the surrogate function from before, now with the acquisition function plotted below. The maximum of the acquisition function is the point that will usually be evaluated next. (Image Source: [Medium.com](https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083), Jan 27, 2025)|

There are numerous acquisition functions, with the most popular being `Expected Improvement` (EI):

- EI is defined as the expected improvement over the current best observation:

$$EI(\boldsymbol{x}) = \mathbb{E}[\max(0, f(\boldsymbol{x}) - f(\boldsymbol{x}^+))]$$

And `Probability of Improvement` (PI):

- PI is defined as the probability that the surrogate function is better than the current best observation:

$$PI(\boldsymbol{x}) = P(f(\boldsymbol{x}) > f(\boldsymbol{x}^+))$$

where $f(\boldsymbol{x})$ is the surrogate function and $f(\boldsymbol{x}^+)$ is the best observation so far.

To read more about BO, please refer to this [Bayesian Optimization tutorial](https://arxiv.org/abs/1807.02811) or this article on [Towards Data Science](https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f).

## BO in NePS

BO is a widely used optimization technique in AutoML, as it can handle expensive-to-evaluate, noisy, high-dimensional and black-box objectives, all of which are common challenges in AutoML. It is used in the optimization of hyperparameters, neural architectures, and the entire pipeline.

!!! info
    Therefore, BO is chosen as the default optimizer in NePS when there is no [Prior](../search_algorithms/prior.md) or [Multi-Fidelity](../search_algorithms/multifidelity.md) information available.
