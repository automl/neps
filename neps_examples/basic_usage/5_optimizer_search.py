"""
This example demonstrates optimizer search using NePS to design custom PyTorch optimizers
and find the best configuration. The search space defines a custom optimizer with three
gradient transformation functions sampled from {sqrt, log, exp, sign, abs}, a learning
rate sampled logarithmically from [0.0001, 0.01], and gradient clipping from [0.5, 1.0].
NePS evaluates each optimizer by optimizing a quadratic function to discover the most
effective combination of gradient transformations and hyperparameters for convergence.

Search Space Structure:
    optimizer_class: optimizer_constructor(
        <sampled from {sqrt, log, exp, sign, abs}>,
        <sampled from {sqrt, log, exp, sign, abs}>,
        <sampled from {sqrt, log, exp, sign, abs}>,
        learning_rate=<sampled from [0.0001, 0.01] log-scale>,
        gradient_clipping=<sampled from [0.5, 1.0]>
    )
"""

import neps
import torch
import logging


def optimizer_constructor(*functions, gradient_clipping: float, learning_rate: float):
    # Build a simple optimizer that applies a sequence of functions to the gradients
    class CustomOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            defaults = dict(
                gradient_clipping=gradient_clipping, learning_rate=learning_rate
            )
            super().__init__(params, defaults)

        def step(self, _closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    for func in functions:
                        grad = func(grad)
                    # Apply gradient clipping
                    grad = torch.clamp(
                        grad, -group["gradient_clipping"], group["gradient_clipping"]
                    )
                    # Update parameters
                    p.data.add_(grad, alpha=-group["learning_rate"])

    return CustomOptimizer


# The search space defines the optimizer class constructed with sampled hyperparameters
# and functions
class OptimizerSpace(neps.PipelineSpace):

    # Parameters with prefixed _ are internal and will not be given to the evaluation
    # function
    _gradient_clipping = neps.Float(0.5, 1.0)
    _learning_rate = neps.Float(0.0001, 0.01, log=True)

    _functions = neps.Categorical(
        choices=(torch.sqrt, torch.log, torch.exp, torch.sign, torch.abs)
    )

    # The optimizer class constructed with the sampled functions and hyperparameters
    optimizer_class = neps.Operation(
        operator=optimizer_constructor,
        args=(
            neps.Resampled(_functions),
            neps.Resampled(_functions),
            neps.Resampled(_functions),
        ),
        kwargs={
            "learning_rate": neps.Resampled(_learning_rate),
            "gradient_clipping": neps.Resampled(_gradient_clipping),
        },
    )


# In the pipeline, we optimize a simple quadratic function using the sampled optimizer
def evaluate_pipeline(optimizer_class) -> float:
    x = torch.ones(size=[1], requires_grad=True)
    optimizer = optimizer_class([x])

    # Optimize for a few steps
    for _ in range(10):
        optimizer.zero_grad()
        y = x**2 + 2 * x + 1
        y.backward()
        optimizer.step()

    return y.item()


# Run NePS with the defined pipeline and space and show the best configuration
if __name__ == "__main__":
    pipeline_space = OptimizerSpace()

    logging.basicConfig(level=logging.INFO)
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/optimizer_search_example",
        evaluations_to_spend=5,
    )
    neps.status(
        root_directory="results/optimizer_search_example",
        print_summary=True,
    )
