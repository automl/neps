"""
This example demonstrates how to use NePS to search for an optimizer
"""

import neps
import torch


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


class OptimizerSpace(neps.PipelineSpace):

    _gradient_clipping = neps.Float(0.5, 1.0)
    _learning_rate = neps.Float(0.0001, 0.01, log=True)

    _functions = neps.Categorical(
        choices=(torch.sqrt, torch.log, torch.exp, torch.sign, torch.abs)
    )

    # The optimizer class constructed with sampled hyperparameters
    # and functions
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
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/optimizer_search_example",
        evaluations_to_spend=5,
        overwrite_root_directory=True,
    )
    neps.status(
        root_directory="results/optimizer_search_example",
        print_summary=True,
    )
