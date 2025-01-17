# Multi-Fidelity and Prior Optimizers

This section concerns optimizers that use both Multi-Fidelity and Priors. They combine the advantages and disadvantages of both methods to exploit all available information.
For a detailed explanation of Multi-Fidelity and Priors, please refer [here](../../reference/search_algorithms/multifidelity.md) and [here](../../reference/search_algorithms/prior.md).

## Optimizers using Multi-Fidelity and Priors

### 1 `PriorBand`

Detailed explanation of `priorband`:

Link to PiBO for its explanation.
Link to HB for its explanation.
Write on how PriorBand uses PiBO to choose the next config for its HB iteration
Explain initial reliance for high fidelities on the Prior and how the incumbent takes over when there is one of max_f.

Write about what to consider when using PB in Neps (Neeratyoy)

### 2 `PriorBand+BO`

Detailed explanation of `priorband+bo`:

Link to PB for its explanation.
Write on how PB can be used together with BO for a model-based version (Neeratyoy)

Write about what to consider when using PB+BO in Neps (Neeratyoy)

### 3 `Successive Halving + Priors`

Detailed explanation of `successive halving + priors`:
Link to SH for its explanation.
Write on how Priors are used to influence the config selection.

### 4 `ASHA + Priors`

Detailed explanation and examples of `asha + priors`:
Link to ASHA for its explanation.
Write on how Priors are used to influence the config selection.
