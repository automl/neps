# Multi-Fidelity Optimizers

## 1 `Successive Halving`

Detailed explanation of `successive halving`:

Write on the concept of SH, keeping the best 1/x algorithms alive.

Explain the problems?

## 2 `HyperBand`

Detailed explanation of `hyperband`:

Link to Sucessive Halving for its explanation.
Write how its called as subroutine and the advantages.

Explain parallelization mode

## 3 `ASHA`

Detailed explanation of `asha`:

Link to Sucessive Halving for its explanation.
Explain the problems of SH for parallelization and how rungs are used to maximize compute use and quick promotion

## 4 `Mobster`

Detailed explanation of `mobster`:

Link to BO-explanation (extern), to explain BO.

Explain the problem when parallelizing BO and how Mobster fantasizes outcomes via joint GP
Explain the difference between using promotion and stopping and advantages of each (good defaults vs conservative)
-> To consider when using Mobster in Neps.

## 5 `IfBO`

Detailed explanation of `IfBO`:

Link to FT, PFNs and BO-explanation (extern), to explain BO, In-Context-Learning and Freeze-Thaw.

Explain the combination of the a PFN as surrogate for Freeze-Thaw-BO.

Write about what to consider when using IfBO  in Neps
___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
