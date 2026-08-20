from __future__ import annotations

import torch

# NOTE: The GP-based optimizers (bayesian_optimization, pibo, ifbo) spend nearly all
# of their time in many *small* linear-algebra calls -- fitting a GP over a handful of
# trials means repeated Cholesky factorizations of matrices that are only tens of rows
# wide. Handing work that small to torch's default intra-op thread pool costs far more
# in thread synchronization than it saves: on a 16-core machine the same 20 BO asks
# take ~80s with 16 threads and ~3s with 1.
#
# The tests run one trial at a time, so there is nothing to gain from intra-op
# parallelism here, and pinning to a single thread also keeps the run well-behaved
# under `pytest -n` where every worker would otherwise spin up its own full pool.
torch.set_num_threads(1)
