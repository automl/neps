from __future__ import annotations

import torch

# NOTE: The GP-based optimizers: fitting a GP over a handful of
# trials means repeated Cholesky factorizations of matrices that are only tens of rows
# wide. Handling work that small to torch's default intra-op thread
# pool costs far more than it saves.
torch.set_num_threads(1)
