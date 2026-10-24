"""Build the tokenized pre-training cache used for training (`val.bin` and `train.bin`).

Does nothing if the cache already holds enough tokens.

    python pipeline/download_data.py                              # 50M training tokens, the default
    python pipeline/download_data.py --n_train_tokens 100000000   # for larger n_tokens choices
    python pipeline/download_data.py --cache_dir /work/$USER/llm_tokens

The default location can also be set via `NEPS_LLM_CACHE_DIR`.
"""

import argparse
from pathlib import Path

from common import TOKEN_CACHE_DIR, prepare_tokens

# #CHANGE_ME: must cover the largest `n_tokens` choice in `generate_configs.py`.
DEFAULT_N_TRAIN_TOKENS = 50_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_train_tokens", type=int, default=DEFAULT_N_TRAIN_TOKENS)
    parser.add_argument("--cache_dir", type=Path, default=TOKEN_CACHE_DIR)
    args = parser.parse_args()

    prepare_tokens(args.n_train_tokens, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
