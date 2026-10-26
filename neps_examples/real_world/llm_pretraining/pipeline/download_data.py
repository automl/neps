"""Build the tokenized pre-training cache used for training (`val.bin` and `train.bin`).

Does nothing if the cache already holds enough tokens.

    python pipeline/download_data.py                              # 50M training tokens, the default
    python pipeline/download_data.py --n-train-tokens 100000000   # for larger n_tokens choices

The default location can also be set via `NEPS_LLM_CACHE_DIR`.
"""

import argparse

from common import prepare_tokens

# #CHANGE_ME: must cover the largest `n_tokens` choice in `generate_configs.py`.
DEFAULT_N_TRAIN_TOKENS = 50_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-train-tokens", type=int, default=DEFAULT_N_TRAIN_TOKENS)
    args = parser.parse_args()
    prepare_tokens(args.n_train_tokens)


if __name__ == "__main__":
    main()
