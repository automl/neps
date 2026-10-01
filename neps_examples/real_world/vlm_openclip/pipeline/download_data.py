"""Build the resized LAION parquet cache used for training (resumable).

Uses the existing cache if full, else local `.tar` shards, else streams from the HF Hub.

    python download_data.py                       # 100k samples, the default
    python download_data.py --n_samples 20000     # smaller cache to try things out
    python download_data.py --shards_dir /path/to/train_data
    python download_data.py --cache_dir /work/$USER/laion_cache

Defaults can also be set via `NEPS_LAION_CACHE_DIR` / `NEPS_LAION_SHARDS`.
"""

import argparse
from pathlib import Path

from common import LAION_CACHE_DIR, LAION_REPO, LAION_SHARDS_DIR, local_shard_path, prepare_laion

# #CHANGE_ME: how many image/caption pairs to cache. The scaling study wants
# enough data that each worker has real work to do -- see N_TRAIN in
# `scaling_study/train.py`, which this must cover.
DEFAULT_N_SAMPLES = 100_000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--cache_dir", type=Path, default=LAION_CACHE_DIR,
        help="Where the prepared parquet cache lives (and is reused from).",
    )
    parser.add_argument(
        "--shards_dir", type=Path, default=LAION_SHARDS_DIR,
        help="Local directory of webdataset .tar shards; falls back to the Hub if absent.",
    )
    parser.add_argument(
        "--first_shard", type=int, default=0,
        help="Shard index to start from; only affects which shards a fresh cache pulls.",
    )
    args = parser.parse_args()

    if local_shard_path(args.first_shard, args.shards_dir) is not None:
        print(f"Using local shards from {args.shards_dir} (no download).")
    else:
        print(f"No local shards at {args.shards_dir}; will stream from {LAION_REPO} if needed.")

    prepare_laion(
        args.n_samples,
        first_shard=args.first_shard,
        cache_dir=args.cache_dir,
        shards_dir=args.shards_dir,
    )


if __name__ == "__main__":
    main()
