"""One-time preparation of the LAION pre-training cache.

LAION webdataset shards hold full-size JPEGs (~150 MB-1 GB per shard). Training
directly off those would spend most of its time decoding large images on the
CPU -- fatal for the scaling study in `scaling_study/`, where the whole point
is to measure what the *GPUs* do as their number grows.

So this script reads the shards once, resizes every image to
`common.IMAGE_SIZE`, re-encodes it small, and writes a compact local parquet
cache (roughly 530 MB for 100k samples). Shards are never copied to disk.

Nothing is fetched that is already available. In order:

  1. If the parquet cache (`--cache_dir`) already holds enough samples, this
     exits immediately -- no shards read, no download.
  2. Otherwise it fills the cache from local `.tar` shards (`--shards_dir`),
     e.g. a LAION-400M copy already staged on the cluster. No network needed.
  3. Only if those are missing does it stream shards from the Hugging Face Hub.

    python download_data.py                       # 100k samples, the default
    python download_data.py --n_samples 20000     # smaller cache to try things out
    python download_data.py --shards_dir /path/to/train_data
    python download_data.py --cache_dir /work/$USER/laion_cache

`--cache_dir` and `--shards_dir` default to `common.LAION_CACHE_DIR` /
`common.LAION_SHARDS_DIR`, which also honour the `NEPS_LAION_CACHE_DIR` and
`NEPS_LAION_SHARDS` environment variables -- set those in your sbatch scripts
so training jobs read the same cache this script writes.

It is resumable and idempotent: one parquet part is written per source shard,
and re-running only fills in the parts that are missing.
"""

import argparse
from pathlib import Path

from common import LAION_CACHE_DIR, LAION_REPO, LAION_SHARDS_DIR, local_shard_path, prepare_laion

# #CHANGE_ME: how many image/caption pairs to cache. The scaling study wants
# enough data that each GPU has real work to do -- see N_TRAIN in
# `scaling_study/train_ddp.py`, which this must cover.
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
