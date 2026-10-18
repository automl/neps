"""Shared settings and the tokenized dataset for the LLM pre-training example"""

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent / ".data"

# #CHANGE_ME: where the token cache lives (also settable via NEPS_LLM_CACHE_DIR).
TOKEN_CACHE_DIR = Path(os.environ.get("NEPS_LLM_CACHE_DIR", DATA_DIR / "tokens"))

# #CHANGE_ME: the pre-training corpus, streamed once by `download_data.py`. A local
# directory of parquet/json files works too (with DATASET_CONFIG = None).
DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"

# GPT-2's tokenizer: its 50,257 token ids fit in uint16, which halves the cache size.
TOKENIZER = "gpt2"
VOCAB_SIZE = 50_257

# Every trial trains on BATCH_SIZE sequences of SEQ_LEN tokens per step, and is
# scored on the same N_VAL_TOKENS validation tokens.
SEQ_LEN = 256
BATCH_SIZE = 32
N_VAL_TOKENS = 1_000_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def autocast():
    """bf16 mixed precision on GPUs with native bf16 (Ampere or newer), else float32."""
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# What the cache was built from; a cache built from anything else is not reused.
_CACHE_SOURCE = {"dataset": DATASET, "dataset_config": DATASET_CONFIG, "tokenizer": TOKENIZER}


def _n_cached(split, cache_dir):
    """Tokens of `split` in the cache, or 0 if it was built from another source."""
    meta_path, path = cache_dir / "meta.json", cache_dir / f"{split}.bin"
    if not (meta_path.exists() and path.exists()):
        return 0
    if json.loads(meta_path.read_text()) != _CACHE_SOURCE:
        return 0
    return path.stat().st_size // 2  # 2 bytes per uint16 token


def prepare_tokens(n_train_tokens, cache_dir=None):
    """Ensure `val.bin` and `train.bin` hold N_VAL_TOKENS and `n_train_tokens` tokens.

    Documents are streamed in order and joined with an end-of-text token. The first
    N_VAL_TOKENS tokens are the validation set, so it is the same in every cache.
    """
    cache_dir = TOKEN_CACHE_DIR if cache_dir is None else Path(cache_dir)
    if (
        _n_cached("val", cache_dir) >= N_VAL_TOKENS
        and _n_cached("train", cache_dir) >= n_train_tokens
    ):
        print(f"Token cache at {cache_dir} is already big enough; nothing to do.")
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = iter(load_dataset(DATASET, DATASET_CONFIG, split="train", streaming=True))
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").unlink(missing_ok=True)  # invalid until fully rebuilt

    for split, n_tokens in (("val", N_VAL_TOKENS), ("train", n_train_tokens)):
        n_written = 0
        # Written to a temporary name, so an interrupted run leaves no partial file.
        tmp_path = cache_dir / f"{split}.bin.tmp"
        with tmp_path.open("wb") as f:
            while n_written < n_tokens:
                ids = tokenizer(next(docs)["text"])["input_ids"] + [tokenizer.eos_token_id]
                tokens = np.array(ids[: n_tokens - n_written], dtype=np.uint16)
                f.write(tokens.tobytes())
                n_written += len(tokens)
        tmp_path.rename(cache_dir / f"{split}.bin")
        print(f"{split}: {n_written:,} tokens written to {cache_dir}")
    (cache_dir / "meta.json").write_text(json.dumps(_CACHE_SOURCE))


def load_tokens(split, n_tokens, cache_dir=None):
    """The first `n_tokens` cached tokens of `split`, memory-mapped. Never downloads."""
    cache_dir = TOKEN_CACHE_DIR if cache_dir is None else Path(cache_dir)
    if _n_cached(split, cache_dir) < n_tokens:
        raise RuntimeError(
            f"The token cache at {cache_dir} holds {_n_cached(split, cache_dir):,} {split} "
            f"tokens from {_CACHE_SOURCE} but {n_tokens:,} are needed. Run "
            "`python pipeline/download_data.py --n_train_tokens <largest n_tokens choice>`."
        )
    return np.memmap(cache_dir / f"{split}.bin", dtype=np.uint16, mode="r")[:n_tokens]


def get_batch(tokens, starts):
    """Inputs and next-token targets for the SEQ_LEN-token windows beginning at `starts`."""
    windows = np.stack([tokens[start : start + SEQ_LEN + 1] for start in starts])
    windows = torch.from_numpy(windows.astype(np.int64)).to(DEVICE)
    return windows[:, :-1], windows[:, 1:]
