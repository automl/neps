"""Shared dataset/model code used by both `train.py` (single-GPU HPO trials)
and `scaling_study/train_ddp.py` (fixed-workload DDP scaling benchmarks).

Pre-training data is LAION image/caption pairs in webdataset-shard format --
the same format a production LAION scaling study trains on. There are three
places it can come from, tried in this order, so nothing is ever re-fetched
that is already on disk:

  1. `LAION_CACHE_DIR` -- the prepared parquet cache. If it already holds
     enough samples, nothing else is touched: no shards are read, nothing is
     downloaded.
  2. `LAION_SHARDS_DIR` -- a local directory of webdataset `.tar` shards, e.g.
     a LAION-400M copy already staged on the cluster. Used to fill the cache
     without any network access.
  3. `LAION_REPO` on the Hugging Face Hub -- streamed over HTTP, the fallback
     for machines that have no local copy.

Shards hold full-size JPEGs, which would make CPU-side image decoding, not the
GPUs, the thing a scaling study measures. So `download_data.py` does the
decode/resize once, offline, and writes the cache as compact
`IMAGE_SIZE`x`IMAGE_SIZE` JPEGs; training then only has to decode those.

All three paths are overridable per run without editing this file:
`NEPS_LAION_CACHE_DIR` and `NEPS_LAION_SHARDS`.

CIFAR is still here, but only as the *downstream* zero-shot benchmark in
`post_hoc_downstream_eval.py` -- nothing trains on it.
"""

import contextlib
import io
import os
import tarfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pandas as pd
import torch
from open_clip import CLIP, CLIPTextCfg, CLIPVisionCfg
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

DATA_DIR = Path(__file__).parent / ".data"

# #CHANGE_ME: where the prepared parquet cache lives. Point this at a shared
# filesystem to build it once and reuse it from every job.
LAION_CACHE_DIR = Path(os.environ.get("NEPS_LAION_CACHE_DIR", DATA_DIR / "laion"))

# #CHANGE_ME: a local directory of webdataset `.tar` shards to build the cache
# from, if you already have LAION staged on the cluster -- then nothing is ever
# downloaded. Set to None (or point `NEPS_LAION_SHARDS` at a missing path) to
# always fall back to the Hub.
LAION_SHARDS_DIR = Path(os.environ.get(
    "NEPS_LAION_SHARDS",
    "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/train_data",
))

# #CHANGE_ME: the Hub fallback, used only when `LAION_SHARDS_DIR` does not
# exist. LAION publishes this one as webdataset tar shards of ~10k samples.
LAION_REPO = "laion/conceptual-captions-12m-webdataset"
LAION_SHARD_URL = f"https://huggingface.co/datasets/{LAION_REPO}/resolve/main/data/{{shard:05d}}.tar"

# Shard filenames are `<index>.tar`, but the zero-padding differs between
# sources (img2dataset writes 8 digits, the Hub repo 5), so both are tried.
SHARD_NAME_WIDTHS = (8, 5)

# Side length images are stored at (and therefore trained at). Also the size
# CIFAR images are upscaled to for the downstream zero-shot eval.
IMAGE_SIZE = 112
# Quality of the re-encoded JPEGs in the local cache. 90 is visually lossless
# at this resolution and keeps a 100k-sample cache well under 1 GB.
CACHE_JPEG_QUALITY = 90

# LAION/OpenAI CLIP normalization statistics.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

CIFAR100_DIR = DATA_DIR / "cifar100"
CIFAR100_URL = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def image_transform():
    """Cache images are already square and `IMAGE_SIZE`-sized, so this is only
    the tensor conversion. CIFAR images (32x32) get resized by the same pipeline.
    """
    return Compose([
        Resize(IMAGE_SIZE),
        CenterCrop(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


# --------------------------------------------------------------------------
# LAION pre-training data
# --------------------------------------------------------------------------


def _resize_encode(raw_jpeg: bytes) -> bytes | None:
    """Decode one shard image, square-crop it to `IMAGE_SIZE`, re-encode small.

    Returns `None` for the handful of truncated/CMYK/animated files every web
    -scraped shard contains, so one bad image never kills a whole download.
    """
    try:
        image = Image.open(io.BytesIO(raw_jpeg)).convert("RGB")
    except Exception:  # noqa: BLE001 - any decode failure means "skip this sample"
        return None

    w, h = image.size
    scale = IMAGE_SIZE / min(w, h)
    image = image.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))), Image.BICUBIC)
    left, top = (image.width - IMAGE_SIZE) // 2, (image.height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=CACHE_JPEG_QUALITY)
    return buffer.getvalue()


def local_shard_path(shard: int, shards_dir: Path | None = None) -> Path | None:
    """The local `.tar` for shard index `shard`, or None if there isn't one.

    Looked up by name rather than by listing the directory: a staged LAION-400M
    copy holds >100k files, and globbing it on a network filesystem is slow.
    """
    shards_dir = LAION_SHARDS_DIR if shards_dir is None else shards_dir
    if shards_dir is None:
        return None
    for width in SHARD_NAME_WIDTHS:
        path = Path(shards_dir) / f"{shard:0{width}d}.tar"
        if path.exists():
            return path
    return None


@contextlib.contextmanager
def _open_shard(shard: int, shards_dir: Path | None = None):
    """Open shard `shard` as a byte stream, preferring the local copy.

    Yields `(fileobj, description)`. Falls back to the Hub only when the shard
    is not on disk, so a staged dataset means zero network traffic.
    """
    path = local_shard_path(shard, shards_dir)
    if path is not None:
        with path.open("rb") as handle:
            yield handle, str(path)
        return

    url = LAION_SHARD_URL.format(shard=shard)
    with urlopen(url) as response:
        yield response, url


def _stream_shard(shard: int, shards_dir: Path | None = None):
    """Yield `(caption, resized_jpeg_bytes)` from one webdataset shard.

    The tar is consumed as a stream (`mode="r|"`), so a downloaded shard is
    never written to disk and a local one is never copied -- only the ~5 KB
    re-encoded images are kept.
    """
    pending: dict[str, dict[str, bytes]] = {}

    with _open_shard(shard, shards_dir) as (stream, _), tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or "." not in member.name:
                continue
            key, _, ext = member.name.partition(".")
            if ext not in ("jpg", "txt"):
                continue

            payload = tar.extractfile(member)
            if payload is None:
                continue
            pending.setdefault(key, {})[ext] = payload.read()

            sample = pending[key]
            if "jpg" in sample and "txt" in sample:
                del pending[key]
                image = _resize_encode(sample["jpg"])
                caption = sample["txt"].decode("utf-8", errors="replace").strip()
                if image is not None and caption:
                    yield caption, image


def prepare_laion(
    n_samples: int, first_shard: int = 0, cache_dir: Path | None = None, shards_dir: Path | None = None,
) -> Path:
    """Ensure the parquet cache holds at least `n_samples` image/caption pairs.

    Returns immediately if the cache is already big enough -- that is the whole
    point of the cache, and it is why re-running this is free. Otherwise it
    tops the cache up from local shards when they exist, and only downloads
    when they do not. One parquet part is written per source shard, so an
    interrupted run resumes at the next missing part instead of starting over.
    """
    cache_dir = LAION_CACHE_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_cached = _cached_sample_count(cache_dir)
    if n_cached >= n_samples:
        print(f"LAION cache at {cache_dir} already holds {n_cached:,} samples; nothing to do.")
        return cache_dir

    shard = first_shard
    while n_cached < n_samples:
        part_path = cache_dir / f"part-{shard:05d}.parquet"
        if part_path.exists():
            shard += 1
            continue

        local = local_shard_path(shard, shards_dir)
        source = f"local shard {local}" if local is not None else f"{LAION_REPO} shard {shard} (download)"
        print(f"Reading {source} ({n_cached:,}/{n_samples:,} samples cached)...")
        captions, images = [], []
        for caption, image in _stream_shard(shard, shards_dir):
            captions.append(caption)
            images.append(image)
            if n_cached + len(images) >= n_samples:
                break

        if not images:
            raise RuntimeError(
                f"Shard {shard} yielded no usable samples. Check LAION_SHARDS_DIR "
                f"({LAION_SHARDS_DIR}) if you meant to read local shards, or that "
                f"{LAION_REPO} is reachable if you meant to download."
            )

        # Written to a temporary name first so an interrupted run never leaves a
        # half-written part that the resume logic would happily skip.
        tmp_path = part_path.with_suffix(".parquet.tmp")
        pd.DataFrame({"caption": captions, "image": images}).to_parquet(tmp_path, index=False)
        tmp_path.rename(part_path)

        n_cached += len(images)
        shard += 1

    print(f"LAION cache ready: {n_cached:,} samples in {cache_dir}")
    return cache_dir


def _cached_parts(cache_dir: Path | None = None) -> list[Path]:
    cache_dir = LAION_CACHE_DIR if cache_dir is None else Path(cache_dir)
    return sorted(cache_dir.glob("part-*.parquet"))


def _cached_sample_count(cache_dir: Path | None = None) -> int:
    """Row count of the cache, read from parquet metadata -- no data is loaded."""
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in _cached_parts(cache_dir))


class LaionClipDataset(Dataset):
    """Image/caption pairs from the local cache, with captions pre-tokenized.

    Tokenizing up front (rather than per `__getitem__`) keeps the dataloader
    workers doing nothing but a small JPEG decode, which is what makes the
    input pipeline cheap enough to not dominate the DDP scaling measurement.
    """

    def __init__(self, df, tokenizer, transform):
        self.images = df["image"].tolist()
        self.tokens = tokenizer(df["caption"].tolist())
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.images[idx])).convert("RGB")
        return self.transform(image), self.tokens[idx]


def load_data(tokenizer, n_train=2000, n_val=500, cache_dir=None):
    """Return `(train_set, val_set)` over disjoint slices of the LAION cache.

    Reads only the prepared cache -- training never touches shards and never
    downloads. The split is deterministic (validation is always the first
    `n_val` cached samples, training the `n_train` after them), so every config
    in a sweep, and every GPU count in the scaling study, sees exactly the same
    data.
    """
    cache_dir = LAION_CACHE_DIR if cache_dir is None else Path(cache_dir)
    parts = _cached_parts(cache_dir)
    if not parts:
        raise RuntimeError(
            f"No LAION cache found at {cache_dir}. Run\n"
            f"    python download_data.py --n_samples {n_train + n_val}\n"
            "once before training (it reads local shards if you have them, and "
            "only downloads otherwise)."
        )

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if len(df) < n_train + n_val:
        raise RuntimeError(
            f"LAION cache at {cache_dir} holds {len(df):,} samples but "
            f"{n_train + n_val:,} were requested. Run "
            f"`python download_data.py --n_samples {n_train + n_val}` to extend it."
        )

    transform = image_transform()
    val = LaionClipDataset(df.iloc[:n_val], tokenizer, transform)
    train = LaionClipDataset(df.iloc[n_val:n_val + n_train], tokenizer, transform)
    return train, val


# --------------------------------------------------------------------------
# CIFAR-100: downstream zero-shot benchmark only, never trained on
# --------------------------------------------------------------------------

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]


class CifarClipDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        return self.transform(image), row["label"]


def load_cifar100_test(n_rows=1000):
    path = CIFAR100_DIR / "test.parquet"
    if not path.exists():
        CIFAR100_DIR.mkdir(parents=True, exist_ok=True)
        urlretrieve(CIFAR100_URL, path)
    return pd.read_parquet(path).rename(columns={"fine_label": "label"}).iloc[:n_rows]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_model(vision_width, vision_layers, text_width, text_layers, embed_dim=128):
    vision_cfg = CLIPVisionCfg(
        image_size=IMAGE_SIZE, layers=vision_layers, width=vision_width, patch_size=8, head_width=32,
    )
    text_cfg = CLIPTextCfg(
        context_length=77, vocab_size=49408, width=text_width, heads=text_width // 32, layers=text_layers,
    )
    return CLIP(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg, output_dict=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_gpu_name():
    return torch.cuda.get_device_properties(DEVICE).name if torch.cuda.is_available() else None


def get_peak_gpu_memory_mb():
    return torch.cuda.max_memory_allocated(DEVICE) / 1e6 if torch.cuda.is_available() else None


def get_model_flops(model, batch_size=1):
    """FLOPs for one forward pass over `batch_size` samples, image + text towers combined."""
    from torch.utils.flop_counter import FlopCounterMode

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    image_input = torch.ones((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), device=device, dtype=dtype)
    text_input = torch.ones((batch_size, model.context_length), device=device, dtype=torch.int64)

    was_training = model.training
    model.eval()
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        model(image_input, text_input)
    model.train(was_training)

    return sum(flop_counter.get_flop_counts()["Global"].values()) / batch_size
