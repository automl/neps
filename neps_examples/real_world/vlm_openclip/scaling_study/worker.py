"""One NePS worker of a scaling-study setting, in its own Slurm job on its own GPU.
Calls `neps.run` on the setting's shared root directory for its share of the sweep.
"""

import argparse
import logging
import time
from pathlib import Path

import neps
from train import evaluate

# How long a follower waits for the first worker to create the NePS state.
STATE_WAIT_TIMEOUT_SEC = 900
STATE_POLL_SEC = 2


class HPOSpace(neps.PipelineSpace):
    # #CHANGE_ME: the searched hyperparameters. A grid of exactly
    # `run_scaling_study.TOTAL_EVALUATIONS` points, so every setting evaluates
    # the same configs and the throughputs stay comparable.
    lr = neps.Categorical(choices=(3e-4, 1e-3))
    wd = neps.Categorical(choices=(1e-5, 1e-4))
    batch_size = neps.Categorical(choices=(256, 512))

    # #CHANGE_ME: fixed for the whole study.
    vision_width = 256
    vision_layers = 6
    text_width = 256
    text_layers = 6
    epoch = 3


def wait_for_state(root_dir: Path) -> None:
    """Block until the first worker has finished creating the NePS state.

    `NePSState.create_or_load` deliberately does not lock the creation path, so
    two workers reaching it at once can read a half-written state. Only the
    first worker of a setting creates it; the rest wait here. `pipeline_space.pkl`
    is the last file creation writes, so its presence means the state is complete.
    """
    marker = root_dir / "pipeline_space.pkl"
    deadline = time.time() + STATE_WAIT_TIMEOUT_SEC
    while not marker.exists():
        if time.time() > deadline:
            raise TimeoutError(
                f"{marker} did not appear within {STATE_WAIT_TIMEOUT_SEC}s. The first "
                "worker of this setting never created the NePS state -- check its log."
            )
        time.sleep(STATE_POLL_SEC)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--evaluations_to_spend", required=True, type=int)
    parser.add_argument("--n_workers", required=True, type=int)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--wait_for_state", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.wait_for_state:
        wait_for_state(args.root_dir)

    def evaluate_pipeline(pipeline_directory, **config):
        return evaluate(
            n_workers=args.n_workers,
            checkpoint_path=Path(pipeline_directory) / "checkpoint.pt",
            **config,
        )

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory=args.root_dir,
        optimizer=("grid_search", {}),
        evaluations_to_spend=args.evaluations_to_spend,
        worker_id=args.worker_id,
    )


if __name__ == "__main__":
    main()
