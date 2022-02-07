import argparse
import logging
from pathlib import Path

from . import read_results

# fmt: off
parser = argparse.ArgumentParser(
    prog="python -m neps.status",
    description="Displays status information about a working directory of a neps.run"
)
parser.add_argument("working_directory", type=Path, help="The working directory given to neps.run")
parser.add_argument("--best_losses", action="store_true", help="Show the trajectory of the best loss across evaluations")
parser.add_argument("--best_configs", action="store_true", help="Show the trajectory of the best configs and their losses across evaluations")
parser.add_argument("--all_configs", action="store_true", help="Show all configs and their losses")
args = parser.parse_args()
# fmt: on

logging.basicConfig(level=logging.WARN)

previous_results, pending_configs, pending_configs_free = read_results(
    args.working_directory, logging.getLogger("neps.status")
)
print(f"#Evaluated configs: {len(previous_results)}")
print(f"#Pending configs: {len(pending_configs)}")
print(f"#Pending configs with worker: {len(pending_configs) - len(pending_configs_free)}")

if len(previous_results) == 0:
    exit(0)

best_loss = float("inf")
best_config_id = None
best_config = None
for config_id, (config, result) in previous_results.items():
    if result["loss"] < best_loss:
        best_loss = result["loss"]
        best_config = config
        best_config_id = config_id

print()
print(f"Best loss: {best_loss}")
print(f"Best config id: {best_config_id}")
print(f"Best config: {best_config}")

if args.best_losses:
    print()
    print("Best loss across evaluations:")
    best_loss_trajectory = args.working_directory / "best_loss_trajectory.txt"
    print(best_loss_trajectory.read_text(encoding="utf-8"))

if args.best_configs:
    print()
    print("Best configs and their losses across evaluations:")
    print(79 * "-")
    best_loss_config = args.working_directory / "best_loss_with_config_trajectory.txt"
    print(best_loss_config.read_text(encoding="utf-8"))

if args.all_configs:
    print()
    print("All evaluated configs and their losses:")
    print(79 * "-")
    all_loss_config = args.working_directory / "all_losses_and_configs.txt"
    print(all_loss_config.read_text(encoding="utf-8"))
