# Cleaning Up Failed Trials

The NePS `clean` command provides a utility to remove failed, crashed, or corrupted trials from your optimization working directory.
This is useful for managing your optimization state and preventing problematic trials from interfering with future optimization runs.

---

## Command Line Usage

### Basic Usage

To clean all non-success stated trials from your optimization directory:

```bash
python -m neps.clean <root_directory>
```

### Dry Run

Preview what will be deleted without making any changes:

```bash
python -m neps.clean <root_directory> --dry_run
```

### Removing Specific Trial IDs

Remove only specific trials by their IDs (regardless of state):

```bash
python -m neps.clean <root_directory> --trial_ids <trial_id_1> <trial_id_2> <trial_id_3>
```

Trial IDs are reported in `metadata.json` within each config directory.

---

## Python API

You can also use the clean functionality programmatically in Python:

### Clean Failed/Crashed/Corrupted Trials

```python
from pathlib import Path
from neps.clean.clean import clean_failed_trials
from neps.state.trial import Trial

root_dir = Path("<root_directory>")
stats = clean_failed_trials(
    root_directory=root_dir,
    desired_states=[
        Trial.State.FAILED,
        Trial.State.CRASHED,
        Trial.State.CORRUPTED,
    ],
    dry_run=False,
)

print(f"Removed {stats['total_removed']} trials")
```

### Clean Specific Trial IDs

```python
from pathlib import Path
from neps.clean.clean import clean_failed_trials

root_dir = Path("<root_directory>")
stats = clean_failed_trials(
    root_directory=root_dir,
    trial_ids=["1", "2"],
    dry_run=False,
)

print(f"Removed {stats['removed']} trials")
print(f"Not found: {stats['not_found']} trials")
```
