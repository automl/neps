# Cleaning Up Failed Trials

The NePS `clean` command provides a utility to clean up failed, crashed, or corrupted trials from your optimization working directory.
By default it resets matched trials to `pending`, keeping their config so they get re-evaluated on the next run. Pass `--delete` to remove them entirely instead.
This is useful for managing your optimization state and preventing problematic trials from interfering with future optimization runs.

---

## Command Line Usage

### Basic Usage

To reset all non-success stated trials in your optimization directory back to pending:

```bash
python -m neps.clean --root-dir <root_directory>
```

If `--root-dir` is omitted, it defaults to `neps_results`.

### Dry Run

Preview what would change without making any changes:

```bash
python -m neps.clean --root-dir <root_directory> --dry-run
```

### Cleaning Specific Trial IDs

Reset only specific trials by their IDs (regardless of state):

```bash
python -m neps.clean --root-dir <root_directory> --trial-ids <trial_id_1> <trial_id_2> <trial_id_3>
```

Trial IDs are reported in `metadata.json` within each config directory.

### Deleting Instead of Resetting

By default, matched trials keep their config on disk and have their state reset to
`pending` (clearing their report), so that a subsequent `neps.run` re-evaluates the
same configuration rather than sampling a new one. Pass `--delete` to instead remove
the trial directory entirely:

```bash
python -m neps.clean --root-dir <root_directory> --delete
```

This can be combined with `--dry-run` and `--trial-ids` as usual, e.g.:

```bash
python -m neps.clean --root-dir <root_directory> --trial-ids 1 2 --delete
```

---

## Python API

You can also use the clean functionality programmatically in Python:

### Reset Failed/Crashed/Corrupted Trials

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

print(f"Reset {stats['total_removed']} trials")
```

### Reset Specific Trial IDs

```python
from pathlib import Path
from neps.clean.clean import clean_failed_trials

root_dir = Path("<root_directory>")
stats = clean_failed_trials(
    root_directory=root_dir,
    trial_ids=["1", "2"],
    dry_run=False,
)

print(f"Reset {stats['removed']} trials")
print(f"Not found: {stats['not_found']} trials")
```

### Deleting Trials

Pass `delete=True` to remove trials entirely instead of resetting them to `pending`:

```python
stats = clean_failed_trials(
    root_directory=root_dir,
    desired_states=[Trial.State.FAILED, Trial.State.CRASHED],
    delete=True,
    dry_run=False,
)
```
