# Introduction and Installation

## Installation

Using pip

```bash
pip install neural-pipeline-search
```

## Optional: Specific torch versions

If you run into any issues regarding versions of the torch ecosystem (like needing cuda enabled versions), you might want to use our utility

```bash
python -m neps.utils.install_torch
```

This script asks for the torch version you want and installs all the torch libraries needed for the neps package with
that version. For the installation `pip` of the active python environment is used.
