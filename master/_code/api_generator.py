"""Generate the code reference pages and navigation.

# https://mkdocstrings.github.io/recipes/
"""

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

# Modules whose members should not include inherited attributes or methods
NO_INHERITS = tuple()

SRCDIR = Path("neps").absolute().resolve()
ROOT = SRCDIR.parent
TAB = "    "


if not SRCDIR.exists():
    raise FileNotFoundError(
        f"{SRCDIR} does not exist, make sure you are running this from the root of the"
        " repository."
    )

for path in sorted(SRCDIR.rglob("*.py")):
    module_path = path.relative_to(ROOT).with_suffix("")
    doc_path = path.relative_to(ROOT).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] in ("__main__", "__version__", "__init__"):
        continue

    if any(part.startswith("_") for part in parts):
        continue

    # Skip neps_spaces/parameters module to avoid conflicts with reference/neps_spaces.md
    if "neps_spaces" in parts and "parameters" in parts:
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

        if ident.endswith(NO_INHERITS):
            fd.write(f"\n{TAB}options:")
            fd.write(f"\n{TAB}{TAB}inherited_members: false")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
