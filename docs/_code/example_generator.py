"""Generate the code reference pages and navigation.

# https://mkdocstrings.github.io/recipes/
"""
from __future__ import annotations

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

SRCDIR = Path("neps").absolute().resolve()
ROOT = SRCDIR.parent
EXAMPLE_FOLDER = ROOT / "neps_examples"
TAB = "    "

if not SRCDIR.exists():
    raise FileNotFoundError(
        f"{SRCDIR} does not exist, make sure you are running this from the root of the repository."
    )

# Write the index page of the examples
EXAMPLE_INDEX = EXAMPLE_FOLDER / "README.md"
if not EXAMPLE_INDEX.exists():
    raise FileNotFoundError(
        f"{EXAMPLE_INDEX} does not exist, make sure you are running this from the root of the repository."
    )

with EXAMPLE_INDEX.open() as fd:
    example_index_contents = fd.read()

DOCS_EXAMPLE_INDEX = Path("examples", "index.md")

with mkdocs_gen_files.open(DOCS_EXAMPLE_INDEX, "w") as fd:
    fd.write(example_index_contents)

mkdocs_gen_files.set_edit_path(DOCS_EXAMPLE_INDEX, EXAMPLE_INDEX)

# Now Iterate through each example folder
for example_dir in EXAMPLE_FOLDER.iterdir():
    if not example_dir.is_dir():
        continue

    readme = next((p for p in example_dir.iterdir() if p.name == "README.md"), None)
    doc_example_dir = Path("examples", example_dir.name)

    # Copy the README.md file to the docs/<example_dir>/index.md
    if readme is not None:
        doc_example_index = doc_example_dir / "index.md"
        with readme.open() as fd:
            contents = fd.read()

        with mkdocs_gen_files.open(doc_example_index, "w") as fd:
            fd.write(contents)

        mkdocs_gen_files.set_edit_path(doc_example_index, readme)

    # Copy the contents of all of the examples to the docs/<example_dir>/examples/<example_name>.md
    for path in example_dir.iterdir():
        if path.suffix != ".py":
            continue

        with path.open() as fd:
            contents = fd.readlines()

        # NOTE: We use quad backticks to escape the code blocks that are present in some of the examples
        escaped_contents = "".join(["````python\n", *contents, "\n````"])

        markdown_example_path = doc_example_dir / f"{path.stem}.md"
        with mkdocs_gen_files.open(markdown_example_path, "w") as fd:
            fd.write(escaped_contents)

        mkdocs_gen_files.set_edit_path(markdown_example_path, path)
