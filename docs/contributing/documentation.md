# Documentation

We use [MkDocs](https://www.mkdocs.org/getting-started/), more specifically [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation.
Source files for the documentation are at [docs](docs) and configuration at  [mkdocs.yml](https://github.com/automl/neps/tree/master/mkdocs.yml).

To build and view the documentation run

```bash
mkdocs build
mkdocs serve
```

and open the URL shown by the `mkdocs serve` command.

To publish the documenation run

```bash
mkdocs gh-deploy
```
