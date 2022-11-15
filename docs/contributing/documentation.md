# Documentation

We use [MkDocs](https://www.mkdocs.org/getting-started/), more specifically [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation. To support documentation for multiple versions, we use the plugin [mike](https://github.com/jimporter/mike).
Source files for the documentation are at [docs](docs) and configuration at  [mkdocs.yml](https://github.com/automl/neps/tree/master/mkdocs.yml).

To build and view the documentation run

```bash
mike deploy 0.5.1 latest
mike serve
```

and open the URL shown by the `mike serve` command.

To publish the documentation run

```bash
mike deploy 0.5.1 latest -p
```
