# Releasing a New Version

There are four steps to releasing a new version of neps:

0. Understand Semantic Versioning
1. Update the Package Version
1. Commit and Push With a Version Tag
1. Update Documentation
1. Publish on PyPI

## 0. Understand Semantic Versioning

We follow the [semantic versioning](https://semver.org) scheme.

## 1. Update the Package Version and CITATION.cff

```bash
poetry version v0.4.10
```

and manually change the version specified in `CITATION.cff`.

## 2. Commit with a Version Tag

First commit and test

```bash
git add pyproject.toml
git commit -m "Bump version from v0.4.9 to v0.4.10"
pytest
```

Then tag and push

```bash
git tag v0.4.10
git push --tags
git push
```

## 3. Update Documentation

First check if the documentation has any issues via

```bash
mkdocs build
mkdocs serve
```

and then looking at it.

Afterwards, publish it via

```bash
mkdocs gh-deploy
```

## 4. Publish on PyPI

To publish to PyPI:

1. Get publishing rights, e.g., asking Danny or Maciej.
1. Be careful, once on PyPI we can not change things.
1. Run

```bash
poetry publish --build
```

This will ask for your PyPI credentials.
