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
poetry version v0.9.0
```

and manually change the version specified in `CITATION.cff`.

## 2. Commit with a Version Tag

First commit and test

```bash
git add pyproject.toml
git commit -m "Bump version from v0.8.4 to v0.9.0"
pytest
```

Then tag and push

```bash
git tag v0.9.0
git push --tags
git push
```

## 3. Update Documentation

First check if the documentation has any issues via

```bash
mike deploy 0.9.0 latest -u
mike serve
```

and then looking at it.

Afterwards, publish it via

```bash
mike deploy 0.9.0 latest -up
```

## 4. Publish on PyPI

To publish to PyPI:

1. Get publishing rights, e.g., asking Danny or Maciej or Neeratyoy.
1. Be careful, once on PyPI we can not change things.
1. Run

```bash
poetry publish --build
```

This will ask for your PyPI credentials.
