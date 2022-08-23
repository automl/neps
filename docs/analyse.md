# Analysing Runs

## Status

To show status information about a neural pipeline search use

```bash
python -m neps.status WORKING_DIRECTORY
```

If you need more status information than is printed per default (e.g., the best config over time), please have a look at

```bash
python -m neps.status --help
```

To show the status repeatedly, on unix systems you can use

```bash
watch --interval 30 python -m neps.status WORKING_DIRECTORY
```
