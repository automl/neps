# Parallelization

In order to run scalefm in parallel on multiple processes or multiple machines, simply call `scalefm.acquire` multiple times.
All calls to `scalefm.acquire` need to use the same `root_directory` on the same filesystem to synchronize between the `scalefm.acquire`'s.

For example, start the logging example in two shells from the same directory as below.

In shell 1:

```bash
python -m scalefm_examples.convenience.logging_additional_info
```

In shell 2:

```bash
python -m scalefm_examples.convenience.logging_additional_info
```
