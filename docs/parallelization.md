# Parallelization

In order to run a neural pipeline search with multiple processes or multiple machines, simply call `neps.run` multiple times.
All calls to `neps.run` need to use the same `root_directory` on the same filesystem, otherwise there is no synchronization between the `neps.run`'s.
