# Seeding

Seeding is only rudimentarily supported in ScaleFM, as we provide a function to capture the global rng state of `Python`, `numpy` and `torch`. It is not yet possible to seed only ScaleFM internally.

See the [Seeding API][scalefm.state.seed_snapshot.SeedSnapshot] for the details on how to [capture][scalefm.state.seed_snapshot.SeedSnapshot.new_capture] and [use][scalefm.state.seed_snapshot.SeedSnapshot.set_as_global_seed_state] this global rng state.
