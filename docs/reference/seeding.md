# Seeding

Seeding is only rudimentarily supported in NePS, as we provide a function to capture the global rng state of `Python`, `numpy` and `torch`. It is not yet possible to seed only NePS internally.

See the [Seeding API][neps.state.seed_snapshot.SeedSnapshot] for the details on how to [capture][neps.state.seed_snapshot.SeedSnapshot.new_capture] and [use][neps.state.seed_snapshot.SeedSnapshot.set_as_global_seed_state] this global rng state.
