# Seeding

Seeding is only rudimentarily supported in AutoScAn, as we provide a function to capture the global rng state of `Python`, `numpy` and `torch`. It is not yet possible to seed only AutoScAn internally.

See the [Seeding API][autoscan.state.seed_snapshot.SeedSnapshot] for the details on how to [capture][autoscan.state.seed_snapshot.SeedSnapshot.new_capture] and [use][autoscan.state.seed_snapshot.SeedSnapshot.set_as_global_seed_state] this global rng state.
