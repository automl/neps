import neps

pipeline_space = dict(
    learning_rate=neps.Float(lower=1e-5, upper=1e-1, log=True),
    epochs=neps.Integer(lower=5, upper=20, is_fidelity=True),
    optimizer=neps.Categorical(choices=["adam", "sgd", "adamw"]),
    batch_size=neps.Constant(value=64)
)
