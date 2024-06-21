import neps

pipeline_space = dict(
    learning_rate=neps.FloatParameter(lower=1e-5, upper=1e-1, log=True),
    epochs=neps.IntegerParameter(lower=5, upper=20, is_fidelity=True),
    optimizer=neps.CategoricalParameter(choices=["adam", "sgd", "adamw"]),
    batch_size=neps.ConstantParameter(value=64)
)
