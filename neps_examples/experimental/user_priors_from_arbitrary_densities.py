import neps

def run_pipeline(some_float, some_integer, some_cat):
    if some_cat != "a":
        y = some_float + some_integer
    else:
        y = -some_float - some_integer
    return y

# ========================================================================================
# Current API
# User prior is given as a default value and a confidence level specified in the parameter itself
pipeline_space = dict(
    some_float=neps.Float(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium"
    ),
    some_integer=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    some_cat=neps.CategoricalParameter(
        choices=["a", "b", "c"], default="a", default_confidence="high"
    )
)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results",
    max_evaluations_total=15,
)

# ========================================================================================
# New API, variant 01
# User prior is passed to neps.run and not specified in the pipeline_space
# The prior is given as one of the following:
# 1) A (non-factorized) density function that returns the likelihood of a given parameter configuration
# 2) A dicttionary of marginal densities for each parameter. Then the factorized density is used.
# 3) A dictionary of default values and confidence levels for each parameter. Then a gaussian prior is used.

pipeline_space = dict(
    some_float=neps.Float(lower=1, upper=1000, log=True),
    some_integer=neps.IntegerParameter(lower=0, upper=50),
    some_cat=neps.CategoricalParameter(choices=["a", "b", "c"])
)

# 1) A (non-factorized) density function that returns the likelihood of a given parameter configuration
def prior_01(some_float, some_integer, some_cat):
    # some exponential distribution
    if some_cat != "a":
        return np.exp(-(some_float + some_integer - 1))
    else:
        return np.exp(-(-some_float - some_integer + 1050))

# 2) A dictionary of marginal densities for each parameter. Then the factorized density is used.
prior_02 = dict(
    some_float=lambda x: 1/400 if 800 < x < 1000 else 1/1600, # prior on interval [800, 1000]
    some_integer=lambda k: 30**k/np.math.factorial(k) * np.exp(-k), # poisson prior on integers k=30
    some_cat=lambda x: 1/2*(x=="b") + 1/3*(x=="c") + 1/6*(x=="a")
)

# 3) A dictionary of default values and confidence levels for each parameter. Then a gaussian prior is used.
prior_03 = dict(
    some_float=dict(default=900, default_confidence="medium"),
    some_integer=dict(default=35, default_confidence="low"),
    some_cat=dict(default="a", default_confidence="high")
)

# Combination of 2) and 3)
prior_04 = dict(
    some_float=dict(default=900, default_confidence="medium"),
    some_integer=lambda k: 30**k/np.math.factorial(k) * np.exp(-k), # poisson prior on integers k=30
    some_cat=dict(default="a", default_confidence="high")
)

# Pass the prior to neps.run

neps.run(
    prior=prior_01, # or prior_02 or prior_03 or prior_04
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results",
    max_evaluations_total=15,
)

# ========================================================================================
# New API, variant 02
# User prior is specfied in the pipeline_space and not directly passed to neps.run
# Same possibiities for priors as in variant 01

# 1) A (non-factorized) density function that returns the likelihood of a given parameter configuration
def prior_01(some_float, some_integer, some_cat):
    # some exponential distribution
    if some_cat != "a":
        return np.exp(-(some_float + some_integer - 1))
    else:
        return np.exp(-(-some_float - some_integer + 1050))

pipeline_space_01 = dict(
    some_float=neps.Float(lower=1, upper=1000, log=True),
    some_integer=neps.IntegerParameter(lower=0, upper=50),
    some_cat=neps.CategoricalParameter(choices=["a", "b", "c"]),
    _prior=prior_01
)

# 2) A dictionary of marginal densities for each parameter. Then the factorized density is used.
pipeline_space_02 = dict(
    some_float=neps.Float(
        lower=1, upper=1000, log=True,
        prior_fun=lambda x: 1/400 if 800 < x < 1000 else 1/1600
    ),
    some_integer=neps.IntegerParameter(lower=0, upper=50,
        prior_fun=lambda k: 30**k/np.math.factorial(k) * np.exp(-k)
),
    some_cat=neps.CategoricalParameter(choices=["a", "b", "c"],
        prior_fun=lambda x: 1/2*(x=="b") + 1/3*(x=="c") + 1/6*(x=="a")
    )
)

# 3) A dictionary of default values and confidence levels for each parameter. Then a gaussian prior is used.
# Same as in the current API
pipeline_space_03 = dict(
    some_float=neps.Float(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium"
    ),
    some_integer=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    some_cat=neps.CategoricalParameter(
        choices=["a", "b", "c"], default="a", default_confidence="high"
    )
)

# Combination of 2) and 3)
pipeline_space_04 = dict(
    some_float=neps.Float(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium",
    ),
    some_integer=neps.IntegerParameter(
        lower=0, upper=50,
        prior_fun=lambda k: 30**k/np.math.factorial(k) * np.exp(-k)
    ),
    some_cat=neps.CategoricalParameter(
        choices=["a", "b", "c"], default="a", default_confidence="high")
)

# Pass the pipeline_space to neps.run
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space_01, # or pipeline_space_02 or pipeline_space_03 or pipeline_space_04
    root_directory="results",
    max_evaluations_total=15,
)
