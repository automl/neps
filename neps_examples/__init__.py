all_main_examples = {  # Used for printing in python -m neps_examples
    "basic_usage": [
        "analyse",
        "architecture",
        "architecture_and_hyperparameters",
        "hyperparameters",
    ],
    "convenience": [
        "logging_additional_info",
        "neps_tblogger_tutorial",
        "running_on_slurm_scripts",
        "neps_x_lightning",
        "running_on_slurm_scripts",
        "working_directory_per_pipeline",
    ],
    "efficiency": [
        "expert_priors_for_hyperparameters",
        "multi_fidelity",
        "multi_fidelity_and_expert_priors",
        "pytorch_native_ddp",
        "pytorch_lightning_ddp",
    ],
}

core_examples = [  # Run locally and on github actions
    "basic_usage/hyperparameters",  # NOTE: This needs to be first for some tests to work
    "basic_usage/analyse",
    "basic_usage/pytorch_nn_example",
    "experimental/expert_priors_for_architecture_and_hyperparameters",
    "efficiency/multi_fidelity",
]

ci_examples = [  # Run on github actions
    "basic_usage/architecture_and_hyperparameters",
    "experimental/hierarchical_architecture",
    "efficiency/expert_priors_for_hyperparameters",
    "convenience/logging_additional_info",
    "convenience/working_directory_per_pipeline",
    "convenience/neps_tblogger_tutorial",
]
