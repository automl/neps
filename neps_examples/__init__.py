all_main_examples = {  # Used for printing in python -m neps_examples
    "basic_usage": [
        "ex1_hyperparameters",
        "ex2_run_analysis",
        "ex3_architecture_search",
        "ex4_architecture_and_hyperparameters",
        "ex5_optimizer_search",
    ],
    "convenience": [
        "create_and_import_custom_config",
        "import_trial",
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
    "basic_usage/ex1_hyperparameters",  # NOTE: This needs to be first for some tests to work
    "basic_usage/ex2_run_analysis",
    "basic_usage/ex3_architecture_search",
    "basic_usage/ex4_architecture_and_hyperparameters",
    "basic_usage/ex5_optimizer_search",
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
