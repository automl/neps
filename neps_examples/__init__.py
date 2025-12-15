all_main_examples = {  # Used for printing in python -m neps_examples
    "basic_usage": [
        "1_hyperparameters",
        "2_run_analysis",
        "3_architecture_search",
        "4_architecture_and_hyperparameters",
        "5_optimizer_search",
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
    "basic_usage/1_hyperparameters",  # NOTE: This needs to be first for some tests to work
    "basic_usage/2_run_analysis",
    "basic_usage/3_architecture_search",
    "basic_usage/4_architecture_and_hyperparameters",
    "basic_usage/5_optimizer_search",
    "efficiency/multi_fidelity",
    "efficiency/expert_priors_for_hyperparameters",
    "efficiency/multi_fidelity_and_expert_priors",
]

ci_examples = [  # Run on github actions
    "convenience/logging_additional_info",
    "convenience/working_directory_per_pipeline",
    "convenience/neps_tblogger_tutorial",
]
