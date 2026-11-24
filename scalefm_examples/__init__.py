all_main_examples = {  # Used for printing in python -m scalefm_examples
    "convenience": [
        "create_and_import_custom_config",
        "import_trials",
        "logging_additional_info",
        "scalefm_tblogger_tutorial",
        "running_on_slurm_scripts",
        "working_directory_per_pipeline",
    ],
    "efficiency": [
        "pytorch_native_ddp",
        "pytorch_native_fsdp",
        "pytorch_lightning_ddp",
        "pytorch_lightning_fsdp",
    ],
    "async_evaluation": [
        "submit",
    ],
    "real_world": [
        "tabpfn_scan_bench",
    ],
}

ci_examples = [  # Run on github actions
    "convenience/logging_additional_info",
    "convenience/working_directory_per_pipeline",
    "convenience/scalefm_tblogger_tutorial",
]
