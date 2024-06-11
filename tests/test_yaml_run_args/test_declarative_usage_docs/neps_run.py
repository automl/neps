import argparse
import neps
from tests.test_yaml_run_args.test_declarative_usage_docs.run_pipeline import \
    run_pipeline_constant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NEPS optimization with run_args.yml."
    )
    parser.add_argument("run_args", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--run_pipeline", action="store_true")
    args = parser.parse_args()

    if args.run_pipeline:
        neps.run(run_args=args.run_args, run_pipeline=run_pipeline_constant)
    else:
        neps.run(run_args=args.run_args)

