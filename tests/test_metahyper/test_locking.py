import concurrent.futures
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from more_itertools import first_true


def run_worker():
    command = "python -m neps_examples.basic_usage.hyperparameters && python -m neps_examples.basic_usage.analyse"

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Return stdout and return code
    return result.stdout, result.returncode


@pytest.mark.metahyper
def test_filelock(n_workers=2) -> None:
    """Test that the filelocking method of parallelization works as intended."""
    # Note: Not using tmpdir
    #
    #   Unfortunatly we can't control this from launching the subprocess so we handle
    #   clean up manualy. This is likely to break if doing multi-processing testing
    #   with something like pytest-forked
    #
    # Note: dependancy on basic_usage example
    #
    #   Not a great idea incase the example changes, ideally each process here would
    #   perform some predefined operation which is known to this test. If the example
    #   changes in some unexpected way, it may break this test
    results_dir = Path("results") / "hyperparameters_example" / "results"
    try:
        assert not results_dir.exists()
        exit_codes = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit each worker function to the concurrent executor
            futures = [executor.submit(run_worker) for _ in range(n_workers)]

            for future in concurrent.futures.as_completed(futures):
                stdout, return_code = future.result()

                exit_codes.append(return_code)
                lines = stdout.splitlines()

                pending_re = r"#Pending configs with worker:\s+(\d+)"
                eval_re = r"#Evaluated configs:\s+(\d+)"

                evaluated = first_true(re.match(eval_re, l) for l in lines)  # noqa
                pending = first_true(re.match(pending_re, l) for l in lines)  # noqa

                assert evaluated is not None
                assert pending is not None

                evaluated_configs = int(evaluated.groups()[0])
                pending_configs = int(pending.groups()[0])

                # Make sure the evaluated configs and the ones pending add up to 15
                assert evaluated_configs + pending_configs == 15

        # Make sure all processes don't fail
        assert np.array_equal(exit_codes, np.zeros(n_workers))

        # Make sure there are 15 completed configurations
        expected = sorted(f"config_{i}" for i in range(1, 16))
        folders = sorted(f.name for f in results_dir.iterdir())
        assert folders == expected

    except Exception as e:
        raise e
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir.parent)


@pytest.mark.summary_csv
def test_summary_csv(n_workers=2):
    # Testing the csv files output.
    summary_dir = Path("results") / "hyperparameters_example" / "summary_csv"
    try:
        assert not summary_dir.exists()
        exit_codes = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit each worker function to the concurrent executor
            futures = [executor.submit(run_worker) for _ in range(n_workers)]

            # Wait for all workers to complete
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.ALL_COMPLETED
            )

            for future in done:
                _, return_code = future.result()
                exit_codes.append(return_code)

        # Make sure all processes don't fail
        assert np.array_equal(exit_codes, np.zeros(n_workers))

        # Make sure the directory is created
        assert summary_dir.is_dir()

        run_data_df = pd.read_csv(summary_dir / "run_status.csv")
        run_data_df.set_index("description", inplace=True)
        num_evaluated_configs_csv = run_data_df.loc["num_evaluated_configs", "value"]
        # Make sure all configs are evaluated (expected)
        assert num_evaluated_configs_csv == 15

        config_data_df = pd.read_csv(summary_dir / "config_data.csv")
        # Make sure the total number of rows in our csv is equal to evaluated configs
        assert config_data_df.shape[0] == 15

        # Make sure that the status of all config is complete, hence all are evaluated
        assert (config_data_df["status"] == "complete").all()
    except Exception as e:
        raise e
    finally:
        if summary_dir.exists():
            shutil.rmtree(summary_dir.parent)
