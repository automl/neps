import re
import shutil
import subprocess
from pathlib import Path

import pytest
from more_itertools import first_true


@pytest.mark.metahyper
def test_filelock() -> None:
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

        # Launch both processes
        p1 = subprocess.Popen(  # pylint: disable=consider-using-with
            "python -m neps_examples.basic_usage.hyperparameters && python -m neps_examples.basic_usage.analyse",
            stdout=subprocess.PIPE,
            shell=True,
            text=True,
        )
        p2 = subprocess.Popen(  # pylint: disable=consider-using-with
            "python -m neps_examples.basic_usage.hyperparameters && python -m neps_examples.basic_usage.analyse",
            stdout=subprocess.PIPE,
            shell=True,
            text=True,
        )

        # Wait for them
        for p in (p1, p2):
            p.wait()
            out, _ = p.communicate()
            lines = out.splitlines()

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

        # Make sure there are 15 completed configurations
        expected = sorted(f"config_{i}" for i in range(1, 16))
        folders = sorted(f.name for f in results_dir.iterdir())
        assert folders == expected

    except Exception as e:
        raise e
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
