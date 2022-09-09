import logging
import os

import pytest
from regression_runner import OPTIMIZERS, TASKS, RegressionRunner


@pytest.fixture(autouse=True)
def use_tmpdir(tmp_path, request):
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


# https://stackoverflow.com/a/59745629
# Fail tests if there is a logging.error
@pytest.fixture(autouse=True)
def no_logs_gte_error(caplog):
    yield
    errors = [
        record for record in caplog.get_records("call") if record.levelno >= logging.ERROR
    ]
    assert not errors


@pytest.mark.regression_all
@pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=OPTIMIZERS)
def test_regression_all(optimizer):
    test_results = {}
    test_results["test_agg"] = 0
    test_results["task_agg"] = 0
    for task in TASKS:
        ks_test, median_test, median_improvement = RegressionRunner(
            optimizer=optimizer, task=task
        ).test()

        test_results[task] = [ks_test, median_test, median_improvement]

        test_results["task_agg"] += (
            1 if (ks_test + median_test == 2) or median_improvement else 0
        )
        test_results["test_agg"] = (
            test_results["test_agg"] + ks_test + median_test + 2 * median_improvement
        )

    result = (
        1
        if test_results["task_agg"] >= 1 and test_results["test_agg"] >= len(TASKS) + 1
        else 0
    )
    assert result == 1, f"Test for {optimizer} didn't pass: {test_results}"

    logging.info(f"Regression test for {optimizer} passed successfully!")
