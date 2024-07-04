from neps.state import Trial
import os
import numpy as np

def test_trial_creation() -> None:
    trial_id = "1"
    time_sampled = 0
    previous_trial = "0"
    worker_id = str(os.getpid())

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=worker_id
    )
    assert trial.state == Trial.State.PENDING
    assert trial.report is None
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id="1",
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=worker_id,
        time_started=None,
        time_submitted=None,
        time_end=None,
    )

def test_trial_as_submitted() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    previous_trial = "0"
    worker_id = str(os.getpid())

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=worker_id
    )
    trial.set_submitted(time_submitted=time_submitted)

    assert trial.state == Trial.State.SUBMITTED
    assert trial.report is None
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=worker_id,
        time_submitted=time_submitted,
        time_started=None,
        time_end=None,
    )

def test_trial_as_in_progress_with_different_evaluating_worker() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    time_started = 2
    previous_trial = "0"
    sampling_worker_id = "42"
    evaluating_worker_id = "43"

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_in_progress(time_started=time_started, worker_id=evaluating_worker_id)

    assert trial.state == Trial.State.IN_PROGRESS
    assert trial.report is None
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        time_started=time_started,
        time_end=None,
    )

def test_trial_as_success_after_being_progress() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    time_started = 2
    time_end = 3
    previous_trial = "0"
    sampling_worker_id = "42"
    evaluating_worker_id = "43"
    loss = 427
    cost = -123.6
    account_for_cost = True
    extra={"picnic": "basket", "counts": [1,2,3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_in_progress(time_started=time_started, worker_id=evaluating_worker_id)
    trial.set_success(
        loss=loss,
        cost=cost,
        account_for_cost=account_for_cost,
        extra=extra,
        time_end=time_end,
    )

    assert trial.state == Trial.State.SUCCESS
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        time_started=time_started,
        time_end=time_end,
    )
    assert trial.report == Trial.Report(
        trial_id=trial_id,
        loss=loss,
        cost=cost,
        account_for_cost=account_for_cost,
        extra=extra,
        err=None,
        tb=None,
        reported_as="success",
    )

def test_trial_as_failed_with_nan_loss_and_in_cost() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    time_started = 2
    time_end = 3
    previous_trial = "0"
    sampling_worker_id = "42"
    evaluating_worker_id = "43"
    loss = np.nan
    cost = np.inf
    account_for_cost = True
    extra={"picnic": "basket", "counts": [1,2,3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_in_progress(time_started=time_started, worker_id=evaluating_worker_id)
    trial.set_fail(
        loss=loss,
        cost=cost,
        account_for_cost=account_for_cost,
        extra=extra,
        time_end=time_end,
    )

    assert trial.state == Trial.State.FAILED
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        time_started=time_started,
        time_end=time_end,
    )
    assert trial.report == Trial.Report(
        trial_id=trial_id,
        loss=loss,
        cost=cost,
        account_for_cost=account_for_cost,
        extra=extra,
        err=None,
        tb=None,
        reported_as="failed",
    )

def test_trial_as_crashed_with_err_and_tb() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    time_started = 2
    time_end = 3
    err = ValueError("Something went wrong")
    tb = "some traceback"
    previous_trial = "0"
    sampling_worker_id = "42"
    evaluating_worker_id = "43"
    extra={"picnic": "basket", "counts": [1,2,3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_in_progress(time_started=time_started, worker_id=evaluating_worker_id)
    trial.set_crashed(
        err=err,
        tb=tb,
        extra=extra,
        time_end=time_end,
    )

    assert trial.state == Trial.State.CRASHED
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        time_started=time_started,
        time_end=time_end,
    )
    assert trial.report == Trial.Report(
        trial_id=trial_id,
        loss=None,
        cost=None,
        account_for_cost=False,
        extra=extra,
        err=err,
        tb=tb,
        reported_as="crashed",
    )
