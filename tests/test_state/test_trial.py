from __future__ import annotations

import os

import numpy as np

from neps.state import Trial


def test_trial_creation() -> None:
    trial_id = "1"
    time_sampled = 0
    previous_trial = "0"
    worker_id = str(os.getpid())

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="1",
        previous_trial_location=None,
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=worker_id,
    )
    assert trial.metadata.state == Trial.State.PENDING
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id="1",
        time_sampled=time_sampled,
        state=Trial.State.PENDING,
        location="1",
        previous_trial_location=None,
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
        previous_trial_location="0",
        location="1",
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=worker_id,
    )
    trial.set_submitted(time_submitted=time_submitted)

    assert trial.metadata.state == Trial.State.SUBMITTED
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        state=Trial.State.SUBMITTED,
        time_sampled=time_sampled,
        previous_trial_location="0",
        location="1",
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
        location="1",
        previous_trial_location="0",
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id,
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_evaluating(time_started=time_started, worker_id=evaluating_worker_id)

    assert trial.metadata.state == Trial.State.EVALUATING
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        state=Trial.State.EVALUATING,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        previous_trial_location="0",
        location="1",
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
    objective_to_minimize = 427
    cost = -123.6
    extra = {"picnic": "basket", "counts": [1, 2, 3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        location="1",
        previous_trial_location="0",
        worker_id=sampling_worker_id,
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_evaluating(time_started=time_started, worker_id=evaluating_worker_id)
    report = trial.set_complete(
        report_as="success",
        objective_to_minimize=objective_to_minimize,
        cost=cost,
        err=None,
        tb=None,
        learning_curve=None,
        evaluation_duration=time_end - time_started,
        extra=extra,
        time_end=time_end,
    )

    assert trial.metadata.state == Trial.State.SUCCESS
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        state=Trial.State.SUCCESS,
        time_sampled=time_sampled,
        previous_trial_location="0",
        location="1",
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        evaluation_duration=time_end - time_started,
        time_submitted=time_submitted,
        time_started=time_started,
        time_end=time_end,
    )
    assert report == Trial.Report(
        trial_id=trial_id,
        objective_to_minimize=objective_to_minimize,
        cost=cost,
        learning_curve=None,
        evaluation_duration=1,
        extra=extra,
        err=None,
        tb=None,
        reported_as="success",
    )


def test_trial_as_failed_with_nan_objective_to_minimize_and_in_cost() -> None:
    trial_id = "1"
    time_sampled = 0
    time_submitted = 1
    time_started = 2
    time_end = 3
    previous_trial = "0"
    sampling_worker_id = "42"
    evaluating_worker_id = "43"
    objective_to_minimize = np.nan
    cost = np.inf
    extra = {"picnic": "basket", "counts": [1, 2, 3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        location="1",
        previous_trial_location="0",
        time_sampled=time_sampled,
        previous_trial=previous_trial,
        worker_id=sampling_worker_id,
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_evaluating(time_started=time_started, worker_id=evaluating_worker_id)
    report = trial.set_complete(
        report_as="failed",
        objective_to_minimize=objective_to_minimize,
        cost=cost,
        learning_curve=None,
        evaluation_duration=time_end - time_started,
        err=None,
        tb=None,
        extra=extra,
        time_end=time_end,
    )
    assert trial.metadata.state == Trial.State.FAILED
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        state=Trial.State.FAILED,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        previous_trial_location="0",
        location="1",
        time_started=time_started,
        time_end=time_end,
        evaluation_duration=time_end - time_started,
    )
    assert report == Trial.Report(
        trial_id=trial_id,
        objective_to_minimize=objective_to_minimize,
        cost=cost,
        learning_curve=None,
        evaluation_duration=time_end - time_started,
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
    extra = {"picnic": "basket", "counts": [1, 2, 3]}

    trial = Trial.new(
        trial_id=trial_id,
        config={"a": "b"},
        time_sampled=time_sampled,
        location="1",
        previous_trial_location="42",
        previous_trial=previous_trial,
        worker_id=sampling_worker_id,
    )
    trial.set_submitted(time_submitted=time_submitted)
    trial.set_evaluating(time_started=time_started, worker_id=evaluating_worker_id)
    report = trial.set_complete(
        report_as="crashed",
        objective_to_minimize=None,
        cost=None,
        learning_curve=None,
        evaluation_duration=time_end - time_started,
        err=err,
        tb=tb,
        extra=extra,
        time_end=time_end,
    )

    assert trial.metadata.state == Trial.State.CRASHED
    assert trial.id == trial_id
    assert trial.config == {"a": "b"}
    assert trial.metadata == Trial.MetaData(
        id=trial_id,
        state=Trial.State.CRASHED,
        time_sampled=time_sampled,
        previous_trial_id=previous_trial,
        sampling_worker_id=sampling_worker_id,
        evaluating_worker_id=evaluating_worker_id,
        time_submitted=time_submitted,
        previous_trial_location="42",
        location="1",
        time_started=time_started,
        time_end=time_end,
        evaluation_duration=time_end - time_started,
    )
    assert report == Trial.Report(
        trial_id=trial_id,
        objective_to_minimize=None,
        cost=None,
        learning_curve=None,
        evaluation_duration=time_end - time_started,
        extra=extra,
        err=err,
        tb=tb,
        reported_as="crashed",
    )
