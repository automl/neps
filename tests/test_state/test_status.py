from __future__ import annotations

import scalefm


class Space(scalefm.PipelineSpace):
    x = scalefm.Float(0.0, 1.0)


def _evaluate(x: float) -> dict:
    return {"objective_to_minimize": x, "cost": 1.0}


def test_status_reads_the_given_run_after_acquiring_another(tmp_path) -> None:
    for name, evaluations in [("first", 2), ("second", 3)]:
        scalefm.acquire(
            evaluate_pipeline=_evaluate,
            pipeline_space=Space(),
            root_directory=tmp_path / name,
            optimizer="random_search",
            total_evaluations_to_spend=evaluations,
        )

    # The process ran `acquire` on "second" last, "first" is still summarized itself
    assert len(scalefm.status(tmp_path / "first")[0]) == 2
    assert len(scalefm.status(tmp_path / "second")[0]) == 3
