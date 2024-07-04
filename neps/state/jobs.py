from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar


@dataclass(kw_only=True)
class Job(ABC):
    jobname: ClassVar[str]
    issued_by: str
    completed_by: str | None = None


@dataclass(kw_only=True)
class SampleJob(Job):
    jobname: ClassVar[str] = "sample"
    issued_by: str
    completed_by: str | None = None

    @classmethod
    def new(cls, issued_by: str) -> SampleJob:
        return cls(issued_by=issued_by)


@dataclass(kw_only=True)
class EvaluateJob(Job):
    jobname: ClassVar[str] = "evaluate"
    issued_by: str
    completed_by: str | None = None

    trial_id: str

    @classmethod
    def new(cls, issued_by: str, trial_id: str) -> EvaluateJob:
        return cls(issued_by=issued_by, trial_id=trial_id)


JOB_MAPPING: dict[str, type[Job]] = {JOB.jobname: JOB for JOB in (SampleJob, EvaluateJob)}
