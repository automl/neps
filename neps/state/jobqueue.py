from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from neps.state.jobs import JOB_MAPPING, Job
from neps.state.shared import Shared


def _serialize_jobqueue_to_jsonl(workqueue: JobQueue, path: Path) -> None:
    filename = path / "jobs.jsonl"
    with filename.open("w") as f:
        lines = [
            json.dumps({"jobname": job.jobname, **asdict(job)}) + "\n"
            for job in workqueue.jobs
        ]
        f.write("\n".join(lines))


def _deserialize_jobqueue_from_jsonl(path: Path) -> JobQueue:
    filename = path / "jobs.jsonl"
    with filename.open("r") as f:
        data = [json.loads(line) for line in f]

    return JobQueue(jobs=deque(JOB_MAPPING[job["jobname"]](**job) for job in data))


@dataclass
class JobQueue:
    """A queue of work that can be consumed by multiple workers."""

    jobs: deque[Job] = field(default_factory=deque)

    def pop(self) -> Job:
        """Remove and return the first job in the queue."""
        return self.jobs.popleft()

    def push(self, job: Job) -> None:
        """Add a job to the queue."""
        self.jobs.append(job)

    def __len__(self) -> int:
        return len(self.jobs)

    def __bool__(self) -> bool:
        return bool(self.jobs)

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return not self.jobs

    def as_filesystem_shared(self, directory: Path) -> Shared[JobQueue, Path]:
        """Return the trial as a shared object."""
        return Shared.using_directory(
            self,
            directory,
            serialize=_serialize_jobqueue_to_jsonl,
            deserialize=_deserialize_jobqueue_from_jsonl,
            lockname=".jobqueue_lock",
            version_filename=".jobqueue_version",
        )
