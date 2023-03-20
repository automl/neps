from __future__ import annotations
import re
import shutil
import subprocess
from typing import TypeVar, Generic
from dataclasses import dataclass
from pathlib import Path

import pytest
from more_itertools import first_true

HERE = Path(__file__).parent
LOCK_GRABBER = HERE / "lock_grabber.py"

CMD = f"python {LOCK_GRABBER}"

def process(cmd) -> subprocess.Popen:
    """Run a command in a subprocess."""
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, text=True)


@dataclass
class Task:
    file: Path

@dataclass
class Writer(Task):
    content: str

    def start(self) -> Worker[Writer]:
        return Worker(self, process(f"{CMD} --file {self.file} --content {self.content}"))

@dataclass
class Reader(Task):

    def start(self) -> Worker[Reader]:
        return Worker(self, process(f"{CMD} --file {self.file}"))

WorkerKind = TypeVar("WorkerKind", bound=Task)

@dataclass
class Worker(Generic[WorkerKind]):
    task: WorkerKind
    process: subprocess.Popen

    def wait(self) -> None:
        self.process.wait()

    def stdout(self) -> str:
        stdout = self.process.stdout
        assert stdout is not None
        return stdout.read()

    def stderr(self) -> str:
        stderr = self.process.stderr
        assert stderr is not None
        return stderr.read()


@pytest.mark.metahyper
def test_many_acquirers(tmp_path: Path) -> None:
    test_dir = tmp_path / "test_many_acquirers"
    test_dir.mkdir()



@pytest.mark.metahyper
def test_example_with_filelock() -> None:
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
        p1 = process("python -m neps_examples.basic_usage.hyperparameters && python -m neps_examples.basic_usage.analyse")
        p2 = process("python -m neps_examples.basic_usage.hyperparameters && python -m neps_examples.basic_usage.analyse")

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
