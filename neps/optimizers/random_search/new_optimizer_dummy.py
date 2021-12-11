import logging

from ..core.new_optimizer import Optimizer


class _DummySearcher(Optimizer):
    def __init__(self, pipeline_space):
        self.pipeline_space = pipeline_space
        self.logger = logging.getLogger(__name__)

    def new_result(self, job):
        if job.exception is not None:
            self.logger.warning(f"job {job.id} failed with exception\n{job.exception}")

        # job.kwargs["config"]
        # job.result["loss"]

    def get_config(self):  # pylint: disable=no-self-use
        return dict()
