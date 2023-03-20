import time
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass
class Watch:
    pred: Callable[[], bool]

    def __call__(
        self, *, timeout: float | None = None, poll: float = 0.05
    ) -> Iterator[bool]:
        start = time.perf_counter()
        pred = self.pred

        # Keep checking every `poll` seconds until the file exists
        while True:
            if pred():
                yield True
                break

            if timeout is not None and time.perf_counter() - start > timeout:
                yield False
                break

            time.sleep(poll)
