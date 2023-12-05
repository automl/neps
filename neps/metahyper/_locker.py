import atexit
import time
from contextlib import contextmanager

import portalocker as pl


class Locker:
    def __init__(self, lock_path, logger):
        self.logger = logger
        atexit.register(self.__del__)
        self.lock_path = lock_path
        self.lock_file = self.lock_path.open("a")  # a for security

    def __del__(self):
        self.lock_file.close()

    def release_lock(self):
        self.logger.debug(f"Release lock for {self.lock_path}")
        pl.unlock(self.lock_file)

    def acquire_lock(self):
        try:
            pl.lock(self.lock_file, pl.LOCK_EX | pl.LOCK_NB)
            self.logger.debug(f"Acquired lock for {self.lock_path}")
            return True
        except pl.exceptions.LockException:
            self.logger.debug(f"Failed to acquire lock for {self.lock_path}")
            return False

    @contextmanager
    def acquire_force(self, time_step=1):
        while not self.acquire_lock():
            time.sleep(time_step)
        yield True
        self.release_lock()
