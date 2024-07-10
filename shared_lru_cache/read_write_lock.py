from __future__ import annotations

from multiprocessing import Condition, Lock, Manager, Value


class ReadWriteLock:
    readers: Value[int]
    readers_lock: Lock
    writers_lock: Lock
    readers_ok: Condition

    def __init__(self, manager: Manager):
        self.readers = manager.Value("i", 0)
        self.readers_lock = manager.Lock()
        self.writers_lock = manager.Lock()
        self.readers_ok = manager.Condition(self.readers_lock)

    def read_lock(self):
        return ReadLock(self)

    def write_lock(self):
        return WriteLock(self)


class ReadLock:
    rw_lock: ReadWriteLock

    def __init__(self, rw_lock):
        self.rw_lock = rw_lock

    def __enter__(self):
        with self.rw_lock.readers_lock:
            self.rw_lock.readers.value += 1
            if self.rw_lock.readers.value == 1:
                self.rw_lock.writers_lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.rw_lock.readers_lock:
            self.rw_lock.readers.value -= 1
            if self.rw_lock.readers.value == 0:
                self.rw_lock.writers_lock.release()


class WriteLock:
    rw_lock: ReadWriteLock

    def __init__(self, rw_lock):
        self.rw_lock = rw_lock

    def __enter__(self):
        self.rw_lock.writers_lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.writers_lock.release()
