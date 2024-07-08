import functools
import pickle
from multiprocessing import Manager


class SharedLRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.read_lock = self.manager.RLock()
        self.write_lock = self.manager.Lock()
        self.order = self.manager.list()
        self.data_store = self.manager.dict()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, frozenset(kwargs.items())))

            with self.read_lock:
                if key in self.cache:
                    with self.write_lock:
                        if (
                            key in self.order
                        ):  # Check if key is in order before removing
                            self.order.remove(key)
                        self.order.append(key)
                    return pickle.loads(self.data_store[key])

            result = func(*args, **kwargs)
            serialized_result = pickle.dumps(result)

            with self.write_lock:
                # Check again in case another process has updated the cache
                if key in self.cache:
                    return result

                self.cache[key] = key
                self.data_store[key] = serialized_result
                self.order.append(key)

                while len(self.order) > self.maxsize:
                    oldest = self.order.pop(0)
                    self.cache.pop(oldest, None)
                    self.data_store.pop(oldest, None)

            return result

        wrapper.cache = self.cache
        wrapper.order = self.order
        wrapper.data_store = self.data_store
        return wrapper


def shared_lru_cache(maxsize=128):
    return SharedLRUCache(maxsize)
