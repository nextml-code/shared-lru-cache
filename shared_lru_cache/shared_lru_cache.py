import functools
import pickle
from multiprocessing import Manager

import numpy as np
import torch

from .read_write_lock import ReadWriteLock


class SharedLRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.lock = ReadWriteLock(self.manager)
        self.order = self.manager.list()
        self.data_store = self.manager.dict()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, frozenset(kwargs.items())))

            with self.lock.read_lock():
                if key in self.cache:
                    hit = True
                    serialized_result, obj_info = self.data_store[key]
                else:
                    hit = False

            if hit:
                return self.deserialize(serialized_result, obj_info)

            result = func(*args, **kwargs)
            serialized_result, obj_info = self.serialize(result)

            with self.lock.read_lock():
                if key in self.cache:
                    return result

            with self.lock.write_lock():
                self.cache[key] = key
                self.data_store[key] = (serialized_result, obj_info)
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

    def serialize(self, obj):
        if isinstance(obj, np.ndarray):
            obj_info = ("numpy", obj.shape, obj.dtype.str)
            return obj.tobytes(), obj_info
        elif isinstance(obj, torch.Tensor):
            obj.byte()
            numpy_array = obj.cpu().numpy()
            obj_info = ("torch", numpy_array.shape, numpy_array.dtype.str)
            return numpy_array.tobytes(), obj_info
        else:
            obj_info = ("other",)
            return pickle.dumps(obj), obj_info

    def deserialize(self, data, obj_info):
        obj_type, *info = obj_info
        if obj_type == "numpy":
            shape, dtype = info
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
        elif obj_type == "torch":
            shape, dtype = info
            dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype
            numpy_array = np.frombuffer(data, dtype=dtype).reshape(shape)
            return torch.from_numpy(numpy_array)
        else:
            return pickle.loads(data)


def shared_lru_cache(maxsize=128):
    return SharedLRUCache(maxsize)
