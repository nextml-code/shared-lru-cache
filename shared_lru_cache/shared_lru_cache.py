import functools
import pickle
from multiprocessing import Manager, shared_memory

import numpy as np
import torch

from .read_write_lock import ReadWriteLock


class SharedLRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.rw_lock = ReadWriteLock(self.manager)
        self.order = self.manager.list()
        self.data_store = {}

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, frozenset(kwargs.items())))

            with self.rw_lock.read_lock():
                if key in self.cache:
                    hit = True
                    shm_name, obj_info = self.data_store.get(key, (None, None))
                    if shm_name is None:
                        hit = False
                        serialized_result = None
                    else:
                        existing_shm = shared_memory.SharedMemory(name=shm_name)
                        serialized_result = bytes(existing_shm.buf[: existing_shm.size])
                        existing_shm.close()
                else:
                    hit = False
                    serialized_result = None
                    shm_name = None

            if hit:
                print(
                    f"hit: {key}, current cache keys: {list(self.cache.keys())}",
                    flush=True,
                )
                return self.deserialize(serialized_result, obj_info)
            else:
                print(
                    f"miss: {key}, shm_name: {shm_name}, current cache keys: {list(self.cache.keys())}",
                    flush=True,
                )

            result = func(*args, **kwargs)
            serialized_result, obj_info = self.serialize(result)

            # with self.rw_lock.read_lock():
            #     if key in self.cache:
            #         return result

            shm = shared_memory.SharedMemory(create=True, size=len(serialized_result))
            shm.buf[: len(serialized_result)] = serialized_result

            with self.rw_lock.write_lock():
                if key not in self.cache:
                    self.cache[key] = key
                    self.data_store[key] = (shm.name, obj_info)
                    self.order.append(key)

                    print(f"added: {key}", flush=True)

                    # while len(self.order) > self.maxsize:
                    #     oldest = self.order.pop(0)
                    #     self.cache.pop(oldest, None)
                    #     shm_name, _ = self.data_store.pop(oldest, (None, None))
                    #     if shm_name:
                    #         old_shm = shared_memory.SharedMemory(name=shm_name)
                    #         old_shm.close()
                    #         old_shm.unlink()

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
            print("numpy", flush=True)
            shape, dtype = info
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
        elif obj_type == "torch":
            print("torch", flush=True)
            shape, dtype = info
            dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype
            numpy_array = np.frombuffer(data, dtype=dtype).reshape(shape)
            return torch.from_numpy(numpy_array)
        else:
            print("other", flush=True)
            return pickle.loads(data)


def shared_lru_cache(maxsize=128):
    return SharedLRUCache(maxsize)
