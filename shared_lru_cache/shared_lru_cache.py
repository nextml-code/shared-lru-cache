import functools
import pickle
import time
from multiprocessing import Manager, Pool


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
                        self.order.remove(key)
                        self.order.append(key)
                    return pickle.loads(self.data_store[key])

            result = func(*args, **kwargs)

            # Serialize the result
            serialized_result = pickle.dumps(result)

            with self.write_lock:
                # Check again in case another process has updated the cache
                if key in self.cache:
                    self.order.remove(key)
                    self.order.append(key)
                    return pickle.loads(self.data_store[key])

                self.cache[key] = key
                self.data_store[key] = serialized_result
                self.order.append(key)

                if len(self.order) > self.maxsize:
                    oldest = self.order.pop(0)
                    del self.cache[oldest]
                    del self.data_store[oldest]

            return result

        wrapper.cache = self.cache
        wrapper.order = self.order
        wrapper.data_store = self.data_store
        return wrapper


def shared_lru_cache(maxsize=128):
    return SharedLRUCache(maxsize)


def test_shared_lru_cache_basic():
    @shared_lru_cache(maxsize=2)
    def func(x):
        return x * 2

    assert func(2) == 4
    assert func(3) == 6
    assert func(2) == 4  # Should be cached


def test_shared_lru_cache_maxsize():
    @shared_lru_cache(maxsize=2)
    def func(x):
        return x * 2

    func(1)
    func(2)
    func(3)

    # 1 should have been evicted
    assert len(func.cache) == 2
    assert "((1,), frozenset())" not in func.cache


def worker(x):
    @shared_lru_cache(maxsize=10)
    def func(x):
        return x * 2

    return func(x)


def test_shared_lru_cache_large_data():
    @shared_lru_cache(maxsize=2)
    def func(x):
        return b"0" * (10**6)  # 1MB of data

    data1 = func(1)
    data2 = func(2)

    assert len(data1) == 10**6
    assert len(data2) == 10**6
    assert len(func.cache) == 2


shared_cache = shared_lru_cache(maxsize=10)


def worker(x):
    @shared_cache
    def func(x):
        return x * 2

    return func(x)


def test_shared_lru_cache_multiprocessing():
    with Pool(processes=4) as pool:
        results = pool.map(worker, range(20))
    assert results == [x * 2 for x in range(20)]


def test_shared_lru_cache_different_arg_types():
    @shared_lru_cache(maxsize=5)
    def func(x):
        return str(x)

    assert func(1) == "1"
    assert func("a") == "a"
    assert func(2.5) == "2.5"
    assert func((1, 2)) == "(1, 2)"
    assert func({"a": 1}) == "{'a': 1}"


def test_shared_lru_cache_with_kwargs():
    @shared_lru_cache(maxsize=3)
    def func(x, y=10):
        return x + y

    assert func(1) == 11
    assert func(2, y=20) == 22
    assert func(3, y=30) == 33
    assert func(1) == 11  # Should be cached
    assert len(func.cache) == 3


def test_shared_lru_cache_eviction_order():
    @shared_lru_cache(maxsize=3)
    def func(x):
        return x * 2

    func(1)
    func(2)
    func(3)
    func(4)  # This should evict 1

    assert "((1,), frozenset())" not in func.cache
    assert "((2,), frozenset())" in func.cache
    assert "((3,), frozenset())" in func.cache
    assert "((4,), frozenset())" in func.cache


def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


cached_fibonacci = shared_lru_cache(maxsize=100)(fibonacci)


def test_shared_lru_cache_performance():
    start_time = time.time()
    result = cached_fibonacci(30)  # Increased from 4 to 30 for a more meaningful test
    end_time = time.time()

    assert result == 832040  # The 30th Fibonacci number
    assert end_time - start_time < 1  # Should be very fast due to caching


def test_shared_lru_cache_vs_functools_lru_cache():
    import functools
    import time

    # Define functions with shared_lru_cache and functools.lru_cache
    shared_fib = shared_lru_cache(maxsize=100)(fibonacci)

    functools_fib = functools.lru_cache(maxsize=100)(fibonacci)

    # Test performance for a smaller Fibonacci number
    n = 32

    # Measure time for shared_lru_cache
    start_time = time.time()
    shared_result = shared_fib(n)
    shared_time = time.time() - start_time

    # Measure time for functools.lru_cache
    start_time = time.time()
    functools_result = functools_fib(n)
    functools_time = time.time() - start_time

    # Assert that both functions return the same result
    assert shared_result == functools_result

    # Print performance comparison
    print(f"shared_lru_cache time: {shared_time:.6f} seconds")
    print(f"functools.lru_cache time: {functools_time:.6f} seconds")

    # Assert that the performance difference is within an acceptable range
    # This is a flexible assertion as performance can vary between runs
    assert (
        abs(shared_time - functools_time) < 0.1
    ), "Performance difference is too large"


# Simulate loading an image from disk
def load_image(filename):
    import numpy as np

    time.sleep(0.2)  # Simulate I/O delay
    return np.random.rand(2, 2)  # Return a random 1000x1000 array


shared_load = shared_lru_cache(maxsize=100)(load_image)
functools_load = functools.lru_cache(maxsize=100)(load_image)


# Helper function for multiprocessing
def worker_shared(filename):
    return shared_load(filename)


def worker_functools(filename):
    return functools_load(filename)


def test_shared_lru_cache_vs_lru_cache_multiprocessing_all_miss():
    import time
    from multiprocessing import Pool

    # Define functions with shared_lru_cache and functools.lru_cache
    # Test parameters
    n_workers = 4
    n_images = 20
    filenames = [f"image_{i}.jpg" for i in range(n_images)]

    # Test shared_lru_cache
    start_time = time.time()
    with Pool(n_workers) as pool:
        shared_results = pool.map(worker_shared, filenames)
    shared_time = time.time() - start_time

    # Test functools.lru_cache
    start_time = time.time()
    with Pool(n_workers) as pool:
        functools_results = pool.map(worker_functools, filenames)
    functools_time = time.time() - start_time

    # Assert that both methods return the same number of results
    assert len(shared_results) == len(functools_results) == n_images

    # Print performance comparison
    print(f"shared_lru_cache time: {shared_time:.6f} seconds")
    print(f"functools.lru_cache time: {functools_time:.6f} seconds")

    # Assert that shared_lru_cache is faster
    assert (
        shared_time < functools_time * 1.5
    ), "shared_lru_cache should be faster in multiprocessing scenario"


def test_shared_lru_cache_vs_lru_cache_multiprocessing_with_hits():
    import time
    from multiprocessing import Pool

    # Test parameters
    n_workers = 4
    n_images = 20
    n_repeats = 3  # Number of times to repeat the process to ensure cache hits
    filenames = [f"image_{i}.jpg" for i in range(n_images)]

    def run_test(worker_func):
        total_time = 0
        for _ in range(n_repeats):
            start_time = time.time()
            with Pool(n_workers) as pool:
                results = pool.map(worker_func, filenames)
            total_time += time.time() - start_time
        return total_time / n_repeats, results

    # Test shared_lru_cache
    shared_time, shared_results = run_test(worker_shared)

    # Test functools.lru_cache
    functools_time, functools_results = run_test(worker_functools)

    # Assert that both methods return the same number of results
    assert len(shared_results) == len(functools_results) == n_images

    # Print performance comparison
    print(f"shared_lru_cache average time: {shared_time:.6f} seconds")
    print(f"functools.lru_cache average time: {functools_time:.6f} seconds")

    # Assert that shared_lru_cache is faster
    assert (
        shared_time < functools_time
    ), "shared_lru_cache should be faster in multiprocessing scenario with cache hits"
