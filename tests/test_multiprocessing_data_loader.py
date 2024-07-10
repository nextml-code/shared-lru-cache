import time
from functools import lru_cache
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset

from shared_lru_cache import shared_lru_cache

MAX_INDEX = 8
CACHE_SIZE = 16


def load_image(idx):
    time.sleep(5)  # Simulate some data loading time
    return torch.zeros((1024, 2500), dtype=torch.uint8)


lru_cached_load_image = lru_cache(maxsize=CACHE_SIZE)(load_image)
shared_lru_cached_load_image = shared_lru_cache(maxsize=CACHE_SIZE)(load_image)


class LRUCachedDataset(Dataset):
    def __init__(self, size=128):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return lru_cached_load_image(idx % MAX_INDEX)


class SharedLRUCachedDataset(Dataset):
    def __init__(self, size=128):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return shared_lru_cached_load_image(idx % MAX_INDEX)


def getitem(input):
    shared_lru_dataset, index = input
    return shared_lru_dataset[index]


def test_shared_lru_cache_with_pool():
    dataset_size = 32
    batch_size = 1
    num_workers = 3

    shared_lru_dataset = SharedLRUCachedDataset(size=dataset_size)

    start_time = time.time()
    with Pool(num_workers) as pool:
        # pool.map(process_batch, shared_lru_dataset)
        pool.map(
            getitem, [(shared_lru_dataset, index) for index in range(dataset_size)]
        )
    shared_lru_time = time.time() - start_time

    print(f"Shared LRU cache DataLoader with Pool time: {shared_lru_time:.6f} seconds")
