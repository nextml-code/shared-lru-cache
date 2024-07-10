import time
from functools import lru_cache
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader, Dataset

from shared_lru_cache import shared_lru_cache

MAX_INDEX = 8
CACHE_SIZE = 16


def load_image(idx):
    time.sleep(5)  # Simulate some data loading time
    return torch.zeros((1024, 2500), dtype=torch.uint8)


lru_cached_load_image = lru_cache(maxsize=CACHE_SIZE)(load_image)
shared_lru_cached_load_image = shared_lru_cache(
    maxsize=CACHE_SIZE, update_on_hit=False
)(load_image)


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


def test_shared_lru_cache_vs_standard_dataloader():
    dataset_size = 32
    batch_size = 1
    num_workers = 3

    # Standard DataLoader with LRUCachedDataset
    # lru_dataset = LRUCachedDataset(size=dataset_size)
    # lru_loader = DataLoader(
    #     lru_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     collate_fn=list,
    # )

    # start_time = time.time()
    # for _ in lru_loader:
    #     pass
    # lru_time = time.time() - start_time

    # Shared LRU cache DataLoader
    shared_lru_dataset = SharedLRUCachedDataset(size=dataset_size)
    shared_lru_loader = DataLoader(
        shared_lru_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,
    )

    start_time = time.time()
    for _ in shared_lru_loader:
        pass
    shared_lru_time = time.time() - start_time

    # print(f"Standard LRU cache DataLoader time: {lru_time:.6f} seconds")
    print(f"Shared LRU cache DataLoader time: {shared_lru_time:.6f} seconds")

    # assert (
    #     shared_lru_time < lru_time
    # ), "Shared LRU cache should be faster than standard LRU cache DataLoader"
