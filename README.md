# shared lru cache

## Install

```bash
poetry add shared-lru-cache
```

## Usage

```
from shared_lru_cache import shared_lru_cache


@shared_lru_cache(maxsize=8)
def load_image():
    pass
```
