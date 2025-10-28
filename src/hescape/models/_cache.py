# hescape/models/_cache.py
"""Shared caching utility for frozen encoder embeddings."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Any

import torch

# Find project root
project_root = next(p for p in Path(__file__).parents if (p / '.git').exists())


class EmbeddingCache:
    """Simple disk-based cache for frozen encoder embeddings."""

    def __init__(self, cache_name: str):
        """
        Initialize cache in project_root/cache/{cache_name}/

        Args:
            cache_name: Name of the cache (e.g., 'tile_embeddings', 'cpgpt_embeddings')
        """
        self.cache_dir = project_root / "cache" / cache_name

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pt"

    def get(self, key: str) -> Any | None:
        """
        Load cached data for a key.

        Args:
            key: Cache key (e.g., file path, slide directory)

        Returns:
            Cached data if exists, None otherwise
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        return None

    def set(self, key: str, data: Any) -> None:
        """
        Save data to cache.

        Args:
            key: Cache key (e.g., file path, slide directory)
            data: Data to cache (must be torch.save compatible)
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(key)
        torch.save(data, cache_path)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        Get from cache or compute if not cached.

        Args:
            key: Cache key
            compute_fn: Function to compute the value if not cached (takes no args)

        Returns:
            Cached or computed data
        """
        cached = self.get(key)
        if cached is not None:

            return cached

        # Cache miss - compute
        data = compute_fn()
        self.set(key, data)
        return data
