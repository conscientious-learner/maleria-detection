"""Persistent image verification, perceptual hashes, and duplicate grouping."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn

from .config import HASH_PROGRESS_INTERVAL


LOGGER = logging.getLogger("malaria.data")
_IMAGE_ERRORS = (OSError, ValueError, SyntaxError, UnidentifiedImageError)


@dataclass
class _HashNode:
    """One indexed perceptual hash in the BK-tree."""

    value: int
    indices: list[int] = field(default_factory=list)
    children: dict[int, _HashNode] = field(default_factory=dict)


class _BKTree:
    """Query perceptual hashes within a small Hamming distance."""

    def __init__(self) -> None:
        self.root: _HashNode | None = None

    def add(self, value: int, index: int) -> None:
        """Insert a hash and retain all source indices with identical hashes."""
        if self.root is None:
            self.root = _HashNode(value, [index])
            return

        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _HashNode(value, [index])
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        """Return prior hash indices and distances within ``radius``."""
        matches: list[tuple[int, int]] = []
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= radius:
                matches.extend((index, distance) for index in node.indices)
            low, high = distance - radius, distance + radius
            stack.extend(
                child for edge, child in node.children.items() if low <= edge <= high
            )
        return matches


class _UnionFind:
    """Join near-duplicate pairs so each connected set stays in one split."""

    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        """Return a component root with path compression."""
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, first: int, second: int) -> None:
        """Merge two components."""
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


def perceptual_hash(path: str | Path) -> int:
    """Compute a 64-bit DCT perceptual hash from a grayscale image."""
    with Image.open(path) as source:
        pixels = np.asarray(
            source.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    low = dctn(pixels, type=2, norm="ortho")[:8, :8].flatten()
    median = np.median(low[1:])
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    """Count differing bits in two perceptual hashes."""
    return (first ^ second).bit_count()


def _inspect_image(path: str) -> dict[str, Any]:
    """Verify an image and compute its hash, returning an invalid record on decode errors."""
    try:
        with Image.open(path) as image:
            image.verify()
        return {"valid": True, "phash": perceptual_hash(path)}
    except _IMAGE_ERRORS:
        return {"valid": False, "phash": None}


def _read_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Read cache records, treating an invalid cache as a rebuild request."""
    if not cache_path.is_file():
        return {}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Could not read image cache %s: %s", cache_path, error)
        return {}
    return cache.get("entries", {}) if isinstance(cache, dict) else {}


def _scan_changed(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Inspect uncached images concurrently while preserving deterministic path order."""
    workers = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(executor.map(_inspect_image, paths))
    return dict(zip(paths, inspected))


def _write_cache(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Persist fingerprints and hashes atomically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    temporary_path.replace(cache_path)


def cached_image_records(
    paths: Sequence[str | Path], cache_path: Path
) -> dict[str, dict[str, Any]]:
    """Return verified-image records keyed by absolute path, reusing unchanged hashes."""
    previous = _read_cache(cache_path)
    current: dict[str, dict[str, Any]] = {}
    changed: list[str] = []
    for source in paths:
        path = str(Path(source).resolve())
        try:
            stat = Path(path).stat()
        except OSError as error:
            LOGGER.warning("Skipping inaccessible image %s: %s", path, error)
            continue
        old = previous.get(path, {})
        if old.get("mtime_ns") == stat.st_mtime_ns and old.get("size") == stat.st_size:
            current[path] = old
        else:
            changed.append(path)

    for path, result in _scan_changed(changed).items():
        stat = Path(path).stat()
        current[path] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, **result}

    _write_cache(cache_path, current)
    return current


def find_near_duplicates(
    paths: Sequence[str],
    max_distance: int,
    hashes: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Find pHash matches and build component IDs for leakage-safe splitting."""
    path_hashes = hashes or _parallel_hashes(paths)
    tree = _BKTree()
    components = _UnionFind(len(paths))
    duplicate_pairs: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        image_hash = path_hashes[path]
        for previous, distance in tree.query(image_hash, max_distance):
            components.union(index, previous)
            duplicate_pairs.append(
                {"first": paths[previous], "second": path, "hamming_distance": distance}
            )
        tree.add(image_hash, index)
        if (index + 1) % HASH_PROGRESS_INTERVAL == 0:
            LOGGER.info("Perceptual duplicate scan: %d / %d images", index + 1, len(paths))
    groups = np.asarray([components.find(index) for index in range(len(paths))], dtype=np.int64)
    return duplicate_pairs, groups


def _parallel_hashes(paths: Sequence[str]) -> dict[str, int]:
    """Hash a direct caller's images concurrently when no persistent records are supplied."""
    workers = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(zip(paths, executor.map(perceptual_hash, paths)))
