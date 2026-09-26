from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

from .hilbert import HilbertFactorExtractor

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


FactorTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


class ArrayCache:
    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self.values: OrderedDict[Tuple[Any, ...], FactorTuple] = OrderedDict()

    @staticmethod
    def _value_nbytes(value: FactorTuple) -> int:
        return sum(int(item.nbytes) for item in value)

    def clear(self) -> None:
        self.values.clear()
        self.current_bytes = 0

    def get(self, key: Tuple[Any, ...]) -> Optional[FactorTuple]:
        value = self.values.get(key)
        if value is None:
            return None
        self.values.move_to_end(key)
        return value

    def put(self, key: Tuple[Any, ...], value: FactorTuple) -> FactorTuple:
        previous = self.values.pop(key, None)
        if previous is not None:
            self.current_bytes -= self._value_nbytes(previous)
        value_bytes = self._value_nbytes(value)
        if value_bytes > self.max_bytes:
            return value
        self.values[key] = value
        self.current_bytes += value_bytes
        self.values.move_to_end(key)
        while self.current_bytes > self.max_bytes:
            _, removed = self.values.popitem(last=False)
            self.current_bytes -= self._value_nbytes(removed)
        return value


class DiskFactorCache:
    def __init__(
        self,
        root: Path,
        metadata: Dict[str, Any],
        length: int,
        num_features: int,
        imf_count: int,
        input_len: int,
        feature_dim: int,
        read_only: bool = False,
    ) -> None:
        if length <= 0:
            raise ValueError("disk factor cache length must be positive")
        self.root = root
        self.length = int(length)
        self.num_features = int(num_features)
        self.read_only = bool(read_only)
        if self.read_only and not root.is_dir():
            raise FileNotFoundError(
                f"read-only factor cache directory does not exist: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                saved = json.load(file)
            if saved != metadata:
                raise ValueError(
                    f"factor cache metadata mismatch in {root}; use a different "
                    "factor_cache_dir or remove the stale cache directory"
                )
        elif self.read_only:
            raise FileNotFoundError(
                f"read-only factor cache metadata does not exist: {metadata_path}"
            )
        else:
            metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
        self.candidates = self._open_array(
            "candidates.float16.mmap",
            (length, num_features, imf_count, input_len),
            np.float16,
        )
        self.features = self._open_array(
            "features.float16.mmap",
            (length, num_features, imf_count, input_len, feature_dim),
            np.float16,
        )
        self.active = self._open_array(
            "active.uint8.mmap",
            (length, num_features, imf_count),
            np.uint8,
        )
        self.ready = self._open_array(
            "ready.uint8.mmap",
            (length, num_features),
            np.uint8,
        )
        if self.read_only:
            ready_count = int(np.count_nonzero(self.ready))
            expected_count = self.length * self.num_features
            if ready_count != expected_count:
                raise RuntimeError(
                    "distributed refinement requires a complete read-only "
                    f"factor cache: {self.root} contains {ready_count}/"
                    f"{expected_count} factors"
                )

    def _open_array(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> np.memmap:
        path = self.root / name
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if path.exists():
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"factor cache file has unexpected size: {path}; expected "
                    f"{expected_bytes} bytes"
                )
            mode = "r" if self.read_only else "r+"
        elif self.read_only:
            raise FileNotFoundError(f"read-only factor cache file is missing: {path}")
        else:
            mode = "w+"
        array = np.memmap(path, mode=mode, dtype=dtype, shape=shape)
        if mode == "w+" and name == "ready.uint8.mmap":
            array.fill(0)
            array.flush()
        return array

    def get(
        self,
        index: int,
        feature_index: int,
        imf_count: int,
    ) -> Optional[FactorTuple]:
        if index < 0 or index >= self.length:
            raise IndexError(f"factor cache index out of range: {index}")
        if feature_index < 0 or feature_index >= self.num_features:
            raise IndexError(
                f"factor cache feature index out of range: {feature_index}"
            )
        if self.ready[index, feature_index] == 0:
            return None
        requested_imf_count = int(imf_count)
        if requested_imf_count <= 0 or requested_imf_count > self.candidates.shape[2]:
            raise ValueError(
                f"requested IMF count {requested_imf_count} is incompatible with "
                f"cache count {self.candidates.shape[2]}"
            )
        return (
            self.candidates[index, feature_index, :requested_imf_count],
            self.features[index, feature_index, :requested_imf_count],
            self.active[index, feature_index, :requested_imf_count],
        )

    def get_batch(
        self,
        indices: np.ndarray,
        imf_count: int,
    ) -> Optional[FactorTuple]:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("factor cache batch indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= self.length):
            raise IndexError(
                "factor cache batch index out of range: "
                f"min={int(indices.min())}, max={int(indices.max())}, "
                f"cache_length={self.length}"
            )
        contiguous = indices.size == 1 or np.all(np.diff(indices) == 1)
        selection = (
            slice(int(indices[0]), int(indices[-1]) + 1)
            if contiguous
            else indices
        )
        ready = self.ready[selection]
        if not np.all(ready):
            return None
        requested_imf_count = int(imf_count)
        if requested_imf_count <= 0 or requested_imf_count > self.candidates.shape[2]:
            raise ValueError(
                f"requested IMF count {requested_imf_count} is incompatible with "
                f"cache count {self.candidates.shape[2]}"
            )
        return (
            np.asarray(self.candidates[selection, :, :requested_imf_count]),
            np.asarray(self.features[selection, :, :requested_imf_count]),
            np.asarray(self.active[selection, :, :requested_imf_count]),
        )

    def materialize(self) -> None:
        if not np.all(self.ready):
            raise RuntimeError(
                f"cannot materialize incomplete factor cache: {self.root}"
            )
        self.candidates = np.array(self.candidates, copy=True)
        self.features = np.array(self.features, copy=True)
        self.active = np.array(self.active, copy=True)
        self.ready = np.array(self.ready, copy=True)

    def put(
        self,
        index: int,
        feature_index: int,
        factors: FactorTuple,
    ) -> None:
        if self.read_only:
            raise RuntimeError(
                f"cannot write incomplete read-only factor cache: {self.root}"
            )
        candidate, feature, activity = factors
        self.candidates[index, feature_index] = candidate
        self.features[index, feature_index] = feature
        self.active[index, feature_index] = activity
        self.ready[index, feature_index] = 1

    def flush(self) -> None:
        if self.read_only or not isinstance(self.candidates, np.memmap):
            return
        self.candidates.flush()
        self.features.flush()
        self.active.flush()
        self.ready.flush()


class FactorCacheManager:
    def __init__(
        self,
        imf_count: int,
        cache_dir: Optional[str],
        cache_imf_count: Optional[int],
        read_only: bool,
    ) -> None:
        self.imf_count = int(imf_count)
        self.cache_dir = (
            str(Path(cache_dir).expanduser().resolve())
            if cache_dir is not None
            else None
        )
        self.read_only = bool(read_only)
        if self.read_only and self.cache_dir is None:
            raise ValueError("factor_cache_read_only requires factor_cache_dir")
        if cache_imf_count is not None and cache_imf_count < self.imf_count:
            raise ValueError(
                "factor_cache_imf_count must be at least imf_count so cached "
                "factors can satisfy the requested gate inputs"
            )
        self.cache_imf_count = (
            int(cache_imf_count)
            if cache_imf_count is not None
            else self.imf_count
        )
        self.memory = ArrayCache(max_bytes=40 * 1024**3)
        self.disk: Dict[str, DiskFactorCache] = {}

    def clear(self) -> None:
        self.memory.clear()
        self.disk.clear()

    def flush(self) -> None:
        for cache in self.disk.values():
            cache.flush()

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _finite_float16_copy(value: np.ndarray) -> np.ndarray:
        limit = float(np.finfo(np.float16).max)
        finite = np.nan_to_num(
            np.asarray(value, dtype=np.float32),
            copy=True,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        )
        np.clip(finite, -limit, limit, out=finite)
        return finite.astype(np.float16, copy=False)

    def _disk_cache(
        self,
        runner: "BasicTSRunner",
        split: str,
        inputs: torch.Tensor,
        inputs_mask: Optional[torch.Tensor],
    ) -> Optional[DiskFactorCache]:
        if self.cache_dir is None or inputs_mask is not None:
            return None
        loader_name = {
            "train": "train_data_loader",
            "val": "val_data_loader",
            "test": "test_data_loader",
        }[split]
        loader = getattr(runner, loader_name, None)
        dataset = getattr(loader, "dataset", None)
        data = getattr(dataset, "data", None)
        if dataset is None or data is None:
            return None
        cached = self.disk.get(split)
        if cached is not None:
            return cached
        values = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        if values.ndim != 2 or values.shape[0] == 0:
            return None
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(memoryview(values).cast("B"))
        source_hash = digest.hexdigest()
        input_len = int(inputs.shape[1])
        num_features = int(inputs.shape[2])
        if values.shape[1] != num_features:
            raise ValueError(
                f"factor cache feature mismatch: dataset has {values.shape[1]}, "
                f"batch has {num_features}"
            )
        dataset_name = runner.cfg.dataset_name
        if not dataset_name:
            raise ValueError("factor cache requires cfg.dataset_name")
        metadata = {
            "version": 1,
            "dataset": str(dataset_name),
            "split": split,
            "source_sha256": source_hash,
            "window_count": int(len(dataset)),
            "input_len": input_len,
            "num_features": num_features,
            "imf_count": self.cache_imf_count,
            "feature_mode": "qaio",
            "local_window_periods": 1.5,
            "padding_mode": "reflect",
            "padding_ratio": 0.25,
            "spline_kind": "cubic",
        }
        namespace = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        name = f"{metadata['dataset']}_{split}_{namespace}"
        cached = DiskFactorCache(
            Path(self.cache_dir) / name,
            metadata,
            length=int(len(dataset)),
            num_features=num_features,
            imf_count=self.cache_imf_count,
            input_len=input_len,
            feature_dim=4,
            read_only=(self.read_only or torch.distributed.is_initialized()),
        )
        if os.environ.get("DR_FACTOR_CACHE_RAM", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            cached.materialize()
        self.disk[split] = cached
        return cached

    def factor_batch(
        self,
        runner: "BasicTSRunner",
        split: str,
        indices: np.ndarray,
        inputs: torch.Tensor,
        inputs_mask: Optional[torch.Tensor],
        extractor: HilbertFactorExtractor,
    ) -> FactorTuple:
        inputs_array = self._to_numpy(inputs).astype(np.float32, copy=False)
        mask_array = (
            self._to_numpy(inputs_mask).astype(bool, copy=False)
            if inputs_mask is not None
            else None
        )
        batch_size, input_len, num_features = inputs_array.shape
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size != batch_size:
            raise ValueError(
                f"factor indices/batch mismatch: {indices.size} != {batch_size}"
            )
        disk_cache = self._disk_cache(runner, split, inputs, inputs_mask)
        if disk_cache is not None and mask_array is None:
            cached_batch = disk_cache.get_batch(indices, imf_count=self.imf_count)
            if cached_batch is not None:
                return cached_batch
        candidates = np.zeros(
            (batch_size, num_features, self.imf_count, input_len),
            dtype=np.float32,
        )
        features = np.zeros(
            (batch_size, num_features, self.imf_count, input_len, 4),
            dtype=np.float32,
        )
        active = np.zeros(
            (batch_size, num_features, self.imf_count),
            dtype=np.float32,
        )
        records = []
        uncached_signals = []
        uncached_record_indices = []
        for batch_index, sample_index in enumerate(indices):
            for feature_index in range(num_features):
                valid_length = input_len
                if mask_array is not None:
                    valid_positions = np.flatnonzero(
                        mask_array[batch_index, :, feature_index]
                    )
                    valid_length = (
                        int(valid_positions[-1]) + 1
                        if valid_positions.size
                        else 0
                    )
                key = (
                    split,
                    int(sample_index),
                    feature_index,
                    valid_length,
                    (
                        hashlib.blake2b(
                            np.packbits(
                                mask_array[
                                    batch_index, :valid_length, feature_index
                                ]
                            ).tobytes(),
                            digest_size=8,
                        ).hexdigest()
                        if mask_array is not None
                        else None
                    ),
                    self.imf_count,
                    "qaio",
                    1.5,
                )
                factors = (
                    disk_cache.get(
                        int(sample_index),
                        feature_index,
                        imf_count=self.imf_count,
                    )
                    if disk_cache
                    else None
                )
                if factors is None and disk_cache is not None and disk_cache.read_only:
                    raise RuntimeError(
                        "distributed refinement requires a complete precomputed "
                        f"factor cache; missing split={split}, sample={sample_index}, "
                        f"feature={feature_index} in {disk_cache.root}"
                    )
                if factors is None:
                    factors = self.memory.get(key)
                if factors is None:
                    factor_signal = inputs_array[
                        batch_index, :valid_length, feature_index
                    ]
                record = {
                    "batch_index": batch_index,
                    "sample_index": int(sample_index),
                    "feature_index": feature_index,
                    "valid_length": valid_length,
                    "key": key,
                    "factors": factors,
                }
                if factors is None:
                    uncached_record_indices.append(len(records))
                    uncached_signals.append(factor_signal)
                records.append(record)
        extracted = extractor.extract_batch(uncached_signals)
        for record_index, factors_float32 in zip(
            uncached_record_indices,
            extracted,
        ):
            factors = tuple(
                self._finite_float16_copy(value) for value in factors_float32
            )
            records[record_index]["factors"] = factors
            self.memory.put(records[record_index]["key"], factors)
        for record in records:
            sample_index = record["sample_index"]
            feature_index = record["feature_index"]
            valid_length = record["valid_length"]
            factors = record["factors"]
            if (
                disk_cache is not None
                and disk_cache.ready[sample_index, feature_index] == 0
                and disk_cache.candidates.shape[2] == self.imf_count
            ):
                disk_cache.put(sample_index, feature_index, factors)
            candidate, feature, activity = factors
            candidates[
                record["batch_index"], feature_index, :, :valid_length
            ] = candidate
            features[
                record["batch_index"], feature_index, :, :valid_length
            ] = feature
            active[record["batch_index"], feature_index] = activity
        return candidates, features, active
