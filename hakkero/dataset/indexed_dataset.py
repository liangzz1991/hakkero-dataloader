#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
import logging
import os.path
import time

import h5py
import torch.utils.data

from hakkero.dataset.utils import format_size

logger = logging.getLogger(__name__)

try:
    import msgspec

    json_decode = msgspec.json.decode
    json_encode = msgspec.json.encode
except ModuleNotFoundError:
    import json

    json_decode = json.loads
    json_encode = json.dumps


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_retry=1, retry_sleep=5):
        super().__init__()

        self.path = path
        self.data_path = os.path.join(path, "data.jsonl")
        self.index_path = os.path.join(path, "index.h5")
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep

        self._size = None
        self._length = None

        self.init()

    def init(self):
        with h5py.File(self.index_path, "r") as f:
            self._size = f["index"][-1]
            self._length = len(f["index"]) - 1

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self._length

    def safe_key(self, key):
        if isinstance(key, int):
            if not (-len(self) <= key < len(self)):
                raise IndexError(f"index {key} out of range")
            return key
        elif isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("strided indexing not implemented")

            # bound range
            start = min(max(-len(self), 0 if key.start is None else key.start), len(self) - 1)
            stop = min(max(-len(self), len(self) if key.stop is None else key.stop), len(self))

            # remap to positive index
            start = start if start >= 0 else start + len(self)
            stop = stop if stop >= 0 else stop + len(self)
            stop = max(start, stop)

            return slice(start, stop, None)
        else:
            raise TypeError(f"indices must be int or slice, not {type(key)}")

    def __getitem__(self, key):
        key = self.safe_key(key)

        if isinstance(key, slice):
            bytes = self.get_bytes(key, safe=False)
            return self.decode_bytes(key, bytes, safe=False)

        if isinstance(key, int):
            byte = self.get_byte(key, safe=False)
            return self.decode_byte(key, byte, safe=False)

    def read(self, i_or_s, offset, size):
        for retry in itertools.count():
            try:
                # destroy the file identifier to avoid pressure on store
                # buffering=0 to avoid overhead during file.seek() and open()
                with open(self.data_path, "rb", buffering=0) as fin:
                    fin.seek(offset)
                    byte = fin.read(size)

                return byte
            except OSError as e:
                if retry >= self.max_retry:
                    raise OSError(f"reach maximum retry: {retry}, the file system is broken.")

                logger.warning(f"retry loading {self.path}:{i_or_s} in {self.retry_sleep} sec due to error: {e}")
                time.sleep(self.retry_sleep)
            except ValueError as e:
                logger.warning(f"fail to read {self.path}:{i_or_s} due to error: {e}")
                return None

    def get_bounds(self, key):
        for retry in itertools.count():
            try:
                with h5py.File(self.index_path, "r") as fin:
                    return fin["index"][key]
            except OSError as e:
                if retry >= self.max_retry:
                    raise OSError(f"reach maximum retry: {retry}, the file system is broken.")

                logger.warning(f"retry loading {self.index_path} in {self.retry_sleep} sec due to error: {e}")
                time.sleep(self.retry_sleep)
            except ValueError as e:
                logger.warning(f"fail to read {self.index_path} due to error: {e}")
                return None

    def get_byte(self, key, safe=True):
        if safe:
            assert isinstance(key, int)
            key = self.safe_key(key)

        bounds = self.get_bounds(slice(key, key + 2))
        if bounds is None:
            return None

        start, stop = bounds[0], bounds[1]
        return self.read(key, start, stop - start)

    def decode_byte(self, key, byte, safe=True):
        if safe:
            assert isinstance(key, int)
            key = self.safe_key(key)

        if byte is None:
            logger.warning(f"Fail to decode {self.path}:{key}: byte is None")
            return None

        try:
            return json_decode(byte)
        except Exception as e:
            logger.warning(f"Fail to decode {self.path}:{key} due to error {e}, byte: {byte}")
            return None

    def get_bytes(self, key, safe=True):
        if safe:
            assert isinstance(key, slice)
            key = self.safe_key(key)
        if key.start == key.stop:
            # early return empty slice
            return []
        bounds = self.get_bounds(slice(key.start, key.stop + 1))
        if bounds is None:
            return None
        start, stop = bounds[0], bounds[-1]
        byte = self.read(key, start, stop - start)

        if byte is None:
            return [None for _ in bounds[:-1]]
        else:
            shifted_bounds = [b - start for b in bounds]
            return [byte[s:e] for s, e in zip(shifted_bounds[:-1], shifted_bounds[1:])]

    def decode_bytes(self, key, bytes, safe=True):
        if safe:
            assert isinstance(key, slice)
            key = self.safe_key(key)
        if bytes is None:
            logger.warning(f"fail to decode {self.path}:{key}: byte string is None")
            return None
        if not bytes:
            return []

        return [self.decode_byte(i, byte, safe=False) for i, byte in zip(range(key.start, key.stop), bytes)]

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path}): {len(self)} entries, {self.pretty_size}"

    @property
    def pretty_size(self):
        return format_size(self.size)
