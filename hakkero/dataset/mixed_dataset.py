#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
import torch.utils.data
from tabulate import tabulate

from hakkero.dataset.iterable_dataset import Prefetcher
from hakkero.dataset.recipes import default_recipe
from hakkero.dataset.segment_dataset import MixedSegmentDataset
from hakkero.dataset.utils import MultinomialSampler


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return 0

    def __len__(self):
        return 10000


class MixedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        config,
        tokenizer,
        num_epochs=1,
        max_length=1024,
        homogeneous=False,
        seed=-1,
        rank=0,
        world_size=1,
        n_shards=2,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.homogeneous = homogeneous

        self.seed = seed

        self.rank = rank
        self.world_size = world_size
        self.n_shards = n_shards

        self.datasets = []
        self.weights = []
        self.prefetcher = Prefetcher()

        for i, (name, sub_config) in enumerate(config.items()):
            if isinstance(sub_config, str):
                path = sub_config
                n_epoch = 3 * self.num_epochs
                recipe = default_recipe
                weight = None
            else:
                path = sub_config["path"]
                n_epoch = sub_config["epoch"] * self.num_epochs
                recipe = sub_config.get("recipe", default_recipe)

                if "segment" not in recipe:
                    recipe["segment"] = "naive"

                if "tokenize" not in recipe:
                    recipe["tokenize"] = "legacy"

                weight = sub_config.get("weight", None)

            dataset = MixedSegmentDataset(
                path,
                name=name,
                tokenizer=tokenizer,
                prefetcher=self.prefetcher,
                seed=self.seed,
                infinite=True if homogeneous else False,
                max_epoch=n_epoch,
                recipe=recipe,
                max_length=max_length,
                n_shards=n_shards,
                rank=rank,
                world_size=world_size,
            )

            self.datasets.append(dataset)
            # weight by size by default, can be overwritten with load_weights
            self.weights.append(n_epoch * dataset.size if weight is None else weight)

        self.weights = np.array(self.weights, dtype=np.float64)
        self.weights /= self.weights.sum()

        self.sampler = MultinomialSampler(seed)
        self.active = np.ones_like(self.weights)

    def __iter__(self):
        self.sampler = MultinomialSampler(self.seed)
        self.active = np.ones_like(self.weights)
        for d in self.datasets:
            iter(d)

        while True:
            if all(d.exhausted for d in self.datasets):
                return

            # remap the desired token distribution to sample distribution
            # as different dataset has different average #tokens
            # mask exhausted dataset
            n_targets = np.array([d.avg_n_target for d in self.datasets], dtype=np.float64)
            weights = self.active * self.weights / n_targets
            try:
                i = self.sampler.next(weights)
            except StopIteration:
                return

            try:
                segment = next(self.datasets[i])
                segment["task"] = i
                yield segment
            except StopIteration:
                self.active[i] = 0
                continue

    def state_dict(self):
        return {
            "datasets": {d.name: d.state_dict() for d in self.datasets},
            "weights": {d.name: w for d, w in zip(self.datasets, self.weights)},
        }

    def load_state_dict(self, state_dict):
        names = [d.name for d in self.datasets]
        self.weights = [state_dict["weights"].get(name, 1.0) for name in names]
        assert all(w >= 0.0 for w in self.weights), "loaded weights should be greater than 0"
        for dataset in self.datasets:
            if dataset.name not in state_dict["datasets"]:
                continue
            dataset.load_state_dict(state_dict["datasets"][dataset.name])
        missing_keys = list(set(names) - set(state_dict["datasets"].keys()))

        return missing_keys

    def load_weights(self, weight_dict):
        names = [d.name for d in self.datasets]
        missing_keys = list(set(names) - set(weight_dict.keys()))
        if missing_keys:
            raise ValueError(f"missing keys of weight_dict: {missing_keys}")
        self.weights = [weight_dict.get(name, 1.0) for name in names]
        assert all(w >= 0.0 for w in self.weights), "loaded weights should be greater than 0"

    def track(self, task_stats):
        # task_stats = {task: {epoch: [indices]}}
        for task, stats in task_stats.items():
            self.datasets[task].track(stats)

    def __repr__(self):
        entries = (
            (d.name, w * 100, f"{len(d):,}", d.size / 1024**3, d.size / len(d) / 1024)
            for w, d in zip(self.weights, self.datasets)
        )
        header = ["name", "weight(%)", "#entries", "size(GB)", "average len(KB)"]
        table = tabulate(
            sorted(entries, key=lambda x: x[1], reverse=True),
            header,
            "pipe",
            floatfmt=".3f",
            colalign=("left", "right", "right", "right"),
        )
        return f"{self.__class__.__name__} with \n{table}"


def get_dataset(
    config, tokenizer, num_epochs, max_length=4096, homogeneous=False, seed=0, rank=0, world_size=1, n_workers=2
):
    return MixedDataset(
        config,
        tokenizer,
        num_epochs=num_epochs,
        max_length=max_length,
        homogeneous=homogeneous,
        seed=seed,
        rank=rank,
        world_size=world_size,
        n_shards=n_workers * world_size,
    )