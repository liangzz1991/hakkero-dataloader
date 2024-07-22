#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import argparse
import os
import shutil

import h5py

from hakkero.dataset.logger import logger


def build_index(filename, output=None):
    logger.info(f"build indexed dataset from {filename}")

    offset = 0
    bounds = [offset]
    with open(filename, "rb") as fin:
        for idx, line in enumerate(fin):
            logger.info(f"cal offset: {idx}")
            offset += len(line)
            bounds.append(offset)

    if output is not None:
        os.makedirs(output, exist_ok=True)
    else:
        output = os.path.dirname(filename)

    logger.info(f"build index.h5 into {output}")
    with h5py.File(os.path.join(output, "index.h5"), "w") as hf:
        hf.create_dataset("index", data=bounds)

    logger.info(f"build data.jsonl into {output}")
    if not os.path.exists(os.path.join(output, "data.jsonl")):
        shutil.copyfile(filename, os.path.join(output, "data.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="build index for dataset")
    parser.add_argument("--filename", type=str, help="full filename of jsonl file", required=True)
    parser.add_argument("--output", type=str, help="output path for saving data.jsonl and index.h5")

    args = parser.parse_args()

    build_index(args.filename, args.output)


if __name__ == "__main__":
    main()
