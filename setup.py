#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from os import path

import setuptools
from setuptools import find_packages
from setuptools import setup

version = int(setuptools.__version__.split(".")[0])
assert version > 30, "requires setuptools > 30"

this_directory = path.abspath(path.dirname(__file__))


__version__ = "1.1.3"


setup(
    name="hakkero-dataloader",
    url="",
    keywords="llm dataloader",
    version=__version__,
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples", "tests"]),
    zip_safe=False,
    install_requires=["numpy", "torch", "msgpack>=0.5.2", "h5py", "bitarray>=2.9.2", "tabulate", "scipy"],
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#universal-wheels
    options={"bdist_wheel": {"universal": "1"}},
    entry_points={
        "console_scripts": ["hakkero=hakkero.dataset.indexify:main"],
    },
)
