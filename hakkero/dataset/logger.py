#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


import logging
import os

logger = logging.getLogger(__name__)


FMT_LOG = "[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
FMT_DATE = "%Y-%m-%d %H:%M:%S"


def _configure_logger():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=FMT_LOG, datefmt=FMT_DATE)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def set_logging_level(level):
    logger.setLevel(level)


_configure_logger()
set_logging_level(os.environ.get("HAKKERO_LOG_LEVEL", "INFO"))
