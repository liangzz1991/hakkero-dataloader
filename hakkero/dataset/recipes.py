#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from hakkero.dataset.segmentation import concat
from hakkero.dataset.segmentation import integrous
from hakkero.dataset.segmentation import naive
from hakkero.dataset.segmentation import unbiased
from hakkero.dataset.tokenization import chatml_message
from hakkero.dataset.tokenization import chatml_preference
from hakkero.dataset.tokenization import huggingface_message
from hakkero.dataset.tokenization import huggingface_preference
from hakkero.dataset.tokenization import legacy

segment = {
    "integrous": integrous,
    "concat": concat,
    "naive": naive,
    "unbiased": unbiased,
}

tokenize = {
    "legacy": legacy,
    "hg": huggingface_message,
    "hg_preference": huggingface_preference,
    "chatml": chatml_message,
    "chatml_preference": chatml_preference,
}


default_recipe = {"segment": "naive", "tokenize": "legacy"}
