#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import torch

from hakkero.dataset.errors import TokenizationError
from hakkero.dataset.utils import IGNORE_INDEX


def legacy(data, tokenizer):
    assert isinstance(data, dict), "wrong data format, expect {key: value}, " + f"but got {data}"

    context = "\n\n".join(
        [
            data[s].strip()
            for s in ("text", "title", "question", "answer", "abstract", "code")
            if s in data and data[s].strip()
        ]
    )

    target = data.get("label", "").strip()

    input = []
    label = []

    ids = tokenizer.encode(context, max_length=int(1e12), truncation=True)
    input.extend(ids)

    if target:
        if ids[-1] == tokenizer.eos_token_id:
            ids.pop()
        label.extend([IGNORE_INDEX] * len(ids))

        ids = tokenizer.encode(target, max_length=int(1e12), truncation=True)
        if ids[0] == tokenizer.bos_token_id:
            ids.pop(0)
        input.extend(ids)
        label.extend(ids)
    else:
        label.extend(ids)

    if len(input) <= 1:
        raise TokenizationError(
            "No valid keys in input, expect of: ('text', 'title', 'question', 'answer', 'abstract', 'code')"
        )

    return dict(input=torch.tensor(input[:-1], dtype=torch.long), label=torch.tensor(label[1:], dtype=torch.long))


# messages = [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...]
def huggingface_message(messages, tokenizer):
    if not isinstance(messages, list) or not isinstance(messages[0], dict):
        raise ValueError("messages should be [{'role': 'xxx', 'content': 'xxx'}, ...]," + f" but got {messages}")
    assert hasattr(tokenizer, "apply_chat_template"), "tokenizer should have apply_chat_template"
    assert messages[-1]["role"] == "assistant" and messages[-2]["role"] == "user"

    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    ) != tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=False
    ), "add_generation_prompt does not take effect, please modify tokenizer.chat_template"

    context_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

    # hack: separate encoding of the context and response will always lead to prefix space in the response
    response_ids_with_prefix = tokenizer.apply_chat_template(messages[-2:], add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(messages[-2:-1], add_generation_prompt=True)
    response_ids = response_ids_with_prefix[len(prefix_ids) :]

    input = context_ids + response_ids
    label = [IGNORE_INDEX for _ in context_ids] + response_ids

    return dict(input=torch.tensor(input[:-1], dtype=torch.long), label=torch.tensor(label[1:], dtype=torch.long))


# data = {
#   "context": [{"role": "user", "content": xxx}, {"role": "assistant", "content": xxx}, ...],
#   "chosen": "xx",
#   "rejected": "xx"
# }
def huggingface_preference(data, tokenizer):
    assert hasattr(tokenizer, "apply_chat_template")
    assert data["context"][-1]["role"] == "user"

    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    ) != tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=False
    ), "add_generation_prompt does not take effect, please modify tokenizer.chat_template"

    context_ids = tokenizer.apply_chat_template(data["context"], add_generation_prompt=True)

    # hack: separate encoding of the context and response will always lead to prefix space in the response
    prefix_ids = tokenizer.apply_chat_template(data["context"][-1:], add_generation_prompt=True)

    inputs = dict(chosen=[], rejected=[])
    labels = dict(chosen=[], rejected=[])

    for key in ("chosen", "rejected"):
        inputs[key].extend(context_ids)
        labels[key].extend(IGNORE_INDEX for _ in context_ids)
        response_ids_with_prefix = tokenizer.apply_chat_template(
            data["context"][-1:] + [{"role": "assistant", "content": data[key]}], add_generation_prompt=False
        )

        assert response_ids_with_prefix[: len(prefix_ids)] == prefix_ids

        response_ids = response_ids_with_prefix[len(prefix_ids) :]

        inputs[key].extend(response_ids)
        labels[key].extend(response_ids)

    return {
        "inputs": {key: torch.tensor(value[:-1]) for key, value in inputs.items()},
        "labels": {key: torch.tensor(value[1:]) for key, value in labels.items()},
    }
