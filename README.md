hakkero-dataloader
------------------

A general dataloader build on top of Pytorch Dataloader.


## 1. How to use

### 1.1 Build Index

Install `pip install hakkero-dataloader` and run the following command to build index.

```shell
hakkero -h
usage: hakkero [-h] --filename FILENAME [--output OUTPUT]

build index for dataset

options:
  -h, --help           show this help message and exit
  --filename FILENAME  full filename of jsonl file
  --output OUTPUT      output path for saving data.jsonl and index.h5
```

### 1.2 Use In Training

```python
from hakkero.dataset import get_dataset

# pretrain or sft
from hakkero.dataset import PadLoader
from hakkero.dataset import UnpadLoader

# preference
from hakkero.dataset import PreferencePadLoader
from hakkero.dataset import PreferenceUnpadLoader

dp_world_size, dp_rank = 1, 0
tokenizer = ...
batch_size = 4
max_length = 4096
n_workers = 2

dataset = get_dataset(
    config="/path/to/dataset",
    tokenizer=tokenizer,
    num_epochs=-1,
    max_length=max_length,
    homogeneous=True,
    seed=9527,
    rank=dp_rank,
    world_size=dp_world_size,
    n_workers=n_workers,
    # segment and tokenize strategy or set them in `config` and let strategy_segment=None and strategy_tokenize=None: 
    st_segment="naive",
    st_tokenize="legacy",
    # add bos/eos token for legacy tokenize strategy
    add_bos_token=True,
    add_eos_token=True,
    # norm dataset weight with tokens of target
    norm_weight_with_n_targets=False,
)

dataloader = UnpadLoader(dataset, max_total_length=batch_size * max_length)
prefetcher = dataloader.prefetch(n_workers)

for step, batch in enumerate(prefetcher, start=0):
    print(batch)
```

example of `config`: 
```json
{
    "hermes25_1":
    {
        "group": "en",
        "name": "hermes25_1",
        "epoch": 1,
        "path": "hermes25",
        "strategy":
        {
            "st_segment": "integrous",
            "st_tokenize": "hg"
        },
        "weight": 0.5
    },
    "hermes25_2":
    {
        "group": "en",
        "name": "hermes25_1",
        "epoch": 1,
        "path": "hermes25",
        "strategy":
        {
            "st_segment": "integrous",
            "st_tokenize": "hg"
        },
        "weight": 0.5
    }
}
```

## 2. Supported Strategies

See [segmentation.py](./hakkero/dataset/segmentation.py) and [tokenization.py](./hakkero/dataset/tokenization.py) for more details.

### 2.1 Segmentation Strategies

- `integrous`: discard sample that is too long, exceed `max_length`
- `concat`: split long sample, concat it with previous segment, shuffle all segments
  - not support preference data.
- `naive`: split long sample with random length, shuffle all segments
  - not support preference data.
- `unbiased`: split long sample exceed `max_length` with random length, shuffle all segments.
  - not support preference data.

### 2.2 Tokenization Strategies

- `legacy`: `\n\n` as delimiter to join text and use `tokenizer.encode` to encode the input.
  - format of input data
    ```json
    {
      "uid": "xxx",
      "data":
      {
          "title": "xxx",
          "summary": "xxx",
          "abstract": "xxx",
          "text": "xxx",
          "question": "xxx",
          "answer": "xxx",
          "code": "xxx",
          "label": "xxx"
      }
    }
    ```

    - All fields except `label` are stripped and joined with "\n\n" as the context.
    - `label` is the target to learn for finetuning (pretrain data should not have the `label` field).
    - See func `legacy` in [tokenization.py](./hakkero/dataset/tokenization.py) for more details.
  - extra parameters: `add_bos_token`, `add_eos_token`

- `hg`: huggingface message data, use `tokenizer.apply_chat_template` to encode the input.
  - format of input data
    ```json
    {
      "uid": "xx",
      "data": [
        {"role": "user", "content": "xxx"},
        {"role": "assistant", "content": "xxx"},
         ...
      ]
    }
    ```

    See func `huggingface_message` in [tokenization.py](./hakkero/dataset/tokenization.py) for more details.

- `chatml`: chat message data, use chatml to encode the input.
  - format of input data
    ```json
    {
      "uid": "xx",
      "data": [
        {"role": "user", "content": "xxx"},
        {"role": "assistant", "content": "xxx"},
         ...
      ]
    }
    ```

    See func `chatml_message` in [tokenization.py](./hakkero/dataset/tokenization.py) for more details.

- `hg_preference`: preference data, use `tokenizer.apply_chat_template` to encode the input.
  - format of input data
    ```json
    {
      "uid": "xx",
      "data": {
        "context": [
          {"role": "user", "content": "xxx"},
          {"role": "assistant", "content": "xxx"},
          ...
          {"role": "user", "content": "xxx"}
        ],
        "chosen": "chosen response",
        "rejected": "rejected response"
      }
    }
    ```
    
    See func `huggingface_preference` in [tokenization.py](./hakkero/dataset/tokenization.py) for more details.

- `chatml_preference`: preference data, use chatml to encode the input.
  - format of input data
    ```json
    {
      "uid": "xx",
      "data": {
        "context": [
          {"role": "user", "content": "xxx"},
          {"role": "assistant", "content": "xxx"},
          ...
          {"role": "user", "content": "xxx"}
        ],
        "chosen": "chosen response",
        "rejected": "rejected response"
      }
    }
    ```
    
    See func `chatml_preference` in [tokenization.py](./hakkero/dataset/tokenization.py) for more details.