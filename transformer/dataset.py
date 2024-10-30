from typing import List, Dict, Optional, Tuple
import os
import platform
from pprint import pprint

import torch
from datasets import load_dataset


# use ~/huggingface_cache for caching
CACHE_DIR = os.path.expanduser("~/huggingface_cache")


def input_label_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  x: (batch_size, num_chars)
  """
  return x[:, :-1], x[:, 1:]


class FullExcerptDataset(torch.utils.data.IterableDataset):
  OOV = "<OOV>"
  START = "<START>"
  END = "<END>"
  PAD = "<PAD>"

  def __init__(self, *, batch_size: int, char_map_file: str, trunc_len: int = 500):
    self.batch_size = batch_size
    self.ds = None
    self.trunc_len = trunc_len

    # Load the character mapping
    with open(char_map_file, "r") as f:
      self.tokens = list(f.read())
    self.VOCAB_SIZE = len(self.tokens)
    print("Vocab size:", self.VOCAB_SIZE)
    self.tokens.extend([self.OOV, self.START, self.END, self.PAD])
    self.token_to_idx = {c: i for i, c in enumerate(self.tokens)}
    print("Character mapping loaded.")

  @property
  def vocab_size(self):
    return len(self.tokens)

  @property
  def oov_token(self):
    return self.token_to_idx[self.OOV]

  @property
  def start_token(self):
    return self.token_to_idx[self.START]

  @property
  def end_token(self):
    return self.token_to_idx[self.END]

  @property
  def pad_token(self):
    return self.token_to_idx[self.PAD]

  @property
  def non_text_tokens(self):
    return {self.start_token, self.end_token, self.oov_token, self.pad_token}

  def __iter__(self):
    return self

  def chars_to_tokens(self, chars: List[str]) -> List[int]:
    return (
      [self.start_token]
      + [self.token_to_idx.get(c, self.oov_token) for c in chars]
      + [self.end_token]
    )

  def strings_to_batch(self, strings: List[str], trunc_len: int = 500) -> torch.Tensor:
    batch = []
    for s in strings:
      batch.append(self.chars_to_tokens(list(s)))
    # pad to max_len
    max_len = max(len(row) for row in batch)
    for i in range(len(batch)):
      batch[i] = batch[i] + [self.pad_token] * (max_len - len(batch[i]))

    if max_len > trunc_len:
      for i in range(len(batch)):
        batch[i] = batch[i][:trunc_len]

    return torch.tensor(batch, dtype=torch.long)

  def __next__(self):
    if self.ds is None:
      self.ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "CC-MAIN-2013-20",
        streaming=platform.system() != "Darwin",
        cache_dir=CACHE_DIR,
        download_mode="reuse_dataset_if_exists",
      )
      self.ds = iter(self.ds["train"])

    records = [list(next(self.ds)["text"]) for _ in range(self.batch_size)]
    batch = self.strings_to_batch(records, trunc_len=self.trunc_len)

    inputs, labels = input_label_split(batch)
    return {"inputs": inputs, "labels": labels}

  def tokens_to_string(self, tokens: torch.Tensor) -> str:
    # assuming 2D batch tensor
    return [
      "".join([self.tokens[tok] for tok in row if tok not in self.non_text_tokens])
      for row in tokens.tolist()
    ]

  def to_dataloader(self) -> Dict[str, torch.Tensor]:
    return torch.utils.data.DataLoader(self, batch_size=None)


if __name__ == "__main__":
  this_file_directory = os.path.dirname(os.path.realpath(__file__))
  char_map_file = os.path.join(this_file_directory, "character_map.txt")
  ds = FullExcerptDataset(batch_size=5, char_map_file=char_map_file)
  for i, batch in enumerate(ds):
    pprint(ds.tokens_to_string(batch))
    if i > 3:
      break
