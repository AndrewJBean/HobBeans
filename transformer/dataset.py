from typing import List, Dict, Optional, Tuple
import os
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


class ExcerptDataset(torch.utils.data.IterableDataset):
  def __init__(
    self,
    *,
    num_chars: int,
    batch_size: int,
    char_map_file: str,
  ):
    self._num_chars = num_chars
    self.batch_size = batch_size
    self.ds = load_dataset(
      "HuggingFaceFW/fineweb-edu",
      "CC-MAIN-2013-20",
      streaming=False,
      cache_dir=CACHE_DIR,
      download_mode="reuse_dataset_if_exists",
    )
    self.ds = iter(self.ds["train"])

    # Load the character mapping
    with open(char_map_file, "r") as f:
      self.chars = list(f.read())
    self.VOCAB_SIZE = len(self.chars)
    print("Vocab size:", self.VOCAB_SIZE)
    self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
    self.chars.append("<OOV>")
    print("Character mapping loaded.")

    # initialize buffer. We will read the next text from the dataset
    # and then yield characters from this buffer
    self.remaining_chars = []

  @property
  def vocab_size(self):
    # including the OOV token
    return self.VOCAB_SIZE + 1

  @property
  def num_chars(self):
    return self._num_chars

  def __iter__(self):
    return self

  def chars_to_tokens(self, chars: List[str]) -> List[int]:
    return [self.char_to_idx.get(c, self.VOCAB_SIZE) for c in chars]

  def __next__(self):
    batch = []
    while len(batch) < self.batch_size:
      while len(self.remaining_chars) < self._num_chars:
        text = next(self.ds)["text"]
        self.remaining_chars = list(text)
      chars = self.remaining_chars[: self._num_chars]
      self.remaining_chars = self.remaining_chars[self._num_chars :]
      batch.append(self.chars_to_tokens(chars))
    batch = torch.tensor(batch, dtype=torch.long)
    inputs, labels = input_label_split(batch)
    return {"inputs": inputs, "labels": labels}

  def tokens_to_string(self, tokens: torch.Tensor) -> str:
    # assuming 2D batch tensor
    return ["".join([self.chars[i] for i in row]) for row in tokens.tolist()]

  def to_dataloader(self) -> Dict[str, torch.Tensor]:
    return torch.utils.data.DataLoader(self, batch_size=None)


if __name__ == "__main__":
  this_file_directory = os.path.dirname(os.path.realpath(__file__))
  char_map_file = os.path.join(this_file_directory, "character_map.txt")
  ds = ExcerptDataset(num_chars=100, batch_size=5, char_map_file=char_map_file)
  for i, batch in enumerate(ds):
    pprint(ds.tokens_to_string(batch))
    if i > 3:
      break
