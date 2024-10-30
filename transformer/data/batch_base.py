from typing import Self
import torch


class DictionaryBatch(dict):
  """
  consider implementing record_stream, pin_memory, and __repr__
  """

  def to(self, device: torch.device, non_blocking: bool = False) -> Self:
    return DictionaryBatch(
      {k: v.to(device=device, non_blocking=non_blocking) for k, v in self.items()}
    )
