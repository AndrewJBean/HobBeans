from typing import Dict, Iterable

try:
  from typing import Self
except ImportError:
  from typing_extensions import Self
import torch
from torcheval.metrics import Metric


class TopKAccuracy(Metric):
  def __init__(self, *, k: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.k = k
    self._add_state("num_correct", default=torch.tensor(0, dtype=torch.float, device=self.device))
    self._add_state("num_total", default=torch.tensor(0, dtype=torch.float, device=self.device))

  def update(self, preds: Dict[str, torch.Tensor]):
    y = preds["logits"]
    labels = preds["labels"]
    y = y.reshape(-1, y.size(-1))
    labels = labels.reshape([-1])
    _, top_k = torch.topk(y, k=self.k, dim=-1)
    correct = (top_k == labels.unsqueeze(-1)).any(-1).float().sum()
    self.num_correct += correct
    self.num_total += labels.size(0)
    return self

  def compute(self):
    return self.num_correct / self.num_total

  def reset(self):
    self.num_correct.zero_()
    self.num_total.zero_()

  def merge_state(self: Self, metrics: Iterable[Self]) -> Self:
    return super().merge_state(metrics)


class Loss(Metric):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._add_state("cumulative", default=torch.tensor(0, dtype=torch.float, device=self.device))
    self._add_state("count", default=torch.tensor(0, dtype=torch.float, device=self.device))

  def update(self, preds: Dict[str, torch.Tensor]):
    loss = preds["loss"]
    self.cumulative += loss
    self.count += 1
    return self

  def compute(self):
    return self.cumulative / self.count

  def reset(self):
    self.cumulative.zero_()
    self.count.zero_()

  def merge_state(self: Self, metrics: Iterable[Self]) -> Self:
    return super().merge_state(metrics)
