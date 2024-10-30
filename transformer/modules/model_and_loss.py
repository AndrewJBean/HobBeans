from typing import Dict
import torch


class ModelAndLoss(torch.nn.Module):
  def __init__(self, *, model: torch.nn.Module):
    super().__init__()
    self.model = model
    self.loss = torch.nn.CrossEntropyLoss()

  def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    x:
      {
        "inputs": (batch_size, sequence_length),
        "labels": (batch_size, sequence_length),
      }
    Returns:
      {
        "logits": (batch_size, sequence_length, char_vocab_size),
        "loss": torch.Tensor,
        "labels": torch.Tensor,
      }
    """
    output = self.model(x)
    # logits: (batch_size, sequence_length, char_vocab_size)
    logits = output["logits"]
    # squash first 2 dimensions
    y = logits.reshape(-1, logits.size(-1))
    labels = x["labels"]
    loss = self.loss(y, labels.flatten())
    return {"loss": loss, "labels": labels, **output}
