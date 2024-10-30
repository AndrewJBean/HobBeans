from typing import Dict, List
import pydantic
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


class ModelConfig(pydantic.BaseModel):
  char_embedding_dim: int
  char_vocab_size: int
  kernel_sizes: List[int]
  out_sizes: List[int]


class Model(torch.nn.Module):
  def __init__(self, *, config: ModelConfig):
    super().__init__()
    self.config = config
    assert len(config.kernel_sizes) == len(config.out_sizes) + 1

    self.char_embedding = torch.nn.Embedding(
      num_embeddings=config.char_vocab_size,
      embedding_dim=config.char_embedding_dim,
    )

    # we need a causal model.
    # we will pad with kernel_size - 1 and truncate the last kernel_size - 1 characters
    # so that output[i] only depends on up to input[i]
    self.conv_layers = torch.nn.ModuleList()
    self.out_channels = self.config.out_sizes + [self.config.char_vocab_size]
    in_channels = config.char_embedding_dim
    for kernel_size, out_channels in zip(config.kernel_sizes, self.out_channels):
      conv_layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size - 1,
      )
      in_channels = out_channels
      self.conv_layers.append(conv_layer)

  def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    x: (batch_size, num_chars)
    """
    x = x["inputs"]
    # (batch_size, num_chars, char_embedding_dim)
    # we permute the tensor to (batch_size, char_embedding_dim, num_chars)
    # because conv1d expects the channel dimension to be the 2nd dimension
    x = self.char_embedding(x).permute(0, 2, 1)

    # all layers except the last one because of relu
    for kernel_size, conv_layer in zip(self.config.kernel_sizes[:-1], self.conv_layers[:-1]):
      # run the conv layer and apply relu
      x = torch.relu(conv_layer(x))
      # the truncate the last kernel_size - 1 characters
      x = x[:, :, : -kernel_size + 1]

    # run the last conv layer
    x = self.conv_layers[-1](x)
    # truncate the last kernel_size - 1 characters
    x = x[:, :, : -self.config.kernel_sizes[-1] + 1]
    # permute back to (batch_size, num_chars, char_vocab_size)
    x = x.permute(0, 2, 1)
    return {"logits": x}
