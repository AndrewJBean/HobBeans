import signal
import sys
from typing import Tuple, Dict
import datetime

import torch
from HobBeans.transformer.dataset import ExcerptDataset
from HobBeans.transformer.config import ModelConfig
from HobBeans.transformer.trainer import TrainerConfig, Trainer, input_label_split
from HobBeans.transformer.metrics import TopKAccuracy, Loss


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


class ModelAndLoss(torch.nn.Module):
  def __init__(self, *, model: torch.nn.Module):
    super().__init__()
    self.model = model
    self.loss = torch.nn.CrossEntropyLoss()

  def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    x: (batch_size, num_chars)
    """
    output = self.model(x)
    # logits: (batch_size, num_chars-1, char_vocab_size)
    logits = output["logits"]
    # squash first 2 dimensions
    y = logits.reshape(-1, logits.size(-1))
    labels = x["labels"]
    loss = self.loss(y, labels.flatten())
    return {"loss": loss, "labels": labels, **output}


def main():
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  num_chars = 100
  batch_size = 10
  dataset = ExcerptDataset(
    num_chars=num_chars,
    batch_size=batch_size,
    char_map_file="character_map.txt",
  )

  num_layers = 3
  config = ModelConfig(
    num_chars=dataset._num_chars,
    char_embedding_dim=128,
    char_vocab_size=dataset.vocab_size,
    kernel_sizes=[5] * num_layers,
    out_sizes=[256] * (num_layers - 1),
  )
  model = Model(config=config)
  model_and_loss = ModelAndLoss(model=model)
  optimizer = torch.optim.Adam(model_and_loss.parameters(), lr=0.001)
  metrics = {"loss": Loss()}
  metrics.update({f"accuracy(k={k})": TopKAccuracy(k=k) for k in [1, 2, 3]})

  trainer_config = TrainerConfig(
    num_steps=10000,
    log_interval=100,
    eval_interval=500,
    eval_steps=100,
  )
  trainer = Trainer(
    config=trainer_config,
    model_and_loss=model_and_loss,
    optimizer=optimizer,
    metrics=metrics,
    device=device,
  )
  trainer.train(dataset)


# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)
  main()
