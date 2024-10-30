import signal
import sys
from typing import Tuple
import datetime

import torch
from HobBeans.transformer.dataset import ExcerptDataset, FullExcerptDataset
from HobBeans.transformer.config import ModelConfig

# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


def input_label_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  x: (batch_size, num_chars)
  """
  return x[:, :-1], x[:, 1:]


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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch_size, num_chars)
    """
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
    return x


class ModelAndLoss(torch.nn.Module):
  def __init__(self, *, config: ModelConfig):
    super().__init__()
    self.model = Model(config=config)
    self.loss = torch.nn.CrossEntropyLoss()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch_size, num_chars)
    """
    model_input, labels = input_label_split(x)
    # y: (batch_size, num_chars-1, char_vocab_size)
    y = self.model(model_input)
    # squash first 2 dimensions
    y = y.reshape(-1, y.size(-1))
    labels = labels.reshape([-1])
    return self.loss(y, labels)


def train_step(
  model_and_loss: ModelAndLoss, x: torch.Tensor, optimizer: torch.optim.Optimizer
) -> torch.Tensor:
  optimizer.zero_grad()
  loss = model_and_loss(x)
  loss.backward()
  optimizer.step()
  return loss


def eval_step(model_and_loss: ModelAndLoss, x: torch.Tensor) -> torch.Tensor:
  # run model_and_loss.model on x[:, :-1] and compare with x[:, 1:]
  # see how many label characters are in the top K predictions
  k = 1
  with torch.no_grad():
    model_input, labels = input_label_split(x)
    y = model_and_loss.model(model_input)
    # y: (batch_size, num_chars-1, char_vocab_size)
    # squash first 2 dimensions
    y = y.reshape(-1, y.size(-1))
    labels = labels.reshape([-1])
    _, top_k = torch.topk(y, k=k, dim=-1)
    return (top_k == labels.unsqueeze(-1)).any(-1).float().mean()


def run_eval(model_and_loss: ModelAndLoss, ds_iter, num_steps):
  acc_sum = 0
  for step in range(num_steps):
    x = next(ds_iter)
    acc = eval_step(model_and_loss, x)
    acc_sum += acc.item()
  return acc_sum / num_steps


def train_model(
  model_and_loss: ModelAndLoss, dataset: ExcerptDataset, num_steps: int, lr: float
) -> ModelAndLoss:
  optimizer = torch.optim.Adam(model_and_loss.parameters(), lr=lr)
  ds_iter = iter(dataset)
  start_time = datetime.datetime.now()
  log_interval = 100
  for step in range(num_steps):
    x = next(ds_iter)
    loss = train_step(model_and_loss, x, optimizer)
    if step % log_interval == 0:
      end_time = datetime.datetime.now()
      steps_per_sec = log_interval / (end_time - start_time).total_seconds()
      start_time = end_time
      print(f"Step={step}, Steps/s={steps_per_sec:.2f}, Loss={loss.item()}")
    if step % 100 == 0:
      acc = run_eval(model_and_loss, ds_iter, num_steps=100)
      print(f"Step {step}, Accuracy {acc}")
  return model_and_loss


def main1():
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  num_chars = 100
  batch_size = 20
  dataset = ExcerptDataset(
    num_chars=num_chars,
    batch_size=batch_size,
    char_map_file="character_map.txt",
    device=device,
  )

  num_layers = 3
  config = ModelConfig(
    num_chars=dataset._num_chars,
    char_embedding_dim=128,
    char_vocab_size=dataset.vocab_size,
    kernel_sizes=[5] * num_layers,
    out_sizes=[256] * (num_layers - 1),
  )
  model_and_loss = ModelAndLoss(config=config).to(device)
  _ = train_model(model_and_loss, dataset, num_steps=10000, lr=0.001)


def main():
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  batch_size = 5
  dataset = FullExcerptDataset(
    batch_size=batch_size,
    char_map_file="character_map.txt",
    device=device,
  )

  # for i, batch in enumerate(dataset):
  #   strs = dataset.tokens_to_string(batch)
  #   for s in strs:
  #     print("-" * 80)
  #     print(s)
  #     print("-" * 80)
  #   if i > 3:
  #     break

  # time the batches per second
  start_time = datetime.datetime.now()
  num_steps = 5000
  for i, batch in enumerate(dataset):
    if i >= num_steps:
      break
  end_time = datetime.datetime.now()
  print(f"Time taken for {num_steps} steps: {(end_time - start_time).total_seconds()}s")
  print(f"Steps per second: {num_steps / (end_time - start_time).total_seconds()}")


if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)
  main()
