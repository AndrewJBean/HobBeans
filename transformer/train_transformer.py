from typing import Dict
import signal
import sys

from HobBeans.transformer.transformer import (
  AutoregressiveTransformerEncoderConfig,
  AutoregressiveTransformerEncoder,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
  IdentityPositionalEncodingConfig,
)
from HobBeans.transformer.metrics import TopKAccuracy, Loss
from HobBeans.transformer.model import ModelAndLoss
from HobBeans.transformer.trainer import Trainer, TrainerConfig
from HobBeans.transformer.dataset import ExcerptDataset

import torch


class TransformerWrapper(torch.nn.Module):
  def __init__(
    self, *, config: AutoregressiveTransformerEncoderConfig, pad_token: int, vocab_size: int
  ):
    super().__init__()
    self.config = config
    self.pad_token = pad_token
    self.vocab_size = vocab_size
    self.model = AutoregressiveTransformerEncoder(config, pad_token, vocab_size)

  def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    x = batch["inputs"]
    logits = self.model(x)
    return {"logits": logits}


def main():
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  num_chars = 100
  batch_size = 10
  dataset = ExcerptDataset(
    num_chars=num_chars,
    batch_size=batch_size,
    char_map_file="character_map.txt",
  )

  emb_dims = 256
  config = AutoregressiveTransformerEncoderConfig(
    embedding_dims=emb_dims,
    positional_encoding=PositionalEncodingConfig(
      sinusoidal=SinusoidalPositionalEncodingConfig(
        relative_freq_spacing=1.2,
        base_freq=1.0,
      ),
      # identity=IdentityPositionalEncodingConfig(),
    ),
    encoder_block=EncoderBlockConfig(
      multi_head_attention=MultiHeadAttentionConfig(num_heads=8),
      mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
    ),
    num_layers=3,
  )
  model = TransformerWrapper(config=config, pad_token=None, vocab_size=dataset.vocab_size)

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
