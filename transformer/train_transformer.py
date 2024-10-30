from typing import Dict
import platform
import signal
import sys
import pathlib

from HobBeans.transformer.transformer import (
  AutoregressiveTransformerEncoderConfig,
  AutoregressiveTransformerEncoder,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from HobBeans.transformer.metrics import TopKAccuracy, Loss
from HobBeans.transformer.model import ModelAndLoss
from HobBeans.transformer.trainer import Trainer, TrainerConfig, CheckpointingConfig, EvalConfig
from HobBeans.transformer.dataset import FullExcerptDataset

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

  char_map_file = pathlib.Path(__file__).parent / "character_map.txt"
  batch_size = 4
  dataset = FullExcerptDataset(
    batch_size=batch_size,
    char_map_file=str(char_map_file),
    trunc_len=512,
  )

  emb_dims = 1024
  num_layers = 8
  config = AutoregressiveTransformerEncoderConfig(
    embedding_dims=emb_dims,
    positional_encoding=PositionalEncodingConfig(
      sinusoidal=SinusoidalPositionalEncodingConfig(
        relative_freq_spacing=1.2,
        base_freq=1.0,
        random_offset=True,
      ),
    ),
    encoder_block=EncoderBlockConfig(
      multi_head_attention=MultiHeadAttentionConfig(num_heads=8),
      mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
    ),
    num_layers=num_layers,
  )
  model = TransformerWrapper(
    config=config, pad_token=dataset.pad_token, vocab_size=dataset.vocab_size
  )

  model_and_loss = ModelAndLoss(model=model)
  optimizer = torch.optim.Adam(model_and_loss.parameters(), lr=0.0001)
  metrics = {"loss": Loss()}
  metrics.update({f"accuracy(k={k})": TopKAccuracy(k=k) for k in [1, 2, 3]})

  # checkpoint dir in ~/model_checkpoints/transformer_checkpoints
  chkpt_dir = (
    pathlib.Path.home() / "model_checkpoints" / f"transformer_checkpoints_{emb_dims}_{num_layers}"
  )
  trainer_config = TrainerConfig(
    num_steps=100000,
    log_interval=100,
    eval=EvalConfig(interval=500, steps=100),
    checkpointing=CheckpointingConfig(interval=1000, directory=str(chkpt_dir))
    if platform.system() == "Darwin"
    else None,
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
