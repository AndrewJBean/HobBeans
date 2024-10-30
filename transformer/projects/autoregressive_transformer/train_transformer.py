from typing import Dict
import platform
import signal
import sys
import pathlib

from transformer.modules.transformer_modules import (
  AutoregressiveTransformerConfig,
  TransformerWrapper,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from transformer.training.metrics import TopKAccuracy, Loss
from transformer.modules.model_and_loss import ModelAndLoss
from transformer.training.trainer import Trainer, TrainerConfig, CheckpointingConfig, EvalConfig
from transformer.data.dataset import FullExcerptDataset

import torch


def main():
  emb_dims = 1024
  num_layers = 16
  num_heads = 16
  chkpt_dir = (
    pathlib.Path.home()
    / "model_checkpoints"
    / f"transformer_checkpoints_{emb_dims}_{num_layers}_{num_heads}"
  )

  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  dataset = FullExcerptDataset(batch_size=4, trunc_len=512)

  # if the config exists in chkpt_dir / config.json, load it
  # otherwise, create a new one
  # TODO: this could include all configuration, including dataset and training
  if (chkpt_dir / "config.json").exists():
    config = AutoregressiveTransformerConfig.json_load(str(chkpt_dir / "config.json"))
    print("Loaded config from", chkpt_dir / "config.json")
  else:
    config = AutoregressiveTransformerConfig(
      embedding_dims=emb_dims,
      positional_encoding=PositionalEncodingConfig(
        sinusoidal=SinusoidalPositionalEncodingConfig(
          relative_freq_spacing=1.2,
          base_freq=1.0,
        ),
      ),
      encoder_block=EncoderBlockConfig(
        multi_head_attention=MultiHeadAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(layer_dims=[emb_dims, emb_dims]),
      ),
      num_layers=num_layers,
    )
    config.json_dump(str(chkpt_dir / "config.json"))
    print("Saved config to", chkpt_dir / "config.json")

  model = TransformerWrapper(
    config=config, pad_token=dataset.pad_token, vocab_size=dataset.vocab_size
  )

  model_and_loss = ModelAndLoss(model=model)
  optimizer = torch.optim.Adam(model_and_loss.parameters(), lr=0.00005)
  metrics = {"loss": Loss()}
  metrics.update({f"accuracy(k={k})": TopKAccuracy(k=k) for k in [1, 2, 3]})

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
