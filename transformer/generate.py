from typing import Dict, Optional
import signal
import sys
import pathlib

from absl import app, flags

from HobBeans.transformer.transformer import (
  AutoregressiveTransformerEncoderConfig,
  AutoregressiveTransformerEncoder,
  SinusoidalPositionalEncodingConfig,
  PositionalEncodingConfig,
  MultiHeadAttentionConfig,
  MLPConfig,
  EncoderBlockConfig,
)
from HobBeans.transformer.model import ModelAndLoss
from HobBeans.transformer.ds_no_trunc import FullExcerptDataset

import torch


# flag for checpoint number
flags.DEFINE_integer("step", None, "The checkpoint step number to load")


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


def get_latest_checkpoint(chkpt_dir: str, step: Optional[int] = None) -> Optional[str]:
  chkpt_dir = pathlib.Path(chkpt_dir)
  checkpoints = list(chkpt_dir.glob("*.pt"))
  max_chkpt = (-1, None)
  step_to_chkpt = {}
  for chkpt in checkpoints:
    trimmed = str(chkpt).split(".pt")[0]
    pos = len(trimmed) - 1
    while pos >= 0 and trimmed[pos].isdigit():
      pos -= 1
    num = int(trimmed[pos + 1 :])
    step_to_chkpt[num] = chkpt
    if num > max_chkpt[0]:
      max_chkpt = (num, chkpt)
  if step is not None:
    return step_to_chkpt.get(step, None)
  return max_chkpt[1]


def main(argv):
  device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")

  batch_size = 10
  dataset = FullExcerptDataset(
    batch_size=batch_size,
    char_map_file="character_map.txt",
  )

  # emb_dims = 512
  # num_layers = 3
  # emb_dims = 1024
  # num_layers = 4
  # emb_dims = 512
  # num_layers = 8
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
  chkpt_dir = (
    pathlib.Path.home() / "model_checkpoints" / f"transformer_checkpoints_{emb_dims}_{num_layers}"
  )
  checkpoint_path = get_latest_checkpoint(chkpt_dir, step=flags.FLAGS.step)
  print(f"Loading checkpoint from {checkpoint_path}")
  model_and_loss.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model = model_and_loss.model
  # get a prompt from the user
  prompt = input("Enter a prompt: ")
  batch = dataset.strings_to_batch([prompt])[:, :-1]
  tokens = list(batch[0].cpu().numpy())
  print(prompt, end="")

  max_context_length = 10240
  with torch.no_grad():
    model.eval()
    while True:
      batch = {"inputs": torch.tensor(tokens).reshape(1, -1)}
      output = model(batch)
      logits = output["logits"]
      next_token_logits = logits[:, -1, :]
      # next_token = torch.argmax(next_token_logits, dim=-1).reshape(1, 1)
      temperature = 1.0
      next_token = torch.multinomial(
        torch.softmax(next_token_logits / temperature, dim=-1), num_samples=1
      )
      if next_token in dataset.non_text_tokens:
        break
      tokens.append(next_token)
      if len(tokens) > max_context_length:
        tokens = tokens[-max_context_length:]
      next_char = dataset.tokens_to_string(next_token)[0]
      print(next_char, end="", flush=True)


# handle keyboard interrupt
def signal_handler(sig, frame):
  print("\nKeyboardInterrupt, exiting...")
  sys.exit(0)


if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)
  app.run(main)
