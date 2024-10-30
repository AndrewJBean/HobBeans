import signal
import sys
import pathlib

from absl import app, flags

from transformer.modules.transformer_modules import AutoregressiveTransformerConfig
from transformer.modules.model_and_loss import ModelAndLoss
from transformer.data.dataset import FullExcerptDataset
from transformer.modules.transformer_modules import TransformerWrapper
from transformer.training.checkpointing import get_latest_checkpoint

import torch


flags.DEFINE_integer("step", None, "The checkpoint step number to load")
flags.DEFINE_string("dir", None, "The directory to load checkpoints from")


def main(argv):
  if flags.FLAGS.dir is None:
    raise ValueError("Please specify a directory to load checkpoints from")
  chkpt_dir = pathlib.Path(flags.FLAGS.dir)

  dataset = FullExcerptDataset()

  if not (chkpt_dir / "config.json").exists():
    raise ValueError(f"Config file not found in {chkpt_dir}")

  config = AutoregressiveTransformerConfig.json_load(str(chkpt_dir / "config.json"))
  model = TransformerWrapper(
    config=config, pad_token=dataset.pad_token, vocab_size=dataset.vocab_size
  )

  model_and_loss = ModelAndLoss(model=model)
  checkpoint_path = get_latest_checkpoint(chkpt_dir, step=flags.FLAGS.step)
  print(f"Loading checkpoint from {checkpoint_path}")
  model_and_loss.load_state_dict(torch.load(checkpoint_path, weights_only=True))
  model = model_and_loss.model.model

  # get a prompt from the user
  prompt = input("Enter a prompt: ")
  batch = dataset.strings_to_batch([prompt])[:, :-1]
  tokens = list(batch[0].cpu().numpy())
  print(prompt, end="")

  max_context_length = 10240
  # max_context_length = 200
  with torch.no_grad():
    model.eval()
    while True:
      batch = torch.tensor(tokens).reshape(1, -1)
      next_token_logits = model(batch)[:, -1, :]

      temperature = 0.8
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
