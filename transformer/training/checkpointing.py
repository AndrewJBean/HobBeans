from typing import Optional
import pathlib


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
