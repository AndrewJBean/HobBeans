from typing import List, Set, Dict, Tuple, Optional
from collections import Counter
import os

import torch
from datasets import load_dataset


ds = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20")
ds = iter(ds["train"])
char_counts = Counter()
while len(char_counts) < 6000:
  text = next(ds)["text"]
  char_counts.update(list(text))

chars = [c for c, _ in char_counts.most_common(5000)]
print("".join(sorted(chars)))

this_file_directory = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(this_file_directory, "character_map.txt"), "w") as f:
  f.write("".join(sorted(chars)))
