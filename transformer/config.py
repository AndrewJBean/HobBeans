from typing import List
import pydantic


class ModelConfig(pydantic.BaseModel):
  num_chars: int
  char_embedding_dim: int
  char_vocab_size: int
  kernel_sizes: List[int]
  out_sizes: List[int]
