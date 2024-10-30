from typing import Optional
import os
import math

import torch

from HobBeans.transformer.config_util import UnionLikeConfig, BaseConfig
from matplotlib import pyplot as plt


class SinusoidalPositionalEncodingConfig(BaseConfig):
  """
  Config for the sinusoidal positional encoding.
  """

  relative_freq_spacing: float
  base_freq: float = 1.0
  random_offset: bool = False


class RotaryPositionalEncodingConfig(BaseConfig):
  """
  Config for the rotary positional encoding.
  """

  relative_freq_spacing: float
  base_freq: float


class IdentityPositionalEncodingConfig(BaseConfig):
  """
  Config for the identity positional encoding.
  """

  encoding: str = "identity"


class PositionalEncodingConfig(UnionLikeConfig):
  """
  Config for the positional encoding.
  """

  sinusoidal: Optional[SinusoidalPositionalEncodingConfig] = None
  rotary: Optional[RotaryPositionalEncodingConfig] = None
  identity: Optional[IdentityPositionalEncodingConfig] = None


class SinusoidalPositionalEncoding(torch.nn.Module):
  """
  Sinusoidal positional encoding.
  """

  def __init__(self, config: SinusoidalPositionalEncodingConfig):
    super().__init__()
    self.config = config
    if self.config.relative_freq_spacing <= 1.0:
      raise ValueError("Relative frequency spacing must be greater than 1")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch_size, num_tokens, token_dim)
    """
    dev = x.device
    num_tokens = x.size(dim=1)
    token_dim = x.size(dim=2)

    freqs = self.config.base_freq * torch.pow(
      self.config.relative_freq_spacing, -torch.arange(0, token_dim // 2, device=dev, dtype=x.dtype)
    )
    freqs = freqs.unsqueeze(0)
    pos = torch.arange(num_tokens, device=dev, dtype=x.dtype).unsqueeze(1)
    pos = pos * freqs
    # random 0 to 2pi scalar offset
    if self.config.random_offset and self.training:
      offset = torch.rand(1, device=dev, dtype=x.dtype) * 2 * math.pi
      pos = pos + offset
    enc = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1).unsqueeze(0)
    return x + enc


class IdentityPositionalEncoding(torch.nn.Module):
  """
  Identity positional encoding.
  """

  def __init__(self, config: IdentityPositionalEncodingConfig):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x


class PositionalEncoding(torch.nn.Module):
  """
  Positional encoding.
  """

  def __init__(self, config: PositionalEncodingConfig):
    super().__init__()
    if config.sinusoidal is not None:
      self.pos_enc = SinusoidalPositionalEncoding(config.sinusoidal)
    elif config.rotary is not None:
      raise NotImplementedError
    elif config.identity is not None:
      self.pos_enc = IdentityPositionalEncoding(config.identity)
    else:
      raise ValueError("Unknown positional encoding config")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.pos_enc(x)


class MultiHeadAttentionConfig(BaseConfig):
  """
  Config for the multi-head attention.

  dims would be configured at a higher level.
  """

  num_heads: int
  dropout: float = 0.0
  bias: bool = True
  add_bias_kv: bool = False
  add_zero_attn: bool = False
  kdim: Optional[int] = None
  vdim: Optional[int] = None


class MultiHeadAttention(torch.nn.MultiheadAttention):
  """
  Multi-head attention.
  """

  def __init__(self, config: MultiHeadAttentionConfig, dims: int):
    super().__init__(
      embed_dim=dims,
      num_heads=config.num_heads,
      dropout=config.dropout,
      bias=config.bias,
      add_bias_kv=config.add_bias_kv,
      add_zero_attn=config.add_zero_attn,
      kdim=config.kdim if config.kdim is not None else dims,
      vdim=config.vdim if config.vdim is not None else dims,
      batch_first=True,
    )


class MultiHeadAttentionKarpathy(torch.nn.Module):
  """
  Multi-head attention from minGPT by Karpathy.
  """

  def __init__(self, config: MultiHeadAttentionConfig, dims: int):
    super().__init__()
    assert dims % config.num_heads == 0
    # key, query, value projections for all heads, but in a batch
    self.c_attn = torch.nn.Linear(dims, 3 * dims)
    # output projection
    self.c_proj = torch.nn.Linear(dims, dims)
    # regularization
    self.attn_dropout = torch.nn.Dropout(config.dropout)
    self.resid_dropout = torch.nn.Dropout(config.dropout)
    self.n_head = config.num_heads
    self.n_embd = dims

  def forward(self, value, **kwargs):
    x = value
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    bias = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(att.device)
    att = att.masked_fill(bias == 0, float("-inf"))
    att = torch.nn.functional.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

    # output projection
    y = self.resid_dropout(self.c_proj(y))
    return y, None


class MLPConfig(BaseConfig):
  """
  Config for the multi-layer perceptron.
  """

  layer_dims: list[int]
  output_activation: bool = False


class MLP(torch.nn.Module):
  """
  Multi-layer perceptron.
  """

  def __init__(self, config: MLPConfig, dims: int):
    super().__init__()
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip([dims] + config.layer_dims[:-1], config.layer_dims)):
      layers.append(torch.nn.Linear(in_dim, out_dim))
      if i < len(config.layer_dims) - 1 or config.output_activation:
        layers.append(torch.nn.ReLU())
    self.mlp = torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    x can have any shape (*, input_dims).
    output will be (*, output_dims).
    for language models, x would be (batch_size, num_tokens, dims).
    """

    return self.mlp(x)


class EncoderBlockConfig(BaseConfig):
  """
  Config for the encoder block.
  """

  multi_head_attention: MultiHeadAttentionConfig
  mlp: MLPConfig


class EncoderBlock(torch.nn.Module):
  """
  Encoder block.

  Input is (batch_size, num_tokens, dims),
  """

  def __init__(self, config: EncoderBlockConfig, dims: int):
    super().__init__()
    self.multi_head_attention = MultiHeadAttention(config.multi_head_attention, dims)
    self.mlp = MLP(config.mlp, dims)

  def forward(
    self,
    x: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
  ) -> torch.Tensor:
    """
    x: (batch_size, num_tokens, dims)
    """
    # multi-head attention
    attn_output, _ = self.multi_head_attention(
      query=x,
      key=x,
      value=x,
      key_padding_mask=key_padding_mask,
      attn_mask=attn_mask,
      is_causal=is_causal,
    )
    # residual connection
    x = x + attn_output
    # layer normalization
    x = torch.nn.functional.layer_norm(x, x.size()[1:])

    # mlp
    mlp_output = self.mlp(x)
    # residual connection
    x = x + mlp_output
    # layer normalization
    x = torch.nn.functional.layer_norm(x, x.size()[1:])

    return x


class AutoregressiveTransformerEncoderConfig(BaseConfig):
  """
  Config for the autoregressive transformer encoder.

  input embedding
  positional encoding
  encoder blocks
  output projection
  softmax
  """

  embedding_dims: int
  positional_encoding: PositionalEncodingConfig
  encoder_block: EncoderBlockConfig
  num_layers: int


class AutoregressiveTransformerEncoder(torch.nn.Module):
  """
  Autoregressive transformer encoder.
  """

  def __init__(
    self, config: AutoregressiveTransformerEncoderConfig, pad_token: Optional[int], vocab_size: int
  ):
    super().__init__()
    self.pad_token = pad_token
    self.embedding = torch.nn.Embedding(vocab_size, config.embedding_dims)
    self.positional_encoding = PositionalEncoding(config.positional_encoding)
    self.encoder_blocks = torch.nn.ModuleList(
      [EncoderBlock(config.encoder_block, config.embedding_dims) for _ in range(config.num_layers)]
    )
    self.output_projection = torch.nn.Linear(config.embedding_dims, vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    causal autoregressive transformer encoder.
    x: (batch_size, num_tokens) of integers
    output: (batch_size, num_tokens, vocab_size)
    """
    # mask out the padding tokens
    if self.pad_token is not None:
      key_padding_mask = torch.where(x == self.pad_token, float("-inf"), 0.0)
    else:
      key_padding_mask = None
    # (batch_size, num_tokens, embedding_dims)
    x = self.embedding(x)
    x = self.positional_encoding(x)
    attn_mask = torch.triu(
      torch.ones(x.size(1), x.size(1), device=x.device) * float("-inf"), diagonal=1
    )
    for encoder_block in self.encoder_blocks:
      x = encoder_block(x, key_padding_mask=key_padding_mask, is_causal=False, attn_mask=attn_mask)
    x = self.output_projection(x)
    return x


if __name__ == "__main__":
  config = PositionalEncodingConfig(
    sinusoidal=SinusoidalPositionalEncodingConfig(
      relative_freq_spacing=1.2,
      base_freq=1.0,
    )
  )
  print(config.json_dumps(indent=None))
  config.json_dump("pos_enc.zip")
  config2 = PositionalEncodingConfig.json_load("pos_enc.zip")
  assert config == config2
  print(config2.json_dumps(indent=None))

  config.json_dump("pos_enc.json")
  config3 = PositionalEncodingConfig.json_load("pos_enc.json")
  assert config == config3
  print(config3.json_dumps(indent=None))

  # delete the files
  os.remove("pos_enc.zip")
  os.remove("pos_enc.json")

  # test the positional encoding
  pos_enc = SinusoidalPositionalEncoding(config.sinusoidal)
  x = torch.randn(4, 1000, 128)
  y = pos_enc(x)
  enc = (y - x)[0, :, :].detach().cpu().numpy()
  plt.imshow(enc, aspect="auto")
  plt.colorbar()
  plt.show()

  pos_enc2 = PositionalEncoding(config)
  y2 = pos_enc2(x)
  assert torch.allclose(y, y2)
