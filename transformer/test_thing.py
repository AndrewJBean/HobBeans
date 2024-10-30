import torch
import torch.nn as nn


class DecoderOnlyTransformer(nn.Module):
  def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.position_embedding = nn.Embedding(max_seq_length, d_model)

    decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    self.fc_out = nn.Linear(d_model, vocab_size)

    self.d_model = d_model
    self.vocab_size = vocab_size

  def forward(self, x):
    seq_length = x.size(1)
    positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)

    token_embeddings = self.token_embedding(x)
    position_embeddings = self.position_embedding(positions)

    x = token_embeddings + position_embeddings

    # Create a causal mask for the decoder
    mask = torch.triu(
      torch.ones(seq_length, seq_length, device=x.device) * float("-inf"), diagonal=1
    )

    output = self.transformer_decoder(x, x, tgt_mask=mask)
    output = self.fc_out(output)

    return output

  def generate(self, start_tokens, max_length):
    self.eval()
    with torch.no_grad():
      current_seq = start_tokens
      for _ in range(max_length - len(start_tokens)):
        logits = self(current_seq)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)

      return current_seq


# Example usage
vocab_size = 10000
d_model = 256
nhead = 8
num_layers = 6
max_seq_length = 512

model = DecoderOnlyTransformer(vocab_size, d_model, nhead, num_layers, max_seq_length)

# Training example
input_seq = torch.randint(0, vocab_size, (1, 10))  # Batch size 1, sequence length 10
output = model(input_seq)
print("Output shape:", output.shape)

# Generation example
start_tokens = torch.tensor([[1, 2, 3]])  # Assume 1, 2, 3 are valid token ids
generated_seq = model.generate(start_tokens, max_length=20)
print("Generated sequence:", generated_seq)
