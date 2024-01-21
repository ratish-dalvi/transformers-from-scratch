import math

import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax


class MultiHeadMaskedAttention(nn.Module):

    def __init__(self, embedding_size, num_heads):

        super().__init__()

        self.keys_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.queries_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.values_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.output_linear = nn.Linear(embedding_size, embedding_size)

        self.num_heads = num_heads


    def forward(self, x):

        batch_size, seq_length, embedding_size = x.size()
        num_heads = self.num_heads

        # Apply the linear layer of size embedding_size * embedding_size
        K = self.keys_linear(x)
        Q = self.queries_linear(x)
        V = self.values_linear(x)

        # For multi-head attention, split it into separate heads
        embedding_split = embedding_size // num_heads

        # Reshape: split the embedding into `num_heads` parts
        K = K.view(batch_size, seq_length, num_heads, embedding_split)
        Q = Q.view(batch_size, seq_length, num_heads, embedding_split)
        V = Q.view(batch_size, seq_length, num_heads, embedding_split)

        # Create 3D tensors containing arrays of seq_length * embedding_split arrays.
        # The easiest way is to swap 2nd and 3rd dimensions, and reshape them to flatten
        # the first two dimensions (which are batch_size, and num_heads after the swap)
        K = K.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)
        Q = Q.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)
        V = V.transpose(1, 2).reshape(batch_size * num_heads, seq_length, embedding_split)

        # Compute scaled dot product
        scaled_dot_product = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size)

        # Mask everything above the diagonal to prevent lookahead.
        assert scaled_dot_product.size() == (batch_size * num_heads, seq_length, seq_length)
        idxs = torch.triu_indices(seq_length, seq_length, offset=1)
        scaled_dot_product[..., idxs[0], idxs[1]] = float('-inf')

        attention_probs = torch.softmax(scaled_dot_product, dim=2)
        attn_out = torch.bmm(attention_probs, V)

        # Convert back to the original size and send through a linear layer
        attn_out = attn_out.view(batch_size, num_heads, seq_length, embedding_split)
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, seq_length, num_heads * embedding_split)

        return self.output_linear(attn_out)

    
class Block(nn.Module):
    """ A simple Transformer block, inspired by gpt2
    """

    def __init__(self, embedding_size, num_heads, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.ln2 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = MultiHeadMaskedAttention(embedding_size, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    
class Transformer(nn.Module):
    """ A basic Transformer using positional embeddings instead of encodings
    """
    def __init__(self, embedding_size, vocab_size, context_length, num_layers,
                 dropout, num_heads, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.pos_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_size)
    
        blocks = [Block(embedding_size, num_heads, dropout) for i in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(embedding_size)        
        self.lm_head = nn.Linear(embedding_size, vocab_size)  # reuse token embeddings instead

        
    def forward(self, input_ids, labels=None):
        batch_size, context_length = input_ids.size()
        token_embedding = self.token_embedding(input_ids)  # size: (batch_size, context_length, embedding_size)

        p = torch.arange(0, context_length, dtype=torch.long, device=self.device).unsqueeze(0)
        # p has size (1, context_length)
        pos_embedding = self.pos_embedding(p)  # Size: (1, context_length, embedding_size)
        
        # Add position embedding to every sequence in batch
        x = token_embedding + pos_embedding  # # Size: (batch_size, context_length, embedding_size)

        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # TODO Should we reuse token embeddings to save parameters
        
        # Get logits and y both into shape (batch_size * context_length, vocab_size)
        if labels is not None:
            loss = cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            return {"loss": loss}

        return logits

    
    @torch.no_grad()
    def generate(self, x, max_tokens, temperature=1.0):
        self.eval()  # Ensure the model is in evaluation mode

        for _ in range(max_tokens):
            logits = self(x)
            logits = logits[:, -1, :] / temperature
            probs = softmax(logits, dim=-1)
            _, x_next = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, x_next), dim=1)
        return x    
