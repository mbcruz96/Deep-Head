import torch
import torch.nn as nn
import math
from typing import Any
from pathlib import Path

class TextEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # embedding dimensionality
        self.vocab_size = vocab_size    # size of corpus vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)  # learnable vobabulary embedding

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # creating a positional encoding matrix of dimension (sequence length, embedding dimensionality)
        pos_enc = torch.zeros(sequence_len, d_model)
        # creating positions vector of dimenstion (sequence length, 1)
        positions = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, ::2] = torch.sin(positions * divisor)
        pos_enc[:, 1::2] = torch.cos(positions * divisor)

        # creating a positional encoding for a batch of embeddings of size (1, sequence length, embedding dimensionality)
        pos_enc = pos_enc.unsqueeze(0)

        # saving positional encoding as non-learnable parameter in same file as model weights
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalizationBlock(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, device='cuda')) # Multiplicative parameter
        self.beta = nn.Parameter(torch.zeros(1, device='cuda')) # Addative parameter

    # input: (batch_size, seq_len, emb_dim)
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer1 = nn.Linear(d_model, d_ff).to(self.device)  # fc layer of dimension (ff_dim, emb_dim)
        self.layer2 = nn.Linear(d_ff, d_model).to(self.device)  # fc layer of dimension (emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input: (batch_size, seq_len, emb_dim, ff_dim) --> (batch_size, seq_len, ff_dim, emb_dim) --> (batch_size, seq_len, emb_dim, ff_dim)
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert d_model % h == 0, 'Embedding dimensionality not divisible by number of heads'
        self.d_k = d_model // h

        # learnable weight matrices
        self.w_q = nn.Linear(d_model, d_model).to(self.device)
        self.w_k = nn.Linear(d_model, d_model).to(self.device)
        self.w_v = nn.Linear(d_model, d_model).to(self.device)
        self.w_o = nn.Linear(d_model, d_model).to(self.device)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # input: (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1**9)
        # (batch, h, seq_len, seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # input: (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # input: (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadSelfAttentionBlock.attention(query, key, value, mask, self.dropout)

        # input: (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # contiguous allows tensor shape to be changed in contiguous memory block

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizationBlock()

    def forward(self, skip_layer, prev_layer):
        return skip_layer + self.dropout(prev_layer(self.norm(skip_layer)))

class EncoderBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadSelfAttentionBlock, ff_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.ff_block = ff_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadSelfAttentionBlock, cross_attention_block: MultiHeadSelfAttentionBlock, ff_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.ff_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size).to(self.device)

    def forward(self, x):
        # input: (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1) # log softmax for numerical stability

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pos_enc: PositionalEncoding, tgt_pos_enc: PositionalEncoding, src_embed: TextEmbeddings, tgt_embed: TextEmbeddings, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_enc(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask,):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_enc(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x):
        return self.projection_layer(x)

'''

TO-DO: Decide whether skip connection should send unadded & unormalized attentions scores or normalized scores

'''
class DeepHead(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pos_enc: PositionalEncoding, tgt_pos_enc: PositionalEncoding, src_embed: TextEmbeddings, tgt_embed: TextEmbeddings, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer
        self.residual_connection = ResidualConnection(dropout)
        self.prev_layer = None

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_enc(src)
        encoded = self.encoder(src, src_mask)
        if self.prev_layer == None:
            self.prev_layer = encoded
        else:
            # skip connection between transformers
            #unormalized = encoded
            encoded = self.residual_connection(encoded, self.prev_layer)
            self.prev_layer = encoded
        return encoded

    def decode(self, tgt, encoder_output, src_mask, tgt_mask,):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_enc(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x):
        return self.projection_layer(x)

def BuildTransformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, N: int = 6, d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
    # Create embedding layers
    src_embedding = TextEmbeddings(d_model, src_vocab_size)
    tgt_embedding = TextEmbeddings(d_model, tgt_vocab_size)

    # Create positional encodings
    src_pos_enc = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_enc = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadSelfAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # create transformer
    transformer = Transformer(encoder, decoder, src_pos_enc, tgt_pos_enc, src_embedding, tgt_embedding, projection_layer)
    # Initialize transformer parameters for better learning
    for layer in transformer.parameters():
        if layer.dim() > 1:
            nn.init.xavier_uniform_(layer)

    return transformer

'''

Finished implementing multi-head self attention in encoder

'''
def BuildDeepHead(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: list[int] = [64, 32, 16, 8, 4, 2], N: int = 6, d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
    # Create embedding layers
    src_embedding = TextEmbeddings(d_model, src_vocab_size)
    tgt_embedding = TextEmbeddings(d_model, tgt_vocab_size)

    # Create positional encodings
    src_pos_enc = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_enc = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for x in h:
        attention_block = MultiHeadSelfAttentionBlock(d_model, x, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadSelfAttentionBlock(d_model, 8, dropout)
        cross_attention_block = MultiHeadSelfAttentionBlock(d_model, 8, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # create transformer
    transformer = Transformer(encoder, decoder, src_pos_enc, tgt_pos_enc, src_embedding, tgt_embedding, projection_layer)
    # Initialize transformer parameters for better learning
    for layer in transformer.parameters():
        if layer.dim() > 1:
            nn.init.xavier_uniform_(layer)

    return transformer
