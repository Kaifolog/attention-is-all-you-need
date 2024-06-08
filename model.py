import torch
import torch.nn as nn
import math

class VanillaTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_padding_idx, trg_padding_idx,
                 max_seq_len=5000, n=6, d_model=512, d_ff=2048, h=8, p_drop=0.1,
                 **model_params):
        super(VanillaTransformer, self).__init__()

        assert d_model % h == 0
        # According to the paper: d_k = d_v = d_model // h
        # And remember that d_q = d_k. Why?
        # See: https://substack-post-media.s3.amazonaws.com/public/images/b75a8df1-0a82-4f79-8e68-4fe16587063d_1474x1108.png
        d_k = d_v = d_model // h

        print("Init VanillaTransformer :"
              f" {max_seq_len=}, {src_vocab_size=}, {trg_vocab_size=}, {src_padding_idx=}, {trg_padding_idx=},"
              f" {n=}, {d_model=}, {d_ff=}, {h=}, {d_k=}, {d_v=}, {p_drop=}")

        self.d_model = d_model
        self.src_padding_idx = src_padding_idx
        self.trg_padding_idx = trg_padding_idx
        self.src_embedding = Embedding(src_vocab_size, d_model, src_padding_idx)
        self.trg_embedding = Embedding(trg_vocab_size, d_model, trg_padding_idx)
        self.dropout = nn.Dropout(p_drop)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.encoder_stack = Encoder(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.decoder_stack = Decoder(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.projection = nn.Linear(d_model, trg_vocab_size, bias=False)

        # 'diagonal=1' means that all elements on and above the main+1 diagonal are retained
        self.register_buffer('leftward_mask', torch.triu(torch.ones((max_seq_len, max_seq_len)), diagonal=1).bool()) 

    @classmethod
    def _mask_paddings(cls, x, padding_idx):
        return x.eq(padding_idx).unsqueeze(-2).unsqueeze(1)

    def forward(self, src, trg):
        # src.size: (BATCH_SIZE, SRC_SEQ_LEN), trg.size: (BATCH_SIZE, TRG_SEQ_LEN)
        # Batches of vectors of tokens

        # Transformer architecture uses two types of masking:
        # 1. Padding mask: To handle batches with sequences of different lengths, we need to ignore PAD tokens.
        src_padding_mask = VanillaTransformer._mask_paddings(src, self.src_padding_idx)
        trg_padding_mask = VanillaTransformer._mask_paddings(trg, self.trg_padding_idx)
        # 2. Look-ahead mask: During training, we need to prevent the transformer from looking ahead to future tokens. This allows us to compute losses for all positions in parallel.
        trg_self_attn_mask = trg_padding_mask | self.leftward_mask[:trg.size(-1), :trg.size(-1)]    # combine the look-ahead and padding masks

        src = self.src_embedding(src)
        src = self.dropout(src + self.positional_encoding(src))

        x = self.encoder_stack(src, padding_mask=src_padding_mask)

        trg = self.trg_embedding(trg)
        trg = self.dropout(trg + self.positional_encoding(trg))

        x = self.decoder_stack(trg, x, self_attn_mask=trg_self_attn_mask, enc_dec_attn_mask=src_padding_mask)

        x = self.projection(x)
        # x.size: (BATCH_SIZE, TRG_SEQ_LEN, VOCAB_SIZE)
        # The output of the decoder consists of distributions over positions. 
        # During training, we shift our target output by one position to the left. 
        # As a result, during inference, the transformer will predict the next token 
        # based on the last distribution in the output.
        return x


class Encoder(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, padding_mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.add_and_norm_1 = AddAndNorm(d_model, p_drop)
        self.pos_ff = PositionWiseFeedForward(d_model, d_ff)
        self.add_and_norm_2 = AddAndNorm(d_model, p_drop)
        # According to the paper: "We apply dropout [33] to the output of each sub-layer, before it is added to the
        # sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
        # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of P_drop = 0.1."
        # Read about layer normalization in Section 3.1 https://arxiv.org/pdf/1607.06450

    def forward(self, x, padding_mask):
        x = self.add_and_norm_1(x, self.self_attention(x, x, x, mask=padding_mask))
        x = self.add_and_norm_2(x, self.pos_ff(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, x_enc, self_attn_mask, enc_dec_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.add_and_norm_1 = AddAndNorm(d_model, p_drop)
        self.enc_dec_attention = MultiHeadAttention(d_model, d_v, d_v, h)
        self.add_and_norm_2 = AddAndNorm(d_model, p_drop)
        self.pos_ff = PositionWiseFeedForward(d_model, d_ff)
        self.add_and_norm_3 = AddAndNorm(d_model, p_drop)
        # Almost the same as the EncoderLayer

    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        x = self.add_and_norm_1(x, self.self_attention(x, x, x, mask=self_attn_mask))
        x = self.add_and_norm_2(x, self.enc_dec_attention(x, x_enc, x_enc, mask=enc_dec_attn_mask))
        x = self.add_and_norm_3(x, self.pos_ff(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, h * d_k, bias=False)
        self.w_k = nn.Linear(d_model, h * d_k, bias=False)
        self.w_v = nn.Linear(d_model, h * d_v, bias=False)
        self.w_o = nn.Linear(h * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

    def _split_into_heads(self, *xs):
        # (q, k, v) sizes: (BATCH_SIZE, SEQ_LEN, D_MODEL) -> (BATCH_SIZE, H, SEQ_LEN, D_K)
        return [x.view(x.size(0), x.size(1), self.h, -1).transpose(1, 2) for x in xs]

    def forward(self, q, k, v, mask=None):
        # (q, k, v) sizes: (BATCH_SIZE, SEQ_LEN, D_MODEL)

        # 1. Divide the continuous d_model representation into 'heads' and perform all the linear projections
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self._split_into_heads(q, k, v)  # -> (q, k, v) sizes: (BATCH_SIZE, H, SEQ_LEN, D_K)
        # 2. Apply attention across all heads
        x = self.attention(q, k, v, mask)
        # 3. Concatenate the heads back using a view and apply a final linear transformation
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)  # -> x: (BATCH_SIZE, SEQ_LEN, D_MODEL)
        x = self.w_o(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_k ** -0.5

    def forward(self, q, k, v, mask):
        # (q, k, v) sizes: (BATCH_SIZE, H, SEQ_LEN, D_K)

        x = torch.matmul(q, k.transpose(-2, -1))  # -> x: (BATCH_SIZE, H, SEQ_LEN, SEQ_LEN)

        x = x if mask is None else x.masked_fill(mask, float('-inf'))   # masking the scores matrix with a mask of the same size
        x = torch.matmul(torch.softmax(self.scale * x, dim=-1), v)
        # According to the paper: "We suspect that for large values of d_k, the dot products grow large in magnitude, 
        # pushing the softmax function into regions where it has extremely small gradients. 
        # To counteract this effect, we scale the dot products by math.sqrt(d_k)"
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = d_model ** 0.5

    def forward(self, x):
        x = self.embedding(x)
        # The reason for increasing the embedding values (by multiplying with sqrt(d_model)) before the
        # addition is to reduce the impact of positional encoding. This ensures that the original meaning of
        # the embedding vector won’t be lost when we add them together.
        return x * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()

        # The formula used in the paper to compute the position embedding was as follows:
        #
        #   PE(pos, 2i) = sin(pos / (10000 ^ (2i / d_model)))
        #   PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / d_model)))
        #
        # Also see: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        # Theoretically the pytorch multiplication is ~30% faster than a division. 
        # So, lets transform the formula to take advantages of multiplication:
        #
        #   pos / (10000 ^ (2i / d_model)) = pos / ( exp(log(10000)) ^ (2i/d)) = 
        #   = pos / exp( log(10000) * (2i/d) ) = pos * exp( log(10000) * (2i/d) )^(-1) =
        #   = pos * exp( - log(10000) * (2i/d) ) = pos * exp( - 2i * (log(10000) / d) )
        #
        # As a result, now we can divide number by number (log(10000) / d), 
        # simplifying all matrix operations to multiplications.
        # Tbh, the PE matrix computes ones in the program so the trick is kind of useless.
        # I guess it was firstly introduced in: https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

        # Create a positional encoding matrix of shape (seq_len, d_model) filled with zeros
        pe = torch.zeros(max_seq_len, d_model)
        # Create a tensor representing positions (0 to seq_len-1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        #[[0],
        # .
        # .
        # .
        # [seq_len-1]]

        # Division term of the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add an extra dimension for the batch using
        pe = pe.unsqueeze(0)
        # We register this tensor in the model's buffer. This is done when we have a tensor that we
        # want to store in the model not as a learned parameter but as a constant.
        self.register_buffer('pe', pe)

    def forward(self, x):
        return (self.pe[:, :x.size(1), :].requires_grad_(False))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(torch.relu(self.linear_1(x)))


class AddAndNorm(nn.Module):
    def __init__(self, d_model, p_drop):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, inputs, x):
        return self.layer_norm(inputs + self.dropout(x))
