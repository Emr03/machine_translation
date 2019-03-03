import torch
import numpy as np
from .sublayers import *

class EncoderLayer(torch.nn.Module):

    def __init__(self, params):

        super(EncoderLayer, self).__init__()
        self.d_model = params["d_model"]
        self.attn = SelfAttention(params)
        self.ffnn = FFNN(params)
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.dropout = torch.nn.Dropout(params["dropout"])

    def forward(self, x, src_mask):
        """

        :param x: shape is batch_size, seq_len, d_model
        :param src_mask: mask to use for masked attention, to hide padding tokens
        :return:
        """
        x = self.layer_norm_1(self.dropout(self.attn(x, x, x, mask=src_mask)) + x)
        x = self.layer_norm_2(self.dropout(self.ffnn(x)) + x)
        return x

class StackedEncoder(torch.nn.Module):

    def __init__(self, n_layers, params, n_langs, is_shared_emb=True):
        """

        :param n_layers:
        :param params:
        :param lang_ids: list of language ids supported by embedding layers
        """
        super(StackedEncoder, self).__init__()
        self.vocab_size = params["vocab_size"]
        self.d_model = params["d_model"]
        self.n_langs = n_langs

        embd_layer = torch.nn.Embedding(self.vocab_size, self.d_model)

        if is_shared_emb:
            self.embedding_layers = [embd_layer for _ in range(self.n_langs)]

        else:
            self.embedding_layers = [torch.nn.Embedding(self.vocab_size, self.d_model) for _ in range(self.n_langs)]

        # freeze embedding layers
        for l in self.embedding_layers:
            l.weight.requires_grad = False

        self.pos_enc = PositionalEncoding(params)
        self.encoder_layers = [EncoderLayer(params) for _ in range(n_layers)]
        self.emb_scale = np.sqrt(self.d_model)

    def forward(self, input_seq, src_mask, lang_id):

        x = self.emb_scale * self.embedding_layers[lang_id](input_seq)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x


if __name__ == "__main__":

    from src.config import params
    # test encoder layer
    x = torch.zeros(20, 5, 512)
    m = torch.zeros(20, 5).unsqueeze(-2).unsqueeze(-2)
    enc_layer = EncoderLayer(params)
    out = enc_layer(x, src_mask=m)
    print(out.shape)

    # test encoder stack
    x = torch.zeros(20, 5, dtype=torch.int64)
    enc = StackedEncoder(n_layers=6, params=params, n_langs=2)
    out = enc(x, m, 0)
    print(out.shape)








