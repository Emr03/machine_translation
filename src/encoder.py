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

    def forward(self, x):
        """

        :param x: shape is batch_size, seq_len, d_model
        :return:
        """
        x = self.layer_norm_1(self.attn(x, x, x) + x)
        x = self.dropout(x)
        x = self.layer_norm_2(self.ffnn(x) + x)
        x = self.dropout(x)

        return x

class StackedEncoder(torch.nn.Module):

    def __init__(self, n_layers, params):

        super(StackedEncoder, self).__init__()
        self.vocab_size =  params["vocab_size"]
        self.d_model =  params["d_model"]
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        self.pos_enc = PositionalEncoding(params)
        self.encoder = torch.nn.Sequential(*[EncoderLayer(params) for _ in range(n_layers)])

    def forward(self, input_seq):
        x = self.pos_enc(input_seq)
        return self.encoder.forward(x)


if __name__ == "__main__":

    from src.config import params
    # test encoder layer
    x = torch.zeros(20, 5, 512, dtype=torch.float32)

    enc_layer = EncoderLayer(params)
    out = enc_layer(x)
    print(out.shape)

    # test encoder stack
    enc = StackedEncoder(n_layers=6, params=params)
    out = enc(x)
    print(out.shape)








