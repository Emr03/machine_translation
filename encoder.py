import torch
import numpy as np
from sublayers import *

_D_MODEL = params["d_model"]
_D_K = params["d_k"]
_ATT_HEADS = params["h"]
_DFF = params["dff"]
_VOCAB = params["vocab_size"]
_MAX_LEN = params["max_len"]

class EncoderLayer(torch.nn.Module):

    def __init__(self):

        super(EncoderLayer, self).__init__()
        self.d_model = _D_MODEL
        self.pos_enc = PositionalEncoding()
        self.attn = SelfAttention()
        self.ffnn = FFNN()
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, input_seq):
        """

        :param input_seq: shape is batch_size, seq_len, d_model
        :return:
        """
        x = self.pos_enc(input_seq)
        x = self.layer_norm_1(self.attn(x, x, x) + x)
        x = self.layer_norm_2(self.ffnn(x) + x)

        return x

class StackedEncoder(torch.nn.Module):

    def __init__(self, n_layers):

        super(StackedEncoder, self).__init__()
        self.vocab_size = _VOCAB
        self.d_model = _D_MODEL
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        # TODO: initialize embedding layer with cross-lingual embeddings
        self.encoder = torch.nn.Sequential([EncoderLayer() for _ in n_layers])

    def forward(self, x):

        return self.encoder.forward(x)









