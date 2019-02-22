import torch
import numpy as np
from sublayers import *

_D_MODEL = 512
_D_K = 64
_ATT_HEADS = 8
_DFF = 2048
_VOCAB = 30000

class EncoderLayer(torch.nn.Module):

    def __init__(self):

        super(EncoderLayer, self).__init__()
        self.d_model = _D_MODEL
        self.attn = SelfAttention()
        self.ffnn = FFNN()
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, input_seq):
        """

        :param input_seq: shape is batch_size, seq_len, d_model
        :return:
        """
        x = self.layer_norm_1(self.attn(input_seq, input_seq, input_seq) + input_seq)
        x = self.layer_norm_2(self.ffnn(x) + x)

        return x

class StackedEncoder(torch.nn.Module):

    def __init__(self, n_layers):

        super(StackedEncoder, self).__init__()
        self.vocab_size = _VOCAB
        self.d_model = _D_MODEL
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        # TODO: initialize embedding layer with cross-lingual embeddings
        # TODO: add position encoding
        self.encoder = torch.nn.Sequential([EncoderLayer() for _ in n_layers])

    def forward(self, x):
        return self.encoder.forward(x)









