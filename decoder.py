import torch
import numpy as np
from sublayers import *

_D_MODEL = 512
_D_K = 64
_ATT_HEADS = 8
_DFF = 2048
_VOCAB = 30000

class DecoderLayer(torch.nn.Module):

    def __init__(self):

        super(DecoderLayer, self).__init__()
        self.d_model = _D_MODEL
        self.masked_attn = SelfAttention(masked=True)
        self.attn = SelfAttention(mask=None)
        self.ffnn = FFNN()
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_3 = torch.nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, dec_outputs, enc_outputs):

        out = self.layer_norm_1(self.masked_attn(xq=dec_outputs, xk=enc_outputs, xv=enc_outputs) + dec_outputs)
        out = self.layer_norm_2(self.attn(out) + out)
        out = self.layer_norm_3(self.ffnn(out) + out)
        return out

class StackedDecoder():

    def __init__(self, n_layers):

        super(StackedDecoder, self).__init__()
        self.d_model = _D_MODEL
        self.vocab_size = _VOCAB
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        # TODO: initialize embedding layer with cross-lingual embeddings
        # TODO: add position encoding
        self.decoder = torch.nn.Sequential([DecoderLayer() for _ in n_layers]
                                           .append(torch.nn.Linear(self.d_model, self.vocab_size)))

    def forward(self, dec_outputs, enc_outputs):
        return torch.nn.functional.softmax(self.decoder.forward(x), dim=-1)
