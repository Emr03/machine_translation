import torch
import numpy as np
from sublayers import *

_D_MODEL = params["d_model"]
_D_K = params["d_k"]
_ATT_HEADS = params["h"]
_DFF = params["dff"]
_VOCAB = params["vocab_size"]
_MAX_LEN = params["max_len"]

class DecoderLayer(torch.nn.Module):

    def __init__(self):

        super(DecoderLayer, self).__init__()
        self.d_model = _D_MODEL
        self.masked_attn = SelfAttention()
        self.attn = SelfAttention()
        self.ffnn = FFNN()
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_3 = torch.nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, dec_outputs, enc_outputs, mask):

        out = self.layer_norm_1(self.masked_attn(x_q=dec_outputs,
                                                 x_k=dec_outputs,
                                                 x_v=dec_outputs,
                                                 mask=mask) + dec_outputs)

        out = self.layer_norm_2(self.attn(x_q=out,
                                          x_k = enc_outputs,
                                          x_v = enc_outputs) + out)

        out = self.layer_norm_3(self.ffnn(out) + out)
        return out


class StackedDecoder(torch.nn.Module):

    def __init__(self, n_layers):

        super(StackedDecoder, self).__init__()
        self.d_model = _D_MODEL
        self.vocab_size = _VOCAB
        self.pos_enc = PositionalEncoding()
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        # TODO: initialize embedding layer with cross-lingual embeddings

        self.decoder_layers = [DecoderLayer() for _ in range(n_layers)]
        self.linear = torch.nn.Linear(self.d_model, self.vocab_size)

    def forward(self, dec_outputs, enc_outputs, mask):

        dec_outputs = self.pos_enc(dec_outputs)
        for layer in self.decoder_layers:
            dec_outputs = layer(dec_outputs=dec_outputs, enc_outputs=enc_outputs, mask=mask)

        return F.softmax(self.linear(dec_outputs), dim=-1)

if __name__ == "__main__":

    # test decoder layer
    x = torch.zeros(20, 5, 512, dtype=torch.float32)
    m = torch.zeros(1, 5)
    m[:, 0] = 1
    dec_layer = DecoderLayer()
    out = dec_layer(dec_outputs=x, enc_outputs=x, mask=m)
    print(out.shape)

    # test decoder stack
    dec = StackedDecoder(n_layers=6)
    out = dec(x, x, m)
    print(out.shape)
