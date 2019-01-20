import torch
from encoder import Encoder
from decoder import Decoder
import numpy as np

class NMT(torch.nn.module):

    def __init__(self, Ex, Ey):

        assert(Ex.shape[0] == Ey.shape[0])
        self.Ex = Ex.transpose(0, 1)
        self.Ey = Ey.transpose(0, 1)
        self.Kx = Ex.shape[1]
        self.Ky = Ey.shape[1]
        self.emb_dim = Ex.shape[0]

        self.encoder = Encoder(input_size=Ex, hidden_size=1000, n_layers=1)
        self.decoder = Decoder(input_size=2000, context_size=1000, hidden_size=1000, n_layers=1)


    def encode(self, input_seq):
        """

        :param input_seq: shape = batch_size, seq_len, Kx
        :return:
        """
        return self.encoder.forward(input_seq.matmul(self.Ex))

    def decode(self, input_seq):

