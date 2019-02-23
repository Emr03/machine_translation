import torch
import torch.nn.functional as F
from encoder import *
from decoder import *
from config import FLAGS
import numpy as np

params = FLAGS.flag_values_dict()

class Transformer(torch.nn.Module):

    def __init__(self):

        self.d_model = params["d_model"]
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.dff = params["dff"]
        self.d_k = params["d_k"]

        self.encoder = StackedEncoder(n_layers=self.n_layers)
        self.decoder = StackedDecoder(n_layers=self.n_layers)

        self.linear = torch.nn.Linear(self.d_model, self.vocab_size)

    def encode(self, input_seq):

        return self.encoder(input_seq)

    def decode(self, prev_output, latent_seq):

        return self.decoder(prev_output, latent_seq)

    def forward(self, input_seq):

        prev_output = self.dictionary[""]
        latent = self.encode(input_seq)



