import torch
import numpy as np

class NoiseModel(torch.nn.Module):

    def __init__(self, word_drop, permute_window):

        super(NoiseModel, self).__init__()
        self.word_drop = word_drop
        self.permute_window = permute_window

    def forward(self, input):
        # TODO
        pass