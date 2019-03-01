import torch
import numpy as np

class Noise(torch.nn.Module):

    def __init__(self, word_drop, permute_window):

        super(Noise, self).__init__()
        self.word_drop = word_drop
        self.permute_window = permute_window

