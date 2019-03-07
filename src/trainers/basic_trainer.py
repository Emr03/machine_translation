import torch.nn.functional as F
import numpy as np
from src.transformer import Transformer
from src.noise_model import NoiseModel
from src.data_loading import get_parser
from src.data.dataset import *
from src.data.loader import *
from abc import ABC, abstractmethod

class Trainer(ABC):
    
    def __init__(self, transformer):
        
        super().__init__()
        self.transformer = transformer
        self.data = transformer.data
        self.data_params = transformer.data_params
        self.noise_model = NoiseModel(data=self.data, params=self.data_params)
        self.max_len = 100

        self.pad_index = transformer.pad_index
        self.eos_index = transformer.eos_index
        self.bos_index = transformer.bos_index

    def get_src_mask(self, src_batch):
        mask = torch.ones_like(src_batch)
        mask.masked_fill_(src_batch == self.pad_index, 0).unsqueeze_(-2).unsqueeze_(-2)
        #print("mask", mask)
        return mask
    
    def get_tgt_mask(self, tgt_batch):

        batch_size, sent_len = tgt_batch.shape

        # hide future words
        tgt_m = np.tril(np.ones((batch_size, sent_len, sent_len)), k=0).astype(np.uint8)
        #print("tgt_m", tgt_m)

        tgt_m = torch.from_numpy(tgt_m)

        # hide padding
        tgt_m.masked_fill_(tgt_batch.unsqueeze(-1) == self.pad_index, 0).unsqueeze_(1)
        #print("tgt_m", tgt_m)
        return tgt_m
    
    @abstractmethod
    def train(n_iter):
        pass

    def get_train_batch(self, lang):

        get_iterator = self.transformer.train_iterators[lang]
        iterator = get_iterator()

        batch, l = next(iterator)
        # print(batch, l)
        batch = batch.transpose_(0, 1)
        return batch, l
    
    def save_model(self, path):
        torch.save(self.transformer.state_dict(), path)
        
    def load_model(self, path):
        self.transformer.load_state_dict(torch.load(path))
