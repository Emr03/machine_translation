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

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.transformer.cuda()

        else:
            self.device = torch.device('cpu')

        self.data = transformer.data
        self.data_params = transformer.data_params
        self.vocab_size = transformer.vocab_size
        self.noise_model = NoiseModel(data=self.data, params=self.data_params)
        self.max_len = 100

        self.pad_index = transformer.pad_index
        self.eos_index = transformer.eos_index
        self.bos_index = transformer.bos_index
        self.id2lang = transformer.id2lang

        # label smoothing parameters
        self.smoothing = 0.1
        self.confidence = 1.0 - self.smoothing

        self.kl_div_loss = torch.nn.KLDivLoss(size_average=False, reduce=True)

    def compute_kl_div_loss(self, x, target, lang):

        x = F.log_softmax(x, dim=-1)

        print("shape test ")
        print("x ", x.shape)
        print("target ", target.shape)

        # same device and dtype as x, requires_grad = false
        smooth_target = torch.zeros_like(x)
        smooth_target.fill_(self.smoothing / self.vocab_size[lang])
        smooth_target.scatter_(dim=1, index=target.data, value=self.confidence)

        # zero the pad_index for each vector of probabilities
        smooth_target[:, self.pad_index] = 0

        # find where the target word is a pad symbol, returns indices along dim 0
        mask = torch.nonzero(target.squeeze().data == self.pad_index)

        if mask.dim() > 0:
            # fill the entries of pad symbols with 0 prob
            smooth_target.index_fill_(0, mask.squeeze(), 0.0)

        return self.kl_div_loss(x, smooth_target)

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

    def get_lm_iterator(self, lang, train=True, add_noise=True):
        """
        returns batch with relevant masks
        moves everything to device

        :param lang:
        :param add_noise:
        :return:
        """

        if train:
            get_src_iterator = self.transformer.train_iterators[lang]

        else:
            get_src_iterator = self.transformer.val_iterators[lang]

        src_iterator = get_src_iterator()

        def iterator():
            for tgt_batch, tgt_l in src_iterator:

                tgt_batch.transpose_(0, 1)
                if add_noise:
                    src_batch, src_l = self.noise_model.add_noise(tgt_batch, tgt_l, lang)

                else:
                    src_batch = tgt_batch
                    src_l = tgt_l

                src_mask = self.get_src_mask(src_batch)
                tgt_mask = self.get_tgt_mask(tgt_batch)

                # move to cuda
                tgt_batch = tgt_batch.to(self.device)
                src_batch = tgt_batch.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                src_l = src_l.to(self.device)
                tgt_l = tgt_l.to(self.device)

                yield {"src_batch": src_batch,
                       "tgt_batch":tgt_batch,
                       "src_mask":src_mask,
                       "tgt_mask": tgt_mask,
                       "src_l": src_l,
                       "tgt_l": tgt_l}

        return iterator

    def save_model(self, path):
        torch.save(self.transformer.state_dict(), path)

    def load_model(self, path):
        self.transformer.load_state_dict(torch.load(path))

if __name__ == "__main__":

    # test kl_div_loss
    def compute_kl_div_loss(x, target, lang):

        smoothing = 0.1
        vocab_size = 30
        confidence = 1.0 - smoothing
        pad_index = 3
        kl_div_loss = torch.nn.KLDivLoss(size_average=False, reduce=True)


        x = F.log_softmax(x, dim=-1)

        # same device and dtype as x, requires_grad = false
        smooth_target = torch.zeros_like(x)
        smooth_target.fill_(smoothing / vocab_size)
        smooth_target.scatter_(dim=1, index=target.data, value=confidence)

        # zero the pad_index for each vector of probabilities
        smooth_target[:, pad_index] = 0
        print("smooth target", smooth_target)

        # find where the target word is a pad symbol, returns indices along dim 0
        mask = torch.nonzero(target.squeeze().data == pad_index)
        print("mask", mask)

        if mask.dim() > 0:
            # fill the entries of pad symbols with 0 prob
            smooth_target.index_fill_(0, mask.squeeze(), 0.0)

        print("smooth target", smooth_target)
        return kl_div_loss(x, smooth_target)

    x = torch.rand(3, 5)
    target = torch.tensor([[1], [2], [3]])

    print(x)
    print(target)

    x = torch.flatten(x, 0, 1)
    target = torch.flatten(target, 0, 1)

    print(x)
    print(target)

    loss = compute_kl_div_loss(x, target, lang=0)
    print("loss", loss)