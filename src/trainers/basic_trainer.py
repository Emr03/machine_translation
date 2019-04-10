from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import *
from src.data.loader import *
from src.model.noise_model import NoiseModel


class Trainer(ABC):

    def __init__(self, transformer, parallel=True):

        super().__init__()
        self.transformer = transformer
        self.encode = transformer.encode
        self.decode = transformer.decode
        self.logger = transformer.logger
        self.parallel = parallel
        self.is_variational = self.transformer.is_variational

        if torch.cuda.is_available() and not parallel:
            self.device = torch.device('cuda')
            self.transformer.cuda()

        elif torch.cuda.is_available() and parallel:
            n_devices = torch.cuda.device_count()
            self.logger.info("Using %i GPU's" % (n_devices))
            self.device = torch.device("cuda:0")
            self.transformer = nn.DataParallel(self.transformer)
            #self.logger.debug("transformer ", self.transformer)
            self.transformer.to(self.device)

        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
            self.parallel = False

        self.data = transformer.data
        self.data_params = transformer.data_params
        self.vocab_size = transformer.vocab_size
        self.noise_model = NoiseModel(data=self.data, params=self.data_params)
        self.max_len = 175

        self.pad_index = transformer.pad_index
        self.eos_index = transformer.eos_index
        self.bos_index = transformer.bos_index
        self.id2lang = transformer.id2lang

        # label smoothing parameters
        self.smoothing = 0.1
        self.confidence = 1.0 - self.smoothing

        # learning rate parameters
        self.step = 0
        self.warmup = 4000
        self.d_model = 512
        self.factor = 1

        self.opt = torch.optim.Adam(self.transformer.parameters(),
                                    lr=0.0,  betas=(0.9, 0.98), eps=1e-9)

        # save training details to resume
        self.state = {'iter': self.step,
                      'state_dict': self.transformer.state_dict(),
                      'optimizer': self.opt.state_dict()}

        # Todo: decide if you want to size_average, it reduces the loss a lot
        self.kl_div_loss = torch.nn.KLDivLoss(size_average=False, reduce=True)

    def save_model(self, path):

        try:
            torch.save(self.transformer.state_dict(), path)

        except Exception as e:
            self.logger.exception("message")

    def load_model(self, path):

        try:
            self.transformer.load_state_dict(torch.load(path, map_location=self.device))

        except Exception as e:
            self.logger.exception("message")

    def checkpoint(self, filename):

        try:
            self.state = {'iter': self.step,
                          'state_dict': self.transformer.state_dict(),
                          'optimizer': self.opt.state_dict()}

            torch.save(self.state, filename)

        except Exception as e:
            self.logger.exception("message")

    def load_checkpoint(self, filename):

        try:
            self.state = torch.load(filename, map_location=self.device)
            model_state_dict = self.state["state_dict"]
            opt_state_dict = self.state["optimizer"]

            self.transformer.load_state_dict(state_dict=model_state_dict)
            self.opt.load_state_dict(state_dict=opt_state_dict)

        except Exception as e:
            self.logger.exception("message")

    def opt_step(self):

        try:
            self.step += 1
            lr = self.factor * (self.d_model ** (-0.5) * min(self.step ** (-0.5),
                                                             self.step * self.warmup ** (-1.5)))

            for p in self.opt.param_groups:
                p['lr'] = lr

            self.opt.step()

        except Exception as e:
            self.logger.exception("message")


    def compute_kl_div_loss(self, x, target, lang):
        """

        :param x: shape = batch_size, sent_len, vocab_size
        :param target: shape = batch_size, sent_len, 1
        :param lang:
        :return:
        """

        try:
            # apply softmax on last dim, corresponding to words
            x = F.log_softmax(x, dim=-1)

            # reshape to index word by word on dim 0
            x = x.reshape(-1, x.size(-1))
            target = target.reshape(-1, 1)

            # get number of tokens, to scale loss
            normalize = target.size(0)

            # same device and dtype as x, requires_grad = false
            smooth_target = torch.zeros_like(x)
            smooth_target.fill_(self.smoothing / self.vocab_size[lang])

            if self.parallel:
                smooth_target.scatter_(dim=1, index=target.data, value=self.confidence)

            # zero the pad_index for each vector of probabilities
            smooth_target[:, self.pad_index] = 0

            # find where the target word is a pad symbol, returns indices along dim 0
            mask = torch.nonzero(target.squeeze().data == self.pad_index)

            if mask.size(0) != 0:
                # print("mask ", mask.size())
                # fill the entries of pad symbols with 0 prob
                smooth_target.index_fill_(0, mask.squeeze(), 0.0)

            if torch.isnan(self.kl_div_loss(x, smooth_target)).item():
                self.logger.debug("loss is nan")
                self.logger.debug("x", x)
                self.logger.debug("smooth target ", smooth_target)

            return self.kl_div_loss(x, smooth_target) / normalize

        except Exception as e:
            self.logger.exception("message")

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

    def compute_sent_len(self, sentences):
        """
        returns a tensor containing the length of each sentence in the batch
        :param sentences: batch_size, max_len
        :return:
        """
        n_sent = sentences.shape[0]
        # TODO: make sure sentences end with eos
        eos_indices = (sentences == self.eos_index).nonzero()
        assert(eos_indices.shape[0] == n_sent)
        eos_indices = eos_indices[:, 1].unsqueeze_() + 1
        return eos_indices

    def pad_sentences(self, sentences):
        pass

    @abstractmethod
    def train(n_iter):
        pass

    def get_lm_iterator(self, lang_id, train=True, add_noise=True):
        """
        returns batch with relevant masks
        moves everything to device

        :param lang:
        :param add_noise:
        :return:
        """

        lang = self.id2lang[lang_id]

        if train:
            assert (self.data['mono'][lang]['train'] is not None)
            get_src_iterator = self.data['mono'][lang]['train'].get_iterator(shuffle=True, group_by_size=True)

        else:
            assert (self.data['mono'][lang]['valid'] is not None)
            get_src_iterator = self.data['mono'][lang]['valid'].get_iterator(shuffle=True, group_by_size=True)

        src_iterator = get_src_iterator()

        def iterator():
            for tgt_batch, tgt_l in src_iterator:

                tgt_batch.transpose_(0, 1)
                if add_noise:
                    src_batch, src_l = self.noise_model.add_noise(tgt_batch, tgt_l, lang_id)

                else:
                    src_batch = tgt_batch
                    src_l = tgt_l

                # does not create new tensor, input to the decoder during training, without eos token
                prev_output = tgt_batch[:, :-1]
                tgt_batch = tgt_batch[:, 1:]

                src_mask = self.get_src_mask(src_batch)
                # create mask based on input to the decoder
                tgt_mask = self.get_tgt_mask(prev_output)

                # move to cuda
                tgt_batch = tgt_batch.to(self.device)
                src_batch = src_batch.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                src_l = src_l.to(self.device)
                tgt_l = tgt_l.to(self.device)

                yield {"src_batch": src_batch,
                       "tgt_batch": tgt_batch,
                       "prev_output": prev_output,
                       "src_mask": src_mask,
                       "tgt_mask": tgt_mask,
                       "src_l": src_l,
                       "tgt_l": tgt_l}
        
        return iterator

    def get_para_iterator(self, lang1, lang2, train=True, add_noise=False):
        """
        returns training batches to translate from lang1 to lang2
        :param lang1:
        :param lang2:
        :param train:
        :param add_noise:
        :return:
        """

        src_lang = self.id2lang[lang1]
        tgt_lang = self.id2lang[lang2]

        if train:
            assert (self.data['para'][(src_lang, tgt_lang)]['train'] is not None)
            get_iterator = self.data['para'][(src_lang, tgt_lang)]['train'].get_iterator(shuffle=True, group_by_size=True)

        else:
            assert (self.data['para'][(src_lang, tgt_lang)]['valid'] is not None)
            get_iterator = self.data['para'][(src_lang, tgt_lang)]['valid'].get_iterator(shuffle=True, group_by_size=True)

        batch_iterator = get_iterator()

        def iterator():

            for src, tgt in batch_iterator:

                src_batch, src_l = src
                tgt_batch, tgt_l = tgt

                tgt_batch.transpose_(0, 1)
                src_batch.transpose_(0, 1)

                if add_noise:
                    src_batch, src_l = self.noise_model.add_noise(src_batch, src_l, lang1)

                # does not create new tensor
                prev_output = tgt_batch[:, :-1]
                tgt_batch = tgt_batch[:, 1:]

                src_mask = self.get_src_mask(src_batch)
                tgt_mask = self.get_tgt_mask(prev_output)

                # move to cuda
                tgt_batch = tgt_batch.to(self.device)
                src_batch = src_batch.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                src_l = src_l.to(self.device)
                tgt_l = tgt_l.to(self.device)

                yield {"src_batch": src_batch,
                       "tgt_batch": tgt_batch,
                       "prev_output": prev_output,
                       "src_mask": src_mask,
                       "tgt_mask": tgt_mask,
                       "src_l": src_l,
                       "tgt_l": tgt_l}

        return iterator

    def greedy_decoding(self, batch_dict, lang1, lang2):
        """
        testing method, examine output under slightly different conditions from training
        :param batch_dict:
        :param lang1:
        :param lang2:
        :return:
        """
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]

        if tgt_batch.shape[0] > 1:
            tgt_batch = tgt_batch[0, :].unsqueeze(0)
            src_batch = src_batch[0, :].unsqueeze(0)
            src_mask = src_mask[0, :].unsqueeze(0)

        print(tgt_batch.shape)

        latent_code = self.encode(input_seq=src_batch,
                                  src_mask=src_mask,
                                  src_lang=lang1)

        #prev_output = torch.ones(1, self.max_len, dtype=torch.int64) * self.pad_index
        #prev_output[:, 0] = self.bos_index
        #prev_token = self.bos_index

        prev_output = batch_dict["prev_output"]
        prev_output = prev_output.to(self.device)
        prev_token = prev_output[:, 0]

        out = []

        word_count = 0

        while prev_token is not self.eos_index and word_count < self.max_len - 1:

            word_count += 1
            dec_input = prev_output[:, :word_count]
            self.logger.info("dec input ", dec_input)
            dec_logits = self.decode(prev_output=dec_input,
                                     latent_seq=latent_code,
                                     src_mask=src_mask,
                                     tgt_mask=None,
                                     tgt_lang=lang2)

            scores = F.softmax(dec_logits, dim=-1)
            max_score, index = torch.max(scores[:, -1], -1)
            self.logger.info("index", index)

            #prev_output[:, word_count] = index.item()
            prev_token = prev_output[:, word_count].item()
            word = self.data['dico'][self.id2lang[lang2]][index.item()]
            out.append(word)
            self.logger.info("output word", word)
            self.logger.info("GT word", tgt_batch[:, word_count])

        self.logger.info("output", out)
        input = []
        for i in range(src_batch.size(1)):
            idx = src_batch[:, i].item()
            input.append(self.data['dico'][self.id2lang[lang1]][idx])

        self.logger.info("input ", input)


    def output_samples(self, batch_dict, lang1, lang2):
        """
        testing method, used to check outputs under teacher forcing
        :param batch_dict:
        :param lang1:
        :param lang2:
        :return:
        """

        tgt_mask = batch_dict["tgt_mask"]
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]
        prev_output = batch_dict["prev_output"]

        if tgt_batch.shape[0] > 1:
            tgt_batch = tgt_batch[0, :].unsqueeze(0)
            src_batch = src_batch[0, :].unsqueeze(0)
            src_mask = src_mask[0, :].unsqueeze(0)
            prev_output = prev_output[0, :].unsqueeze(0)
            tgt_mask = tgt_mask[0, :].unsqueeze(0)

        output_seq = self.transformer(input_seq=src_batch,
                                      prev_output=prev_output,
                                      src_mask=src_mask,
                                      tgt_mask=tgt_mask,
                                      src_lang=lang1,
                                      tgt_lang=lang2)

        loss = self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)
        self.logger.info("loss", loss)

        scores = F.softmax(output_seq, dim=-1)
        max_score, indices = torch.max(scores, -1)
        #print(indices)
        words = [self.data['dico'][self.id2lang[lang2]][indices[:, i].item()] for i in range(indices.size(1))]
        self.logger.info("output", words)

        input_sent = []
        for i in range(src_batch.size(1)):
            idx = src_batch[:, i].item()
            input_sent.append(self.data['dico'][self.id2lang[lang1]][idx])

        self.logger.info("input ", input_sent)


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
            smooth_target.index_fill_(dim=0, index=mask.squeeze(), value=0.0)

        print("smooth target", smooth_target)
        return kl_div_loss(x, smooth_target)

    x = torch.rand(3, 2, 5)
    target = torch.tensor([[[1], [1]], [[2], [2]], [[3], [3]]])

    print(x)
    print(target)

    x = x.reshape(-1, 5)
    target = target.reshape(-1, 1)

    print(x)
    print(target)

    print(x.shape)
    print(target.shape)

    loss = compute_kl_div_loss(x, target, lang=0)
    print("loss", loss)
