from src.logger import create_logger
import torch.nn.functional as F
import torch.cuda
import numpy as np
from src.transformer import Transformer
from src.noise_model import NoiseModel
from src.data_loading import get_parser
from src.data.dataset import *
from src.data.loader import *
from .basic_trainer import Trainer
import logging

class LanguageModeling(Trainer):

    def __init__(self, transformer):

        super().__init__(transformer)

    def reconstruction_loss(self, batch_dict, lang):

        tgt_mask = batch_dict["tgt_mask"]
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]

        output_seq = self.transformer(input_seq=src_batch,
                                  prev_output=tgt_batch,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_lang=lang,
                                  tgt_lang=lang)
       
        return self.compute_kl_div_loss(x=output_seq, target=src_batch, lang=lang)

    def greedy_decoding(self, batch_dict, lang):

        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]

        if tgt_batch.shape[0] > 1:
            tgt_batch = tgt_batch[0]
            src_batch = src_batch[0]
            src_mask = src_mask[0]

        latent_code = self.transformer.encode(input_seq=src_batch,
                                              src_mask=src_mask,
                                              src_lang=lang)

        prev_output = torch.ones(1, self.max_len, dtype=torch.int64)*self.pad_index
        prev_output = prev_output.to(self.device)
        out = []
        prev_output[:, 0] = self.bos_index
        prev_token = self.bos_index
        word_count = 0

        while prev_token is not self.eos_index and word_count<self.max_len-1:

            word_count+=1
            tgt_mask = torch.ones(1, word_count+1).to(self.device)
            tgt_mask[:, word_count] = 0
            dec_logits = self.transformer.decode(prev_output=prev_output[:, :word_count+1],
                                        latent_seq=latent_code,
                                        src_mask=src_mask,
                                        tgt_mask=tgt_mask,
                                        tgt_lang=lang)

            scores = F.softmax(dec_logits, dim=-1)
            max_score, index = torch.max(scores[:, -1], -1)
            print("index", index)

            prev_output[:, word_count] = index.item()
            prev_token = prev_output[:, word_count].item()
            word = self.data['dico'][self.id2lang[lang]][index.item()]
            out.append(word)

        print("output", out)
        input = []
        for i in range(src_batch.size(1)):
            idx = src_batch[:, i].item()
            input.append(self.data['dico'][self.id2lang[lang]][idx])

        print("input ", input)

        loss = self.compute_kl_div_loss(x=prev_output, target=src_batch, lang=lang)

        print("loss ", loss)

    def train(self, n_iter):

        # for param in self.transformer.parameters():
        #     print(param.get_device())

        lang = 0
        logger.info("Training language model for %s " % (self.id2lang[lang]))

        get_iterator = self.get_lm_iterator(lang=lang, train=True, add_noise=True)
        train_iterator = get_iterator()
        opt = torch.optim.Adam(self.transformer.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)

        for i in range(n_iter):
            opt.zero_grad()

            try:
                batch_dict = next(train_iterator)

            except StopIteration:
                # restart the iterator
                iterator = get_iterator()
                batch_dict = next(iterator)

            loss = self.reconstruction_loss(batch_dict, lang=lang)

            if i % 50 == 0:
                #print("iter ", i, "loss: ", loss)
                self.logger.info("iter %i: loss %40.1f" %(i, loss.item()))

            loss.backward()
            opt.step()

    def test(self, n_tests):
        self.transformer.eval()
        lang = 0
        get_iterator = self.get_lm_iterator(lang=lang, train=True, add_noise=True)
        train_iterator = get_iterator()
        for i in range(n_tests):
            batch_dict = next(train_iterator)
            self.greedy_decoding(batch_dict, lang)

if __name__ == "__main__":

    logger = create_logger("logs/sanity_check.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger, embd_file="corpora/mono/all.en-fr.60000.vec")
    trainer = LanguageModeling(model)

    trainer.train(300000)
    trainer.save_model("en_language_model.pth")
    logger.info("testing trained model")
    trainer.test(10)
    logger.info("testing loaded model")
    trainer.load_model("enlanguage_model.pth")
    trainer.test(10)
