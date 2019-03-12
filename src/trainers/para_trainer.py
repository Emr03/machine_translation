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

class ParallelTrainer(Trainer):

    def __init__(self, transformer):

        super().__init__(transformer)

    def translation_loss(self, batch_dict, lang1, lang2):

        tgt_mask = batch_dict["tgt_mask"]
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]
        prev_output = batch_dict["prev_output"]

        output_seq = self.transformer(input_seq=src_batch,
                                      prev_output=prev_output,
                                      src_mask=src_mask,
                                      tgt_mask=tgt_mask,
                                      src_lang=lang1,
                                      tgt_lang=lang1)

        return self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)

    def greedy_decoding(self, batch_dict, lang1, lang2):

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

        prev_output = torch.ones(1, self.max_len, dtype=torch.int64) * self.pad_index
        prev_output = prev_output.to(self.device)
        out = []
        prev_output[:, 0] = self.bos_index
        prev_token = self.bos_index
        word_count = 0

        while prev_token is not self.eos_index and word_count < self.max_len - 1:
            word_count += 1
            tgt_mask = torch.ones(1, word_count + 1).to(self.device)
            tgt_mask[:, word_count] = 0
            dec_logits = self.decode(prev_output=prev_output[:, :word_count + 1],
                                                        latent_seq=latent_code,
                                                        src_mask=src_mask,
                                                        tgt_mask=tgt_mask,
                                                        tgt_lang=lang2)

            scores = F.softmax(dec_logits, dim=-1)
            max_score, index = torch.max(scores[:, -1], -1)
            # print("index", index)

            prev_output[:, word_count] = index.item()
            prev_token = prev_output[:, word_count].item()
            word = self.data['dico'][self.id2lang[lang2]][index.item()]
            out.append(word)

        print("output", out)
        input = []
        for i in range(src_batch.size(1)):
            idx = src_batch[:, i].item()
            input.append(self.data['dico'][self.id2lang[lang1]][idx])

        print("input ", input)

    def train(self, n_iter):

        # for param in self.transformer.parameters():
        #     print(param.get_device())

        lang1 = 0
        lang2 = 1
        logger.info("Training translation model for %s -> %s " % (self.id2lang[lang1], self.id2lang[lang2]))

        get_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
        train_iterator = get_iterator()

        for i in range(n_iter):
            self.opt.zero_grad()

            try:
                batch_dict = next(train_iterator)

            except StopIteration:
                # restart the iterator
                get_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
                train_iterator = get_iterator()
                batch_dict = next(train_iterator)

            try:
                loss = self.translation_loss(batch_dict, lang1=lang1, lang2=lang2)

                if i % 50 == 0:
                    # print("iter ", i, "loss: ", loss)
                    self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

                loss.backward()
                self.opt_step()

            except Exception as e:
                self.logger.debug("Exception in training loop")
                self.logger.debug(e.message)

    def test(self, n_tests):
        self.transformer.eval()
        lang1 = 0
        lang2 = 1
        get_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
        train_iterator = get_iterator()
        for i in range(n_tests):
            batch_dict = next(train_iterator)
            #self.greedy_decoding(batch_dict, lang1, lang2)
            self.output_samples(batch_dict, lang1, lang2)


if __name__ == "__main__":

    logger = create_logger("logs/para_trainer.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger,
                        init_emb=True, embd_file="corpora/mono/all.en-fr.60000.vec")

    trainer = ParallelTrainer(model)
    # test iterator
    get_iter = trainer.get_para_iterator(0, 1)
    iter = get_iter()

    batch_dict = next(iter)
    prev_output = batch_dict["prev_output"]
    tgt_mask = batch_dict["tgt_mask"]
    tgt_batch = batch_dict["tgt_batch"]

    print("prev_output", prev_output)
    print("tgt_mask", tgt_mask)
    print("tgt_batch", tgt_batch)

    #trainer.train(3000)
    #trainer.save_model("en_fr.pth")
    #logger.info("testing trained model")
    #trainer.test(10)
    logger.info("testing loaded model")
    trainer.load_model("en_fr.pth")
    trainer.test(10)
