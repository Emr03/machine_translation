import torch.nn.functional as F
import torch.cuda
import numpy as np
from src.transformer import Transformer
from src.noise_model import NoiseModel
from src.data_loading import get_parser
from src.data.dataset import *
from src.data.loader import *
from .basic_trainer import Trainer

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

        return F.cross_entropy(input=torch.flatten(output_seq, 0, 1),
                               target=torch.flatten(src_batch), ignore_index=self.pad_index)

    def greedy_decoding(self, sent, src_mask):

        # TODO: check

        assert(sent.shape[0] == 1)
        lang = 0
        src_mask = self.get_src_mask(sent)
        latent_code = self.transformer.encode(input_seq=sent,
                                              src_mask=src_mask,
                                              src_lang=lang)

        prev_output = torch.ones(1, self.max_len, dtype=torch.int64)*self.pad_index
        out = []
        prev_output[:, 0] = self.bos_index
        prev_token = self.bos_index
        word_count = 0

        while prev_token is not self.eos_index and word_count<self.max_len-1:

            word_count+=1
            tgt_mask = np.ones((1, self.max_len))
            tgt_mask = torch.from_numpy(tgt_mask)
            tgt_mask = tgt_mask.masked_fill_(prev_output == self.pad_index, 0)
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
            print(out)

        input = []
        for i in range(sent.size(1)):
            idx = sent[:, i]
            input.append(self.data['dico']['en'][idx])

        print("input ", input)

    def train(self, n_iter):

        for param in self.transformer.parameters():
            print(param.get_device())

        lang = 0
        get_iterator = self.get_lm_iterator(lang=lang, train=True, add_noise=True)
        train_iterator = get_iterator()
        opt = torch.optim.Adam(self.transformer.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)

        for i in range(n_iter):
            opt.zero_grad()
            batch_dict = next(train_iterator)

            loss = self.reconstruction_loss(batch_dict, lang=lang)

            if i % 50 == 0:
                print("iter ", i, "loss: ", loss)

            loss.backward()
            opt.step()

    def test(self, n_tests):
        self.transformer.eval()
        lang=0
        for i in range(n_tests):
            pass
            # batch, l = self.get_train_batch(0)
            # self.greedy_decoding(batch)
            #loss = self.reconstruction_loss(src_batch=batch, lengths=l, lang=lang)


if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, embd_file="corpora/mono/all.en-fr.60000.vec")
    trainer = LanguageModeling(model)
    trainer.train(3000)
    trainer.save_model("sanity_check.pth")
    trainer.load_model("sanity_check.pth")
    trainer.test(10)
