import torch.nn.functional as F
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

    def reconstruction_loss(self, src_batch, lengths, lang, noise=True):

        tgt_mask = self.get_tgt_mask(src_batch)
        tgt_batch = torch.copy_(src_batch)

        if noise:
            src_batch, new_len = self.noise_model.add_noise(src_batch, lengths, lang)

        src_mask = self.get_src_mask(src_batch)

        output_seq = self.transformer(input_seq=src_batch,
                                  prev_output=tgt_batch,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_lang=lang,
                                  tgt_lang=lang)

        return F.cross_entropy(input=torch.flatten(output_seq, 0, 1),
                               target=torch.flatten(src_batch))

    def train(self, n_iter):

        print("transformer ", self.transformer)
        print("transformer parameter list", next(self.transformer.parameters()))
        print("encoder parameter list ", next(self.transformer.encoder.parameters()))
        print("decoder parameter list ", next(self.transformer.decoder.parameters()))

        opt = torch.optim.Adam(self.transformer.parameters(), lr=0.0001)
        lang = 0
        for i in range(n_iter):
            opt.zero_grad()
            batch, l = self.get_train_batch(lang)
            loss = self.reconstruction_loss(src_batch=batch, lengths=l, lang=lang)
            print("iter ", i, "loss: ", loss)
            loss.backward()
            opt.step()

if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, embd_file="corpora/mono/all.en-fr.60000.vec")
    trainer = LanguageModeling(model)
    trainer.train(100)




