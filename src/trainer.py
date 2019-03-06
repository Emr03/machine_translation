import torch
import torch.nn.functional as F
import numpy as np
from .transformer import Transformer
from .noise_model import NoiseModel

class Trainer:

    def __init__(self, transformer):

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

    def back_translation_loss(self, src_batch, lengths, tgt_batch, src_lang, tgt_lang, noise=True):

        if noise:
            corr_translations = self.noise_model.add_noise(src_batch, lengths=lengths, lang_id=src_lang)

        else:
            corr_translations = src_batch

        # compute back-translation
        back_translations = self.translate(src_batch=corr_translations,
                                           tgt_batch=tgt_batch,
                                           src_lang=src_lang,
                                           tgt_lang=tgt_lang,
                                           beam_size=1)

        # compute loss
        return F.cross_entropy(input=torch.flatten(back_translations, 0, 1),
                               target=torch.flatten(src_batch))

    def translate(self, src_batch, tgt_batch, src_lang, tgt_lang, beam_size, teacher_force=False):

        batch_size = src_batch.size(0)
        if tgt_batch is None:
            tgt_batch = torch.new_full(size=batch_size, fill_value=self.pad_index)
            tgt_batch[:, 0] = self.bos_index

        tgt_mask = self.get_tgt_mask(tgt_batch)
        src_mask = self.get_src_mask(src_batch)

        if teacher_force:
            output = self.transformer(input_seq=src_batch,
                                      prev_output=tgt_batch,
                                      src_mask=src_mask,
                                      tgt_mask=tgt_mask,
                                      src_lang=src_lang,
                                      tgt_lang=tgt_lang)

            #return F.softmax(output, dim=-1)

        else:
            enc_out = self.transformer.encode(input_seq=src_batch,
                                              src_mask=src_mask,
                                              src_lang=src_lang)

            self.beam_search(beam_size, enc_out, src_mask, tgt_lang)

    def beam_search(self, beam_size, encoder_outputs, src_mask, tgt_lang):

        batch_size = encoder_outputs.size(0)
        last_token = self.bos_index
        hypotheses = torch.new_full(size=(batch_size, beam_size, self.max_len),
                                     fill_value=self.pad_index)
        hypotheses [:, :, 0] = self.bos_index

        for sent in range(batch_size):
            prev_token = self.bos_index
            curr_len = 1
            while prev_token is not self.eos_index:

                tgt_mask = self.get_tgt_mask(hypotheses[sent, :, :])
                out = self.transformer.decode(prev_output=hypotheses[sent, :, :],
                                        latent_seq=encoder_outputs,
                                        src_mask=src_mask,
                                        tgt_mask=tgt_mask,
                                        tgt_lang=tgt_lang)

                print("out", out)

                scores = F.softmax(out, dim=-1)
                scores.topk(beam_size, dim=-1, largest=True, sorted=True))
                print("scores", scores)

                # use embedding layer of decoder to map output indices back to word embeddings



        


    def train_loop(self, train_iter):
        pass

    def get_batch(self, lang):

        get_iterator = self.train_iterators[lang]
        iterator = get_iterator()

        batch, l = next(iterator)
        print(batch, l)
        batch = batch.transpose_(0, 1)
        return batch, l

if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, embd_file="corpora/mono/all.en-fr.60000.vec")
    trainer = Trainer(model)

    src_batch = trainer.get_batch(lang=0)
    trainer.translate(src_batch=src_batch,
                      tgt_batch=None,
                      src_lang=0,
                      tgt_lang=1,
                      beam_size=3,
                      teacher_force=False)