from src.data.dataset import *
from src.data.loader import *
from src.model.transformer import Transformer
from src.utils.data_loading import get_parser
from src.utils.logger import create_logger
import logging
from .basic_trainer import Trainer
from src.model.beam_search_wrapper import MyBeamSearch
import copy

class UnsupervisedTrainer(Trainer):

    def __init__(self, transformer, exp_name, acc_steps=1,
                 use_distance_loss=True, parallel=True, load_from_checkpoint=False):

        super().__init__(transformer, parallel)

        self.exp_name = exp_name
        self.use_distance_loss = use_distance_loss
        self.acc_steps = acc_steps

        self.beam_search = MyBeamSearch(self.transformer, beam_size=1, logger=logging,
                                        n_best=1, encoding_lengths=512, max_length=175)

        # if self.parallel:
        #     # self.device is the main device where stuff is aggregated
        #     self.beam_search = torch.nn.DataParallel(self.beam_search)

        # don't make beam search parallel, to avoid gather errors
        self.beam_search.to(self.device)
        self.distance_cost = 0

        if self.is_variational:
            self.logger.info("is variational")
            self.kl_cost = 0
            self.kl_cost_rate = 0.0001

        if load_from_checkpoint:
            self.load_checkpoint(exp_name+".pth")

    def reconstruction_loss(self, batch_dict, lang1, lang2):

        tgt_mask = batch_dict["tgt_mask"]
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]
        prev_output = batch_dict["prev_output"]

        try :

            if self.is_variational:

                # returns decoded samples and kl divergence between prior and posterior
                output_seq, kl_div, latent = self.transformer(input_seq=src_batch,
                                              prev_output=prev_output,
                                              src_mask=src_mask,
                                              tgt_mask=tgt_mask,
                                              src_lang=lang1,
                                              tgt_lang=lang2)

                loss = self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)

                kl_div = torch.mean(kl_div)
                logging.info("kl_div %10.2f, kl_cost %10.5f" % (kl_div.item(), self.kl_cost))
                loss += kl_div*self.kl_cost

            else:

                output_seq = self.transformer(input_seq=src_batch,
                                              prev_output=prev_output,
                                              src_mask=src_mask,
                                              tgt_mask=tgt_mask,
                                              src_lang=lang1,
                                              tgt_lang=lang2)

                loss = self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)

            return loss

        except Exception as e:
            logging.exception("message")

    def distance_loss(self, latent_1, latent_2):
        """

        :param latent_1: avg encoder output for src sent
        :param latent_2: avg decoder output for tgt sent
        :return:
        """
        assert(latent_1.size(1) == latent_2.size(1))
        assert(latent_1.size(0) == latent_2.size(0))

        return torch.sum(torch.norm(latent_2 - latent_1, p=2, dim=-1))

    def create_backtranslation_batch(self, batch_dict, src_lang, tgt_lang, add_noise=True):
        """
        translate from src_lang to tgt_lang,
        set the translation as the source and the original as the target

        :param batch_dict: from language modeling
        :return: new batch_dict, with replaced source elements
        """

        # note that bos is missing, works better that way (???)
        x = batch_dict["tgt_batch"]

        src_mask = self.get_src_mask(x)
        y, len = self.generate_parallel(src_batch=x,
                                        src_mask=src_mask,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang)

        # we have to penalize the distance between the source's emb and the output's emb
        distance_penalty = 0
        if self.use_distance_loss:
            src_z = self.transformer.module.get_emb(input_seq=x,
                                            src_mask=src_mask,
                                            src_lang=src_lang)

            tgt_z = self.transformer.module.get_emb(input_seq=y,
                                            src_mask=self.get_src_mask(y),
                                            src_lang=tgt_lang)

            distance_penalty = self.distance_loss(src_z, tgt_z) * self.distance_cost
            self.logger.info("distance penalty %40.2f" % (distance_penalty.item()))

        if add_noise:
            y, len = self.noise_model.add_noise(y.cpu(), len.cpu(), tgt_lang)
            y = y.to(self.device)
            len = len.to(self.device)

        # only the source elements change
        src_mask = self.get_src_mask(y)
        new_batch_dict = copy.deepcopy(batch_dict)
        new_batch_dict["src_batch"] = y
        new_batch_dict["src_mask"] = src_mask
        new_batch_dict["src_l"] = len

        return new_batch_dict, distance_penalty

    def train(self, n_iter):

        lang1 = 0
        lang2 = 1
        self.logger.info("Training translation model for %s , %s " % (self.id2lang[lang1], self.id2lang[lang2]))

        get_lm_iterators = [self.get_lm_iterator(lang_id=lang1, add_noise=True),
                            self.get_lm_iterator(lang_id=lang2, add_noise=True)]

        lm_iterators = [get_iter() for get_iter in get_lm_iterators]

        get_para_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
        para_iterator = get_para_iterator()

        self.opt.zero_grad()
        for i in range(n_iter):

            if self.is_variational:
                self.kl_cost = min(1, self.step*self.kl_cost_rate)

            if self.use_distance_loss:
                self.distance_cost = self.kl_cost

            try:
                lang_batch_dict = next(lm_iterators[0])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[0] = self.get_lm_iterator(lang_id=lang1, add_noise=True)
                lm_iterators[0] = get_lm_iterators[0]()
                lang_batch_dict = next(lm_iterators[0])

            # get lm loss for lang 1
            loss = self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang1, lang2=lang1)
            logging.info("iter %i: reconstruction loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict, distance_loss = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                              src_lang=lang1,
                                                              tgt_lang=lang2)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang2, lang2=lang1) + distance_loss
            logging.info("iter %i: back translation loss %40.1f" % (i, loss.item()))

            try:
                lang_batch_dict = next(lm_iterators[1])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[1] = self.get_lm_iterator(lang_id=lang1, add_noise=True)
                lm_iterators[1] = get_lm_iterators[1]()
                lang_batch_dict = next(lm_iterators[1])

            # get lm loss for lang 2
            loss += self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang2, lang2=lang2)
            logging.info("iter %i: reconstruction loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict, distance_loss = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                                src_lang=lang2,
                                                                tgt_lang=lang1)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang1, lang2=lang2) + distance_loss
            logging.info("iter %i: backtranslation loss %40.1f" % (i, loss.item()))

            try:
                loss.backward()

                # only update params and zero grads after we process a whole batch
                if i % self.acc_steps == 0:
                    self.opt_step()
                    self.opt.zero_grad()

            except Exception as e:
                logging.debug("Exception in training loop")
                logging.exception("message")

            if i % 200 == 0:
                # print("iter ", i, "loss: ", loss)
                logging.info("iter %i: loss %40.1f" % (i, loss.item()))
                trainer.checkpoint(self.exp_name+".pth")

            try:
                para_batch_dict = next(para_iterator)

            except StopIteration:
                # restart the iterator
                get_para_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=True)
                para_iterator = get_para_iterator()
                para_batch_dict = next(para_iterator)

            val_loss = self.reconstruction_loss(para_batch_dict, lang1=lang1, lang2=lang2)
            logging.info("iter %i: val_loss %40.1f" % (i, val_loss.item()))

    def generate_parallel(self, src_batch, src_mask, src_lang, tgt_lang):
        """
        generate sentences for back-translation using greedy decoding
        :param batch_dict: dict of src batch and src mask
        :param src_lang:
        :param tgt_lang:
        :return:
        """
        output, len = self.beam_search(src_batch, src_mask, src_lang=src_lang, tgt_lang=tgt_lang)

        # For verification, what does an output sample look like?
        self.indices_to_words(output[0, :].unsqueeze_(0), tgt_lang)
        self.logger.info("reference: ")
        self.indices_to_words(src_batch[0, :].unsqueeze_(0), src_lang)

        return output, len

    def indices_to_words(self, sent, lang):
        """

        :param sent: 1 x len tensor of a sample sentence, holds indices of words
        :lang: language id
        :return: prints sentence words
        """
        input = []
        for i in range(sent.size(1)):
            idx = sent[:, i].item()
            input.append(self.data['dico'][self.id2lang[lang]][idx])

        self.logger.info("sample sentence in lang %s: %s" % (self.id2lang[lang], ','.join(input)))

    def test(self, n_tests):
        self.transformer.eval()
        lang1 = 0
        lang2 = 1
        get_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
        train_iterator = get_iterator()
        for i in range(n_tests):
            batch_dict = next(train_iterator)
            # self.greedy_decoding(batch_dict, lang1, lang2)
            self.output_samples(batch_dict, lang1, lang2)
            loss = self.translation_loss(batch_dict, lang1, lang2)
            logging.info("translation loss", loss)

if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    exp_name = data_params.exp_name
    is_variational = data_params.variational > 0
    use_distance_loss = data_params.use_distance_loss > 0
    load_from_checkpoint = data_params.load_from_checkpoint > 0

    logging.basicConfig(filename="logs/"+exp_name+".log", level=logging.DEBUG)

    model = Transformer(data_params=data_params, logger=logging,
                        init_emb=True,
                        embd_file="corpora/mono/all.en-fr.60000.vec",
                        is_variational=is_variational)

    trainer = UnsupervisedTrainer(model, exp_name,
                                 use_distance_loss=use_distance_loss,
                                 load_from_checkpoint=load_from_checkpoint)

    trainer.train(2*50000)
    trainer.checkpoint(exp_name+".pth")
