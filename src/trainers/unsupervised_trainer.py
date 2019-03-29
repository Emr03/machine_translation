from src.data.dataset import *
from src.data.loader import *
from src.model.transformer import Transformer
from src.utils.data_loading import get_parser
from src.utils.logger import create_logger
from .basic_trainer import Trainer
from src.model.beam_search_wrapper import MyBeamSearch
import copy

class UnsupervisedTrainer(Trainer):

    def __init__(self, transformer, parallel=True):

        super().__init__(transformer, parallel)

    def reconstruction_loss(self, batch_dict, lang1, lang2):

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
                                      tgt_lang=lang2)

        return self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)

    def create_backtranslation_batch(self, batch_dict, src_lang, tgt_lang, add_noise=True):
        """
        translate from src_lang to tgt_lang,
        set the translation as the source and the original as the target

        :param batch_dict: from language modeling
        :return: new batch_dict, with replaced source elements
        """

        # note that bos is missing
        x = batch_dict["tgt_batch"]
        src_mask = self.get_src_mask(x)
        y, len = self.generate_parallel(src_batch=x,
                                        src_mask=src_mask,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang)

        if add_noise:
            y, len = self.noise_model.add_noise(y, len, tgt_lang)

        # only the source elements change
        src_mask = self.get_src_mask(y)
        new_batch_dict = copy.deepcopy(batch_dict)
        new_batch_dict["src_batch"] = y
        new_batch_dict["src_mask"] = src_mask
        new_batch_dict["src_l"] = len

        return new_batch_dict

    def train(self, n_iter):

        lang1 = 0
        lang2 = 1
        logger.info("Training translation model for %s , %s " % (self.id2lang[lang1], self.id2lang[lang2]))

        get_lm_iterators = [self.get_lm_iterator(lang=lang1, add_noise=True),
                            self.get_lm_iterator(lang=lang2, add_noise=True)]

        lm_iterators = [get_iter() for get_iter in get_lm_iterators]

        for i in range(n_iter):

            self.opt.zero_grad()

            try:
                lang_batch_dict = next(lm_iterators[0])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[0] = self.get_lm_iterator(lang=lang1, add_noise=True)
                lm_iterators[0] = get_lm_iterators[0]()
                lang_batch_dict = next(lm_iterators[0])

            # get lm loss for lang 1
            loss = self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang1, lang2=lang1)
            self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                              src_lang=lang1,
                                                              tgt_lang=lang2)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang2, lang2=lang1)
            self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

            try:
                lang_batch_dict = next(lm_iterators[1])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[1] = self.get_lm_iterator(lang=lang1, add_noise=True)
                lm_iterators[1] = get_lm_iterators[1]()
                lang_batch_dict = next(lm_iterators[1])

            # get lm loss for lang 2
            loss += self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang2, lang2=lang2)
            self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                                src_lang=lang2,
                                                                tgt_lang=lang1)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang1, lang2=lang2)
            self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

            try:

                if i % 50 == 0:
                    # print("iter ", i, "loss: ", loss)
                    self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

                loss.backward()
                self.opt_step()

            except Exception as e:
                self.logger.debug("Exception in training loop")
                self.logger.debug(e.message)

    def generate_parallel(self, src_batch, src_mask, src_lang, tgt_lang):
        """
        generate sentences for back-translation using greedy decoding
        :param batch_dict: dict of src batch and src mask
        :param src_lang:
        :param tgt_lang:
        :return:
        """

        batch_size = src_batch.shape[0]
        assert(src_mask.shape[0] == batch_size)

        beam = MyBeamSearch(self.transformer, tgt_lang, beam_size=1, batch_size=batch_size, n_best=2,
                            mb_device=self.device,
                            encoding_lengths=512, max_length=40)

        output, len = beam.perform(src_batch, src_mask, src_lang=src_lang, tgt_lang=tgt_lang)
        return output, len

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
            self.logger.info("translation loss", loss)

if __name__ == "__main__":

    logger = create_logger("logs/unsupervised_trainer.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger,
                        init_emb=True, embd_file="corpora/mono/all.en-fr.60000.vec")

    trainer = UnsupervisedTrainer(model)

    trainer.train(10000)
    trainer.save_model("en_fr_nonpara.pth")
    logger.info("testing trained model")
    trainer.test(10)
    logger.info("testing loaded model")
    trainer.load_model("en_fr_nonpara.pth")
    trainer.test(10)
