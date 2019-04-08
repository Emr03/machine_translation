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

        self.beam_search = MyBeamSearch(self.transformer, beam_size=1, logger=self.logger,
                                        n_best=1, encoding_lengths=512, max_length=175)

        if self.parallel:
            # self.device is the main device where stuff is aggregated
            self.beam_search = torch.nn.DataParallel(self.beam_search)

        self.beam_search.to(self.device)

        if self.is_variational:
            self.kl_cost = 0
            self.kl_cost_rate = 0.0001

    def reconstruction_loss(self, batch_dict, lang1, lang2):

        tgt_mask = batch_dict["tgt_mask"]
        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]
        prev_output = batch_dict["prev_output"]

        if self.is_variational:

            # returns decoded samples and kl divergence between prior and posterior
            output_seq, kl_div = self.transformer(input_seq=src_batch,
                                          prev_output=prev_output,
                                          src_mask=src_mask,
                                          tgt_mask=tgt_mask,
                                          src_lang=lang1,
                                          tgt_lang=lang2)

            loss = self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)
            loss += self.kl_cost*kl_div

        else:

            output_seq = self.transformer(input_seq=src_batch,
                                          prev_output=prev_output,
                                          src_mask=src_mask,
                                          tgt_mask=tgt_mask,
                                          src_lang=lang1,
                                          tgt_lang=lang2)

            loss = self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)

        return loss

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

        get_lm_iterators = [self.get_lm_iterator(lang_id=lang1, add_noise=True),
                            self.get_lm_iterator(lang_id=lang2, add_noise=True)]

        lm_iterators = [get_iter() for get_iter in get_lm_iterators]

        get_para_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=False)
        para_iterator = get_para_iterator()

        for i in range(n_iter):

            self.opt.zero_grad()

            if self.is_variational:
                self.kl_cost = min(1, i*self.kl_cost_rate)

            try:
                lang_batch_dict = next(lm_iterators[0])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[0] = self.get_lm_iterator(lang_id=lang1, add_noise=True)
                lm_iterators[0] = get_lm_iterators[0]()
                lang_batch_dict = next(lm_iterators[0])

            # get lm loss for lang 1
            loss = self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang1, lang2=lang1)
            self.logger.info("iter %i: reconstruction loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                              src_lang=lang1,
                                                              tgt_lang=lang2)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang2, lang2=lang1)
            self.logger.info("iter %i: back translation loss %40.1f" % (i, loss.item()))

            try:
                lang_batch_dict = next(lm_iterators[1])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[1] = self.get_lm_iterator(lang_id=lang1, add_noise=True)
                lm_iterators[1] = get_lm_iterators[1]()
                lang_batch_dict = next(lm_iterators[1])

            # get lm loss for lang 2
            loss += self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang2, lang2=lang2)
            self.logger.info("iter %i: reconstruction loss %40.1f" % (i, loss.item()))

            # the same for back-translation
            back_batch_dict = self.create_backtranslation_batch(batch_dict=lang_batch_dict,
                                                                src_lang=lang2,
                                                                tgt_lang=lang1)

            loss += self.reconstruction_loss(batch_dict=back_batch_dict, lang1=lang1, lang2=lang2)
            self.logger.info("iter %i: backtranslation loss %40.1f" % (i, loss.item()))

            try:
                loss.backward()
                self.opt_step()

            except Exception as e:
                self.logger.debug("Exception in training loop")
                self.logger.debug(e.message)

            try:

                if i % 50 == 0:
                    # print("iter ", i, "loss: ", loss)
                    self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

                    # TODO: add validation (with parallel and non-parallel)
                    para_batch_dict = next(para_iterator)

            except StopIteration:
                # restart the iterator
                get_para_iterator = self.get_para_iterator(lang1=lang1, lang2=lang2, train=False, add_noise=True)
                para_iterator = get_para_iterator()
                para_batch_dict = next(para_iterator)

            val_loss = self.reconstruction_loss(para_batch_dict, lang1=lang1, lang2=lang2)
            self.logger.info("iter %i: val_loss %40.1f" % (i, val_loss.item()))

    def generate_parallel(self, src_batch, src_mask, src_lang, tgt_lang):
        """
        generate sentences for back-translation using greedy decoding
        :param batch_dict: dict of src batch and src mask
        :param src_lang:
        :param tgt_lang:
        :return:
        """

        batch_size = src_batch.shape[0]
        print("batch size in trainer ", batch_size)
        #assert(src_mask.shape[0] == batch_size)

        output, len = self.beam_search(src_batch, src_mask, src_lang=src_lang, tgt_lang=tgt_lang)

        # For verification, what does an output sample look like?
        self.indices_to_words(output[0, :].unsqueeze_(0), tgt_lang)
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

        logger.info("sample sentence in lang %s: %s" % (self.id2lang[lang], ','.join(input)))

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

    logger = create_logger("logs/variational.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger,
                        init_emb=True, embd_file="corpora/mono/all.en-fr.60000.vec", is_variational=True)

    trainer = UnsupervisedTrainer(model)

    trainer.train(10000)
    trainer.save_model("en_fr_nonpara.pth")
    logger.info("testing trained model")
    trainer.test(10)
    logger.info("testing loaded model")
    trainer.load_model("en_fr_nonpara.pth")
    trainer.test(10)
