from src.data.dataset import *
from src.data.loader import *
from src.model.transformer import Transformer
from src.utils.data_loading import get_parser
from src.utils.logger import create_logger
from .basic_trainer import Trainer


class UnsupervisedTrainer(Trainer):

    def __init__(self, transformer):

        super().__init__(transformer)

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

    def train(self, n_iter):

        lang1 = 0
        lang2 = 1
        logger.info("Training translation model for %s , %s " % (self.id2lang[lang1], self.id2lang[lang2]))

        get_back_para_iterators = [self.get_back_para_iterator(lang1=lang1, lang2=lang2, add_noise=False),
                                   self.get_back_para_iterator(lang1=lang2, lang2=lang1, add_noise=False)]

        get_lm_iterators = [self.get_lm_iterator(lang=lang1, add_noise=True),
                            self.get_lm_iterator(lang=lang2, add_noise=True)]

        back_para_iterators = [get_iter() for get_iter in get_back_para_iterators]
        lm_iterators = [get_iter() for get_iter in get_lm_iterators]

        for i in range(n_iter):

            self.opt.zero_grad()

            try:
                lang1_batch_dict = next(lm_iterators[0])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[0] = self.get_lm_iterator(lang=lang1, add_noise=True)
                lm_iterators[0] = get_lm_iterators[0]()
                lang_batch_dict = next(lm_iterators[0])

            # get lm loss for lang 1
            loss = self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang1, lang2=lang1)

            try:
                lang_batch_dict = next(lm_iterators[1])

            except StopIteration:
                # restart the iterator
                get_lm_iterators[1] = self.get_lm_iterator(lang=lang1, add_noise=True)
                lm_iterators[1] = get_lm_iterators[1]()
                lang_batch_dict = next(lm_iterators[1])

            # get lm loss for lang 2
            loss += self.reconstruction_loss(batch_dict=lang_batch_dict, lang1=lang2, lang2=lang2)

            try:
                para_batch_dict = next(back_para_iterators[0])

            except StopIteration:
                # restart the iterator
                get_back_para_iterators[0] = self.get_back_para_iterator(lang1=lang1, lang2=lang2, add_noise=False)
                back_para_iterators[0] = get_back_para_iterators[0]()
                para_batch_dict = next(back_para_iterators[0])

            loss += self.reconstruction_loss(batch_dict=para_batch_dict, lang1=lang1, lang2=lang2)

            try:
                para_batch_dict = next(back_para_iterators[1])

            except StopIteration:
                # restart the iterator
                get_back_para_iterators[1] = self.get_back_para_iterator(lang1=lang2, lang2=lang1, add_noise=False)
                back_para_iterators[1] = get_back_para_iterators[1]()
                para_batch_dict = next(back_para_iterators[1])

            loss += self.reconstruction_loss(batch_dict=para_batch_dict, lang1=lang2, lang2=lang1)

            try:

                if i % 50 == 0:
                    # print("iter ", i, "loss: ", loss)
                    self.logger.info("iter %i: loss %40.1f" % (i, loss.item()))

                loss.backward()
                self.opt_step()

            except Exception as e:
                self.logger.debug("Exception in training loop")
                self.logger.debug(e.message)

    def generate_parallel(self, batch_dict, src_lang, tgt_lang):
        """
        generate sentences for back-translation using greedy decoding
        :param src_lang:
        :param tgt_lang:
        :return:
        """

        tgt_batch = batch_dict["tgt_batch"]
        src_mask = batch_dict["src_mask"]
        src_batch = batch_dict["src_batch"]

        batch_size = tgt_batch.shape[0]

        latent_code = self.encode(input_seq=src_batch,
                                  src_mask=src_mask,
                                  src_lang=src_lang)

        prev_output = torch.ones(batch_size, self.max_len, dtype=torch.int64) * self.pad_index
        prev_output[:, 0] = self.bos_index
        prev_output = prev_output.to(self.device)

        prev_token = prev_output[:, 0]

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

            # prev_output[:, word_count] = index.item()
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
    logger = create_logger("logs/para_trainer.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger,
                        init_emb=True, embd_file="corpora/mono/all.en-fr.60000.vec")

    trainer = UnsupervisedTrainer(model)
    # test iterator
    # get_iter = trainer.get_para_iterator(lang1=0, lang2=1, train=False, add_noise=False)
    # iter = get_iter()

    # batch_dict = next(iter)
    # prev_output = batch_dict["prev_output"]
    # tgt_mask = batch_dict["tgt_mask"]
    # tgt_batch = batch_dict["tgt_batch"]
    #
    # print("prev_output", prev_output)
    # print("tgt_mask", tgt_mask)
    # print("tgt_batch", tgt_batch)

    trainer.train(10000)
    trainer.save_model("en_fr.pth")
    logger.info("testing trained model")
    trainer.test(10)
    logger.info("testing loaded model")
    trainer.load_model("en_fr.pth")
    trainer.test(10)

if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, embd_file="corpora/mono/all.en-fr.60000.vec")
    trainer = Trainer(model)

    src_batch, l = trainer.get_batch(lang=0)
    trainer.translate(src_batch=src_batch,
                      tgt_batch=None,
                      src_lang=0,
                      tgt_lang=1,
                      beam_size=3,
                      teacher_force=False)
