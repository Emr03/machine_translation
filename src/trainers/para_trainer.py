from src.data.dataset import *
from src.data.loader import *
from src.model.transformer import Transformer
from src.utils.data_loading import get_parser
from src.utils.logger import create_logger
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
                                      tgt_lang=lang2)

        return self.compute_kl_div_loss(x=output_seq, target=tgt_batch, lang=lang2)


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
            loss = self.translation_loss(batch_dict, lang1, lang2)
            self.logger.info("translation loss", loss)


if __name__ == "__main__":

    logger = create_logger("logs/para_trainer.log")
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, logger=logger,
                        init_emb=True, embd_file="corpora/mono/all.en-fr.60000.vec")

    trainer = ParallelTrainer(model)

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
