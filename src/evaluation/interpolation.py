import torch

class Interpolation:

    def __init__(self, transformer, filename):

        self.transformer = transformer.eval()
        self.dictionaries = transformer.dictionaries
        self.id2lang = transformer.id2lang

        self.bos_index = transformer.bos_index
        self.eos_index = transformer.eos_index
        self.pad_index = transformer.pad_index

        self.filename = filename


    def get_endpts(self, sent1, src_mask1, sent2, src_mask2, lang1, lang2):
        """
        returns latent rep (mean) of two batches of sentences
        :param sent1:
        :param sent2:
        :return:
        """

        # TODO: return avg or compute avg,
        z1 = self.transformer.encode(sent1, src_mask1, lang1)
        z2 = self.transformer.encode(sent2, src_mask2, lang2)

        dist = z2 - z1
        eps = 0.1

        while eps < 1:

            z = z1







