import torch
from .data.dataset import *
from .data.loader import *
from .data_loading import get_parser
from .data.dictionary import PAD_WORD, EOS_WORD, BOS_WORD

# TODO: switch to new batch shape

class NoiseModel():

    def __init__(self, data, params):

        super(NoiseModel, self).__init__()

        self.data = data
        self.langs = list(data['dico'].keys())
        self.iterators = {}
        self.params = params
        # initialize BPE subwords
        self.init_bpe()

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = []

        # for each language
        for lang in self.langs:
            # get bpe dictionary for that language
            dico = self.data['dico'][lang]

            # for each token in the dictionary, indicate if it does not end with @@
            # indicate which tokens are at the end of a word
            self.bpe_end.append(np.array([not dico[i].endswith('@@') for i in range(len(dico))]))

        print("bpe_end is " + str(self.bpe_end))

    def word_shuffle(self, x, l, lang_id):
        """
        verified
        :param x: batch of sentences shape (max len, batch_size), containing indices of bpe tokens
        :param l: vector of length for eah sentence
        :param lang_id: language of input sentence batch
        :return:
        """
        if self.params.word_shuffle== 0:
            return x, l

        # define noise word scores
        # use x.size(0) - 1 to avoid eos
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        # get the bpe_end boolean indiators from indices in batch x
        bpe_end = self.bpe_end[lang_id][x]

        # count the number of non-end of words,
        # the number in each cell indicates the word (order) index that this token belongs to
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()#

        # for each sentence i in the batch
        for i in range(l.size(0)):
            # generate a random permutation

            # for each element of word_idx for sent i, add noise
            # noise is indexed using word_idx so tokens from the same word get the same score
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)

            # sort the scores, get indices from sorted array
            # since word_idx assigns the same int to tokens of the same word
            # tokens from the same word remain contiguous
            permutation = scores.argsort()

            # get sentence tokens (except bos and eos and padding)
            # use indies from permutation to shuffle the words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])

        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        verified
        :param x: batch of sentences shape (max len, batch_size), containing indices of bpe tokens
        :param l: vector of length for each sentence
        :param lang_id:
        :return:
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        assert (x[0] == self.params.bos_index[lang_id]).sum() == l.size(0)

        # get boolean array for words to keep, with prob (1 - word_dropout)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # index tokens based on which word they belong to
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        # for each sentence in the batch
        # here "words" refers to bpe tokens
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()

            # randomly drop words from the input
            # use word_idx to index keep, to drop entire words
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]

            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                # this is fine, since tokens of the same word have the same number in word_idx
                new_s.append(words[np.random.randint(1, len(words))])

            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == self.params.bos_index[lang_id] and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))

        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)

        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)

        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths, lang_id):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths, lang_id)
        words, lengths = self.word_dropout(words, lengths, lang_id)
        words, lengths = self.word_blank(words, lengths, lang_id)
        return words, lengths

    def test_noise(self, lang):
        """
        Print out unnoised and noised sentences for a language
        """

        lang_id = self.params.lang2id[lang]
        sent1, len1 = self.get_batch('encdec', lang, None)
        print("sent1 before noise is ")
        print(sent1)
        print("len1 before noise is ")
        print(len1)

        sent1, len1 = self.add_noise(sent1, len1, lang_id)

        print('sent1 after noise for ' + lang + ' is')
        print(sent1)
        print('len1 for ' + lang + " is ")
        print(len1)

    def get_iterator(self, iter_name, lang1, lang2, back):
        """
        Create a new iterator for a dataset.
        """
        assert back is False or lang2 is not None
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '')
        # logger.info("Creating new training %s iterator ..." % key)
        if lang2 is None:
            dataset = self.data['mono'][lang1]['train']
        elif back:
            dataset = self.data['back'][(lang1, lang2)]
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k]['train']
        iterator = dataset.get_iterator(shuffle=True, group_by_size=self.params.group_by_size)()
        self.iterators[key] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2, back=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert back is False or lang2 is not None
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '')
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, back)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, back)
            batch = next(iterator)
        return batch if (lang2 is None or lang1 < lang2 or back) else batch[::-1]


def main(params):
    check_all_data_params(params)
    data = load_data(params)
    noiseModel = NoiseModel(data, params)

    # what's in bpe_end
    noiseModel.init_bpe()

    for lang in params.langs:
        noiseModel.test_noise(lang)

    #print(data)

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    print(params)
    main(params)

