from .config import params
from .encoder import *
from .decoder import *
from .sublayers import *
from .noise_model import *
from .data.loader import *
from .data_loading import get_parser
from .data.dataset import *
from .pretrain_embeddings import *


class Transformer(torch.nn.Module):

    def __init__(self, data_params, logger,  embd_file=None, init_emb=True, is_shared_emb=True):
        """
        :param n_langs: number of supported languages
        :param is_shared_emb: languages use shared embeddings
        """
        super(Transformer, self).__init__()
        assert (type(is_shared_emb) is bool)

        self.logger = logger
        self.d_model = params["d_model"]
        self.n_layers = params["n_layers"]
        self.dff = params["dff"]
        self.d_k = params["d_k"]

        self.is_shared_emb = is_shared_emb

        # will be set in load_data
        self.data = None
        self.bos_index = None
        self.eos_index = None
        self.pad_index = None
        self.noise_model = None
        self.languages = None
        self.dictionaries = None
        self.vocab_size = None

        self.load_data(data_params)
        self.n_langs = len(self.languages)

        self.encoder = StackedEncoder(n_layers=self.n_layers,
                                      params=params,
                                      n_langs=self.n_langs,
                                      vocab_size=self.vocab_size,
                                      is_shared_emb=is_shared_emb,
                                      freeze_emb=init_emb)

        self.decoder = StackedDecoder(n_layers=self.n_layers,
                                      params=params,
                                      n_langs=self.n_langs,
                                      vocab_size=self.vocab_size,
                                      is_shared_emb=is_shared_emb,
                                      freeze_emb=init_emb)

        linear = torch.nn.Linear(self.d_model, self.vocab_size[0])

        if self.is_shared_emb:
            self.linear_layers = torch.nn.ModuleList([linear for _ in range(self.n_langs)])

        else:
            self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(self.d_model, self.vocab_size[l]) for l in range(self.n_langs)])

        def init_weights(m):

            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)

                if m.bias is not None:
                    torch.nn.init.constant(m.bias, 0)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        if init_emb and embd_file is not None:
            self.initialize_embeddings(embedding_file=embd_file)

    def encode(self, input_seq, src_mask, src_lang):

        return self.encoder(input_seq, src_mask=src_mask, lang_id=src_lang)

    def decode(self, prev_output, latent_seq, src_mask, tgt_mask, tgt_lang):

        return self.decoder(prev_output, latent_seq, src_mask=src_mask, tgt_mask=tgt_mask, lang_id=tgt_lang)

    def forward(self, input_seq, prev_output, src_mask, tgt_mask, src_lang, tgt_lang):

        latent = self.encode(input_seq, src_mask, src_lang)

        dec_outputs = self.decode(prev_output=prev_output,
                                  latent_seq=latent,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  tgt_lang=tgt_lang)

        return self.linear_layers[tgt_lang](dec_outputs)

    def load_data(self, data_params):

        all_data = load_data(data_params)
        self.data = all_data
        self.data_params = data_params
        print(all_data)

        self.languages = list(all_data['dico'].keys())
        self.id2lang = {i: lang for i, lang in enumerate(self.languages)}

        self.mono_data_train = [all_data['mono'][self.languages[0]]['train'],
                                all_data['mono'][self.languages[1]]['train']]

        #self.mono_data_valid = [all_data['mono'][self.languages[0]]['valid'],
        #                        all_data['mono'][self.languages[1]]['valid']]

        self.dictionaries = all_data['dico']
        self.vocab_size = [len(self.dictionaries[l].word2id) for l in self.languages]

        # by construction, special indices are the same for all languages
        self.pad_index = data_params.pad_index
        self.eos_index = data_params.eos_index
        self.bos_index = data_params.bos_index

        print("pad_index", self.pad_index)
        print("eos_index", self.eos_index)
        print("bos_index", self.bos_index)
        print("unk_index", data_params.unk_index)
        print("blank_index", data_params.blank_index)

        self.train_iterators = [self.mono_data_train[l].get_iterator(shuffle=True, group_by_size=True)
                                for l in range(len(self.languages))]

        #self.val_iterators = [self.mono_data_valid[l].get_iterator(shuffle=True, group_by_size=True)
        #                      for l in range(len(self.languages))]

    def initialize_embeddings(self, embedding_file):

        if embedding_file == '':
            return

        split = embedding_file.split(',')

        # for shared embeddings
        if len(split) == 1:
            assert os.path.isfile(embedding_file)
            pretrained_0, word2id_0 = reload_embeddings(embedding_file, self.d_model)

            # replicate shared embeddings for each language
            pretrained = [pretrained_0 for _ in range(self.n_langs)]

            # replicate shared dictionary for each language
            word2id = [word2id_0 for _ in range(self.n_langs)]

        else:
            assert len(split) == self.n_langs
            assert not self.is_shared_emb
            assert all(os.path.isfile(x) for x in split)
            pretrained = []
            word2id = []
            for path in split:
                pretrained_i, word2id_i = reload_embeddings(path, self.emb_dim)
                pretrained.append(pretrained_i)
                word2id.append(word2id_i)

        found = [0 for _ in range(self.n_langs)]
        lower = [0 for _ in range(self.n_langs)]

        # for every language
        for i, lang in enumerate(self.languages):

            # if shared embeddings across languages, just do this once
            if self.is_shared_emb and i > 0:
                break

            # define dictionary / parameters to update
            dico = self.data['dico'][lang]

            # update the embedding layer of the encoder & decoder, for language i
            to_update = [self.encoder.embedding_layers[i].weight.data]
            to_update.append(self.decoder.embedding_layers[i].weight.data)
            to_update.append(self.linear_layers[i].weight.data)

            # for every word in that language
            for word_id in range(self.vocab_size[i]):
                word = dico[word_id]

                # if word is in the dictionary of that language
                if word in word2id[i]:

                    # count the number of words found for each language
                    found[i] += 1

                    # get the embedding vector for that word
                    vec = torch.from_numpy(pretrained[i][word2id[i][word]])

                    # for each embedding layer to update
                    # set the word_id's word vector to vec
                    for x in to_update:
                        x[word_id] = vec

                # if word requires lowercasing
                elif word.lower() in word2id[i]:
                    found[i] += 1
                    lower[i] += 1
                    vec = torch.from_numpy(pretrained[i][word2id[i][word.lower()]])
                    for x in to_update:
                        x[word_id] = vec

        # print summary
        for i in range(self.n_langs):
            _found = found[0 if self.is_shared_emb  else i]
            _lower = lower[0 if self.is_shared_emb  else i]
            self.logger.info(
                "Initialized %i / %i word embeddings for \"%s\" (including %i "
                "after lowercasing)." % (_found, self.vocab_size[i], i, _lower)
            )

if __name__ == "__main__":
    # test transformer

    # x = torch.zeros(20, 5, dtype=torch.int64)
    # y = torch.zeros(20, 7, dtype=torch.int64)
    # tgt_m = np.tril(np.ones((1, 7, 7)), k=0).astype(np.uint8)
    # tgt_m = torch.from_numpy(tgt_m)
    #
    # src_m = torch.zeros(20, 5).unsqueeze(-2).unsqueeze(-2)

    # out = model(input_seq=x, prev_output=y, src_mask=src_m, tgt_mask=tgt_m, src_lang=1, tgt_lang=0)
    # print(out.shape)

    # test lm_loss
    # x = torch.ones(2, 5, dtype=torch.int64)
    # x[:, -2:] = model.pad_index
    # loss = model.lm_loss(x, 1)
    # print(loss)

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    model = Transformer(data_params=data_params, embd_file="corpora/mono/all.en-fr.60000.vec")

    batch, l = model.get_batch(lang=0)
    print("batch", batch)
    print("l", l)
    loss = model.lm_loss(src_batch=batch, lengths=l, lang=0)
    print(loss)
