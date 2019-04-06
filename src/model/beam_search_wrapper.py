from src.model.transformer import *
from src.onmt.translate.beam import *
from src.onmt.translate.beam_search import BeamSearch
from logging import getLogger
from src.utils.beam_search_utils import tile

class MyBeamSearch(torch.nn.Module):
    '''
    Wrapper around OpenNMT beam search that suits our purposes
        dico: the dictionary of vocabulary
        beam_size: beam size parameter for beam search
        batch_size: batch size
        n_best: don't stop until we reached n_best sentences (sentences that hit EOS)
        mb_device: the type of device. See https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        encoding_lengths: LongTensor of encoding lengths
        max_length: Longest acceptable sequence, not counting begin-of-sentence (presumably there has been no EOS yet if max_length is used as a cutoff)
    '''
    def __init__(self, transformer, tgt_lang, beam_size, batch_size, n_best,
                 mb_device, encoding_lengths, max_length):

        super(MyBeamSearch, self).__init__()
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_length = max_length

        self.pad_index = transformer.module.pad_index
        self.eos_index = transformer.module.eos_index
        self.bos_index = transformer.module.bos_index[tgt_lang]
        self.id2lang = transformer.module.id2lang
        self.transformer = transformer.module
        self.device = mb_device

        #pad, bos, and eos are based on values from Dictionary.py.
        # GMTGlobalScorer for length penalty
        # TODO: what is n_best?
        self.beamSearch = BeamSearch(beam_size, batch_size,
                                     pad=self.pad_index, bos=self.bos_index,
                                     eos=self.eos_index,
                                     n_best=n_best, mb_device=mb_device,
                                     global_scorer=GNMTGlobalScorer(0.7, 0., "avg", "none"),
                                     min_length=0, max_length=max_length, return_attention=False,
                                     block_ngram_repeat=0,
                                     exclusion_tokens=set(),
                                     memory_lengths=encoding_lengths,
                                     stepwise_penalty=False, ratio=0.)

    '''
    Performs beam search on a batch of sequences
    Adapted from _translate_batch in translator.py from onmt
    Returns: hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    '''
    def forward(self, batch, src_mask, src_lang, tgt_lang, random=False):

        assert(batch.size(0) == self.batch_size)
        transformer = self.transformer.eval()

        # disable gradient tracking
        with torch.set_grad_enabled(False):

            # (1) Run the encoder on the src.
            enc_out = transformer.encode(batch, src_mask, src_lang)

            # (2) Repeat src objects `beam_size` times. along dim 0
            # We use batch_size x beam_size
            enc_out = enc_out.repeat(self.beam_size, 1, 1)
            src_mask = src_mask.repeat(self.beam_size, 1, 1, 1)
            #print("enc out", enc_out[:, :, 0])

            # dec_output should be batch_size x beam_size, dec_seq_len
            # in this first case it should be batch_size x 1 x hidden_size since it's just the first word generated
            dec_out = torch.ones(self.batch_size*self.beam_size, 1, dtype=torch.int64)*self.bos_index
            dec_out = dec_out.to(self.device)

            # sanity check
            # print("sanity check")
            # print(self.beamSearch.current_predictions)
            # print(self.beamSearch.current_origin)
            # dec_out = self.beamSearch.current_predictions.view(-1, 1)

            for step in range(self.max_length):

                # decoder_input = self.beamSearch.current_predictions.view(-1, 1)
                # print("decoder_input", decoder_input.shape)

                # in case of inference tgt_len = 1, batch = beam times batch_size
                log_probs = transformer.decode(dec_out, enc_out, src_mask,
                                               tgt_mask=None, tgt_lang=tgt_lang)[:, -1, :]

                log_probs = F.log_softmax(log_probs, dim=-1)
                #print("log probs", log_probs.shape)

                #advance takes input of size batch_size*beam_size x vocab_size
                self.beamSearch.advance(log_probs, None)

                # check if any beam is finished (last output selected was eos)
                any_beam_is_finished = self.beamSearch.is_finished.any()
                if any_beam_is_finished:
                    self.beamSearch.update_finished()
                    if self.beamSearch.done:
                        break

                # get chosen words by beam search
                next_word = self.beamSearch.current_predictions.view(self.batch_size*self.beam_size, -1)

                # get indices of expanded nodes, for each input sentence
                select_indices = self.beamSearch.current_origin
                #print("select_indices", select_indices)

                # select previous output of expanded nodes
                dec_out = dec_out[select_indices]
                #print("dec_out", dec_out)

                #dec out should be batch_size x (previous_sentence_len + 1) x hidden_size
                dec_out = torch.cat((dec_out, next_word), 1)
                #print("current predictions" + str(self.beamSearch.current_predictions))
                #print("dec out", dec_out)

        # (batch_size) list of (beam_size) lists of tuples
        hypotheses = self.beamSearch.hypotheses
        sentences, len = self.format_sentences(hypotheses=hypotheses)
        return sentences, len

    def format_sentences(self, hypotheses, random=False):
        """

        :param hypotheses: list of lists
        :param random:
        :return:
        """

        # get lengths of sentences
        if random:
            indices = np.random.randint(low=0, high=self.beam_size, size=self.batch_size)
            sentences = list(map(lambda beams: beams[indices[i]][1]) for i, beams in enumerate(hypotheses))
            lengths = torch.LongTensor([s[indices[i]].shape[0] + 2 for i, s in enumerate(hypotheses)])

        else:
            sentences = list(map(lambda beams: beams[-1][1], hypotheses))
            lengths = torch.LongTensor([s.shape[0] + 2 for s in sentences])

        # fill unused sentence spaces with pad token
        sent = torch.LongTensor(lengths.size(0), lengths.max()).fill_(self.pad_index)

        # copy sentence tokens, don't overwrite bos, add eos
        for i, s in enumerate(sentences):
            sent[i, 1:lengths[i] - 1, ].copy_(s)
            sent[i, lengths[i] - 1] = self.eos_index

        return sent, lengths


if __name__ == "__main__":

    from src.utils.config import params
    #batch_size x seq_len
    x = torch.zeros(2, 5, dtype=torch.int64)
    x[1, :] = torch.ones(5, dtype=torch.int64)

    src_m = torch.ones(2, 5)
    src_m[:, -2:-1] = 0
    src_m = src_m.unsqueeze(-2).unsqueeze(-2)

    # parser = get_parser()
    # data_params = parser.parse_args()
    # check_all_data_params(data_params)
    transformer = Transformer(data_params=None, logger=getLogger(), embd_file=None).eval()

    beam = MyBeamSearch(transformer, beam_size=3, batch_size=2, n_best=2,
                        mb_device=torch.device("cpu"),
                        encoding_lengths=512, max_length=40)

    sent, len = beam.perform(x, src_m, src_lang=1, tgt_lang=1)
    print(sent, len)
