from src.model.transformer import *
from src.onmt.translate.beam import *
from src.onmt.translate.beam_search import BeamSearch
from logging import getLogger
from src.utils.beam_search_utils import tile

class MyBeamSearch:
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
    def __init__(self, beam_size, batch_size, n_best,
                 mb_device, encoding_lengths, max_length):

        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_length = max_length

        self.pad_index = transformer.pad_index
        self.eos_index = transformer.eos_index
        self.bos_index = transformer.bos_index
        self.id2lang = transformer.id2lang

        #pad, bos, and eos are based on values from Dictionary.py.
        # GMTGlobalScorer for length penalty
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
    def perform(self, transformer, batch, src_mask, src_lang, tgt_lang):

        assert(batch.size(0) == self.batch_size)
        assert(len(batch.shape) == 2)

        # disable gradient tracking
        with torch.set_grad_enabled(False):

            # (1) Run the encoder on the src.
            enc_out = transformer.encode(batch, src_mask, src_lang)

            # (2) Repeat src objects `beam_size` times. along dim 1
            # We use batch_size x beam_size
            enc_out = tile(enc_out, self.beam_size, dim=0)
            src_mask = tile(src_mask, self.beam_size, dim=0)
            print("enc out", enc_out.shape)

            # dec_output should be batch_size x beam_size, dec_seq_len
            # in this first case it should be batch_size x 1 x hidden_size since it's just the first word generated
            dec_out = torch.ones(self.batch_size*self.beam_size, 1, dtype=torch.int64)*self.bos_index

            for step in range(self.max_length):

                # decoder_input = self.beamSearch.current_predictions.view(-1, 1)
                # print("decoder_input", decoder_input.shape)

                # in case of inference tgt_len = 1, batch = beam times batch_size
                log_probs = transformer.decode(dec_out, enc_out, src_mask,
                                               tgt_mask=None, tgt_lang=tgt_lang)[:, -1, :]

                log_probs = F.log_softmax(log_probs, dim=-1)
                print("log probs", log_probs.shape)

                #advance takes input of size batch_size*beam_size x vocab_size
                self.beamSearch.advance(log_probs, None)

                next_word = self.beamSearch.current_predictions.view(self.batch_size*self.beam_size, -1)
                #dec out should be batch_size x (previous_sentence_len + 1) x hidden_size
                dec_out = torch.cat((dec_out, next_word), 1)
                #print("current predictions shape is " + str(self.beamSearch.current_predictions))
                print("dec out", dec_out)

                # indices indicate which beams correspond to the same batch
                select_indices = self.beamSearch.current_origin
                print("select indices", select_indices)

                # checks if any beam is finished, then updates state.
                any_beam_is_finished = self.beamSearch.is_finished.any()
                if any_beam_is_finished:
                    self.beamSearch.update_finished()
                    # done if all beams are finished
                    if self.beamSearch.done:
                        break

        return self.beamSearch.hypotheses

if __name__ == "__main__":

    from src.utils.config import params
    #batch_size x seq_len
    x = torch.zeros(2, 5, dtype=torch.int64)

    src_m = torch.ones(2, 5)
    src_m[:, -2:-1] = 0
    src_m = src_m.unsqueeze(-2).unsqueeze(-2)

    # parser = get_parser()
    # data_params = parser.parse_args()
    # check_all_data_params(data_params)
    transformer = Transformer(data_params=None, logger=getLogger(), embd_file=None).eval()

    beam = MyBeamSearch(beam_size=3, batch_size=2, n_best=2,
                        mb_device=torch.device("cpu"),
                        encoding_lengths=512, max_length=40)

    print(beam.perform(transformer.eval(), x, src_m, src_lang=1, tgt_lang=1))
