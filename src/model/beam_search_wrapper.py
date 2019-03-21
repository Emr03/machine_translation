from src.model.transformer import *
from src.onmt.translate.beam import *
from src.onmt.translate.beam_search import BeamSearch
from src.utils.logger import *

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
    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best,
                 mb_device, encoding_lengths, max_length, src_lang_id, tgt_lang_id):

        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_length = max_length
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        #pad, bos, and eos are based on values from Dictionary.py.
        # GMTGlobalScorer for length penalty
        self.beamSearch = BeamSearch(beam_size, batch_size, pad=pad, bos=bos, eos=eos,
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
    def perform(self, transformer, batch, src_mask):

        encoder = transformer.encoder
        decoder = transformer.decoder

        enc_out = encoder(batch, src_mask, self.src_lang_id)
        # dec_output should be batch_size x dec_seq_len x hidden_size
        dec_out = decoder(batch, enc_out, src_mask, None)
        #log_probs should be batch_size x dec_seq_len x vocab_size
        log_probs = torch.log(transformer.decode(batch, enc_out, src_mask, None, self.tgt_lang_id))
        for step in range(self.max_length):
            #advance takes something of size batch_size * beam_size, 1
            #attention is only used for coverage penalty, which we're not using
            self.beamSearch.advance(log_probs, None)
            print("current predictions shape is " + str(self.beamSearch.current_predictions.shape))

            # checks if any beam is finished, then updates state.
            any_beam_is_finished = self.beam.is_finished.any()
            if any_beam_is_finished:
                self.beamSearch.update_finished()
                # done if all beams are finished
                if self.beamSearch.done:
                    break

            #pass in stuff to decoder, get log probabilities and new decoder states back
            dec_out = decoder(dec_out, enc_out, src_mask, tgt_mask=None,
                                 lang_id=self.tgt_lang_id)

            log_probs = torch.log(transformer.decode(dec_out, enc_out, src_mask, None, self.tgt_lang_id))



        return self.beamSearch.hypotheses

if __name__ == "__main__":

    from src.utils.config import params
    #batch_size x seq_len
    x = torch.zeros(2, 5, dtype=torch.int64)

    src_m = torch.ones(2, 5)
    src_m[:, -2:-1] = 0
    src_m = src_m.unsqueeze(-2).unsqueeze(-2)
    #print(src_m.shape)
    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    transformer = Transformer(data_params=data_params, logger=create_logger(".log.txt"), embd_file="data/mono/all.en-fr.60000.vec")


    beam = MyBeamSearch(beam_size=3, batch_size=2, pad=2,bos=0, eos=1, n_best=2, mb_device=torch.device("cpu"), encoding_lengths=512, max_length=40, src_lang_id=0, tgt_lang_id=1)
    print(beam.perform(transformer, x, src_m))