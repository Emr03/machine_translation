from src.model.decoder import *
from src.model.encoder import *
from src.onmt.translate.beam import *
from src.onmt.translate.beam_search import BeamSearch


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
    def perform(self, encoder, decoder, batch, src_mask):

        enc_out = encoder(batch, src_mask, self.src_lang_id)

        for step in range(self.max_length):
            # TODO: make beam object, verify implementation
            decoder_in = self.beam.current_predictions.view(1, -1, 1)

            #pass in stuff to decoder, get log probabilities back
            dec_output = decoder(decoder_in, enc_out, src_mask, tgt_mask=None,
                                 lang_id=self.tgt_lang_id)

            log_probs = torch.log(dec_output)
            #attention is only used for coverage penalty, which we're not using
            self.beam.advance(log_probs, None)

            #checks if any beam is finished, then updates state.
            any_beam_is_finished = self.beam.is_finished.any()
            if any_beam_is_finished:
                self.beam.update_finished()
                #done if all beams are finished
                if self.beam.done:
                    break

        return self.beam.hypotheses

if __name__ == "__main__":

    from src.utils.config import params

    x = torch.zeros(2, 5, dtype=torch.int64)

    src_m = torch.ones(2, 5)
    src_m[:, -2:-1] = 0
    src_m = src_m.unsqueeze(-2).unsqueeze(-2)
    #print(src_m.shape)

    enc = StackedEncoder(n_layers=6, vocab_size=[90, 90], params=params, n_langs=2)
    dec = StackedDecoder(n_layers=6, vocab_size=[90, 90], params=params, n_langs=2, is_shared_emb=False)

    beam = MyBeamSearch(beam_size=3, batch_size=10, pad=2,bos=0, eos=1, n_best=2, mb_device=torch.device("cpu"), encoding_lengths=None, max_length=40, src_lang_id=0, tgt_lang_id=1)
    print(beam.perform(enc, dec, x, src_m))