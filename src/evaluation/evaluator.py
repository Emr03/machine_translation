# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn
from src.data.loader import *
from src.model.transformer import Transformer
from src.utils.data_loading import get_parser
from src.utils.logger import create_logger
from src.trainers.unsupervised_trainer import UnsupervisedTrainer
import logging
from src.model.beam_search_wrapper import MyBeamSearch


logger = getLogger()

TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = 'src/evaluation/multi-bleu.perl'
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH

def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s"
    subprocess.Popen(restore_cmd % path, shell=True).wait()

class EvaluatorMT(object):

    def __init__(self, transformer, params, exp_name):
        """
        Initialize evaluator.
        """
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.decode = transformer.decode

        self.data = transformer.data
        self.dico = transformer.data['dico']
        self.params = params
        self.exp_name = exp_name

        # create reference files for BLEU evaluation
        self.create_reference_files()

        self.beam_search = MyBeamSearch(transformer, beam_size=3, logger=logging,
                                        n_best=1, encoding_lengths=512, max_length=175)

        self.beam_search.to(self.device)

    def get_pair_for_mono(self, lang):
        """
        Find a language pair for monolingual data.
        """
        candidates = [(l1, l2) for (l1, l2) in self.data['para'].keys() if l1 == lang or l2 == lang]
        assert len(candidates) > 0
        return sorted(candidates)[0]

    def get_src_mask(self, src_batch):

        mask = torch.ones_like(src_batch)
        mask.masked_fill_(src_batch == self.params.pad_index, 0).unsqueeze_(-2).unsqueeze_(-2)
        #print("mask", mask)
        return mask

    def get_tgt_mask(self, tgt_batch):

        batch_size, sent_len = tgt_batch.shape

        # hide future words
        tgt_m = np.tril(np.ones((batch_size, sent_len, sent_len)), k=0).astype(np.uint8)
        #print("tgt_m", tgt_m)

        tgt_m = torch.from_numpy(tgt_m)

        # hide padding
        tgt_m.masked_fill_(tgt_batch.unsqueeze(-1) == self.params.pad_index, 0).unsqueeze_(1)
        #print("tgt_m", tgt_m)
        return tgt_m

    def compute_sent_len(self, sentences):
        """
        returns a tensor containing the length of each sentence in the batch
        :param sentences: batch_size, max_len
        :return:
        """
        n_sent = sentences.shape[0]
        eos_indices = (sentences == self.eos_index).nonzero()
        assert(eos_indices.shape[0] == n_sent)
        eos_indices = eos_indices[:, 1].unsqueeze_() + 1
        return eos_indices

    def generate_parallel(self, src_batch, src_mask, src_lang, tgt_lang):
        """
        generate sentences for back-translation using greedy decoding
        :param batch_dict: dict of src batch and src mask
        :param src_lang:
        :param tgt_lang:
        :return:
        """
        output, len = self.beam_search(src_batch, src_mask, src_lang=src_lang, tgt_lang=tgt_lang)

        return output, len

    def mono_iterator(self, data_type, lang):
        """
        If we do not have monolingual validation / test sets, we take one from parallel data.
        """
        dataset = self.data['mono'][lang][data_type]
        if dataset is None:
            pair = self.get_pair_for_mono(lang)
            dataset = self.data['para'][pair][data_type]
            i = 0 if pair[0] == lang else 1
        else:
            i = None
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch if i is None else batch[i]

    def get_iterator(self, data_type, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['valid', 'test']
        if lang2 is None or lang1 == lang2:
            for batch in self.mono_iterator(data_type, lang1):
                yield batch if lang2 is None else (batch, batch)
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k][data_type]
            dataset.batch_size = 32
            for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
                yield batch if lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2
            lang1_id = params.lang2id[lang1]
            lang2_id = params.lang2id[lang2]

            for data_type in ['valid', 'test']:

                lang1_path = os.path.join(self.exp_name, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_type))
                lang2_path = os.path.join(self.exp_name, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type))

                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_type, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico[lang1], lang1_id, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico[lang2], lang2_id, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

                # store data paths
                params.ref_paths[(lang2, lang1, data_type)] = lang1_path
                params.ref_paths[(lang1, lang2, data_type)] = lang2_path

    def eval_para(self, lang1, lang2, data_type, scores):
        """
        Evaluate lang1 -> lang2 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # hypothesis
        txt = []

        n_words2 = self.params.n_words[lang2_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang2):

            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.transpose_(0, 1).cuda(), sent2.transpose_(0, 1).cuda()

            src_mask = self.get_src_mask(sent1.cpu()).cuda()
            tgt_mask = self.get_tgt_mask(sent2.cpu()).cuda()

            # encode / decode / generatef
            sent2_ , len2_= self.generate_parallel(src_batch=sent1, src_mask=src_mask, src_lang=lang1_id, tgt_lang=lang2_id)

            # cross-entropy loss
            #xe_loss += loss_fn2(decoded.view(-1, n_words2), sent2[1:].view(-1)).item()
            count += (len2 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type)
        hyp_path = os.path.join(self.exp_name, hyp_name)
        ref_path = params.ref_paths[(lang1, lang2, data_type)]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)

        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

        # update scores
        scores['ppl_%s_%s_%s' % (lang1, lang2, data_type)] = np.exp(xe_loss / count)
        scores['bleu_%s_%s_%s' % (lang1, lang2, data_type)] = bleu


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


def convert_to_text(batch, lengths, dico, lang_id, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


if __name__ == "__main__":

    parser = get_parser()
    data_params = parser.parse_args()
    check_all_data_params(data_params)
    exp_name = data_params.exp_name
    is_variational = data_params.variational > 0
    use_distance_loss = data_params.use_distance_loss > 0
    load_from_checkpoint = data_params.load_from_checkpoint > 0

    logging.basicConfig(filename="logs/" + exp_name + "_eval" + ".log", level=logging.DEBUG)

    model = Transformer(data_params=data_params, logger=logging,
                        init_emb=True,
                        embd_file="corpora/mono/all.en-fr.60000.vec",
                        is_variational=is_variational)

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)
        # self.logger.debug("transformer ", self.transformer)
        model.to(device)

    else:
        print("Not using cuda")
        device = torch.device('cpu')

    filename = exp_name+".pth"
    state = torch.load(filename, map_location=device)
    model_state_dict = state["state_dict"]

    model.load_state_dict(state_dict=model_state_dict)

    eval = EvaluatorMT(transformer=model.module, params=data_params, exp_name=exp_name)

    scores = OrderedDict({'epoch': 1})
    eval.eval_para(lang1='en', lang2='fr', data_type='test', scores=scores)
    eval.eval_para(lang1='en', lang2='fr', data_type='valid', scores=scores)