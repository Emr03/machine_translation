# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import argparse

from src.data.dictionary import EOS_WORD, UNK_WORD

def create_word_masks(params, data):
    """
    Create masks for allowed / forbidden output words.
    """
    if not hasattr(params, 'vocab') or len(params.vocab) == 0:
        return
    params.vocab_mask_pos = []
    params.vocab_mask_neg = []
    for lang, n_words in zip(params.langs, params.n_words):
        dico = data['dico'][lang]
        vocab = data['vocab'][lang]
        words = [EOS_WORD, UNK_WORD] + list(vocab)
        mask_pos = set([dico.index(w) for w in words])
        mask_neg = [i for i in range(n_words) if i not in mask_pos]
        params.vocab_mask_pos.append(torch.LongTensor(sorted(mask_pos)))
        params.vocab_mask_neg.append(torch.LongTensor(sorted(mask_neg)))


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
