# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

from .dictionary import EOS_WORD, UNK_WORD

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

def get_iterator(self, iter_name, lang1, lang2, back):
    """
    Create a new iterator for a dataset.
    """
    assert back is False or lang2 is not None
    key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '')
    #logger.info("Creating new training %s iterator ..." % key)
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
