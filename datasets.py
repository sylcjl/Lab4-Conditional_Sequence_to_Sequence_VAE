#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pathlib
import numpy as np
import torch
from collections import namedtuple
from torch.utils.data import Dataset


Vocab = namedtuple("Vocab", ("vocab", "tense"))
VocabPair = namedtuple("VocabPair", ("input", "target"))


class VocabTenseDataset(Dataset):

    def __init__(self, path: str,
                 test_label_path: str = None,
                 device: torch.device = torch.device("cpu"),
                 SOS=0, EOS=1):

        self.path = path
        self.test_label_path = test_label_path
        self.device = device
        self.SOS = [SOS]
        self.EOS = [EOS]

        self.vocab_pairs = []
        self.embedded_pairs = []
        self._load_vocabs(pathlib.Path(path))

    def _load_vocabs(self, path: str):
        vocabs = np.loadtxt(path, dtype=np.str)

        if not self.test_label_path:
            for vocab_tenses in vocabs:
                embedded_vocabs = [self._do_embedding(vocab) for vocab in vocab_tenses]
                ### vocab_tenses = ['abandon',
                #                   'abandons',
                #                   'abandoning',
                #                   'abandoned']
                ### embedded_tenses = [[2, 3, 2, 15, 5, 16, 15],
                #                      [2, 3, 2, 15, 5, 16, 15, 20],
                #                      [2, 3, 2, 15, 5, 16, 15, 10, 15, 8],
                #                      [2, 3, 2, 15, 5, 16, 15, 6, 5]]
                for tense in range(4):
                    vocab = Vocab(vocab_tenses[tense],
                                  tense)
                    embedded_vocab = Vocab(torch.tensor(embedded_vocabs[tense], dtype=torch.long, device=self.device),
                                           torch.tensor([tense], dtype=torch.long, device=self.device))
                    self.vocab_pairs.append(VocabPair(vocab, vocab))
                    self.embedded_pairs.append(VocabPair(embedded_vocab, embedded_vocab))
        else:
            vocabs_tense = np.loadtxt(self.test_label_path, dtype=np.int)
            if len(vocabs) != len(vocabs):
                raise ValueError("The number of vocabularies ({}) and "
                                 "the number of vocabularies' label {} do not match.".format(len(vocabs),
                                                                                             len(vocabs_tense)))

            for vocab_pair, tense_pair in zip(vocabs, vocabs_tense):
                ### vocab_pair      = ['abet', 'abetting']
                ### tense_pair      = [ 0,      2]
                ### embedded_vocabs = [[2, 3, 6, 21], [2, 3, 6, 21, 21, 10, 15, 8]]
                embedded_vocabs = [torch.tensor(self._do_embedding(vocab), dtype=torch.long, device=self.device)
                                   for vocab in vocab_pair]
                tense_pair_tensor = [torch.tensor([tense], dtype=torch.long, device=self.device) for tense in tense_pair]

                self.vocab_pairs.append(VocabPair(Vocab(vocab_pair[0], tense_pair[0]),
                                                  Vocab(vocab_pair[1], tense_pair[1])))
                self.embedded_pairs.append(VocabPair(Vocab(embedded_vocabs[0], tense_pair_tensor[0]),
                                                     Vocab(embedded_vocabs[1], tense_pair_tensor[1])))

    def _do_embedding(self, vocab):
        return self.SOS + [self._char_to_num(char) for char in vocab.lower()] + self.EOS

    def _char_to_num(self, char):
        return ord(char) - 97 + 2  # odr("a") = 97

    def to_vocab(self, chars):
        return "".join(list(map(self._num_to_char, chars)))

    def _num_to_char(self, num):
        if [num] in (self.SOS, self.EOS):
            return ""
        else:
            return chr(num - 2 + 97)

    def __getitem__(self, index):
        return self.vocab_pairs[index], self.embedded_pairs[index]

    def __len__(self):
        return len(self.vocab_pairs)


if __name__ == "__main__":
    import os
    data = VocabTenseDataset(os.path.join("lab4_dataset", "train.txt"))

    data = VocabTenseDataset(path=os.path.join("lab4_dataset", "test.txt"),
                             test_label_path=os.path.join("lab4_dataset", "test_tense.txt"))
    pass