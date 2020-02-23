#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Input manipulation for the MIMIC-III ICD-9 prediction task

Converts the input text to token indices.
Adapted from the `vocab.py` file in the CS224N assignments.


Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training discharge summaries
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]

"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List

class DischargeVocab(object):
    """
    Discharge note vocabulary representing terms from the discharge notes ("the input")
    """

    def __init__(self, word2id=None):
        """
        Constructor for the vocabulary representation
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0  # Pad Token
            self.word2id['<unk>'] = 1  # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.pad_token = self.word2id['<pad>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """
        Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """
        Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of sentences of words into list of list of indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[int]]): sentence(s) in indices
        """
        return [[self[w] for w in s] for s in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_var = torch.tensor(word_ids, dtype=torch.long, device=device)
        return sents_var

    @staticmethod
    def from_source_text(source_text, size, freq_cutoff=2):
        """
        Given a corpus construct a Vocab Entry.
        @param source_text (list[list[str]]): corpus of padded sentences produced by read_source_text function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        word_freq = Counter(chain(*source_text))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        vocab_entry = DischargeVocab()
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            if word != vocab_entry.pad_token:
                vocab_entry.add(word)
        return vocab_entry

    def save(self, out_path):
        """
        Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(self.word2id, open(out_path, 'w'), indent=2)

    @staticmethod
    def load_previous_vocab(file_path):
        """
        Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        return DischargeVocab(word2id=entry)


def main():
    """
    Main func.
    """
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])

    src_sents = read_corpus(args['--train-src'], source='src')

    vocab = Vocab.from_source_text(src_sents, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words' % (len(vocab.src)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])

if __name__ == '__main__':
    main()

