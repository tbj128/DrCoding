#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Input manipulation for the MIMIC-III ICD-9 prediction task

Preprocesses the Glove embeddings into a memory efficient format


Usage:
    pretrained_embeddings.py GLOVE_FILE GLOVE_OUTPUT_FILE GLOVE_OUTPUT_MAP_FILE

Options:
    -h --help                  Show this screen.

"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
import bcolz
import numpy as np
import pickle

class PretrainedEmbeddings(object):

    def __init__(self, glove, word_to_pos):
        """
        """
        self.glove = glove
        self.word_to_pos = word_to_pos
        self.embed_size = len(glove["<unk>"])

    def get(self, word):
        if word not in self.word_to_pos:
            return self.glove[self.word_to_pos["<unk>"]]
        else:
            return self.glove[self.word_to_pos[word]]

    @staticmethod
    def load(glove_path, glove_map_path, use_bcolz=True):
        if use_bcolz:
            vectors = bcolz.open(glove_path)[:]
        else:
            with open(glove_path, 'rb') as f:
                vectors = pickle.load(f)
        with open(glove_map_path, 'rb') as f:
            word_to_pos = pickle.load(f)
        return PretrainedEmbeddings(vectors, word_to_pos)

    @staticmethod
    def load_glove_from_source(glove_path, glove_output, glove_map_output, use_bcolz=True):
        """
        """
        word_to_pos = {}

        # Get the total number of words
        with open(glove_path, 'r') as f:
            for i, line in enumerate(f):
                pass
        num_words = i + 2 # to account for the additional <unk> word we are adding
        embedding_size = len(line.split(' ')) - 1

        vectors = np.zeros((num_words, embedding_size), dtype=np.float32)
        tot_vec = None # used to compute the 'average vector' that will represent unknown words
        with open(glove_path, 'r') as f:
            i = 0
            for line in f:
                line_arr = line.split(" ")
                word = line_arr[0]
                vector = np.array([float(w) for w in line_arr[1:]])
                vectors[i] = vector
                word_to_pos[word] = i

                if tot_vec is None:
                    tot_vec = np.array(vector)
                else:
                    np.add(tot_vec, np.array(vector))

                i += 1

        avg_vec = np.divide(tot_vec, num_words)
        vectors[i] = avg_vec
        word_to_pos["<unk>"] = i

        if use_bcolz:
            vectors = bcolz.carray(vectors, rootdir=glove_output, mode='w')
            vectors.flush()
        else:
            with open(glove_output, 'wb') as f:
                pickle.dump(vectors.tolist(), f)

        with open(glove_map_output, 'wb') as f:
            pickle.dump(word_to_pos, f)


def main():
    """
    Main func.
    """
    args = docopt(__doc__)

    print("Processing the Glove file...")

    PretrainedEmbeddings.load_glove_from_source(args["GLOVE_FILE"], args["GLOVE_OUTPUT_FILE"], args["GLOVE_OUTPUT_MAP_FILE"])

    print("Finished processing")

if __name__ == '__main__':
    main()

