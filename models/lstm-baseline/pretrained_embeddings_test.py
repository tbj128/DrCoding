#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Unit test for the pretrained embeddings
"""
from typing import List

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as f

from pretrained_embeddings import PretrainedEmbeddings

class TestPretrainedEmbeddings(unittest.TestCase):

    def test(self):
        print ("-"*80)
        print("Running Sanity Check for Pretrained Glove embeddings")
        print ("-"*80)

        embeddings = PretrainedEmbeddings.load("../../embeddings/glove.6B.50d.txt.out.dat", "../../embeddings/glove.6B.50d.txt.map.txt")
        print(embeddings.get("the"))


if __name__ == '__main__':
    unittest.main()
