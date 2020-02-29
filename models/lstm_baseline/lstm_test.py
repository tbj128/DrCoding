#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Unit test for the LSTM baseline model
"""
from typing import List

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as f

from vocab import DischargeVocab
from lstm import DischargeLSTM

class TestLSTMBaseline(unittest.TestCase):

    def test_lstm_sanity_check(self):
        print ("-"*80)
        print("Running Sanity Check for LSTM baseline")
        print ("-"*80)

        source_text = [["butterflies", "fly", "towards", "sun"], ["apples", "fly", "towards", "ground"], ["oranges", "are", "yummy", "<pad>"]]
        vocab = DischargeVocab.from_source_text(source_text, 100, freq_cutoff=0)

        source_text = vocab.to_input_tensor(source_text, torch.device("cpu"))
        lstm = DischargeLSTM(vocab=vocab, embed_size=10, hidden_size=10, dropout_rate=0.5, num_output_classes=3)

        source_lengths = torch.tensor([4] * 3)

        output = lstm(source_text, source_lengths)
        self.assertEquals(output.shape, torch.Size([3, 3]))


if __name__ == '__main__':
    unittest.main()
