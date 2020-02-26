#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding | Unit test for the baseline model
"""
from typing import List

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as f

from vocab import DischargeVocab
from lstm import DischargeLSTM
import baseline

class TestBaseline(unittest.TestCase):

    def test_evaluate_model_with_dev(self):
        print ("-"*80)
        print("Running Sanity Check for evaluate_model_with_dev")
        print ("-"*80)

        device = torch.device("cpu")
        source_text = [["butterflies", "fly", "towards", "sun"], ["apples", "fly", "towards", "ground"], ["oranges", "are", "yummy", "<pad>"]]
        vocab = DischargeVocab.from_source_text(source_text, 100, freq_cutoff=0)
        source_text = vocab.to_input_tensor(source_text, torch.device("cpu"))

        source_lengths = torch.tensor([4] * 3)
        actual_icds = torch.tensor([[0, 1, 0], [0, 1, 1], [0, 1, 1]])
        pos_to_icd = {
            0: '110',
            1: '332',
            2: '313'
        }

        dev_data = list(zip(source_text, source_lengths, actual_icds))

        lstm = DischargeLSTM(vocab=vocab, embed_size=10, hidden_size=10, dropout_rate=0.5, pos_to_icd=pos_to_icd)
        lstm(source_text, source_lengths)

        threshold = 0.3
        self.assertEqual(0.35, baseline.evaluate_model_with_dev(lstm, dev_data, threshold, device).round(2))



if __name__ == '__main__':
    unittest.main()
