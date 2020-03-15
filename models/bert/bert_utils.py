# Adapted from: https://github.com/google-research/bert/blob/master/extract_features.py
# NOTES:
# - Updated the Input classes to hold sample metadata
# - Updated the conversion method to handle and extract sample metadata

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, metadata_text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text sequence to be classified.
            metadata_text: (Optional) string. The untokenized text of the metadata/context text sequence
        """
        self.guid = guid
        self.text = text
        self.metadata_text = metadata_text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ids_metadata=None, input_mask_metadata=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ids_metadata = input_ids_metadata
        self.input_mask_metadata = input_mask_metadata


def convert_examples_to_features(examples, max_seq_length, tokenizer, max_metadata_length=36):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_text = tokenizer.tokenize(example.text)

        tokens_metadata_text = None
        if example.metadata_text:
            tokens_metadata_text = tokenizer.tokenize(example.metadata_text)
            # tokens_metadata_text = tokenizer.tokenize(example.metadata_text + " " + example.text)
            if len(tokens_metadata_text) > max_metadata_length:
                tokens_metadata_text = tokens_metadata_text[:(max_metadata_length)]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_text) > max_seq_length - 2:
            tokens_text = tokens_text[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens_text = ["[CLS]"] + tokens_text + ["[SEP]"]
        segment_ids = [0] * len(tokens_text)

        input_ids_text = tokenizer.convert_tokens_to_ids(tokens_text)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_text = [1] * len(input_ids_text)

        # Zero-pad up to the sequence length.
        padding_text = [0] * (max_seq_length - len(input_mask_text))
        input_ids_text += padding_text
        input_mask_text += padding_text
        segment_ids += padding_text

        input_ids_metadata_text = None
        input_mask_metadata_text = None

        if tokens_metadata_text:
            input_ids_metadata_text = tokenizer.convert_tokens_to_ids(tokens_metadata_text)
            input_mask_metadata_text = [1] * len(input_ids_metadata_text)
            padding_metadata_text = [0] * (max_metadata_length - len(input_mask_metadata_text))
            input_ids_metadata_text += padding_metadata_text
            input_mask_metadata_text += padding_metadata_text
            assert len(input_ids_metadata_text) == max_metadata_length
            assert len(input_mask_metadata_text) == max_metadata_length

        assert len(input_ids_text) == max_seq_length
        assert len(input_mask_text) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index == 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_text]))
            if tokens_metadata_text:
                logger.info("metadata tokens: %s" % " ".join([str(x) for x in tokens_metadata_text]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_text]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask_text]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        if input_ids_metadata_text and input_mask_metadata_text:
            features.append(InputFeatures(input_ids=input_ids_text,
                                          input_mask=input_mask_text,
                                          segment_ids=segment_ids,
                                          input_ids_metadata=input_ids_metadata_text,
                                          input_mask_metadata=input_mask_metadata_text))
        else:
            features.append(InputFeatures(input_ids=input_ids_text,
                                          input_mask=input_mask_text,
                                          segment_ids=segment_ids))
    return features
