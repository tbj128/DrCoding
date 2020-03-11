"""
Customized BERT-based classifiers for DrCoding
"""

import sys

import random
import tqdm
import gzip
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertPreTrainedModel, BertModel
from transformer_common.positional_encoding import PositionalEncoding
from bert.modeling_bert_with_metadata import BertModel as BertModelWithMetadata
from bert.modeling_bert_with_metadata_xs import BertModel as BertModelWithMetadataXS
from vocab import Vocab
from utils import create_embedding_from_glove


class BertClassifier(BertPreTrainedModel):
    """
    BERT multi-label classifier
    """
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Forward pass of the BERT classifier
        :param input_ids: the input IDs (bs, seq len)
        :param token_type_ids: (not used) a tensor of zeros indicating which sequence in sequence pairs (bs, seq len)
        :param attention_mask: tensor of one if not pad token, zero otherwise (bs, seq len)
        :return: logits corresponding to each output class (bs, )
        """
        _, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def freeze_bert_encoder(self):
        """
        Prevents further backpropagation (used when testing)
        """
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """
        Re-enables backpropagation (used when training)
        """
        for param in self.bert.parameters():
            param.requires_grad = True

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        # Only save the model and not the entire pretrained Bert
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), path)

    @staticmethod
    def load(model_path: str, bert_pretrained_path: str, num_labels: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """

        state_dict = torch.load(model_path)
        model = BertClassifier.from_pretrained(bert_pretrained_path, state_dict=state_dict, num_labels=num_labels)
        return model


class BertClassifierWithMetadata(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifierWithMetadata, self).__init__(config)
        self.bert = BertModelWithMetadata(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, metadata_input_ids=None):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            metadata_ids=metadata_input_ids
        )
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        # Only save the model and not the entire pretrained Bert
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), path)

    @staticmethod
    def load(model_path: str, bert_pretrained_path: str, num_labels: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """

        state_dict = torch.load(model_path)
        model = BertClassifierWithMetadata.from_pretrained(bert_pretrained_path, state_dict=state_dict, num_labels=num_labels)
        return model


class BertClassifierWithMetadataXS(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifierWithMetadataXS, self).__init__(config)
        self.bert = BertModelWithMetadataXS(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, metadata_input_ids=None):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            metadata_ids=metadata_input_ids[:,:24]
        )
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        # Only save the model and not the entire pretrained Bert
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), path)

    @staticmethod
    def load(model_path: str, bert_pretrained_path: str, num_labels: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """
        state_dict = torch.load(model_path)
        model = BertClassifierWithMetadata.from_pretrained(bert_pretrained_path, state_dict=state_dict, num_labels=num_labels)
        return model

