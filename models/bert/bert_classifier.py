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
from transformers import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from transformer_common.positional_encoding import PositionalEncoding
from vocab import Vocab
from utils import create_embedding_from_glove


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
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
    def load(model_path: str, bert_pretrained_path: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """

        state_dict = torch.load(model_path)
        model = BertClassifier.from_pretrained(bert_pretrained_path, state_dict=state_dict)
        return model


class BertClassifierMetadataOption1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
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
    def load(model_path: str, bert_pretrained_path: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """

        state_dict = torch.load(model_path)
        model = BertClassifier.from_pretrained(bert_pretrained_path, state_dict=state_dict)
        return model
