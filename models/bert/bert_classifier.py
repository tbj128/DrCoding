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
        self.bert = BertModelWithMetadata(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, metadata_input_ids=None, metadata_len=None):
        # Reduce input_ids into a series of batches of sizes equal to the len of the original metadata_input_ids
        batch_size, seq_len = input_ids.size()

        input_ids = input_ids[:, 1:] # remove CLS

        pad_column = torch.tensor([0] * batch_size, device=input_ids.device).unsqueeze(1)
        input_ids = torch.cat((input_ids, pad_column), dim=1)

        if metadata_len == None:
            metadata_len = seq_len

        metadata_ids = metadata_input_ids[:, :metadata_len]
        batch_increase_factor = int(seq_len / metadata_len)

        cls_column = torch.tensor([101] * (batch_size * batch_increase_factor), device=input_ids.device).unsqueeze(1)
        r_input_ids = input_ids.view(batch_size * batch_increase_factor, metadata_len)
        r_input_ids = torch.cat((cls_column, r_input_ids), dim=1)
        attn_mask = attention_mask.view(batch_size * batch_increase_factor, metadata_len)

        zeros_column = torch.zeros((batch_size * batch_increase_factor), 1, device=input_ids.device).type(torch.long)
        ones_column = torch.ones((batch_size * batch_increase_factor), 1, device=input_ids.device).type(torch.long)

        r_attn_mask = torch.cat((ones_column, attn_mask), dim=1)

        r_tokens = token_type_ids.view(batch_size * batch_increase_factor, metadata_len)
        r_tokens = torch.cat((r_tokens, zeros_column), dim=1)

        r_meta = torch.repeat_interleave(metadata_ids, repeats=batch_increase_factor, dim=0)
        r_meta = torch.cat((r_meta, zeros_column), dim=1)

        _, pooled_output = self.bert(
            input_ids=r_input_ids,
            attention_mask=r_attn_mask,
            token_type_ids=r_tokens,
            metadata_ids=r_meta
        ) # bs * batch_increase_factor, dim

        # mask = (r_input_ids == 0) # (batch size, seq length)

        # Don't account for completely zeroed out sub-batches
        mask = (attn_mask.view(batch_size, batch_increase_factor, metadata_len).sum(dim=2) > 0).type(torch.float).unsqueeze(2)
        pooled_output = pooled_output.view(batch_size, batch_increase_factor, -1)
        pooled_output = pooled_output * mask
        pooled_output = torch.sum(pooled_output, dim=1) / torch.sum(mask.squeeze(2).type(torch.float), dim=1)

        # mask = (mask.view(batch_size, batch_increase_factor, metadata_len).sum(dim=2) == metadata_len).unsqueeze(2) # bs, batch_increase_factor
        # pooled_output = pooled_output.view(batch_size, batch_increase_factor, -1)
        # pooled_output = pooled_output * (~mask).type(torch.float)
        # pooled_output = torch.sum(pooled_output, dim=1) / torch.sum((mask == False).type(torch.float), dim=1)

        # pooled_output = pooled_output.view(batch_size, batch_increase_factor, -1).mean(dim=1)
        # pooled_output = pooled_output.mean(dim=1)
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
        # model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), path)

    @staticmethod
    def load(model_path: str, bert_pretrained_path: str, num_labels: int):
        """ Load a fine-tuned model from a file.
        @param model_path (str): path to model
        """
        state_dict = torch.load(model_path)
        model = BertClassifierWithMetadataXS.from_pretrained(bert_pretrained_path, state_dict=state_dict, num_labels=num_labels)
        return model

