import sys

from reformer.reformer_pytorch import Reformer
from reformer.reformer_pytorch_with_metadata import Reformer as ReformerWithMetadata
from transformer_common.positional_encoding import PositionalEncoding

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
from vocab import Vocab
from utils import read_source_text

class ReformerClassifier(nn.Module):
    """
    Custom classifier built on top of the Reformer model. Alternative, smaller memory-footprint model when compared
    to the Transformer model
    """

    def __init__(self, vocab, embed_size, depth, max_seq_len, use_metadata = False, num_heads = 8, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0.1, layer_dropout = 0.1, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False, full_attn_thres = 0, num_mem_kv = 0):
        """
        Constructor
        :vocab: vocab object
        :embed_size: size of the word embeddings
        :depth: the number of layers in the Reformer model
        :max_seq_len: the length of the input sequence
        :use_metadata: True if the ICD descriptions should be used as part of the attention scheme, False otherwise

        See the the Reformer class for additional parameter descriptions
        """
        super().__init__()
        self.vocab = vocab
        self.embed_size = embed_size
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.ff_chunks = ff_chunks
        self.attn_chunks = attn_chunks
        self.causal = causal
        self.weight_tie = weight_tie
        self.lsh_dropout = lsh_dropout
        self.layer_dropout = layer_dropout
        self.random_rotations_per_head = random_rotations_per_head
        self.twin_attention = twin_attention
        self.use_scale_norm = use_scale_norm
        self.use_full_attn = use_full_attn
        self.full_attn_thres = full_attn_thres
        self.num_mem_kv = num_mem_kv
        self.use_metadata = use_metadata

        self.num_output_classes = len(self.vocab.icd)
        num_tokens = len(vocab.discharge)

        self.encoder = nn.Embedding(num_tokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, transpose=False)

        if not self.use_metadata:
            self.reformer = Reformer(embed_size, depth, max_seq_len, heads=num_heads, bucket_size=bucket_size,
                                     n_hashes=n_hashes, ff_chunks=ff_chunks, attn_chunks=attn_chunks, causal=causal,
                                     weight_tie=weight_tie, lsh_dropout=lsh_dropout, layer_dropout=layer_dropout,
                                     random_rotations_per_head=random_rotations_per_head, twin_attention=twin_attention,
                                     use_scale_norm=use_scale_norm, use_full_attn=use_full_attn,
                                     full_attn_thres=full_attn_thres, num_mem_kv=num_mem_kv)

        else:
            self.reformer = ReformerWithMetadata(embed_size, depth, max_seq_len, heads=num_heads, bucket_size=bucket_size,
                                     n_hashes=n_hashes, ff_chunks=ff_chunks, attn_chunks=attn_chunks, causal=causal,
                                     weight_tie=weight_tie, lsh_dropout=lsh_dropout, layer_dropout=layer_dropout,
                                     random_rotations_per_head=random_rotations_per_head, twin_attention=twin_attention,
                                     use_scale_norm=use_scale_norm, use_full_attn=use_full_attn,
                                     full_attn_thres=full_attn_thres, num_mem_kv=num_mem_kv)

        self.pre_classifier = nn.Linear(embed_size, embed_size)
        self.classifier = nn.Linear(embed_size, self.num_output_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, source_lengths, metadata_ids=None, **kwargs):
        """
        The forward pass of the Reformer classifier
        :param src: the padded input text (bs, seq len)
        :param source_lengths: the lengths of the original input text (bs, 1)
        :param metadata_ids: the metadata text, requried if self.use_metadata is set to True (bs, seq len)
        :return: logits of the Transformer (bs, num_output_classes)
        """
        mask = (src == self.vocab.discharge.pad_token) # (batch size, seq length)
        src = self.encoder(src)

        #
        # Uncomment below if we want to use the positional encoding
        # Empirical evidence suggests position encoding does not
        # perform as well in sequence classification problems
        #
        # src = self.encoder(src) * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)

        if self.use_metadata:
            metadata_ids = self.encoder(metadata_ids)
            hidden_state = self.reformer(src, input_mask=mask, metadata_ids=metadata_ids)  # (bs, seq length, dim)
        else:
            hidden_state = self.reformer(src, input_mask=mask)  # (bs, seq length, dim)

        #
        # Uncomment below if we want to use the CLS token to perform classification
        # Testing shows that the CLS token does not perform as well as if we had
        # took the mean over the hidden states
        #
        # pooled_output = hidden_state[:, 0, :].squeeze()  # (bs, dim) - we take the first character (the CLS token)

        hidden_state = hidden_state * (~mask).type(torch.float).unsqueeze(2)
        pooled_output = torch.sum(hidden_state, dim=1) / torch.sum(mask == False, dim=1).type(torch.float).unsqueeze(1)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_output_classes)

        return logits

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = ReformerClassifier(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """
        Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        args = dict(
            embed_size=self.embed_size,
            depth=self.depth,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            bucket_size=self.bucket_size,
            n_hashes=self.n_hashes,
            ff_chunks=self.ff_chunks,
            attn_chunks=self.attn_chunks,
            causal=self.causal,
            weight_tie=self.weight_tie,
            lsh_dropout=self.lsh_dropout,
            layer_dropout=self.layer_dropout,
            random_rotations_per_head=self.random_rotations_per_head,
            twin_attention=self.twin_attention,
            use_scale_norm=self.use_scale_norm,
            use_full_attn=self.use_full_attn,
            full_attn_thres=self.full_attn_thres,
            num_mem_kv=self.num_mem_kv
        )

        params = {
            'args': args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
