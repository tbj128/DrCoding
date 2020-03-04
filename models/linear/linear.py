import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSentiment(nn.Module):
    def __init__(self, vocab, num_embeddings, embed_dim, num_class):
        super().__init__()
        self.vocab = vocab
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.embedding = nn.EmbeddingBag(len(vocab.discharge), embed_dim)
        # self.embedding = nn.EmbeddingBag(5, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        batch_size = text.shape[0]
        seq_len = text.shape[1]
        offsets = []
        i = 0
        for j in range(batch_size):
            offsets.append(i)
            i += seq_len
        text = text.flatten()
        embedded = self.embedding(text, torch.tensor(offsets))
        return self.fc(embedded)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TextSentiment(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """
        Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(num_embeddings=self.num_embeddings, embed_dim=self.embed_dim, num_class=self.num_class),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
