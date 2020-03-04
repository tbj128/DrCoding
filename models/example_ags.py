import spacy
import torch
import torchtext
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import text_classification, TextClassificationDataset
from torchtext import data
import pandas as pd
from tqdm import tqdm
import io
NGRAMS = 2
import os

train_file = '../data_split_text_class/text_classification.tiny.train'
test_file = '../data_split_text_class/text_classification.tiny.test'
# train_file = '../data_split_text_class/text_classification.train'
# test_file = '../data_split_text_class/text_classification.test'



# dataset = Dataset(config)
# train_dataset, test_dataset = dataset.load_data(train_file, test_file)

#
# if not os.path.isdir('./.data-ag'):
#     os.mkdir('./.data-ag')
# train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
#     root='./.data-ag', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
SEQ_LEN = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab

def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]), ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)

def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                token_ids = [vocab[token] for token in tokens]
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
            if len(token_ids) == 0:
                print('Row contains no tokens.')

            token_ids = token_ids[:SEQ_LEN]
            while len(token_ids) < SEQ_LEN:
                token_ids.append(0)

            tokens = torch.tensor(token_ids)
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)

def _setup_datasets(ngrams=1, vocab=None, include_unk=False):
    train_csv_path = train_file
    test_csv_path = test_file

    if vocab is None:
        print('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    print('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))

train_dataset, test_dataset = _setup_datasets()

class TextSentimentLinear(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text) # (bs, seq, embed_dim)
        embedded = torch.mean(embedded, dim=1) # (bs, dim)
        embedded = nn.Tanh()(embedded)  # (bs, dim)
        embedded = self.fc(embedded)
        return F.softmax(embedded, dim=1)

import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text) # (bs, seq length, dim)

        embedded = embedded.permute(1, 0, 2) # (seq length, bs, dim)

        embedded = self.transformer_encoder(embedded) # (seq length, bs, dim)
        embedded = torch.mean(embedded, dim=0) # (bs, dim)
        embedded = nn.Tanh()(embedded)  # (bs, dim)
        embedded = self.fc(embedded) # (bs, dim)
        return F.softmax(embedded, dim=1)

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
# model = TextSentimentLinear(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.stack(text)
    return text, offsets, label

import time
from torch.utils.data.dataset import random_split, Dataset

N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)



train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print('\tLoss: {}(train)\t|\tAcc: {}(train)'.format(train_loss, train_acc))
    print('\tLoss: {}(valid)\t|\tAcc: {}(valid)'.format(valid_loss, valid_acc))
