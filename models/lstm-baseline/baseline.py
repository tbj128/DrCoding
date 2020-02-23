#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Kuangcong Liu <cecilia4@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --word-embed-size=<int>                 word embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --threshold=<int>                       probability threshold of accepting an ICD code [default: 0.3]
"""
import math
import sys
import pickle
import time
import re

from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from lstm import DischargeLSTM
from utils import batch_iter, read_source_text, read_icd_codes

import torch
import torch.nn.utils
import torch.nn.functional as f

from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import f1_score


def evaluate_scores(references: List[List[str]], predicted: List[List[str]]):
    """
    Given set of references and predicted ICD codes, return the precision, recall, f1, and accuracy statistics
    """
    assert len(references) == len(predicted)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_accuracy = 0

    for i in range(len(references)):
        joint = len(set(references[i]).intersection(set(predicted[i])))
        union = len(set(references[i] + predicted[i]))
        num_references = len(references[i])
        num_predicted = len(predicted[i])
        precision = joint / num_references
        total_precision += precision

        recall = joint / num_predicted
        total_recall += recall

        f1 = 2 * joint / (num_predicted + num_references)
        total_f1 += f1

        accuracy = joint / union
        total_accuracy += accuracy

    precision = total_precision / len(references)
    recall = total_recall / len(references)
    f1 = total_f1 / len(references)
    accuracy = total_accuracy / len(references)

    return precision, recall, f1, accuracy


def evaluate_model_with_dev(model, dev_data, threshold, batch_size=32):
    """

    """
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    total_f1 = 0
    with torch.no_grad():
        for src_text, actual_icds in batch_iter(dev_data, batch_size):
            model_out = model(src_text)
            likelihoods = -f.log_softmax(model_out)
            for row in likelihoods:
                icd_preds = []
                for i in range(len(likelihoods[row])):
                    if likelihoods[row][i] >= threshold:
                        icd_preds.append(model.pos_to_icd[i])
                f1 = f1_score(actual_icds[row], icd_preds, average='macro')
                total_f1 += f1

    avg_f1 = total_f1 / len(dev_data)

    if was_training:
        model.train()

    return avg_f1


def train(args: Dict):
    """
    Train the baseline LSTM model
    @param args (Dict): args from cmd line
    """
    vocab = DischargeVocab.load_previous_vocab(args['--vocab'])

    train_source_text = read_source_text(args['--train-src'], target_length=int(args['--target-length']), pad_token=vocab.pad_token)
    train_icd_codes = read_icd_codes(args['--train-tgt'])

    dev_source_text = read_source_text(args['--dev-src'], target_length=int(args['--target-length']), pad_token=vocab.pad_token)
    dev_icd_codes = read_icd_codes(args['--dev-tgt'])

    train_data = list(zip(train_source_text, train_icd_codes))
    dev_data = list(zip(dev_source_text, dev_icd_codes))

    train_batch_size = int(args['--batch-size'])

    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    model = DischargeLSTM(vocab=vocab,
                          embed_size=int(args['--embed-size']),
                          hidden_size=int(args['--hidden-size']),
                          dropout_rate=float(args['--dropout']),
                          num_output_classes=)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Using device: %s' % device, file=sys.stderr)

    model = model.to(device)

    lossFunc = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = total_loss = total_processed_words = 0
    total_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    print('Starting baseline training...')

    while True:
        epoch += 1

        for batch_src_text, batch_icd_codes in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(batch_src_text)

            batch_src_text_tensor = self.vocab.to_input_tensor(batch_src_text, device)

            model_output = model(batch_src_text_tensor, batch_icd_codes)
            example_losses = -lossFunc(model_output)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            total_loss += batch_losses_val
            total_processed_words += sum(len(s) for s in batch_src_text)
            total_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f ' \
                      'total examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         total_loss / total_examples,
                                                                                         total_examples,
                                                                                         total_processed_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                total_examples = total_loss = total_processed_words = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('begin validation ...', file=sys.stderr)

                dev_f1 = evaluate_model_with_dev(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = dev_f1

                print('validation: iter %d, dev. f1 %f' % (train_iter, dev_f1), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reseet patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def predict_icd_codes(args: Dict[str, str]):
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_source_text = read_source_text(args['TEST_SOURCE_FILE'], source='src')
    print("load test icd codes from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
    test_icd_codes = read_icd_codes(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = DischargeLSTM.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Using device: %s' % device, file=sys.stderr)

    threshold = args['--threshold']

    hypotheses = []
    with torch.no_grad():
        for src_text in tqdm(test_source_text, desc='Decoding', file=sys.stdout):
            batch_src_text_tensor = self.vocab.to_input_tensor([src_text], device)
            model_out = model([batch_src_text_tensor])
            likelihoods = f.log_softmax(model_out)
            icd_preds = []
            for i in range(likelihoods):
                if likelihoods[i] >= threshold:
                    icd_preds.append(model.pos_to_icd[i])
            hypotheses.append(icd_preds)

    if was_training: model.train(was_training)

    precision, recall, f1, accuracy = evaluate_scores(test_icd_codes, hypotheses)
    print('Precision {}, recall {}, f1 {}, accuracy: {}'.format(precision, recall, f1, accuracy), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for hyps in hypotheses:
            f.write(",".join(hyps) + '\n')


def main():
    """
    Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
