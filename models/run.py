#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Usage:
    baseline.py train --train-src=<file> --train-icd=<file> --dev-src=<file> --dev-icd=<file> --vocab=<file> [options]
    baseline.py predict [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    baseline.py predict [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --model=<str>                           the type of model to train [default: baseline]
    --train-src=<file>                      train source file
    --train-icd=<file>                      train ICD codes file
    --dev-src=<file>                        dev source file
    --dev-icd=<file>                        dev ICD codes file
    --vocab=<file>                          vocab file
    --target-length=<int>                   max length of each input sequence [default: 1000]
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
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --verbose                               show additional logging
"""
import math
import sys
import pickle
import time
import re

from docopt import docopt
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from lstm_baseline.lstm import DischargeLSTM
from reformer.reformer_classifier import ReformerClassifier
from utils import batch_iter, read_source_text, read_icd_codes

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from vocab import Vocab


def evaluate_scores(references: List[str], predicted: List[str]):
    """
    Given set of references and predicted ICD codes, return the precision, recall, f1, and accuracy statistics
    """
    assert len(references) == len(predicted)

    f1 = f1_score(references, predicted, average="micro")
    precision = precision_score(references, predicted, average="micro")
    recall = recall_score(references, predicted, average="micro")
    accuracy = accuracy_score(references, predicted)

    return precision, recall, f1, accuracy

def predict_output(args, model, dev_data, device, batch_size=32):
    preds = []
    icds = []
    with torch.no_grad():
        for src_text, src_lengths, actual_icds in batch_iter(dev_data, batch_size):
            batch_src_text_tensor = model.vocab.discharge.to_input_tensor(src_text, device)
            batch_src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)

            if args['--model'] == 'baseline':
                model_out = model(batch_src_text_tensor, batch_src_lengths)
            else:
                model_out = model(batch_src_text_tensor)
            top_prediction_indices = torch.argmax(F.softmax(model_out, dim=1), dim=1)  # bs x 1

            for ind in top_prediction_indices.cpu().tolist():
                top_output_icd = model.vocab.icd.get_icd(ind)
                preds.append(top_output_icd)
            for actual_icd in actual_icds:
                icds.append(actual_icd)
    return preds, icds

def evaluate_model_with_dev(args, model, dev_data, device, batch_size=32):
    """

    """
    was_training = model.training
    model.eval()

    # no_grad() signals backend to throw away all gradients
    preds, icds = predict_output(args, model, dev_data, device, batch_size)

    f1 = f1_score(preds, icds, average='micro')

    if was_training:
        model.train()

    return f1


def train(args: Dict):
    """
    Train the baseline LSTM model
    @param args (Dict): args from cmd line
    """

    vocab = Vocab.load(args['--vocab'])
    use_cls = args['--model'] != "baseline"

    train_source_text, train_source_lengths = read_source_text(args['--train-src'], target_length=int(args['--target-length']), pad_token=vocab.discharge.pad_token, use_cls=use_cls)
    train_icd_codes = read_icd_codes(args['--train-icd'])

    dev_source_text, dev_source_lengths = read_source_text(args['--dev-src'], target_length=int(args['--target-length']), pad_token=vocab.discharge.pad_token, use_cls=use_cls)
    dev_icd_codes = read_icd_codes(args['--dev-icd'])

    train_data = list(zip(train_source_text, train_source_lengths, train_icd_codes))
    dev_data = list(zip(dev_source_text, dev_source_lengths, dev_icd_codes))

    train_batch_size = int(args['--batch-size'])

    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    model_type = args['--model']
    if model_type == "baseline":
        model = DischargeLSTM(vocab=vocab,
                              embed_size=int(args['--word-embed-size']),
                              hidden_size=int(args['--hidden-size']),
                              dropout_rate=float(args['--dropout']))
    elif model_type == "reformer":
        model = ReformerClassifier(
            vocab=vocab,
            dim=int(args['--word-embed-size']),
            depth=6,
            max_seq_len=int(args['--target-length']),
            num_heads=8,
            bucket_size=64,
            n_hashes=4,
            ff_chunks=10,
            lsh_dropout=0.1,
            weight_tie=True,
            causal=True,
            use_full_attn=False # set this to true for comparison with full attention
        )
        if args['--cuda']:
            model.cuda()
    else:
        raise NotImplementedError("Invalid model type")

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

        for batch_src_text, batch_src_lengths, batch_icd_codes in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(batch_src_text)

            batch_src_text_tensor = model.vocab.discharge.to_input_tensor(batch_src_text, device)
            batch_src_lengths = torch.tensor(batch_src_lengths, dtype=torch.long, device=device)
            batch_icd_codes = model.vocab.icd.to_one_hot(batch_icd_codes, device)

            if args['--verbose']:
                print("  > epoch {} iter {} batch_src_text {}".format(epoch, train_iter, batch_src_text_tensor.shape))

            if args['--model'] == 'baseline':
                model_output = model(batch_src_text_tensor, batch_src_lengths)
            else:
                model_output = model(batch_src_text_tensor)
            example_losses = lossFunc(model_output, batch_icd_codes)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            if args['--verbose']:
                print("  > epoch {} iter {} loss {}".format(epoch, train_iter, loss.item()))

            loss.backward()

            if args['--verbose']:
                print("  > epoch {} iter {} after loss backward".format(epoch, train_iter))

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if args['--verbose']:
                print("  > epoch {} iter {} after optimizer step".format(epoch, train_iter))

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
            if train_iter % valid_niter == 0 or epoch == int(args['--max-epoch']):
                print('begin validation ...', file=sys.stderr)

                dev_f1 = evaluate_model_with_dev(args, model, dev_data, device, batch_size=128)   # dev batch size can be a bit larger
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

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def predict_icd_codes(args: Dict[str, str]):
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    use_cls = args["--model"] != "baseline"
    test_source_text, test_source_lengths = read_source_text(args['TEST_SOURCE_FILE'], use_cls=use_cls)

    if args['TEST_TARGET_FILE']:
        print("load test icd codes from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_icd_codes = read_icd_codes(args['TEST_TARGET_FILE'])
    else:
        test_icd_codes = None

    test_data = list(zip(test_source_text, test_source_lengths, test_icd_codes))

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = DischargeLSTM.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Using device: %s' % device, file=sys.stderr)

    preds, icds = predict_output(args, model, test_data, device, batch_size=128)

    if was_training: model.train(was_training)

    if test_icd_codes is not None:
        precision, recall, f1, accuracy = evaluate_scores(icds, preds)
        print('Precision {}, recall {}, f1 {}, accuracy: {}'.format(precision, recall, f1, accuracy), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for pred in preds:
            f.write(pred + '\n')


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
    elif args['predict']:
        predict_icd_codes(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
