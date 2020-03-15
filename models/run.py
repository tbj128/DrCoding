#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrCoding: Predicting ICD-9 Codes from Discharge Summaries

Partially adapted from the run.py from CS224N Assignment 5

Usage:
    run.py train --train-src=<file> --dev-src=<file> --vocab=<file> [options]
    run.py predict [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --model=<str>                           the type of model to train [default: baseline]
    --glove-path=<file>                     the glove embedding file [default: NONE]
    --train-src=<file>                      train source file
    --dev-src=<file>                        dev source file
    --vocab=<file>                          vocab file [default: vocab.json]
    --vocab-test=<file>                     vocab file [default: vocab.json]
    --target-length=<int>                   max length of each input sequence [default: 1000]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --word-embed-size=<int>                 word embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --threshold=<int>                       threshold to predict presence of an ICD code. Must specify if --top-k is not specified.
    --top-k=<int>                           take the top k ICD codes in the prediction. Must specify if --threshold is not specified.
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --dropout=<float>                       dropout [default: 0.5]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --transformer-depth=<int>               number of Transformer encoder layers to use [default: 1]
    --transformer-heads=<int>               number of Transformer heads to use [default: 8]
    --base-bert-path=<file>                 path to the pre-trained BERT model
    --icd-desc-file=<file>                  file containing hadmid to ICD descriptions [default: NONE]
    --max-metadata-length=<int>             maximum length of the metadata text
    --verbose                               show additional logging
    --meta-len=<file>                       metadata length [default: 32]
"""
import math
import sys
import pickle
import time
import re
import signal

from docopt import docopt
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from transformers import BertTokenizer

from bert.bert_classifier import BertClassifier, BertClassifierWithMetadataXS
from lstm_baseline.lstm import DischargeLSTM
from reformer.reformer_classifier import ReformerClassifier
from utils import batch_iter, read_source_text, read_source_text_for_bert_with_metadata, read_icd_descs_for_testing

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from vanilla_transformer.transformer_classifier import TransformerClassifier
from vocab import Vocab
import logging
import csv

################################################################################
# LOGGER setup

logger = logging.getLogger('drcoding')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('drcoding.log')
fh.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

logger.info('Starting run.py')

################################################################################

def evaluate_scores(references: List[List[str]], predicted: List[List[str]]):
    """
    Given set of references and predicted ICD codes, return the precision, recall, f1, and accuracy statistics
    :param references: list of list of ICD code IDs
    :param predicted: list of list of ICD code IDs
    :return: precision, recall, f1, accuracy (averaged across all samples)
    """
    assert len(references) == len(predicted)

    f1 = 0.
    precision = 0.
    recall = 0.
    accuracy = 0.
    for i in range(len(references)):
        f1 += f1_score(references[i], predicted[i])
        precision += precision_score(references[i], predicted[i])
        recall += recall_score(references[i], predicted[i])
        accuracy += accuracy_score(references[i], predicted[i])

    return precision / len(references), recall / len(references), f1 / len(references), accuracy / len(references)


def predict_output(args, model, dev_data, device, batch_size=32, tokenizer=None):
    """
    Applies the provided model to the test data
    :param args: the input arguments
    :param model: the pre-trained model
    :param dev_data: zipped collection of discharge notes, ICD codes, ICD descriptions
    :param device: the torch device
    :param batch_size: the batch size to use
    :return: preds (the model-predicted ICD codes), icds (the actual ICD codes)
    """
    preds = []
    icds = []
    completed = 0
    with torch.no_grad():
        for src_text, src_lengths, actual_icds, actual_icd_descs in batch_iter(dev_data, batch_size):
            if args['--model'] == "bert":
                input_ids = torch.tensor([f.input_ids for f in src_text], dtype=torch.long, device=device)
                input_mask = torch.tensor([f.input_mask for f in src_text], dtype=torch.long, device=device)
                segment_ids = torch.tensor([f.segment_ids for f in src_text], dtype=torch.long, device=device)
                model_out = model(input_ids, segment_ids, input_mask)
            elif args['--model'] == "bert-metadata":
                # Original Version
                input_ids = torch.tensor([f.input_ids for f in src_text], dtype=torch.long, device=device)
                input_mask = torch.tensor([f.input_mask for f in src_text], dtype=torch.long, device=device)
                segment_ids = torch.tensor([f.segment_ids for f in src_text], dtype=torch.long, device=device)
                model_out = model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    metadata_input_ids=input_ids,
                    metadata_len=int(args['--meta-len'])
                )
            elif args['--model'] == "reformer-metadata":
                # batch_src_text_tensor = model.vocab.discharge.to_input_tensor(src_text, device)
                # batch_src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
                # model_out = model(batch_src_text_tensor, batch_src_lengths, metadata_ids=batch_src_text_tensor)

                batch_src_text_tensor = model.vocab.discharge.to_input_tensor(src_text, device)
                batch_src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
                model_out = model(batch_src_text_tensor, batch_src_lengths, metadata_ids=batch_src_text_tensor)
            else:
                batch_src_text_tensor = model.vocab.discharge.to_input_tensor(src_text, device)
                batch_src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
                model_out = model(batch_src_text_tensor, batch_src_lengths)
            output_scores = F.softmax(model_out, dim=1)  # bs x classes

            for output_score_arr in output_scores.cpu().tolist():
                one_hot = []
                if args["--threshold"]:
                    thresh = float(args['--threshold'])
                    for score in output_score_arr:
                        one_hot.append(1 if score >= thresh else 0)
                elif args["--top-k"]:
                    top_k = int(args['--top-k'])
                    top_indices = set((-np.array(output_score_arr)).argsort()[:top_k])
                    i = 0
                    for score in output_score_arr:
                        one_hot.append(1 if i in top_indices else 0)
                        i += 1
                else:
                    raise NotImplementedError("Invalid prediction type")


                preds.append(one_hot)
            for actual_icd_one_hot in actual_icds:
                icds.append(list(actual_icd_one_hot))

            completed += len(src_text)
            if completed % 100 == 0:
                print("Completed {}/{}".format(completed, len(dev_data)))
            if completed % 10 == 0:
                precision, recall, f1, accuracy = evaluate_scores(icds, preds)
                print('  SO FAR: Precision {}, recall {}, f1 {}, accuracy: {}'.format(precision, recall, f1, accuracy))

    return preds, icds


def evaluate_model_with_dev(args, model, dev_data, device, batch_size=32, tokenizer=None):
    """
    Evaluate the model and return the precision, recall, f1 and accuracy metrics
    """
    was_training = model.training
    model.eval()

    preds, icds = predict_output(args, model, dev_data, device, batch_size=batch_size, tokenizer=tokenizer)
    precision, recall, f1, accuracy = evaluate_scores(icds, preds)

    if was_training:
        model.train()

    return precision, recall, f1, accuracy


def train(args):
    """
    Train the model with the training data
    :param args: the args from the command line
    """

    vocab = Vocab.load(args['--vocab'])
    model_type = args['--model']

    # Change the below flag if we want to inject a 'CLS' token when parsing the input
    # Note that this is done automatically if using the BERT tokenizer
    # Also note that 'CLS' tokens do not work very well with the Reformer model
    use_cls = False

    tokenizer = None
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args['--base-bert-path'])
        train_source_text, train_source_lengths, train_icd_codes, train_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['--train-src'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer)

        dev_source_text, dev_source_lengths, dev_icd_codes, dev_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['--dev-src'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer)
    elif model_type == "bert-metadata":
        tokenizer = BertTokenizer.from_pretrained(args['--base-bert-path'])
        train_source_text, train_source_lengths, train_icd_codes, train_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['--train-src'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer,
            metadata_file_path=args['--icd-desc-file']
        )

        dev_source_text, dev_source_lengths, dev_icd_codes, dev_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['--dev-src'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer,
            metadata_file_path=args['--icd-desc-file']
        )
    else:
        train_source_text, train_source_lengths, train_icd_codes, train_icd_descs = read_source_text(args['--train-src'], target_length=int(args['--target-length']), use_cls=use_cls, metadata_file_path=args['--icd-desc-file'])
        dev_source_text, dev_source_lengths, dev_icd_codes, dev_icd_descs = read_source_text(args['--dev-src'], target_length=int(args['--target-length']), use_cls=use_cls, metadata_file_path=args['--icd-desc-file'])

    train_data = list(zip(train_source_text, train_source_lengths, train_icd_codes, train_icd_descs))
    dev_data = list(zip(dev_source_text, dev_source_lengths, dev_icd_codes, dev_icd_descs))

    train_batch_size = int(args['--batch-size'])

    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    if model_type == "baseline":
        model = DischargeLSTM(vocab=vocab,
                              embed_size=int(args['--word-embed-size']),
                              hidden_size=int(args['--hidden-size']),
                              dropout_rate=float(args['--dropout']),
                              glove_path=args['--glove-path'],
                              device=device)
    elif model_type == "reformer":
        model = ReformerClassifier(
            vocab=vocab,
            embed_size=int(args['--word-embed-size']),
            depth=int(args['--transformer-depth']),
            max_seq_len=int(args['--target-length']),
            num_heads=int(args['--transformer-heads']),
            bucket_size=64,
            n_hashes=4,
            ff_chunks=10,
            lsh_dropout=0.1,
            layer_dropout=float(args['--dropout']),
            weight_tie=True,
            causal=True,
            use_full_attn=False
        )
    elif model_type == "reformer-metadata":
        # Use a forked version of the Reformer model that uses metadata from the ICD codes during attention
        model = ReformerClassifier(
            vocab=vocab,
            embed_size=int(args['--word-embed-size']),
            depth=int(args['--transformer-depth']),
            # max_seq_len=int(args['--target-length']),
            max_seq_len=128,
            num_heads=int(args['--transformer-heads']),
            bucket_size=64,
            n_hashes=4,
            ff_chunks=10,
            lsh_dropout=0.1,
            layer_dropout=float(args['--dropout']),
            weight_tie=True,
            causal=True,
            use_full_attn=False,
            use_metadata=True
        )
    elif model_type == "transformer":
        # model = ReformerClassifier(
        #     vocab=vocab,
        #     embed_size=int(args['--word-embed-size']),
        #     depth=int(args['--transformer-depth']),
        #     max_seq_len=128,
        #     num_heads=int(args['--transformer-heads']),
        #     bucket_size=64,
        #     n_hashes=4,
        #     ff_chunks=10,
        #     lsh_dropout=0.1,
        #     layer_dropout=float(args['--dropout']),
        #     weight_tie=True,
        #     causal=True,
        #     use_full_attn=True,
        #     use_metadata=False
        # )
        model = TransformerClassifier(
            vocab=vocab,
            ninp=int(args['--word-embed-size']),
            nhid=int(args['--hidden-size']),
            nlayers=int(args['--transformer-depth']),
            nhead=int(args['--transformer-heads'])
        )
    elif model_type == "bert":
        model = BertClassifier.from_pretrained(args['--base-bert-path'], num_labels=len(vocab.icd))
        model.unfreeze_bert_encoder()
    elif model_type == "bert-metadata":
        model = BertClassifierWithMetadataXS.from_pretrained(args['--base-bert-path'], num_labels=len(vocab.icd))
        model.unfreeze_bert_encoder()
    else:
        raise NotImplementedError("Invalid model type")

    if args['--cuda']:
        model.cuda()

    model.train()

    if model_type != "bert" and model_type != "bert-metadata":
        # We are pre-loading weights from the BERT models so no need to set our random weights
        uniform_init = float(args['--uniform-init'])
        if np.abs(uniform_init) > 0.:
            print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
            for p in model.parameters():
                p.data.uniform_(-uniform_init, uniform_init)

    print('Using device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = total_loss = total_processed_words = 0
    total_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    print('Starting training {}...'.format(model_type))
    logger.info('Starting training {}...'.format(model_type))

    while True:
        epoch += 1

        for batch_src_text, batch_src_lengths, batch_icd_codes, batch_icd_descs in batch_iter(train_data, batch_size=train_batch_size, shuffle=False):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(batch_src_text)

            if args['--model'] == "bert":
                input_ids = torch.tensor([f.input_ids for f in batch_src_text], dtype=torch.long, device=device)
                input_mask = torch.tensor([f.input_mask for f in batch_src_text], dtype=torch.long, device=device)
                segment_ids = torch.tensor([f.segment_ids for f in batch_src_text], dtype=torch.long, device=device)
                model_output = model(input_ids, segment_ids, input_mask)
            elif args['--model'] == "bert-metadata":
                input_ids = torch.tensor([f.input_ids for f in batch_src_text], dtype=torch.long, device=device)
                input_mask = torch.tensor([f.input_mask for f in batch_src_text], dtype=torch.long, device=device)
                segment_ids = torch.tensor([f.segment_ids for f in batch_src_text], dtype=torch.long, device=device)
                input_ids_metadata = torch.tensor([f.input_ids_metadata for f in batch_src_text], dtype=torch.long, device=device)
                model_output = model(input_ids, segment_ids, input_mask,
                                     metadata_input_ids=input_ids_metadata,
                                     metadata_len=int(args['--meta-len']))
            elif args['--model'] == "reformer-metadata":
                batch_src_text_tensor = model.vocab.discharge.to_input_tensor(batch_src_text, device)
                batch_icd_descs_tensor = model.vocab.discharge.to_input_tensor(batch_icd_descs, device)
                batch_src_lengths = torch.tensor(batch_src_lengths, dtype=torch.long, device=device)
                model_output = model(batch_src_text_tensor, batch_src_lengths, metadata_ids=batch_icd_descs_tensor, metadata_len=int(args['--meta-len']))
            else:
                batch_src_text_tensor = model.vocab.discharge.to_input_tensor(batch_src_text, device)
                batch_src_lengths = torch.tensor(batch_src_lengths, dtype=torch.long, device=device)
                model_output = model(batch_src_text_tensor, batch_src_lengths)

            batch_icd_codes = torch.tensor(batch_icd_codes, dtype=torch.float, device=device)

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(model_output.view(-1, len(vocab.icd)), batch_icd_codes.view(-1, len(vocab.icd)))
            batch_loss = loss
            print("Loss is {}".format(loss))

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

            if model_type == "bert" or model_type == "bert-metadata":
                total_processed_words += sum(len(s.input_ids) for s in batch_src_text)
            else:
                total_processed_words += sum(len(s) for s in batch_src_text)
            total_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f ' \
                      'total examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         total_loss,
                                                                                         total_examples,
                                                                                         total_processed_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                logger.info('epoch %d, iter %d, avg. loss %.2f ' \
                      'total examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         total_loss,
                                                                                         total_examples,
                                                                                         total_processed_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))

                train_time = time.time()
                total_examples = total_loss = total_processed_words = 0.

            # Perform validation at the pre-configured rate, or if we have reached the maximum number of epochs
            if train_iter % valid_niter == 0 or epoch == int(args['--max-epoch']):
                precision, recall, f1, accuracy = evaluate_model_with_dev(args, model, dev_data, device, batch_size=int(args["--batch-size"]), tokenizer=tokenizer)   # dev batch size can be a bit larger
                valid_metric = f1

                print('VALIDATION: {} | Precision {}, recall {}, f1 {}, accuracy: {}'.format(train_iter, precision, recall, f1, accuracy), file=sys.stderr)
                logger.info('VALIDATION: {} | Precision {}, recall {}, f1 {}, accuracy: {}'.format(train_iter, precision, recall, f1, accuracy))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    logger.info('save currently the best model to [%s]' % model_save_path)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    logger.info('hit patience %d' % patience)

                    print("Saving latest model...")
                    model.save(model_save_path + ".latest")

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        logger.info('hit #%d trial' % num_trial)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            logger.info('early stop!')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        if model_type == "bert" or model_type == "bert-metadata":
                            model.load_state_dict(params)
                        else:
                            model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
                else:
                    print("Saving latest model...")
                    model.save(model_save_path + ".latest")

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                logger.info('reached maximum number of epochs!')
                exit(0)

def predict_icd_codes(args: Dict[str, str]):
    """
    Freezes the model and predicts the ICD codes. Writes the evaluation scores to console and writes predictions to file
    :param args: the command line arguments
    """
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    model_type = args["--model"]

    # Set to True if we want to inject a 'CLS' token as part of the input
    # Recommended to use False
    use_cls = False

    vocab = Vocab.load(args['--vocab-test'])

    tokenizer = None
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args['--base-bert-path'])
        test_source_text, test_source_lengths, test_icd_codes, test_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['TEST_SOURCE_FILE'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer)
    elif model_type == "bert-metadata":
        tokenizer = BertTokenizer.from_pretrained(args['--base-bert-path'])
        test_source_text, test_source_lengths, test_icd_codes, test_icd_descs = read_source_text_for_bert_with_metadata(
            file_path=args['TEST_SOURCE_FILE'],
            target_length=int(args['--target-length']),
            tokenizer=tokenizer,
            metadata_file_path=args['--icd-desc-file']
        )
    else:
        test_source_text, test_source_lengths, test_icd_codes, test_icd_descs = read_source_text(args['TEST_SOURCE_FILE'], target_length=int(args["--target-length"]), use_cls=use_cls, metadata_file_path=args['--icd-desc-file'])
    test_data = list(zip(test_source_text, test_source_lengths, test_icd_codes, test_icd_descs))

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)

    # TODO: Remove vocab from models
    if args["--model"] == "baseline":
        model = DischargeLSTM.load(args['MODEL_PATH'])
    elif args["--model"] == "reformer":
        model = ReformerClassifier.load(args['MODEL_PATH'])
    elif args["--model"] == "reformer-metadata":
        model = ReformerClassifier.load(args['MODEL_PATH'])
    elif args["--model"] == "transformer":
        # model = ReformerClassifier.load(args['MODEL_PATH'])
        model = TransformerClassifier.load(args['MODEL_PATH'])
    elif args["--model"] == "bert":
        model = BertClassifier.load(args['MODEL_PATH'], args['--base-bert-path'], num_labels=len(vocab.icd))
        model.freeze_bert_encoder()
    elif args["--model"] == "bert-metadata":
        model = BertClassifierWithMetadataXS.load(args['MODEL_PATH'], args['--base-bert-path'], num_labels=len(vocab.icd))
        model.freeze_bert_encoder()
    else:
        raise NotImplementedError("Not implemented")

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    was_training = model.training
    model.eval()

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Using device: %s' % device, file=sys.stderr)

    preds, icds = predict_output(args, model, test_data, device, batch_size=int(args["--batch-size"]), tokenizer=tokenizer)

    if was_training: model.train(was_training)

    precision, recall, f1, accuracy = evaluate_scores(icds, preds)
    print('Precision {}, recall {}, f1 {}, accuracy: {}'.format(precision, recall, f1, accuracy), file=sys.stderr)
    logger.info('Precision {}, recall {}, f1 {}, accuracy: {}'.format(precision, recall, f1, accuracy))

    with open(args['OUTPUT_FILE'], 'w') as f:
        csv_writer = csv.writer(f)
        decoded_icd_preds = vocab.icd.from_one_hot(preds)
        for decoded_icd_pred in decoded_icd_preds:
            csv_writer.writerow(decoded_icd_pred)


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

    logger.info("Running with the following args: {}".format(args))

    if args['train']:
        train(args)
    elif args['predict']:
        predict_icd_codes(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
