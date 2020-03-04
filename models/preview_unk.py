"""
DrCoding
Given a note file, create a preview of what the model sees after passing through the vocab

Tom Jin <tomjin@stanford.edu>

Usage:
    create_icd_descriptions.py NOTE_FILE VOCAB OUTPUT_FILE [options]

Options:
    -h --help                               show this screen.
    --use_cls=<bool>                        use classification token? [default: True]
    --target-length=<int>                   max length of each input sequence [default: 1000]
"""
import csv

import re

import torch
from docopt import docopt

from utils import read_source_text
from vocab import Vocab


class PreviewUnk(object):
    def __init__(self, note_file, vocab_file, output_file, use_cls, target_length):
        self.note_file = note_file
        self.vocab = Vocab.load(vocab_file)
        self.output_file = output_file
        self.use_cls = use_cls
        self.target_length = target_length

    def _write(self, suffix, arr):
        with open(self.output_file + suffix, 'w') as f:
            writer = csv.writer(f, delimiter=",")
            for row in arr:
                writer.writerow(row)

    def run(self):
        print("Reading ICD file...")
        train_source_text, train_source_lengths = read_source_text(self.note_file,
                                                                   target_length=int(self.target_length),
                                                                   use_cls=self.use_cls)
        print("Finished reading ICD file.")
        self._write(".orig", train_source_text)

        output_inds = self.vocab.discharge.words2indices(train_source_text)
        output = []
        for line in output_inds:
            rehydrated = self.vocab.discharge.indices2words(line)
            output.append(rehydrated)

        self._write(".rehydrate", output)

def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = PreviewUnk(args["NOTE_FILE"], args["VOCAB"], args["OUTPUT_FILE"], args["--use_cls"], args["--target-length"])
    split.run()


if __name__ == '__main__':
    main()
