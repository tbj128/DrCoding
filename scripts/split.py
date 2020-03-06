"""
DrCoding
Splits the preprocessed data into train, dev, and test sets. Also creates a tiny set for debugging purposes

Tom Jin <tomjin@stanford.edu>

Usage:
    split.py NOTE_FILE ICD_FILE OUTPUT_PATH [options]

Options:
    -h --help                               show this screen.
    --size=<int>                            number of entries to generate for the tiny set
"""
import json

from docopt import docopt
from sklearn.model_selection import train_test_split
import csv

class Split(object):
    def __init__(self, note_file, icd_file, output_folder, size):
        self.icd_file = icd_file
        self.note_file = note_file
        self.output_folder = output_folder
        self.size = size

    def _write(self, out_file, tiny_out_file, x, y):
        with open(self.output_folder + '/' + out_file, 'w') as f:
            with open(self.output_folder + '/' + tiny_out_file, 'w') as fs:
                f_writer = csv.writer(f)
                fs_writer = csv.writer(fs)
                i = 0
                for line in x:
                    out = []
                    out.append(line[0]) # hadmid
                    out.append(line[1]) # text
                    out.extend(y[i]) # ICD codes
                    f_writer.writerow(out)
                    if i < self.size:
                        fs_writer.writerow(out)
                    i += 1

    def to_one_hot(self, icd_to_index, icds):
        retval = [0] * len(icd_to_index)
        for icd in icds:
            retval[icd_to_index[icd]] = 1
        return retval

    def run(self):
        notes = []
        icds = []
        icd_to_index = {}

        with open(self.icd_file, 'r') as fi:
            icd_reader = csv.reader(fi, delimiter="\t")
            for row in icd_reader:
                for icd in row[1:]:
                    if icd not in icd_to_index:
                        icd_to_index[icd] = len(icd_to_index)

        with open(self.note_file, 'r') as fn:
            note_reader = csv.reader(fn, delimiter="\t")
            with open(self.icd_file, 'r') as fi:
                icd_reader = csv.reader(fi, delimiter="\t")
                print("Reading note file...")
                for row in note_reader:
                    notes.append(row)
                print("Finished reading note file.")
                print("Reading ICD file...")
                for row in icd_reader:
                    icds.append(self.to_one_hot(icd_to_index, row[1:]))
                print("Finished reading ICD file.")

        print("Splitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(notes, icds, test_size=0.2)
        print("Finished splitting train/test")

        self._write('test.data', 'test.tiny.data', X_test, y_test)

        print("Splitting train/dev...")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        self._write('train.data', 'train.tiny.data', X_train, y_train)
        self._write('dev.data', 'dev.tiny.data', X_val, y_val)

        print("   Generated {} train entries".format(len(X_train)))
        print("   Generated {} val entries".format(len(X_val)))
        print("   Generated {} test entries".format(len(X_test)))

        with open(self.output_folder + '/icd.txt', 'w') as fs:
            for icd, index in sorted(icd_to_index.items(), key=lambda x: x[1]):
                fs.write(icd + "\n")

def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = Split(args["NOTE_FILE"], args["ICD_FILE"], args["OUTPUT_PATH"], int(args["--size"]))
    split.run()


if __name__ == '__main__':
    main()
