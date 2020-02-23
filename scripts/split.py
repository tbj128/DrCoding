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

from docopt import docopt
from sklearn.model_selection import train_test_split


class Split(object):
    def __init__(self, note_file, icd_file, output_folder, size):
        self.icd_file = icd_file
        self.note_file = note_file
        self.output_folder = output_folder
        self.size = size

    def _write(self, out_file, tiny_out_file, arr):
        with open(self.output_folder + '/' + out_file, 'w') as f:
            f.writelines(arr)

        if tiny_out_file is not None:
            with open(self.output_folder + '/' + tiny_out_file, 'w') as f:
                for i, row in enumerate(arr):
                    if i >= self.size:
                        break
                    f.write(row + "\n")

    def run(self):
        notes = []
        icds = []
        with open(self.note_file, 'r') as fn:
            with open(self.icd_file, 'r') as fi:
                print("Reading note file...")
                for row in fn:
                    notes.append(row)
                print("Finished reading note file.")
                print("Reading ICD file...")
                for row in fi:
                    icds.append(row)
                print("Finished reading ICD file.")

        print("Splitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(notes, icds, test_size=0.2)
        print("Finished splitting train/test")

        self._write('note.test', 'note.tiny.test', X_test)
        self._write('icd.test', 'icd.tiny.test', y_test)

        print("Splitting train/dev...")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        self._write('note.dev', 'note.tiny.dev', X_val)
        self._write('icd.dev', 'icd.tiny.dev', y_val)
        self._write('note.train', 'note.tiny.train', X_train)
        self._write('icd.train', 'icd.tiny.train', y_train)

        # Write a tiny 'test' file that contains a portion of the tiny train data
        # (this will be used to test our model to make sure it is working correctly)
        self._write('note.overfit.tiny.test', None, X_train[:10])
        self._write('icd.overfit.tiny.test', None, y_train[:10])

        print("   Generated {} train entries".format(len(X_train)))
        print("   Generated {} val entries".format(len(X_val)))
        print("   Generated {} test entries".format(len(X_test)))


def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = Split(args["NOTE_FILE"], args["ICD_FILE"], args["OUTPUT_PATH"], int(args["--size"]))
    split.run()


if __name__ == '__main__':
    main()
