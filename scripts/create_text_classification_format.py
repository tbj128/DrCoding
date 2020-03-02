"""
DrCoding
Creates a tiny two-class dataset from the pre-split data for debugging purposes

Tom Jin <tomjin@stanford.edu>

Usage:
    create_text_classification_format.py NOTE_FILE ICD_FILE OUTPUT_PATH [options]

Options:
    -h --help                               show this screen.
"""

from docopt import docopt
from sklearn.model_selection import train_test_split


class CreateTextClassificationFormat(object):
    def __init__(self, note_file, icd_file, output_folder):
        self.icd_file = icd_file
        self.note_file = note_file
        self.output_folder = output_folder

    def _write(self, out_file, arr):
        with open(self.output_folder + '/' + out_file, 'w') as f:
            for row in arr:
                f.write(row + "\n")

    def run(self):
        notes = []
        icds = []
        with open(self.note_file, 'r') as fn:
            with open(self.icd_file, 'r') as fi:
                print("Reading note file...")
                for row in fn:
                    notes.append(row.strip())
                print("Finished reading note file.")
                print("Reading ICD file...")
                for row in fi:
                    icds.append(row.strip())
                print("Finished reading ICD file.")

        output_train = []
        output_test = []
        for i, icd in enumerate(icds):
            if i < len(notes) * 0.7:
                output_train.append(icds[i] + "," + notes[i])
            else:
                output_test.append(icds[i] + "," + notes[i])

        self._write("text_classification.train", output_train)
        self._write("text_classification.test", output_test)



def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = CreateTextClassificationFormat(args["NOTE_FILE"], args["ICD_FILE"], args["OUTPUT_PATH"])
    split.run()


if __name__ == '__main__':
    main()
