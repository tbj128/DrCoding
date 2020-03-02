"""
DrCoding
Creates a tiny two-class dataset from the pre-split data for debugging purposes

Tom Jin <tomjin@stanford.edu>

Usage:
    create_tiny.py NOTE_FILE ICD_FILE OUTPUT_PATH [options]

Options:
    -h --help                               show this screen.
    --classes=<int>                         number of classes that the tiny set should contain
    --size=<int>                            number of entries to generate for the tiny set
"""

from docopt import docopt
from sklearn.model_selection import train_test_split


class CreateTiny(object):
    def __init__(self, note_file, icd_file, output_folder, num_classes, size):
        self.icd_file = icd_file
        self.note_file = note_file
        self.output_folder = output_folder
        self.num_classes = num_classes
        self.size = size

    def _write(self, out_file, arr):
        with open(self.output_folder + '/' + out_file, 'w') as f:
            for row in arr:
                f.write(row + "\n")

    def run(self):
        notes = []
        icds = []
        classes = set()
        with open(self.note_file, 'r') as fn:
            with open(self.icd_file, 'r') as fi:
                print("Reading note file...")
                for row in fn:
                    notes.append(row.strip())
                print("Finished reading note file.")
                print("Reading ICD file...")
                for row in fi:
                    icds.append(row.strip())
                    if row not in classes and len(classes) < self.num_classes:
                        classes.add(row.strip())
                print("Finished reading ICD file.")

        print("Classes {}".format(classes))

        output_notes = []
        output_icds = []
        icd_to_count = {}
        row = 0
        for i, icd in enumerate(icds):
            if icd in classes:
                output_notes.append(notes[i])
                output_icds.append(icds[i])
                if icd not in icd_to_count:
                    icd_to_count[icd] = 1
                else:
                    icd_to_count[icd] += 1
                row += 1
            if row >= self.size:
                break

        print("Output notes had size {}".format(len(output_notes)))

        self._write('note.tiny.train', output_notes)
        self._write('icd.tiny.train', output_icds)

        self._write('note.tiny.dev', output_notes[0:min(10, self.size)])
        self._write('icd.tiny.dev', output_icds[0:min(10, self.size)])

        for k, v in icd_to_count.items():
            print("ICD {} had {} items".format(k, v))


def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = CreateTiny(args["NOTE_FILE"], args["ICD_FILE"], args["OUTPUT_PATH"], int(args["--classes"]), int(args["--size"]))
    split.run()


if __name__ == '__main__':
    main()
