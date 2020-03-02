"""
DrCoding
Statistics

Tom Jin <tomjin@stanford.edu>

Usage:
    statistics.py NOTE_FILE ICD_FILE [options]

Options:
    -h --help                               show this screen.
"""

from docopt import docopt
import numpy as np


class Statistics(object):
    def __init__(self, note_file, icd_file):
        self.icd_file = icd_file
        self.note_file = note_file

    def run(self):
        icd_to_count = {}
        words = []
        with open(self.note_file, 'r') as fn:
            for row in fn:
                row_arr = row.split(" ")
                words.append(len(row_arr))

        avg = np.mean(words)
        median = np.median(words)
        print("Mean words {}".format(avg))
        print("Median words {}".format(median))

        with open(self.icd_file, 'r') as fi:
            for row in fi:
                if row.strip() not in icd_to_count:
                    icd_to_count[row.strip()] = 0
                icd_to_count[row.strip()] += 1

        for k, v in icd_to_count.items():
            print("ICD {} had {} items".format(k, v))


def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = Statistics(args["NOTE_FILE"], args["ICD_FILE"])
    split.run()


if __name__ == '__main__':
    main()
