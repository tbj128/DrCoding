"""
DrCoding
Given a ICD code file, create another file with the same order but containing the descriptions instead

Tom Jin <tomjin@stanford.edu>

Usage:
    create_icd_descriptions.py D_ICD_FILE ICD_FILE OUTPUT_FILE [options]

Options:
    -h --help                               show this screen.
"""
import csv

import re
from docopt import docopt


class CreateICDDescriptions(object):
    def __init__(self, d_icd_file, icd_file, output_file):
        self.d_icd_file = d_icd_file
        self.icd_file = icd_file
        self.output_file = output_file
        self.icd_to_desc = {}

    def _write(self, arr):
        with open(self.output_file, 'w') as f:
            for row in arr:
                f.write(row + "\n")

    def _raw_icd_to_icd(self, raw_icd):
        """
        Convert the ICD into its top three digit category
        """
        icd = raw_icd[:3]
        return icd

    def get_icd_map(self):
        """
        Example ICD map row:

        189,"01190","Pulmonary TB NOS-unsp...

        """
        print("> Getting the ICD map description...")
        regex = re.compile(r"[^a-zA-Z]+", re.IGNORECASE)
        with open(self.d_icd_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # Skip headers
            for line in reader:
                raw_icd = line[1]
                desc = line[3]
                icd = self._raw_icd_to_icd(raw_icd)
                if icd not in self.icd_to_desc:
                    self.icd_to_desc[icd] = ""

                uniq_words = set(self.icd_to_desc[icd].split(" "))
                for word in desc.split(" "):
                    w = regex.sub("", word.strip()).lower()
                    if w not in uniq_words:
                        self.icd_to_desc[icd] += " " + w
                self.icd_to_desc[icd] = self.icd_to_desc[icd].strip()

        print("> Finished getting the ICD map description. Found {}".format(len(self.icd_to_desc)))

    def run(self):
        output = []
        self.get_icd_map()

        with open(self.icd_file, 'r') as fi:
            print("Reading ICD file...")
            for row in fi:
                icd = row.strip()
                if icd in self.icd_to_desc:
                    output.append(self.icd_to_desc[icd])
                else:
                    print("FOUND UNKNOWN")
                    output.append("<unk>")
            print("Finished reading ICD file.")
        self._write(output)

def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    split = CreateICDDescriptions(args["D_ICD_FILE"], args["ICD_FILE"], args["OUTPUT_FILE"])
    split.run()


if __name__ == '__main__':
    main()
