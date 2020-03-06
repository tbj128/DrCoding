"""
DrCoding
Takes the MIMIC-III data and preprocesses it in a format that can be used in our model

Tom Jin <tomjin@stanford.edu>

Usage:
    preprocess.py --icdmap=<file> --icd=<file> --notes=<file> --top-k=<int> --top-k-per-patient=<int> [--sample]

Options:
    -h --help                               show this screen.
    --icdmap=<file>                         path to the D_ICD_DIAGNOSES.csv containing description of ICD-9 codes
    --icd=<file>                            path to the DIAGNOSES_ICD.csv containing ICD-9 codes for each patient stay
    --notes=<file>                          path to the NOTEEVENTS.csv containing the clinical notes
    --output=<file>                         output directory
    --sample                                if set, looks at only one discharge summary from the clinical notes
    --top-k=<int>                           filter overall data to keep patients with an ICD code within the top k most frequently occurring ICD codes [default: 50]
    --top-k-per-patient=<int>               the number of ICD codes kept for each patient. eg. value of 1 means to produce the most relevant ICD code for each patient. -1 means that all ICD codes are kept. [default: -1]
"""

from docopt import docopt
import csv
import re

class Preprocess():
    def __init__(self, icdmap, icd, notes, output, take_sample, top_k, top_k_per_patient):
        self.f_icdmap = icdmap
        self.f_icd = icd
        self.f_notes = notes
        self.f_output = output
        self.take_sample = take_sample
        self.top_k = top_k
        self.top_k_per_patient = top_k_per_patient
        self.icd_to_desc = {}
        self.condensed_icd_to_desc = {}
        self.hadmid_with_discharge = set()
        self.hadmid_to_icds = {}
        self.top_icd_codes = set()

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
        with open(self.f_icdmap, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # Skip headers
            for line in reader:
                self.icd_to_desc[line[1]] = line[2]
                condensed_icd = self._raw_icd_to_icd(line[1])
                self.condensed_icd_to_desc[condensed_icd] = line[3]

        print("> Finished getting the ICD map description. Found {}".format(len(self.icd_to_desc)))

    def extract_top_icd_codes(self):
        """
        Extracts the top k ICD codes based on their frequency

        """
        print("> Extracting the top {} ICD codes...".format(self.top_k))
        icd_to_count = {}
        with open(self.f_icd, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # Skip headers
            for line in reader:
                hadmid = line[2]
                if hadmid in self.hadmid_with_discharge:
                    raw_icd = line[4]
                    if raw_icd not in self.icd_to_desc:
                        # This might be an invalid ICD code or a procedure code (which we don't care about)
                        continue
                    if raw_icd.startswith("V") or raw_icd.startswith("E"):
                        # We want to exclude "external" and "supplemental" ICD-9 codes
                        continue

                    # Convert the ICD into its top three digit category
                    icd = self._raw_icd_to_icd(raw_icd)
                    if icd not in icd_to_count:
                        icd_to_count[icd] = 1
                    else:
                        icd_to_count[icd] += 1

        i = 0
        for key, value in sorted(icd_to_count.items(), key=lambda kv: kv[1], reverse=True):
            if i >= self.top_k:
                break
            print("   ICD {} has count {}".format(key, value))
            self.top_icd_codes.add(key)
            i += 1

        print("> Extracted ICD codes: {}".format(list(self.top_icd_codes)))

    def extract_icd_codes(self):
        """
        Example ICD row:

        "ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"
        1297,109,172335,1,"40301"

        """
        print("> Extracting the ICD codes...")

        num_hadmid_with_icd = 0
        num_icds = 0
        with open(self.f_icd, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # skip header
            for line in reader:
                if line[3] == '':
                    continue
                hadmid = line[2]
                seq = int(line[3])
                raw_icd_code = line[4]
                icd_code = self._raw_icd_to_icd(raw_icd_code)
                if icd_code not in self.top_icd_codes:
                    # Passing on this ICD-9 code as it's not in our top K codes
                    continue

                if hadmid not in self.hadmid_with_discharge:
                    # Passing on this patient stay as there was no discharge summary
                    continue

                if hadmid not in self.hadmid_to_icds:
                    self.hadmid_to_icds[hadmid] = {}
                    num_hadmid_with_icd += 1

                if icd_code not in self.hadmid_to_icds[hadmid]:
                    self.hadmid_to_icds[hadmid][icd_code] = seq
                else:
                    # Always take the ICD code that has the lower seq number (higher priority)
                    if self.hadmid_to_icds[hadmid][icd_code] > seq:
                        self.hadmid_to_icds[hadmid][icd_code] = seq

        for k, v in self.hadmid_to_icds.items():
            num_icds += len(v)

        print("> Wrote {} hadmids".format(num_hadmid_with_icd))
        print("> {} ICD codes per hadmid".format(float(num_icds) / float(num_hadmid_with_icd)))

    def extract_hadmids_with_discharge_summaries(self):
        """
        Reads the clinical notes, filtering for discharge notes and removing artifacts
        """
        print("> Extracting the hadmids with discharge summaries")
        num_notes = 0
        with open(self.f_notes, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # Skip headers

            for line in reader:
                hadmid = line[2]
                category = line[6]
                if category == "Discharge summary":
                    self.hadmid_with_discharge.add(hadmid)
                    if self.take_sample and num_notes > 100:
                        break
                if num_notes % 500 == 0:
                    print("   > Processed {} lines so far".format(num_notes), end='\r', flush=True)
                num_notes += 1
        print("> Number of notes = {}".format(num_notes))

    def extract_discharge_summaries(self):
        """
        Reads the clinical notes, filtering for discharge notes and removing artifacts
        """
        print("> Extracting and converting the discharge summaries...")
        i = 0
        num_discharge_summaries = 0

        hadmid_to_note = {}
        with open(self.f_notes, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader, None) # Skip headers

            # Pattern to remove all occurrences of [** name **], which are placeholders for names (HIPAA regulations)
            hipaa_regex = re.compile(r"\[\*\*.*?\*\*\]", re.IGNORECASE)

            # Pattern to keep only alpha characters, some punctuation, and whitespace
            # This has the benefit of removing the hidden ICD-9 codes in some summaries
            regex = re.compile(r"[^a-zA-Z \-,\.]+", re.IGNORECASE)

            phrases_to_remove = ['admission date', 'discharge date', 'date of birth', 'service',
                                 'chief complaint', 'history of present illness',
                                 'past medical history', 'admission diagnosis',
                                 'history of the present illness', 'attending', 'cc', 'dictated by medquist job']
            useless_words = re.compile(r"|".join(phrases_to_remove))

            num_duplicates = 0
            for line in reader:
                hadmid = line[2]
                category = line[6]
                text = line[10]
                if category == "Discharge summary" and hadmid in self.hadmid_to_icds and len(
                        self.hadmid_to_icds[hadmid]) >= self.top_k_per_patient:
                    text = text.lower().replace("\n", " ")  # Convert to lower case and make one line
                    text = re.sub('/', ' ', text)  # Split slashes into two words
                    text = hipaa_regex.sub("", text)  # Remove name placeholders
                    text = regex.sub("", text)  # Remove punctuation, numbers, etc.
                    text = ' '.join([w for w in text.split() if len(w) > 1])  # Remove single letters
                    text = useless_words.sub("", text)
                    if hadmid in hadmid_to_note:
                        hadmid_to_note[hadmid] += " " + text
                        num_duplicates += 1
                    else:
                        hadmid_to_note[hadmid] = text

                    num_discharge_summaries += 1

                    if self.take_sample and num_discharge_summaries > 100:
                        break
                if i % 5000 == 0:
                    print("   > Processed {} lines so far".format(i), end='\r', flush=True)
                i += 1

        print("Duplicates found and merged {}".format(num_duplicates))

        with open(self.f_output + "/notes.processed.txt", "w") as fw:
            notes_writer = csv.writer(fw, delimiter="\t")
            with open(self.f_output + "/icd.processed.txt", "w") as icd_fw:
                icd_writer = csv.writer(icd_fw, delimiter="\t")
                with open(self.f_output + "/icd-desc.processed.txt", "w") as icd_desc_fw:
                    icd_desc_writer = csv.writer(icd_desc_fw, delimiter="\t")

                    for hadmid, text in hadmid_to_note.items():
                        known_icds = []
                        for icd, seq in self.hadmid_to_icds[hadmid].items():
                            known_icds.append({
                                "icd": icd,
                                "seq": seq
                            })
                        known_icds = sorted(known_icds, key=lambda k: k['seq'])
                        known_icds = list(map(lambda x: x["icd"], known_icds))
                        if self.top_k_per_patient == -1:
                            patient_icds = known_icds
                        else:
                            patient_icds = known_icds[:self.top_k_per_patient]

                        icd_writer.writerow([hadmid] + patient_icds)
                        desc_row = [hadmid]
                        for icd in patient_icds:
                            desc_row.append(self.condensed_icd_to_desc[icd])
                        icd_desc_writer.writerow(desc_row)

                        notes_writer.writerow([hadmid, text])
        print("> Finished extracting discharge summaries")
        print("> Number of discharge summaries = {}".format(num_discharge_summaries))


def main():
    """
    Entry point of tool.
    """
    args = docopt(__doc__)
    pp = Preprocess(args["--icdmap"], args["--icd"], args["--notes"], args["--output"], args["--sample"], int(args["--top-k"]), int(args["--top-k-per-patient"]))
    pp.get_icd_map()
    pp.extract_hadmids_with_discharge_summaries()
    pp.extract_top_icd_codes()
    pp.extract_icd_codes()
    pp.extract_discharge_summaries()

if __name__ == '__main__':
    main()
