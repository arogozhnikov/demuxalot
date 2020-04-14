import time

import numpy as np
import pandas as pd


def hash_string(s):
    """
    Used to compress UB (molecule barcodes) to group identical ones.
    Mapping is deterministic, unique and fast for UBs used.
    """
    result = 0
    for c in s:
        result *= 5
        result += ord(c)
    return result


base_lookup = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def compress_base(base):
    return base_lookup[base]


def decompress_base(base_index):
    return 'ACGTN'[base_index]


for base, base_index in base_lookup.items():
    assert decompress_base(compress_base(base)) == base


def fast_np_add_at_1d(x, indices, weights):
    x[:] = x + np.bincount(indices, weights=weights, minlength=len(x))


class BarcodeHandler:
    def __init__(self, barcodes):
        """Barcode handler is needed to compress barcodes to integers,
        because strings take too much space  """
        assert not isinstance(barcodes, str), 'construct by passing list of possible barcodes'
        barcodes = list(barcodes)
        assert len(set(barcodes)) == len(barcodes), "all passed barcodes should be unique"
        self.ordered_barcodes = barcodes
        self.barcode2index = {bc: i for i, bc in enumerate(barcodes)}

    @staticmethod
    def from_file(barcodes_filename):
        """
        :param barcodes_filename: path to barcodes.csv or barcodes.csv.gz where each line is a barcode
        """
        barcodes = pd.read_csv(barcodes_filename, header=None)[0].values
        return BarcodeHandler(barcodes)


def read_vcf_to_header_and_pandas(vcf_filename):
    """
    :param vcf_filename: vcf filename
    :return: list of commented lines (VCF header), and dataframe with all SNPs.
        Positions in the output are 0-based
    """
    header_lines = []
    with open(vcf_filename) as f:
        for line in f:
            if line.startswith("##"):
                header_lines.append(line)
            else:
                break

    df = pd.read_csv(vcf_filename, sep="\t", skiprows=len(header_lines))
    assert list(df.columns[:8]) == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    # switching to zero-based
    df["POS"] -= 1
    return header_lines, df.rename(columns={"#CHROM": "CHROM"})


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.time_taken = time.time() - self.start_time
        print("Timer {} completed in  {:.3f} seconds".format(self.name, self.time_taken))