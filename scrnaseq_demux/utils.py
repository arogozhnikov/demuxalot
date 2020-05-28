import time

import numpy as np
import pandas as pd
import pysam


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
    def __init__(self, barcodes, RG_tags=None):
        """Barcode handler is needed to compress barcodes to integers,
        because strings take too much space
        :param barcodes: list of strings, each one is barcode (corresponds to barcode in cellranger)
        :param RG_tags: optional list of the same length, used when RG tag should be used as a part of barcode identity.
          RG tag shows original file when reads are merged from multiple bam files into one, because .
          This may be very handy when you merge several bamfiles (e.g. for reproducible unbiased training of genotypes)
        """
        assert not isinstance(barcodes, str), 'construct by passing list of possible barcodes'
        barcodes = list(barcodes)
        self.use_rg = False
        if RG_tags is not None:
            RG_tags = list(RG_tags)
            assert len(barcodes) == len(RG_tags), 'RG tags should be the same length as '
            barcodes = [(barcode, rg) for barcode, rg in zip(barcodes, RG_tags)]
            self.use_rg = True

        assert len(set(barcodes)) == len(barcodes), "all passed barcodes should be unique"
        self.ordered_barcodes = list(sorted(barcodes))
        self.barcode2index = {bc: i for i, bc in enumerate(self.ordered_barcodes)}

    def get_barcode_index(self, read: pysam.AlignedRead):
        """ Returns None if barcode is not in the whitelist, otherwise a small integer """
        if not read.has_tag("CB"):
            return None
        if self.use_rg:
            # require RG tag to be available for each read
            barcode = read.get_tag("CB"), read.get_tag('RG')
        else:
            barcode = read.get_tag("CB")
        return self.barcode2index.get(barcode, None)

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