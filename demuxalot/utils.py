import time
from collections import Counter
from pathlib import Path

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


def compress_base(base: str) -> int:
    return base_lookup[base]


def decompress_base(base_index: int) -> str:
    return 'ACGTN'[base_index]


def fast_np_add_at_1d(x, indices, weights):
    x[:] = x + np.bincount(indices, weights=weights, minlength=len(x))


class BarcodeHandler:
    def __init__(self, barcodes, RG_tags=None, tag="CB"):
        """Barcode handler is needed to compress barcodes to integers,
        because strings take too much space
        :param barcodes: list of strings, each one is barcode (corresponds to barcode in cellranger)
        :param RG_tags: optional list of the same length, used when RG tag should be used as a part of barcode identity.
          RG tag shows original file when reads are merged from multiple bam files into one.
          This is very handy when you merge several bamfiles (e.g. for reproducible unbiased training of genotypes).
          Don't forget to pass '-r' as an argument to samtools merge.
        :param tag: tag in BAM-file that keeps (corrected) barcode. Default is 'CB' (from cellranger)
        """
        assert not isinstance(barcodes, (str, Path)), 'construct by passing list of possible barcodes'
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
        self.tag = tag

    @property
    def n_barcodes(self):
        return len(self.barcode2index)

    def get_barcode_index(self, read: pysam.AlignedRead):
        """ Returns None if barcode is not in the whitelist, otherwise a small integer """
        if not read.has_tag(self.tag):
            return None
        if self.use_rg:
            # require RG tag to be available for each read
            barcode = read.get_tag(self.tag), read.get_tag("RG")
        else:
            barcode = read.get_tag(self.tag)
        return self.barcode2index.get(barcode, None)

    @staticmethod
    def from_file(barcodes_filename):
        """
        :param barcodes_filename: path to barcodes.csv or barcodes.csv.gz where each line is a barcode
        """
        barcodes = pd.read_csv(barcodes_filename, header=None)[0].values
        return BarcodeHandler(barcodes)

    def filter_to_rg_value(self, rg_value):
        """ Create a copy of this handler with only barcodes specific to one original file described by RG tag """
        assert self.use_rg
        result = BarcodeHandler(self.barcode2index, tag=self.tag)
        result.barcode2index = {
            # replace inappropriate barcodes with dummy values to keep order
            (barcode if rg == rg_value else index): index
            for (barcode, rg), index in self.barcode2index.items()
        }
        result.ordered_barcodes = list(result.barcode2index)
        result.use_rg = False
        return result

    def __repr__(self):
        if not self.use_rg:
            return f'<BarcodeHandler with {self.n_barcodes} barcodes>'
        else:
            rg_stats = Counter(rg for barcode, rg in self.barcode2index)
            stats = ''
            for rg_code, count in rg_stats.most_common():
                stats += f'{rg_code}: {count} \n'
            return f'<BarcodeHandler with {self.n_barcodes} barcodes. Number of barcodes for RG codes: {rg_stats}>'


def read_vcf_to_header_and_pandas(vcf_filename):
    """
    Not super-reliable, but quite efficient and convenient method to parse VCF files.
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


def as_str(filename):
    assert isinstance(filename, (str, Path))
    return str(filename)


