import numpy as np
import pandas as pd


def hash_string(s):
    """
    Used to compress UB (molecule barcodes) to group identical ones.
    Mapping is unique and fast for
    """
    result = 0
    for c in s:
        result *= 5
        result += ord(c)
    return result


def fast_np_add_at_1d(x, indices, weights):
    x[:] = x + np.bincount(indices, weights=weights, minlength=len(x))


class BarcodeHandler:
    def __init__(self, barcodes):
        """Barcode handler is needed to compress barcodes to integers,
        because strings take too much space  """
        barcodes = list(barcodes)
        assert len(set(barcodes)) == len(barcodes), "all passed barcodes should be unique"
        self.barcode2index = {bc: i for i, bc in enumerate(barcodes)}
        self.index2barcode = {i: bc for i, bc in enumerate(barcodes)}


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
