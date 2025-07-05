import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import pysam
import urllib.request


def hash_string(s) -> int:
    """
    Used to compress UB (molecule barcodes) to group identical ones.
    Mapping is deterministic, unique and fast for UBs used.
    """
    result = 0
    for c in s:
        result *= 5
        result += ord(c)
    # make sure we fit into int32, as a residue by largest prime below 2 ** 32 - 1
    return result % 2147483629

base_lookup = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def compress_base(base: str) -> int:
    return base_lookup[base]


def decompress_base(base_index: int) -> str:
    return 'ACGTN'[base_index]


def fast_np_add_at_1d(x, indices, weights) -> None:
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

    def get_barcode_index(self, read: pysam.AlignedRead) -> Optional[int]:
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
    def from_file(barcodes_filename, **kwargs):
        """
        :param barcodes_filename: path to barcodes.csv or barcodes.csv.gz where each line is a barcode
        :param **kwargs: optional additional keyword arguments to pass down to BarcodeHandler.__init__
        """
        barcodes = pd.read_csv(barcodes_filename, header=None)[0].values.astype("str")
        return BarcodeHandler(barcodes, **kwargs)

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


def as_str(filename) -> str:
    assert isinstance(filename, (str, Path))
    return str(filename)


def download_file(url, local_filename) -> str:
    """ Utility used only in examples """
    if Path(local_filename).exists():
        print(f'file {local_filename} already exists locally')
    else:
        Path(local_filename).parent.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(url, local_filename)
        print(f'downloaded to {local_filename}')
    return local_filename


def summarize_counted_SNPs(snp_counts: Dict[str, 'CompressedSNPCalls']):
    """
    helper function to show number of calls/transcripts available for each barcode
    """
    records = []
    barcode2number_of_calls = Counter()
    barcode2number_of_transcripts = Counter()

    for chromosome, calls in snp_counts.items():
        records.append(dict(
            chromosome=chromosome,
            n_molecules=calls.n_molecules,
            n_snp_calls=calls.n_snp_calls,
        ))

        barcode2number_of_transcripts.update(Counter(calls.molecules['compressed_cb']))
        barcodes = calls.molecules['compressed_cb'][calls.snp_calls['molecule_index']]
        barcode2number_of_calls.update(Counter(barcodes))

    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[12, 5])

    def truncate_at_perc(x, percentile=99.5):
        x = np.asarray(list(x))
        return x.clip(0, np.percentile(x, percentile))

    ax1.hist(
        truncate_at_perc(barcode2number_of_calls.values()),
        histtype='step', bins=20,
    )
    ax1.set_ylabel('barcodes')
    ax1.set_xlabel('SNP calls per droplet')

    ax2.hist(
        truncate_at_perc(barcode2number_of_transcripts.values()),
        histtype='step', bins=20,
    )
    ax2.set_ylabel('number of barcodes')
    ax2.set_xlabel('transcripts per droplet')
    fig.show()

    return pd.DataFrame(records).sort_values('chromosome').set_index('chromosome')


class FeatureLookup:
    """
    Allows efficiently represent a number of integer features as a single dense integer.
    Instance of this class learns a mapping and then allows compressing/uncompressing
    """
    def __init__(self, *features):
        self.n_categories = [np.max(f) + 1 for f in features]
        total_categories = np.prod(self.n_categories)
        if total_categories < 2 ** 7:
            self.dtype = 'int8'
        elif total_categories < 2 ** 15:
            self.dtype = 'int16'
        elif total_categories < 2 ** 31:
            self.dtype = 'int32'
        elif total_categories < 2 ** 63:
            self.dtype = 'int64'
        else:
            raise RuntimeError('too many combinations')

        self._lookup = np.unique(self._to_internal_compressed(*features))

    @property
    def nvalues(self):
        return len(self._lookup)

    def _to_internal_compressed(self, *features):
        result = np.zeros(len(features[0]), dtype=self.dtype)
        assert len(features) == len(self.n_categories)
        for f, n_cats in zip(features, self.n_categories):
            assert f.max() < n_cats
            result *= n_cats
            result += f.astype(self.dtype)
        return result

    def _from_internal_compressed(self, indices):
        result = []
        for n_cats in self.n_categories[::-1]:
            result.append(indices % n_cats)
            indices //= n_cats

        assert np.all(indices == 0)
        return result[::-1]

    def lookup_for_individual_features(self):
        return self._from_internal_compressed(self._lookup)

    def compress(self, *features):
        compressed_index = np.searchsorted(self._lookup, self._to_internal_compressed(*features))
        for reconstructed, original in zip(self.uncompress(compressed_index), features):
            # redundant, checking just in case
            np.testing.assert_equal(original, reconstructed)
        counts_of_compressed = np.bincount(compressed_index, minlength=len(self._lookup))
        return compressed_index, counts_of_compressed

    def uncompress(self, compressed_index):
        return self._from_internal_compressed(self._lookup[compressed_index])


def _compute_qualities(probs: pd.DataFrame, barcode2possible_donors: dict):
    """
    Computes metrics to detect. Input is pd.DataFrame with cols=samples, index=barcode.
    We a priori know that some samples are possible and others are not (possible are passed as true+sample_names).
    Logloss also checks that probabilities are well-calibrated (because typically those are not).
    In the context of this function, doublet G1 + G2 is yet-another-genotype and function makes no difference.
    You need to provide all possible singlet and doublet genotypes
    """
    assert probs.index.isin(barcode2possible_donors).all(), " probs index barcodes should be in the dict "
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-2), "probabilities should sum to one for each barcode"

    donors_in_columns = set(probs.columns)
    # check that all donors are in the list
    for _, donors in barcode2possible_donors.items():
        assert all(d in donors_in_columns for d in donors), f'some of donors not found in probabilities: {donors}'

    loglosses = []
    is_correct = []

    for barcode, sample_probs in probs.iterrows():
        sample_probs: pd.Series = sample_probs
        possible_donors: List[str] = barcode2possible_donors[barcode]
        prob = sample_probs[possible_donors].sum()
        loglosses.append(-np.log(prob.clip(1e-4)))
        # process draws here
        is_correct.append(sample_probs.idxmax() in possible_donors)

    return {
        "logloss": np.mean(loglosses),
        "accuracy": np.mean(is_correct),
        "error rate": 1 - np.mean(is_correct),
    }
