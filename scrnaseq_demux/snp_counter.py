from collections import defaultdict
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pysam

from scrnaseq_demux.cellranger_specific import compute_p_misaligned, discard_read
from scrnaseq_demux.utils import hash_string, BarcodeHandler, compress_base


class ChromosomeSNPLookup:
    def __init__(self, positions: np.ndarray):
        """
        Allows fast checking of intersection with SNPs, a bit memory-inefficient, but quite fast.
        Important note is that information from only one chromosome can be stored, so only positions are kept.
        :param positions: zero-based (!) positions of SNPs on chromosomes. Aligns with pysam enumeration
        """
        assert isinstance(positions, np.ndarray)
        assert np.array_equal(positions, np.sort(positions))
        assert len(positions) < 2 ** 31, "can't handle that big chromosome - will consume too much memory"
        # lookup takes a lot of space, so would be better to either minimize dtype
        # or to keep only a part of it
        self.lookup = np.cumsum(np.bincount(positions + 1)).astype("uint16")
        self.positions = positions

    def snips_exist(self, start, end):
        """ end is excluded, start is included """
        if start >= len(self.lookup):
            return False
        end = min(end, len(self.lookup) - 1)
        return self.lookup[end] != self.lookup[start]

    def get_snps(self, read):
        snps = []  # list of tuples: (reference_position, nucleotide, base mapping quality)
        # fast check first
        if not self.snips_exist(read.reference_start, read.reference_end + 1):
            return snps

        seq = read.seq
        qual = read.query_qualities

        read_position = 0
        refe_position = read.pos

        for code, l in read.cigartuples:
            if code in [0, 7, 8]:
                # coincides or does not coincide
                if self.snips_exist(refe_position, refe_position + l):
                    lo = np.searchsorted(self.positions, refe_position)
                    hi = np.searchsorted(self.positions, refe_position + l)
                    assert hi != lo
                    for ref_position in self.positions[lo:hi]:
                        position_in_read = read_position + (ref_position - refe_position)
                        snps.append((ref_position, seq[position_in_read], qual[position_in_read]))

                refe_position += l
                read_position += l
            elif code in (2, 3):
                # should we include deletion as snp?
                refe_position += l
            elif code in (1, 4, 5, 6):
                read_position += l
            else:
                raise NotImplementedError(f"cigar code unknown {code}")
        return snps


def double_array(array):
    return np.concatenate([array, array], axis=0)


# TODO not ignoring duplicates!
class CompressedSNPCalls:
    def __init__(self, start_snps_size=1000):
        # important. This does NOT remove duplicates
        self.mindex2cb_ub_p_group_misaligned = []
        self.n_snps = 0

        self.sindex2snp_call = np.array(
            [(-1, -1, 255, -1.)] * start_snps_size,
            dtype=[('molecule_index', 'int32'), ('snp_position', 'int32'), ('base_index', 'uint8'),
                   ('p_base_wrong', 'float32')]
        )

    def add_group_snps(self, compressed_cb, compressed_ub, p_group_misaligned, snps):
        if len(snps) + self.n_snps > len(self.sindex2snp_call):
            # double array size while keeping original size
            self.sindex2snp_call = np.concatenate([self.sindex2snp_call, self.sindex2snp_call], axis=0)
        molecule_index = len(self.mindex2cb_ub_p_group_misaligned)
        self.mindex2cb_ub_p_group_misaligned.append((compressed_cb, compressed_ub, p_group_misaligned))

        for reference_position, base, p_base_wrong in snps:
            self.sindex2snp_call[self.n_snps] = (molecule_index, reference_position, compress_base(base), p_base_wrong)
            self.n_snps += 1

    def minimize_memory_footprint(self):
        self.sindex2snp_call = self.sindex2snp_call[:self.n_snps].copy()

    @staticmethod
    def merge(chromosome_snpcalls, chrom_pos2sindex):
        for chromosome, snp_calls in chromosome_snpcalls:
            assert isinstance(snp_calls, CompressedSNPCalls)

        merged_snps = np.array(
            [(-1, -1, 255, -1.)] * sum(x.n_snps for _, x in chromosome_snpcalls),
            dtype=[('mindex', 'int32'), ('sindex', 'int32'), ('base_index', 'uint8'), ('p_base_wrong', 'float32')]
        )
        mindex2bindex = []
        fragment_start = 0
        for chromosome, snp_calls in chromosome_snpcalls:
            fragment_end = fragment_start + snp_calls.n_snps
            fragment = merged_snps[fragment_start:fragment_end]
            source = snp_calls.sindex2snp_call[:snp_calls.n_snps]
            fragment['mindex'] = source['molecule_index'] + len(mindex2bindex)
            fragment['sindex'] = [chrom_pos2sindex[chromosome, pos] for pos in source['snp_position']]
            fragment['base_index'] = source['base_index']
            fragment['p_base_wrong'] = source['p_base_wrong']
            mindex2bindex.extend([cb for cb, ub, p_group_misaligned in snp_calls.mindex2cb_ub_p_group_misaligned])
            fragment_start = fragment_end

        assert fragment_end == len(merged_snps)
        return mindex2bindex, merged_snps


def compress_cbub_reads_group_to_snips(
        reads,
        snp_lookup: ChromosomeSNPLookup,
        compute_p_read_misaligned,
        skip_complete_duplicates=True,
) -> Tuple[float, list]:
    """
    Takes a group of reads and leaves only information about SNP positions
    """
    p_group_misaligned = 1
    processed_positions = set()
    SNPs = {}  # position to list of pairs character, quality
    for read in reads:
        p_read_misaligned = compute_p_read_misaligned(read)
        pos = (read.reference_start, read.reference_end, read.get_tag("AS"))
        if skip_complete_duplicates and (pos in processed_positions):
            # ignoring duplicates with same begin/end and number of mismatches
            # relatively cheap way to exclude complete duplicates, as those do not contribute
            continue
        processed_positions.add(pos)
        p_group_misaligned *= p_read_misaligned

        for reference_position, base, base_qual in snp_lookup.get_snps(read):
            SNPs.setdefault(reference_position, []).append((base, base_qual, p_read_misaligned))

    compressed_snps = []  # (position, base, p_wrong)
    for snp_position, bases_probs in SNPs.items():
        base2p_wrong = defaultdict(lambda: 1)
        for base, base_qual, _p_read_misaligned in bases_probs:
            base2p_wrong[base] *= 0.1 ** (0.1 * min(base_qual, 40))

        if len(base2p_wrong) > 1:
            # molecule should have only one candidate, this this is artifact
            # of reverse transcription or amplification or sequencing
            best_prob = min(base2p_wrong.values())
            # drop poorly sequenced candidate(s), this resolves some obvious conflicts
            base2p_wrong = {
                base: p_wrong
                for base, p_wrong in base2p_wrong.items()
                if p_wrong * 0.01 <= best_prob or p_wrong < 0.001
            }

        # if #candidates is still not one, discard this sample
        if len(base2p_wrong) != 1:
            continue
        (base, p_wrong), = base2p_wrong.items()
        compressed_snps.append((snp_position, base, p_wrong))

    return p_group_misaligned, compressed_snps


def compress_old_cbub_groups(
        threshold_position: int,
        cbub2position_and_reads,
        # cbub2qual_and_snps,
        compressed_snp_calls,
        snp_lookup: ChromosomeSNPLookup,
        compute_p_read_misaligned,
):
    """
    Compresses groups of reads that already left region of our consideration. We keep snp list instead of reads.
    Compressed groups are removed from cbub2position_and_reads, compressed version written to cbub2qual_and_snps
    """
    to_remove = []
    for cbub, (position, reads) in cbub2position_and_reads.items():
        if position < threshold_position:
            to_remove.append(cbub)
            if not snp_lookup.snips_exist(
                    min(read.reference_start for read in reads), max(read.reference_end for read in reads) + 1,
            ):
                # no SNPs in this fragment, just skip it
                continue
            p_group_misaligned, snips = compress_cbub_reads_group_to_snips(
                reads, snp_lookup, compute_p_read_misaligned=compute_p_read_misaligned
            )
            if len(snips) == 0:
                # there is no reason to care about this group, it provides no genotype information
                continue
            # keep only one alignment group per cbub while treating others as misalignment
            compressed_snp_calls.add_group_snps(cbub[0], cbub[1], p_group_misaligned, snips)
            # if (cbub not in cbub2qual_and_snps) or (cbub2qual_and_snps[cbub][0] > p_group_misaligned):
            #     cbub2qual_and_snps[cbub] = p_group_misaligned, snips

    for cbub in to_remove:
        cbub2position_and_reads.pop(cbub)


def count_call_variants_for_chromosome(
        bamfile_or_filename,
        chromosome,
        chromosome_snps_zero_based,
        cellbarcode_compressor,
        compute_p_read_misaligned,
        discard_read,
) -> CompressedSNPCalls:
    prev_segment = None
    # cbub2qual_and_snps = {}
    compressed_snp_calls = CompressedSNPCalls()
    cbub2position_and_reads = {}
    snp_lookup = ChromosomeSNPLookup(chromosome_snps_zero_based)
    if isinstance(bamfile_or_filename, str):
        bamfile_or_filename = pysam.AlignmentFile(bamfile_or_filename)

    for read in bamfile_or_filename.fetch(chromosome):
        curr_segment = read.pos // 10 ** 6
        if curr_segment != prev_segment:
            compress_old_cbub_groups(
                read.pos - 10 ** 6, cbub2position_and_reads, compressed_snp_calls, snp_lookup,
                compute_p_read_misaligned,
            )
            prev_segment = curr_segment

        if discard_read(read):
            continue
        if not read.has_tag("CB"):
            continue
        cb = cellbarcode_compressor(read.get_tag("CB"))
        if cb is None:
            continue
        if not read.has_tag("UB"):
            continue
        ub = hash_string(read.get_tag("UB"))
        cbub = cb, ub
        if cbub not in cbub2position_and_reads:
            cbub2position_and_reads[cbub] = [read.reference_end, [read]]
        else:
            cbub2position_and_reads[cbub][0] = max(read.reference_end, cbub2position_and_reads[cbub][0])
            cbub2position_and_reads[cbub][1].append(read)
    compress_old_cbub_groups(
        np.inf, cbub2position_and_reads, compressed_snp_calls, snp_lookup, compute_p_read_misaligned,
    )
    compressed_snp_calls.minimize_memory_footprint()
    return compressed_snp_calls


def count_snps(
        bamfile_location: str,
        chromosome2positions: Dict[str, np.ndarray],
        barcode_handler: BarcodeHandler,
        joblib_n_jobs=-1,
        joblib_verbosity=11,
        compute_p_misaligned=compute_p_misaligned,
        discard_read=discard_read,
) -> Dict[str, CompressedSNPCalls]:
    """
    Computes which molecules can provide information about SNPs

    :param bamfile_location: bam file, local path. It's going to be extensively read in multiple threads
    :param chromosome2positions: which positions are of interest for each chromosome,
        dictionary mapping chromosome name to np.ndarray of SNP positions within chromosome
    :param barcode_handler: handler which picks
    :param joblib_n_jobs: how many threads to run in parallel
    :param joblib_verbosity: verbosity level as interpreted by joblib
    :param compute_p_misaligned: callback that estimates probability of read being misaligned
    :param discard_read: callback that decides if aligned read should be discarded from demultiplexing analysis

    see cellranger_specific.py for examples of callbacks specific for cellranger

    :return: returns an object which stores information about molecules, their SNPs and barcodes,
        that can be used by demultiplexer
    """
    chromosome2positions = list(chromosome2positions.items())
    with pysam.AlignmentFile(bamfile_location) as f:
        f: pysam.AlignmentFile = f
        chromosome2nreads = {contig.contig: contig.mapped for contig in f.get_index_statistics()}

        def chromosome_order(chromosome, positions):
            # simple estimate for number of SNP calls
            return -chromosome2nreads[chromosome] * len(positions) / f.get_reference_length(chromosome)

        chromosome2positions = list(sorted(chromosome2positions, key=lambda chr_pos: chromosome_order(*chr_pos)))

    with joblib.Parallel(n_jobs=joblib_n_jobs, verbose=joblib_verbosity) as parallel:
        _cbub2qual_and_snps = parallel(
            joblib.delayed(count_call_variants_for_chromosome)(
                bamfile_location,
                chromosome,
                positions,
                cellbarcode_compressor=lambda cb: barcode_handler.barcode2index.get(cb, None),
                compute_p_read_misaligned=compute_p_misaligned,
                discard_read=discard_read,
            )
            for chromosome, positions in chromosome2positions
        )
    chromosome2compressed_snp_calls = dict(zip([chrom for chrom, _ in chromosome2positions], _cbub2qual_and_snps))
    return chromosome2compressed_snp_calls
