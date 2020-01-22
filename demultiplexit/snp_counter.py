from typing import Tuple

import numpy as np
import pysam

from demultiplexit.cellranger_specific import compute_p_mistake, discard_read
from demultiplexit.utils import hash_string


class ChromosomeSNPLookup:
    def __init__(self, positions: np.ndarray):
        """
        Allows fast checking of intersection with SNPs, a bit memory-inefficient, but quite fast.
        Important note is that information from only one chromosome can be stored, so only positions are stored.
        :param positions: zero-based (!) positions of SNPs on chromosomes
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


def compress_cbub_reads_group_to_snips(
        reads,
        snp_lookup: ChromosomeSNPLookup,
        compute_p_read_misaligned,
        skip_complete_duplicates=True
) -> Tuple[float, dict]:
    """
    Take a group of reads and leaves only information about SNP positions
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

    return p_group_misaligned, SNPs


def compress_old_cbub_groups(
        threshold_position: int,
        cbub2position_and_reads,
        cbub2qual_and_snps,
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
                # no reads in this fragment, why caring? just forget about it
                continue
            p_group_misaligned, snips = compress_cbub_reads_group_to_snips(
                reads, snp_lookup, compute_p_read_misaligned=compute_p_read_misaligned
            )
            if len(snips) == 0:
                # there is no reason to care about this one
                continue
            if (cbub not in cbub2qual_and_snps) or (cbub2qual_and_snps[cbub][0] > p_group_misaligned):
                cbub2qual_and_snps[cbub] = p_group_misaligned, snips

    for cbub in to_remove:
        cbub2position_and_reads.pop(cbub)


def count_call_variants_for_chromosome(
        bamfile_or_filename,
        chromosome,
        chromosome_snps_zero_based,
        cellbarcode_compressor,
        compute_p_read_misaligned=compute_p_mistake,
        discard_read=discard_read,
        verbose=False,
):
    prev_segment = None
    cbub2qual_and_snps = {}
    cbub2position_and_reads = {}
    snp_lookup = ChromosomeSNPLookup(chromosome_snps_zero_based)
    if isinstance(bamfile_or_filename, str):
        bamfile_or_filename = pysam.AlignmentFile(bamfile_or_filename)

    for read in bamfile_or_filename.fetch(chromosome):
        curr_segment = read.pos // 10 ** 6
        if curr_segment != prev_segment:
            if verbose:
                print(f"before clearing: {len(cbub2position_and_reads):10} {len(cbub2qual_and_snps):10}", read.pos)
            compress_old_cbub_groups(
                read.pos - 10 ** 6, cbub2position_and_reads, cbub2qual_and_snps, snp_lookup, compute_p_read_misaligned,
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
        np.inf, cbub2position_and_reads, cbub2qual_and_snps, snp_lookup, compute_p_read_misaligned,
    )
    return cbub2qual_and_snps
