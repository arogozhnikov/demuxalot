from collections import defaultdict
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pysam
from pathlib import Path

from .cellranger_specific import compute_p_misaligned, discard_read
from .utils import hash_string, BarcodeHandler, compress_base, as_str


class ChromosomeSNPLookup:
    def __init__(self, positions: np.ndarray):
        """
        Allows fast checking of intersection with SNPs, a bit memory-inefficient, but quite fast.
        Information from only one chromosome can be stored, so only positions are kept.
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

        # aligned pair of positions in read and in reference
        read_position = 0
        refe_position = read.pos

        for code, l in read.cigartuples:
            if code in [0, 7, 8]:
                # coincides or does not coincide
                if self.snips_exist(refe_position, refe_position + l):
                    lo, hi = np.searchsorted(self.positions, [refe_position, refe_position + l])
                    for ref_position in self.positions[lo:hi]:
                        position_in_read = read_position + (ref_position - refe_position)
                        snps.append((ref_position, seq[position_in_read], qual[position_in_read]))

                refe_position += l
                read_position += l
            elif code in (2, 3):
                # deletions
                refe_position += l
            elif code in (1, 4, 5, 6):
                # insertions and skips
                read_position += l
            else:
                raise NotImplementedError(f"cigar code unknown {code}")
        return snps


def double_array(array):
    # double array size while keeping original information and preserve dtype. Minimal implementation
    return np.concatenate([array, array], axis=0)


class CompressedSNPCalls:
    def __init__(self, start_snps_size=1024, start_molecule_size=128):
        """
        Keeps calls from one chromosome in a compressed format
        :param start_snps_size:
        :param start_molecule_size:
        """
        # internal representation of molecules and calls is structured numpy arrays
        # numpy can't add elements dynamically so we do that manually.
        # First elements of array are occupied, number of elements is kept in counters
        self.n_molecules = 0
        self.molecules = np.array(
            [(-1, -1, -1.)] * start_molecule_size,
            dtype=[('compressed_cb', 'int32'), ('compressed_ub', 'int32'), ('p_group_misaligned', 'float32')]
        )

        self.n_snp_calls = 0
        self.snp_calls = np.array(
            [(-1, -1, 255, -1.)] * start_snps_size,
            dtype=[('molecule_index', 'int32'), ('snp_position', 'int32'), ('base_index', 'uint8'),
                   ('p_base_wrong', 'float32')]
        )

    def add_calls_from_read_group(self, compressed_cb, compressed_ub, p_group_misaligned, snps):
        if len(snps) + self.n_snp_calls > len(self.snp_calls):
            self.snp_calls = double_array(self.snp_calls)
        if self.n_molecules == len(self.molecules):
            self.molecules = double_array(self.molecules)

        molecule_index = self.n_molecules
        self.molecules[molecule_index] = (compressed_cb, compressed_ub, p_group_misaligned)
        self.n_molecules += 1

        for reference_position, base, p_base_wrong in snps:
            self.snp_calls[self.n_snp_calls] = (molecule_index, reference_position, compress_base(base), p_base_wrong)
            self.n_snp_calls += 1

    def minimize_memory_footprint(self):
        self.snp_calls = self.snp_calls[:self.n_snp_calls].copy()
        self.molecules = self.molecules[:self.n_molecules].copy()
        assert np.all(self.molecules['p_group_misaligned'] != -1)
        assert np.all(self.snp_calls['p_base_wrong'] != -1)

    @staticmethod
    def concatenate(snp_calls_list: List['CompressedSNPCalls']) -> 'CompressedSNPCalls':
        """ concatenates snp calls from the same chromosome """
        n_molecules = 0
        collected_calls = []
        collected_molecules = []
        for calls in snp_calls_list:
            variant_calls = calls.snp_calls[:calls.n_snp_calls].copy()
            variant_calls['molecule_index'] += n_molecules

            collected_calls.append(variant_calls)
            collected_molecules.append(calls.molecules[:calls.n_molecules])
            # return none
            n_molecules += calls.n_molecules

        result = CompressedSNPCalls()
        result.molecules = np.concatenate(collected_molecules)
        result.n_molecules = len(result.molecules)
        result.snp_calls = np.concatenate(collected_calls)
        result.n_snp_calls = len(result.snp_calls)
        return result


def compress_cbub_reads_group_to_snips(
        reads,
        snp_lookup: ChromosomeSNPLookup,
        compute_p_read_misaligned,
        skip_complete_duplicates=True,
) -> Tuple[float, list]:
    """
    Takes a group of reads with identical CB+UB=UMI (assuming all reads came from the same molecule)
    and leaves only information about SNP positions by aggregating bases across reads
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
            # molecule should have only one candidate, this is an artifact
            # of reverse transcription or amplification or sequencing
            best_prob = min(base2p_wrong.values())
            # drop poorly sequenced candidate(s), this resolves some obvious conflicts
            base2p_wrong = {
                base: p_wrong
                for base, p_wrong in base2p_wrong.items()
                if p_wrong * 0.01 <= best_prob or p_wrong < min(0.001, 1e8 * best_prob)
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
        compressed_snp_calls,
        snp_lookup: ChromosomeSNPLookup,
        compute_p_read_misaligned,
):
    """
    Compresses groups of reads that already left region of our consideration.
    Compressed groups are removed from cbub2position_and_reads and only snps are kept in composed_snp_calls.
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
            compressed_snp_calls.add_calls_from_read_group(cbub[0], cbub[1], p_group_misaligned, snips)

    for cbub in to_remove:
        cbub2position_and_reads.pop(cbub)


def count_call_variants_for_chromosome(
        bamfile_or_filename,
        chromosome,
        chromosome_snps_zero_based,
        barcode_handler: BarcodeHandler,
        compute_p_read_misaligned,
        discard_read,
        start=None,
        stop=None,
) -> Tuple[str, CompressedSNPCalls]:
    prev_segment = None
    compressed_snp_calls = CompressedSNPCalls()
    cbub2position_and_reads = {}
    snp_lookup = ChromosomeSNPLookup(chromosome_snps_zero_based)
    if isinstance(bamfile_or_filename, (str, Path)):
        bamfile_or_filename = pysam.AlignmentFile(as_str(bamfile_or_filename))

    for read in bamfile_or_filename.fetch(chromosome, start=start, stop=stop):
        curr_segment = read.pos // 1000
        if curr_segment != prev_segment:
            compress_old_cbub_groups(
                read.pos - 1000, cbub2position_and_reads, compressed_snp_calls, snp_lookup,
                compute_p_read_misaligned,
            )
            prev_segment = curr_segment

        if discard_read(read):
            continue
        cb = barcode_handler.get_barcode_index(read)
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
    return chromosome, compressed_snp_calls


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

    :param bamfile_location: local path to bam file. File is going to be extensively read in multiple threads
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
    jobs = prepare_counting_tasks(bamfile_location, chromosome2positions, barcode_handler=barcode_handler)
    with joblib.Parallel(n_jobs=joblib_n_jobs, verbose=joblib_verbosity, pre_dispatch='all') as parallel:
        chromosome2compressed_snp_calls = parallel(
            joblib.delayed(count_call_variants_for_chromosome)(
                bamfile,
                chromosome,
                positions,
                start=start,
                stop=stop,
                barcode_handler=barcode_handler,
                compute_p_read_misaligned=compute_p_misaligned,
                discard_read=discard_read,
            )
            for bamfile, chromosome, start, stop, positions, barcode_handler in jobs
        )
    _chr2calls = defaultdict(list)
    for chromosome, calls in chromosome2compressed_snp_calls:
        _chr2calls[chromosome].append(calls)

    chromosome2compressed_snp_calls = {
        chromosome: CompressedSNPCalls.concatenate(chromosome_calls)
        for chromosome, chromosome_calls
        in _chr2calls.items()
    }

    return chromosome2compressed_snp_calls


def prepare_counting_tasks(
        bamfile_location,
        chromosome2positions: Dict[str, np.ndarray],
        barcode_handler: BarcodeHandler,
        n_reads_per_job: int = 10_000_000,
        minimum_fragment_length_per_job: int = 5_000,
        minimum_overlap: int = 100,
):
    """
    Split calling of a file into subtasks.
    Each subtask defined by genomic region and non-empty list of positions
    """
    if isinstance(bamfile_location, dict):
        rg2bamfile_location = bamfile_location
        tasks = []
        assert barcode_handler.use_rg, 'barcode handler should use RG tag'
        for rg in set(rg for tag, rg in barcode_handler.barcode2index):
            assert rg in rg2bamfile_location, f'{rg} has no matching path in bamfile_location parameter'
            tasks.extend(prepare_counting_tasks(
                rg2bamfile_location[rg],
                chromosome2positions=chromosome2positions,
                barcode_handler=barcode_handler.filter_to_rg_value(rg),
                n_reads_per_job=n_reads_per_job,
                minimum_fragment_length_per_job=minimum_fragment_length_per_job,
                minimum_overlap=minimum_overlap,
            ))
        return tasks

    with pysam.AlignmentFile(as_str(bamfile_location)) as f:
        chromosome2n_reads = {contig.contig: contig.mapped for contig in f.get_index_statistics()}

        tasks = []  # chromosome, start, stop, positions
        for chromosome, positions in chromosome2positions.items():
            length = f.get_reference_length(chromosome)
            n_jobs = min(
                chromosome2n_reads[chromosome] // n_reads_per_job,
                length // minimum_fragment_length_per_job
            )
            n_jobs = max(1, n_jobs)

            # now need to find a good split
            split_ids = np.searchsorted(positions, np.linspace(0, length, n_jobs + 1)[1:-1])
            for positions_subset in np.split(positions, split_ids):
                if len(positions_subset) == 0:
                    continue
                start = max(0, min(positions_subset) - minimum_overlap)
                stop = min(length, max(positions_subset) + minimum_overlap)
                task = (bamfile_location, chromosome, start, stop, positions_subset, barcode_handler)
                # very naive and strange heuristic for how long each task will take
                # needed to assign more weight to small regions with deep coverage and many SNPs
                complexity = len(positions_subset) * chromosome2n_reads[chromosome] / length ** 0.5
                tasks.append((-complexity, task))

    # complex tasks first
    tasks = [task for complexity, task in sorted(tasks)]
    return tasks
