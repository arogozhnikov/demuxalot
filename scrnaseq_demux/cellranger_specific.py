"""
This file defines callbacks that are optimized for cellranger
and can overcome some of issues in cellranger output.

If some other aligner is used, you can use simpler callbacks (e.g. using only mapq).
"""
from pysam import AlignedRead


def discard_read(read: AlignedRead) -> bool:
    if read.get_tag("AS") <= len(read.seq) - 8:
        # more than 2 edits
        return True
    if read.get_tag("NH") > 1:
        # multi-mapped
        return True
    return False


def compute_p_misaligned(read: AlignedRead) -> float:
    if read.get_tag("AS") <= len(read.seq) - 8:
        # more than 2 edits. Suspicious
        # This cuts out information about too small genes, but it will be ignored only during demultiplexing
        return 1
    if read.get_tag("NH") > 1:
        # multi-mapped
        return 1
    if read.mapq < 20:
        # this one should not be triggered because of NH, but just in case
        return 1
    # by default.
    return 0.01