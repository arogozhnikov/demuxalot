"""
This file defines callbacks that are optimized for cellranger
and can overcome some of issues in cellranger output.

If some other aligner is used, you can use simpler callbacks (e.g. using only mapq).
"""
from pysam import AlignedRead
from typing import Optional, Tuple

from demuxalot.utils import hash_string


def parse_read(read: AlignedRead) -> Optional[Tuple[float, int]]:
    """
    returns None if read should be ignored.
    Read still can be ignored if it is not in the barcode list
    """
    if read.get_tag("AS") <= len(read.seq) - 8:
        # more than 2 edits
        return None
    if read.get_tag("NH") > 1:
        # multi-mapped
        return None
    if not read.has_tag("UB"):
        # does not have molecule barcode
        return None
    if read.mapq < 20:
        # this one should not be triggered because of NH, but just in case
        return None

    p_misaligned = 0.01  # default value
    ub = hash_string(read.get_tag("UB"))
    return p_misaligned, ub
