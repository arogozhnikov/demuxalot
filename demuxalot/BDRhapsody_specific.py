"""
This file defines callbacks that are optimized for BD Rhapsody WTA (RNA) assays
and can overcome some of the issues in BD Rhapsody output.

If some other aligner is used, you can use simpler callbacks (e.g. using only mapq).
"""
from pysam import AlignedRead
from typing import Optional, Tuple

from demuxalot.utils import hash_string


def parse_read(read: AlignedRead, umi_tag="MA", nhits_tag="NH", score_tag="AS",
               score_diff_max = 8, mapq_threshold = 20,
               # max. 2 edits --^
               p_misaligned_default = 0.01) -> Optional[Tuple[float, int]]:
    """
    returns None if read should be ignored.
    Read still can be ignored if it is not in the barcode list
    """
    if read.get_tag(score_tag) <= len(read.seq) - score_diff_max:
        # too many edits
        return None
    if read.get_tag(nhits_tag) > 1:
        # multi-mapped
        return None
    if not read.has_tag(umi_tag):
        # does not have molecule barcode
        return None
    if read.mapq < mapq_threshold:
        # this one should not be triggered because of NH, but just in case
        return None

    p_misaligned = p_misaligned_default  # default value
    ub = hash_string(read.get_tag(umi_tag))
    return p_misaligned, ub
