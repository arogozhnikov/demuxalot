def discard_read(read) -> bool:
    if read.get_tag("AS") <= len(read.seq) - 8:
        # more than 2 edits
        return True
    if read.get_tag("NH") > 1:
        # multi-mapped
        return True
    return False


def compute_p_mistake(read):
    if read.get_tag("AS") <= len(read.seq) - 8:
        # more than 2 edits
        return 1
    if read.get_tag("NH") > 1:
        # multi-mapped
        return 1
    if read.mapq < 20:
        # this one should not be triggered because of NH, but just in case
        return 1
    # by default. TODO encounter amount of edits? How to avoid bias in this case?
    return 0.01