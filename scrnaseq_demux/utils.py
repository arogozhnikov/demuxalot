import numpy as np


def hash_string(s):
    """ used to compress UB (molecule barcodes) to group identical ones"""
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