__version__ = '0.2.0'
__author__ = 'Alex Rogozhnikov'

from .utils import BarcodeHandler
from .snp_counter import count_snps
from .demux import Demultiplexer, ProbabilisticGenotypes
from .snp_detection import detect_snps_positions
