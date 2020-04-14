import json
import unittest
from pathlib import Path

import numpy as np

from scrnaseq_demux import ProbabilisticGenotypes, count_snps, BarcodeHandler
from scrnaseq_demux.utils import Timer
from hashlib import md5

here = Path(__file__).parent


class TestClass(unittest.TestCase):
    def setUp(self):
        with open(here / 'calibration/lane2guessed_donor.json') as f:
            lane2inferred_genotypes = json.load(f)

        self.bamfile_location = str(here / 'composed_1_perlane_sorted.bam')

        with Timer('barcodes'):
            self.barcode_handler = BarcodeHandler.from_file(here / 'composed_barcodes_1.tsv')

        with Timer('construct genotypes from GSA'):
            self.vcf_filename = here / 'system1_merged_v4_mini_cleared.vcf'

            self.used_genotypes_names = list(np.unique(sum(lane2inferred_genotypes.values(), [])).tolist())
            # genotypes_used = ProbabilisticGenotypes_old(self.used_genotypes_names)
            genotypes_used = ProbabilisticGenotypes(self.used_genotypes_names)
            genotypes_used.add_vcf(self.vcf_filename, prior_strength=100)

        with Timer('update genotypes with new SNPs'):
            # extend genotypes with added SNPs
            # saving learnt genotypes to a separate file
            self.prior_filename = here / 'new_snps.csv'
            genotypes_used.add_prior_betas(self.prior_filename, prior_strength=10)

        self.genotypes_used = genotypes_used

    def test_counting(self):
        chromosome2positions = self.genotypes_used.get_chromosome2positions()
        with Timer('new_snp_counting'):
            chromosome2compressed_snp_calls = count_snps(
                bamfile_location=self.bamfile_location,
                chromosome2positions=chromosome2positions,
                barcode_handler=self.barcode_handler,
            )

        counter = {
            chromosome: (calls.n_molecules, calls.n_snp_calls)
            for chromosome, calls in chromosome2compressed_snp_calls.items()
        }

        # checking match of molecules
        assert counter == {
            'GRCh38_______1': (3918, 4380),
            'GRCh38_______10': (765, 837),
            'GRCh38_______11': (7688, 8605),
            'GRCh38_______12': (1393, 1428),
            'GRCh38_______13': (376, 383),
            'GRCh38_______14': (858, 1195),
            'GRCh38_______15': (1261, 1747),
            'GRCh38_______16': (1882, 2075),
            'GRCh38_______17': (2838, 3587),
            'GRCh38_______18': (124, 124),
            'GRCh38_______19': (5428, 6432),
            'GRCh38_______2': (2171, 2664),
            'GRCh38_______20': (1377, 1437),
            'GRCh38_______21': (95, 95),
            'GRCh38_______22': (1764, 2133),
            'GRCh38_______3': (2945, 3517),
            'GRCh38_______4': (1011, 1121),
            'GRCh38_______5': (739, 754),
            'GRCh38_______6': (2122, 2613),
            'GRCh38_______7': (2306, 3372),
            'GRCh38_______8': (2201, 2412),
            'GRCh38_______9': (853, 932),
            'GRCh38_______MT': (24873, 51053),
            'GRCh38_______X': (384, 404)
        }

        # checking all calls belong to positions passed
        positions_set = {
            (chromosome, position)
            for chromosome, positions in chromosome2positions.items()
            for position in positions
        }

        all_calls = []
        for chromosome, calls in chromosome2compressed_snp_calls.items():
            calls.minimize_memory_footprint()
            assert len(calls.molecules) == calls.n_molecules
            assert len(calls.snp_calls) == calls.n_snp_calls

            for molecule_index, snp_position, base_index, p_base_wrong in calls.snp_calls:
                cb, ub, p_misaligned = calls.molecules[molecule_index]
                assert (chromosome, snp_position) in positions_set
                # TODO return this check?
                # assert decompress_base(base_index) in 'ACGT'
                all_calls.append(
                    (chromosome, int(snp_position), int(base_index), float(p_base_wrong), float(p_misaligned)))

        all_calls = list(sorted(all_calls))
        all_calls_without_group_p = [call[:4] for call in all_calls]
        all_calls_without_probs = [call[:3] for call in all_calls]

        hash_no_probs = md5(json.dumps(all_calls_without_probs, sort_keys=True).encode('utf-8')).hexdigest()
        assert hash_no_probs == '18815f3bfee5e3dedc8797520676f9bd'
        print('hash_no_probs', hash_no_probs)

        hash_no_group_misalignment = \
            md5(json.dumps(all_calls_without_group_p, sort_keys=True).encode('utf-8')).hexdigest()
        print('hash_no_misalignment', hash_no_group_misalignment)
        assert hash_no_group_misalignment == 'ad9fe6edf69c58b487aef03d07d0be64'

        # just verifying reproducibility of hash
        hash1 = md5(json.dumps(all_calls, sort_keys=True).encode('utf-8')).hexdigest()
        hash2 = md5(json.dumps(all_calls, sort_keys=True).encode('utf-8')).hexdigest()
        print('hash = ', hash1)
        assert hash1 == hash2, 'hashing irreproducible, super weird'
        assert hash1 == '575eb6f063564596d0eb3f79a8eb5d34'
