import json
import pprint
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

from scrnaseq_demux import ProbabilisticGenotypes, Demultiplexer, count_snps, BarcodeHandler
from scrnaseq_demux.utils import Timer

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
        self.chromosome2snp_calls = self.count_snps()

    @staticmethod
    def check_genotypes_are_identical(genotypes1: ProbabilisticGenotypes, genotypes2: ProbabilisticGenotypes):
        assert len(genotypes1.snp2snpid) == len(genotypes2.snp2snpid)
        assert genotypes1.genotype_names == genotypes2.genotype_names

        assert genotypes1.n_variants == genotypes2.n_variants

        snp2sindex1, _beta_priors1 = genotypes1._generate_canonical_representation()
        snp2sindex2, _beta_priors2 = genotypes2._generate_canonical_representation()
        assert snp2sindex1 == snp2sindex2
        assert np.allclose(_beta_priors1, _beta_priors2)

    def test_export_and_load_of_genotypes(self):
        with Timer('check export'):
            posterior_filename = here / '_temp_exported_prior.tsv'
            self.genotypes_used.save_betas(posterior_filename)

        with Timer('check import'):
            genotypes_loaded = ProbabilisticGenotypes(self.used_genotypes_names)
            genotypes_loaded.add_prior_betas(posterior_filename, prior_strength=1.)

        with Timer('verifying agreement'):
            self.check_genotypes_are_identical(self.genotypes_used, genotypes_loaded)

    def test_reverse_order_of_addition_to_genotypes(self):
        genotypes_straight = ProbabilisticGenotypes(self.used_genotypes_names)
        genotypes_straight.add_prior_betas(self.prior_filename, prior_strength=10)
        genotypes_straight.add_vcf(self.vcf_filename, prior_strength=100)

        genotypes_reverse = ProbabilisticGenotypes(self.used_genotypes_names)
        genotypes_reverse.add_prior_betas(self.prior_filename, prior_strength=10)
        genotypes_reverse.add_vcf(self.vcf_filename, prior_strength=100)

        self.check_genotypes_are_identical(genotypes_straight, genotypes_reverse)

        kwargs = dict(
            chromosome2compressed_snp_calls=self.chromosome2snp_calls,
            barcode_handler=self.barcode_handler,
            only_singlets=True,
        )

        posteriors1 = Demultiplexer.predict_posteriors(**kwargs, genotypes=genotypes_straight)
        posteriors2 = Demultiplexer.predict_posteriors(**kwargs, genotypes=genotypes_reverse)
        assert np.all(posteriors1[0] == posteriors2[0])
        assert np.all(posteriors1[1] == posteriors2[1])

    def test_exporting_and_loading_genotypes(self):
        import tempfile
        with tempfile.NamedTemporaryFile() as temp:
            self.genotypes_used.save_betas(temp.name)
            loaded_genotypes = ProbabilisticGenotypes(genotype_names=self.genotypes_used.genotype_names,
                                                      default_prior=self.genotypes_used.default_prior)
            loaded_genotypes.add_prior_betas(temp.name, prior_strength=1.)
        self.check_genotypes_are_identical(self.genotypes_used, loaded_genotypes)

    def count_snps(self):
        with Timer('snp_counting'):
            chromosome2compressed_snp_calls = count_snps(
                bamfile_location=self.bamfile_location,
                chromosome2positions=self.genotypes_used.get_chromosome2positions(),
                barcode_handler=self.barcode_handler,
            )

        counter = {
            chromosome: calls.n_molecules
            for chromosome, calls in chromosome2compressed_snp_calls.items()
        }

        for chromosome, calls in chromosome2compressed_snp_calls.items():
            assert len(calls.molecules) == calls.n_molecules

        pprint.pprint(counter)
        assert counter == {
            'GRCh38_______1': 3918,
            'GRCh38_______10': 765,
            'GRCh38_______11': 7688,
            'GRCh38_______12': 1393,
            'GRCh38_______13': 376,
            'GRCh38_______14': 858,
            'GRCh38_______15': 1261,
            'GRCh38_______16': 1882,
            'GRCh38_______17': 2838,
            'GRCh38_______18': 124,
            'GRCh38_______19': 5428,
            'GRCh38_______2': 2171,
            'GRCh38_______20': 1377,
            'GRCh38_______21': 95,
            'GRCh38_______22': 1764,
            'GRCh38_______3': 2945,
            'GRCh38_______4': 1011,
            'GRCh38_______5': 739,
            'GRCh38_______6': 2122,
            'GRCh38_______7': 2306,
            'GRCh38_______8': 2201,
            'GRCh38_______9': 853,
            'GRCh38_______MT': 24873,
            'GRCh38_______X': 384,
        }

        return chromosome2compressed_snp_calls

    def test_demultiplexing_singlets_vs_doublets(self):
        with Timer('checking alignments of singlets and doublets'):
            kwargs = dict(
                chromosome2compressed_snp_calls=self.chromosome2snp_calls,
                genotypes=self.genotypes_used,
                barcode_handler=self.barcode_handler,
            )
            logits_singlet, prob_singlet = Demultiplexer.predict_posteriors(**kwargs, only_singlets=True)
            logits_doublet, prob_doublet = Demultiplexer.predict_posteriors(**kwargs, only_singlets=False)
            assert np.allclose(logits_singlet, logits_doublet.loc[:, logits_singlet.columns])
            assert np.allclose(prob_singlet.sum(axis=1), 1, atol=1e-4), prob_singlet.sum(axis=1)
            assert np.allclose(prob_doublet.sum(axis=1), 1, atol=1e-4), prob_doublet.sum(axis=1)
            # checking that setting doublet prior to something small is identical to just singlets
            logits_doublet_pseudo, prob_doublet_pseudo = Demultiplexer.predict_posteriors(
                **kwargs, only_singlets=False, doublet_prior=1e-8)
            print(prob_singlet / prob_doublet_pseudo.loc[:, prob_singlet.columns])
            assert np.allclose(prob_singlet, prob_doublet_pseudo.loc[:, prob_singlet.columns], atol=1e-3)

    def test_demultiplexing_agaisnt_historical_result(self):
        with Timer('demultiplexing'):
            for barcode_posterior_probs_df, debug_info in Demultiplexer.staged_genotype_learning(
                    chromosome2compressed_snp_calls=self.chromosome2snp_calls,
                    genotypes=self.genotypes_used,
                    barcode_handler=self.barcode_handler,
                    n_iterations=3):
                print('one more iteration complete')
                logits = debug_info['barcode_logits']
                print('logits info', logits.min())

            mean_probs = softmax(logits, axis=1).mean(axis=0)

            reference_mean_probs = \
                [0.023656892, 0.038816728, 0.039463125, 0.010899138, 0.029516088, 0.08418094, 0.027043846, 0.048838142,
                 0.110589385, 0.012225234, 0.0654298, 0.07042614, 0.081185535, 0.08959608, 0.09606744, 0.01548529,
                 0.015876694, 0.045034245, 0.059479408, 0.023949377, 0.012240634]

            print('mean probs  [', ', '.join(str(x) for x in mean_probs), ']')

            # assert np.allclose(mean_probs, reference_mean_probs, atol=0.01)

        with Timer('demultiplexing again and exporting difference'):
            learnt_genotypes_filename = here / '_learnt_beta_contributions.csv'
            for _, debug_info2 in Demultiplexer.staged_genotype_learning(
                    chromosome2compressed_snp_calls=self.chromosome2snp_calls,
                    genotypes=self.genotypes_used,
                    barcode_handler=self.barcode_handler,
                    n_iterations=3,
                    save_learnt_genotypes_to=str(learnt_genotypes_filename)):
                print('one more iteration complete')

        # check that learnt genotypes are identical
        assert np.allclose(debug_info['genotype_snp_posterior'], debug_info2['genotype_snp_posterior'])

        with Timer('importing difference'):
            genotypes_learnt = ProbabilisticGenotypes(self.used_genotypes_names)
            genotypes_learnt.add_prior_betas(learnt_genotypes_filename, prior_strength=1.)

            # checking snps are identical, but not checking betas
            assert genotypes_learnt._generate_canonical_representation()[0] == \
                   self.genotypes_used._generate_canonical_representation()[0]

            logits2, _ = Demultiplexer.predict_posteriors(
                chromosome2compressed_snp_calls=self.chromosome2snp_calls,
                genotypes=genotypes_learnt,
                barcode_handler=self.barcode_handler,
                only_singlets=True)

            assert np.allclose(logits, logits2)
