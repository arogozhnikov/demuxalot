import json
import pprint
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import unittest
import time
from scipy.special import softmax

from scrnaseq_demux import ProbabilisticGenotypes, Demultiplexer, count_snps, BarcodeHandler
from scrnaseq_demux.demux import ProbabilisticGenotypes_old

here = Path(__file__).parent


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.time_taken = time.time() - self.start_time
        print("Timer {} completed in  {:.3f} seconds".format(self.name, self.time_taken))


class TestClass(unittest.TestCase):
    def setUp(self):
        with open(here / 'lane2guessed_donor.json') as f:
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
            self.prior_filename = self.prepare_prior_file(genotypes_used)
            genotypes_used.add_prior_betas(self.prior_filename, prior_strength=10)

        self.genotypes_used = genotypes_used
        self.chromosome2snp_calls = self.count_snps()

    def prepare_prior_file(self, genotypes_used: ProbabilisticGenotypes):
        chrom2snp_positions_and_stats = joblib.load(
            here / 'chrom2possible0basedpositions_based_on_donors.joblib.pkl')
        df = defaultdict(list)
        for chromosome, (snp_positions, snp_stats) in chrom2snp_positions_and_stats.items():
            for snp_position, snp_stat in zip(snp_positions, snp_stats):
                if any((chromosome, snp_position, base) in genotypes_used.snp2snpid for base in 'ACGT'):
                    continue

                _, _, alt, ref = np.argsort(snp_stat)
                alt_count, ref_count = snp_stat[alt], snp_stat[ref]
                alt_count, ref_count = np.asarray([alt_count, ref_count]) / (alt_count + ref_count)
                df['CHROM'].append(chromosome)
                df['POS'].append(snp_position)
                df['BASE'].append('ACGT'[ref])
                df['DEFAULT_PRIOR'].append(ref_count)

                df['CHROM'].append(chromosome)
                df['POS'].append(snp_position)
                df['BASE'].append('ACGT'[alt])
                df['DEFAULT_PRIOR'].append(alt_count)
        prior_filename = here / 'new_snips.csv'
        pd.DataFrame(df).to_csv(prior_filename, sep='\t', index=False)
        print(f'Added {len(df["CHROM"]) // 2} new snps in total')
        return prior_filename

    def check_genotypes_are_identical(self, genotypes1: ProbabilisticGenotypes, genotypes2: ProbabilisticGenotypes):
        assert len(genotypes1.snp2snpid) == len(genotypes2.snp2snpid)
        assert genotypes1.genotype_names == genotypes2.genotype_names

        assert genotypes1.n_variants == genotypes2.n_variants

        snp2sindex1, snp2ref_alt1, _beta_priors1 = genotypes1.generate_genotype_snp_beta_prior()
        snp2sindex2, snp2ref_alt2, _beta_priors2 = genotypes2.generate_genotype_snp_beta_prior()
        assert snp2sindex1 == snp2sindex2
        assert snp2ref_alt1 == snp2ref_alt2
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

    def check_reverse_order_of_addition(self):
        genotypes_reverse = ProbabilisticGenotypes(self.used_genotypes_names)
        genotypes_reverse.add_prior_betas(self.prior_filename, prior_strength=10)
        genotypes_reverse.add_vcf(self.vcf_filename, prior_strength=100)
        self.check_genotypes_are_identical(self.genotypes_used, genotypes_reverse)

        kwargs = dict(
            chromosome2compressed_snp_calls=self.chromosome2snp_calls,
            barcode_handler=self.barcode_handler,
            only_singlets=True,
        )

        posteriors1 = Demultiplexer.predict_posteriors(**kwargs, genotypes=self.genotypes_used)
        posteriors2 = Demultiplexer.predict_posteriors(**kwargs, genotypes=genotypes_reverse)
        assert np.all(posteriors1[0] == posteriors2[0])
        assert np.all(posteriors1[1] == posteriors2[1])

    def count_snps(self):
        with Timer('new_snp_counting'):
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

    def test_old_vs_new_genotypes(self):
        genotypes_new = ProbabilisticGenotypes(self.used_genotypes_names)
        genotypes_new.add_vcf(self.vcf_filename, prior_strength=100)

        genotypes_old = ProbabilisticGenotypes_old(self.used_genotypes_names)
        genotypes_old.add_vcf(self.vcf_filename, prior_strength=100)

        self.prior_filename = self.prepare_prior_file(genotypes_new)

        for _ in range(2):
            snp2sindex1, snp2ref_alt1, genotype_snp_beta_prior1 = genotypes_new.generate_genotype_snp_beta_prior()
            snp2sindex2, snp2ref_alt2, genotype_snp_beta_prior2 = genotypes_old.generate_genotype_snp_beta_prior()
            assert snp2sindex1 == snp2sindex2
            assert snp2ref_alt1 == snp2ref_alt2
            # allows off by one
            coincided = np.isclose(genotype_snp_beta_prior1, genotype_snp_beta_prior2, atol=1.1, rtol=1e-1) \
                        | (genotype_snp_beta_prior1 == genotype_snp_beta_prior2)
            assert np.all(coincided)
            genotypes_new.add_prior_betas(self.prior_filename, prior_strength=10)
            genotypes_old.add_prior_betas(self.prior_filename, prior_strength=10)

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

            # checking snps are identical
            assert genotypes_learnt.generate_genotype_snp_beta_prior()[:2] == \
                   self.genotypes_used.generate_genotype_snp_beta_prior()[:2]

            logits2, _ = Demultiplexer.predict_posteriors(
                chromosome2compressed_snp_calls=self.chromosome2snp_calls,
                genotypes=genotypes_learnt,
                barcode_handler=self.barcode_handler,
                only_singlets=True)

            assert np.allclose(logits, logits2)
