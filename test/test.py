import json
import pprint
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import unittest
import time

from scrnaseq_demux import ProbabilisticGenotypes, Demultiplexer, count_snps, BarcodeHandler

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
            genotypes_used = ProbabilisticGenotypes(self.used_genotypes_names)
            genotypes_used.add_vcf(self.vcf_filename, prior_strength=100)

        with Timer('update genotypes with new SNPs'):
            # extend genotypes with added SNPs
            # saving learnt genotypes to a separate file
            self.prior_filename = self.prepare_prior_file(genotypes_used)
            genotypes_used.add_prior_betas(self.prior_filename, prior_strength=10)
        self.genotypes_used = genotypes_used
        self.chromosome2cbub2qual_and_snps = self.count_snps()

    def prepare_prior_file(self, genotypes_used):
        chrom2snp_positions_and_stats = joblib.load(
            here / 'chrom2possible0basedpositions_based_on_donors.joblib.pkl')
        df = defaultdict(list)
        for chromosome, (snp_positions, snp_stats) in chrom2snp_positions_and_stats.items():
            for snp_position, snp_stat in zip(snp_positions, snp_stats):
                if (chromosome, snp_position) in genotypes_used.snips:
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

    def check_genotypes_are_identical(self, genotypes1, genotypes2):
        assert len(genotypes1.snips) == len(genotypes2.snips)
        assert genotypes1.genotype_names == genotypes2.genotype_names
        for (chrom, pos), (ref, alt, priors) in genotypes1.snips.items():
            ref2, alt2, priors2 = genotypes2.snips[chrom, pos]
            assert alt == alt2
            assert ref == ref2
            assert np.allclose(priors, priors2)

        snp2sindex1, _, _beta_priors1 = genotypes1.generate_genotype_snp_beta_prior()
        snp2sindex2, _, _beta_priors2 = genotypes2.generate_genotype_snp_beta_prior()
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

    def check_reverse_order_of_addition(self):
        genotypes_reverse = ProbabilisticGenotypes(self.used_genotypes_names)
        genotypes_reverse.add_prior_betas(self.prior_filename, prior_strength=10)
        genotypes_reverse.add_vcf(self.vcf_filename, prior_strength=100)
        self.check_genotypes_are_identical(self.genotypes_used, genotypes_reverse)

        kwargs = dict(
            chromosome2cbub2qual_and_snps=self.chromosome2cbub2qual_and_snps,
            barcode_handler=self.barcode_handler,
            only_singlets=True,
        )

        posteriors1 = Demultiplexer.predict_posteriors(**kwargs, genotypes=self.genotypes_used)
        posteriors2 = Demultiplexer.predict_posteriors(**kwargs, genotypes=genotypes_reverse)
        assert np.all(posteriors1[0] == posteriors2[0])
        assert np.all(posteriors1[1] == posteriors2[1])

    def count_snps(self):
        with Timer('new_snp_counting'):
            chromosome2cbub2qual_and_snps = count_snps(
                bamfile_location=self.bamfile_location,
                chromosome2positions=self.genotypes_used.get_chromosome2positions(),
                barcode_handler=self.barcode_handler,
            )

            counter = {
                chromosome: len(cals) for chromosome, cals in chromosome2cbub2qual_and_snps.items()
            }
            pprint.pprint(counter)

        assert counter == {
            'GRCh38_______1': 3920,
            'GRCh38_______10': 765,
            'GRCh38_______11': 7687,
            'GRCh38_______12': 1393,
            'GRCh38_______13': 376,
            'GRCh38_______14': 851,
            'GRCh38_______15': 1261,
            'GRCh38_______16': 1884,
            'GRCh38_______17': 2838,
            'GRCh38_______18': 124,
            'GRCh38_______19': 5429,
            'GRCh38_______2': 2167,
            'GRCh38_______20': 1378,
            'GRCh38_______21': 95,
            'GRCh38_______22': 1765,
            'GRCh38_______3': 2926,
            'GRCh38_______4': 1011,
            'GRCh38_______5': 740,
            'GRCh38_______6': 2122,
            'GRCh38_______7': 2306,
            'GRCh38_______8': 2202,
            'GRCh38_______9': 854,
            'GRCh38_______MT': 24862,
            'GRCh38_______X': 384,
        }

        for chromosome, cbub2qual_and_snps in chromosome2cbub2qual_and_snps.items():
            print(chromosome, len(cbub2qual_and_snps))
        return chromosome2cbub2qual_and_snps

    def test_demultiplexing_singlets_vs_doublets(self):
        with Timer('checking alignments of singlets and doublets'):
            kwargs = dict(
                chromosome2cbub2qual_and_snps=self.chromosome2cbub2qual_and_snps,
                genotypes=self.genotypes_used,
                barcode_handler=self.barcode_handler,
            )
            logits_singlet, prob_singlet = Demultiplexer.predict_posteriors(**kwargs, only_singlets=True)
            logits_doublet, prob_doublet = Demultiplexer.predict_posteriors(**kwargs, only_singlets=False)
            assert np.allclose(logits_singlet, logits_doublet.loc[:, logits_singlet.columns])
            assert np.allclose(prob_singlet.sum(axis=1), 1)
            assert np.allclose(prob_doublet.sum(axis=1), 1)

    def test_demultiplexing_agaisnt_historical_result(self):
        with Timer('demultiplexing'):
            for barcode_posterior_probs_df, debug_info in Demultiplexer.staged_genotype_learning(
                    chromosome2cbub2qual_and_snps=self.chromosome2cbub2qual_and_snps,
                    genotypes=self.genotypes_used,
                    barcode_handler=self.barcode_handler,
                    n_iterations=3):
                print('one more iteration complete')
                logits = debug_info['barcode_logits']

            print(list(np.max(logits, axis=0)))

            reference_logits = [-4.1255603, -4.2695103, -9.513682, -3.824123, -4.157081, -4.3482676, -3.9152608,
                                -8.900769, -4.4701805, -3.8421426, -4.585875, -4.1683974, -3.307035, -4.4903097,
                                -4.402192, -3.8523905, -3.9055922, -3.9764569, -3.7411292, -3.964514, -3.770989]

            assert np.allclose(np.max(logits, axis=0), reference_logits)

        with Timer('demultiplexing again and exporting difference'):
            learnt_genotypes_filename = here / '_learnt_beta_contributions.csv'
            for _, debug_info2 in Demultiplexer.staged_genotype_learning(
                    chromosome2cbub2qual_and_snps=self.chromosome2cbub2qual_and_snps,
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
                chromosome2cbub2qual_and_snps=self.chromosome2cbub2qual_and_snps,
                genotypes=genotypes_learnt,
                barcode_handler=self.barcode_handler,
                only_singlets=True)

            assert np.allclose(logits, logits2)
