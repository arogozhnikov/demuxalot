import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from orgalorg.rnaseq.demultiplexing.demultiplexing import compute_qualities

from scrnaseq_demux import ProbabilisticGenotypes, Demultiplexer, count_snps, BarcodeHandler
from scrnaseq_demux.snp_counter import CompressedSNPCalls

here = Path(__file__).parent


class TestClass(unittest.TestCase):
    def setUp(self):
        if Path(here / 'composed_10_perlane_sorted.bam').exists():
            with open(here / 'lane2guessed_donor.json') as f:
                self.lane2inferred_genotypes = {k.split('_')[-1]: v for k, v in json.load(f).items()}
            self.bamfile_location = str(here / 'composed_10_perlane_sorted.bam')
            self.barcode_handler = BarcodeHandler.from_file(here / 'composed_barcodes_10.tsv')
            self.vcf_filename = here / 'system1_merged_v5_for_demultiplexing.vcf'
        else:
            with open('/data/rnaseq/composed/lane2guessed_donor.json') as f:
                self.lane2inferred_genotypes = {k.split('_')[-1]: v for k, v in json.load(f).items()}
            self.bamfile_location = str('/data/rnaseq/composed/composed_200_perlane_sorted.bam')
            self.barcode_handler = BarcodeHandler.from_file('/data/rnaseq/composed/composed_barcodes_200.tsv')
            self.vcf_filename = '/data/genetics/system1_merged_v5_for_demultiplexing.vcf'

        self.used_genotypes_names = [
            'D10', 'D100', 'D101', 'D102', 'D103', 'D104', 'D105', 'D109', 'D11', 'D110', 'D111', 'D112', 'D113',
            'D114', 'D115', 'D116', 'D117', 'D118', 'D119', 'D12', 'D120', 'D121', 'D122', 'D123', 'D124', 'D125',
            'D126', 'D127', 'D13', 'D133', 'D134', 'D139', 'D14', 'D141', 'D143', 'D145', 'D15', 'D16', 'D17', 'D18',
            'D19', 'D199', 'D20', 'D200', 'D201', 'D202', 'D203', 'D204', 'D205', 'D206', 'D207', 'D208', 'D21', 'D22',
            'D3', 'D6', 'D74', 'D75', 'D76', 'D8', 'D84', 'D85', 'D86', 'D87', 'D88', 'D89', 'D9', 'D90', 'D91', 'D93',
            'D94', 'D95', 'D96', 'D97'
        ]

        # self.used_genotypes_names = list(np.unique(sum(self.lane2inferred_genotypes.values(), [])).tolist())
        self.barcode2possible_donors = {
            barcode: self.lane2inferred_genotypes[barcode.split('_')[1]] for barcode in
            self.barcode_handler.barcode2index if barcode.split('_')[1] in self.lane2inferred_genotypes
        }
        # print(self.barcode_handler.ordered_barcodes[:50])
        # print(self.barcode2possible_donors)
        # print(self.lane2inferred_genotypes)
        # assert 0 == 1

        print('loading genotypes')
        self.genotypes_used = ProbabilisticGenotypes(self.used_genotypes_names)
        self.genotypes_used.add_vcf(self.vcf_filename, prior_strength=100)
        # self.genotypes_used.save_betas(here / 'system1_merged_v5_for_demultiplexing.tsv')
        # self.genotypes_used.add_prior_betas(here / 'system1_merged_v5_for_demultiplexing.tsv', prior_strength=100)
        print('loaded genotypes')

    def test_calibration(self):
        chrom2position = self.genotypes_used.get_chromosome2positions()
        chromosome2compressed_snp_calls = count_snps(
            bamfile_location=self.bamfile_location,
            chromosome2positions=chrom2position,
            barcode_handler=self.barcode_handler,
        )

        chromosome2compressed_snp_calls_aggressive = {
            chromosome: self.deduplicate(calls, aggressive=True) for chromosome, calls in
            chromosome2compressed_snp_calls.items()
        }

        chromosome2compressed_snp_calls_nonaggressive = {
            chromosome: self.deduplicate(calls, aggressive=False) for chromosome, calls in
            chromosome2compressed_snp_calls.items()
        }

        calls_collection = [
            ('original', chromosome2compressed_snp_calls),
            # ('non-aggressive', chromosome2compressed_snp_calls_nonaggressive),
            # ('aggressive', chromosome2compressed_snp_calls_aggressive)
        ]
        counts = {
            name: {chrom: c.n_snp_calls for chrom, c in calls.items()} for name, calls in calls_collection
        }
        print(pd.DataFrame(counts))

        def use_molecules(molecule_calls, barcode_calls):
            return molecule_calls

        def use_molecules_zero_mistake(molecule_calls, barcode_calls):
            molecule_calls = molecule_calls.copy()
            molecule_calls['p_base_wrong'] = 1e-10
            return molecule_calls

        def use_molecules_001_mistake(molecule_calls, barcode_calls):
            molecule_calls = molecule_calls.copy()
            molecule_calls['p_base_wrong'] = 0.01
            return molecule_calls

        def use_molecules_01_mistake(molecule_calls, barcode_calls):
            molecule_calls = molecule_calls.copy()
            molecule_calls['p_base_wrong'] = 0.1
            return molecule_calls

        def use_molecules_prob_sum(molecule_calls, barcode_calls):
            molecule_calls = molecule_calls.copy()
            molecule_calls['p_base_wrong'] = (
                    molecule_calls['p_molecule_aligned_wrong'] + molecule_calls['p_base_wrong']).clip(0, 1)
            return molecule_calls

        def use_molecules_prob_sum_sqrt(molecule_calls, barcode_calls):
            molecule_calls = molecule_calls.copy()
            molecule_calls['p_base_wrong'] = (
                    molecule_calls['p_molecule_aligned_wrong'] ** 0.5 + molecule_calls['p_base_wrong']).clip(0, 1)
            return molecule_calls

        def use_barcodes(molecule_calls, barcode_calls):
            return barcode_calls

        def use_barcodes_zero_mistake(molecule_calls, barcode_calls):
            barcode_calls = barcode_calls.copy()
            barcode_calls['p_base_wrong'] = 1e-10
            return barcode_calls

        def use_barcodes_001_mistake(molecule_calls, barcode_calls):
            barcode_calls = barcode_calls.copy()
            barcode_calls['p_base_wrong'] = 0.01
            return barcode_calls

        def use_barcodes_01_mistake(molecule_calls, barcode_calls):
            barcode_calls = barcode_calls.copy()
            barcode_calls['p_base_wrong'] = 0.1
            return barcode_calls

        def deduplicate_all_molecules_to_barcodes_and_001(molecule_calls, barcode_calls):
            cbub2lowest_p = {}
            for variant_id, compressed_cb, compressed_ub, p_base_wrong, p_molecule_aligned_wrong in molecule_calls:
                cbub2lowest_p[compressed_cb, compressed_ub] = min(
                    p_molecule_aligned_wrong,
                    cbub2lowest_p.get((compressed_cb, compressed_ub), 10.)
                )

            barcode_calls = {}  # variant_id, barcode_id -> p_base_wrong
            for variant_id, compressed_cb, compressed_ub, p_base_wrong, p_molecule_aligned_wrong in molecule_calls:
                if p_molecule_aligned_wrong > cbub2lowest_p[compressed_cb, compressed_ub]:
                    continue
                barcode_calls[variant_id, compressed_cb] = \
                    barcode_calls.get((variant_id, compressed_cb), 1) * p_base_wrong

            barcode_calls = np.array(
                [(variant_id, cb, p_base_wrong) for (variant_id, cb), p_base_wrong in barcode_calls.items()],
                dtype=[('variant_id', 'int32'), ('compressed_cb', 'int32'), ('p_base_wrong', 'float32')],
            )

            return barcode_calls

        methods = [
            use_molecules, use_molecules_zero_mistake, use_molecules_01_mistake, use_molecules_001_mistake,
            use_molecules_prob_sum, use_molecules_prob_sum_sqrt,
            use_barcodes, use_barcodes_zero_mistake, use_barcodes_01_mistake, use_barcodes_001_mistake,
            deduplicate_all_molecules_to_barcodes_and_001,
        ]
        results = {}
        for method in methods:
            for calls_name, calls in calls_collection:
                logits, probs = self.predict_posteriors(
                    chromosome2compressed_snp_calls=calls,
                    genotypes=self.genotypes_used,
                    barcode_handler=self.barcode_handler,
                    compute_calls=method,
                )
                quals = compute_qualities(
                    probs=probs.loc[list(self.barcode2possible_donors)],
                    barcode2possible_donors=self.barcode2possible_donors,
                )
                results[method.__name__, calls_name] = quals
            print(method.__name__)
        pd.set_option("display.max_colwidth", 500)
        pd.set_option("display.max_columns", 500)
        print(pd.DataFrame(results).T)

    @staticmethod
    def predict_posteriors(
            chromosome2compressed_snp_calls,
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            compute_calls,
            p_genotype_clip=0.01,
    ):
        from scrnaseq_demux.demux import fast_np_add_at_1d, softmax
        variant_index2snp_index, variant_index2betas, molecule_calls, barcode_calls = \
            Demultiplexer.pack_calls(chromosome2compressed_snp_calls, genotypes)

        calls = compute_calls(molecule_calls, barcode_calls)

        n_genotypes = len(genotypes.genotype_names)

        genotype_prob = Demultiplexer._compute_probs_from_betas(
            variant_index2snp_index, variant_index2betas, p_genotype_clip=p_genotype_clip)
        assert np.isfinite(genotype_prob).all()

        barcode_posterior_logits = np.zeros([len(barcode_handler.ordered_barcodes), n_genotypes], dtype="float32")

        column_names = []
        for gindex, genotype in enumerate(genotypes.genotype_names):
            p = genotype_prob[calls['variant_id'], gindex]
            log_penalties = np.log(p * (1 - calls['p_base_wrong']) + calls['p_base_wrong'].clip(1e-4))
            fast_np_add_at_1d(barcode_posterior_logits[:, gindex], calls['compressed_cb'], log_penalties)
            column_names += [genotype]

        logits_df = pd.DataFrame(
            data=barcode_posterior_logits,
            index=list(barcode_handler.ordered_barcodes), columns=column_names,
        )
        logits_df.index.name = 'BARCODE'
        probs_df = pd.DataFrame(
            data=softmax(barcode_posterior_logits, axis=1),
            index=list(barcode_handler.ordered_barcodes), columns=column_names,
        )
        probs_df.index.name = 'BARCODE'
        return logits_df, probs_df

    @staticmethod
    def deduplicate(calls: CompressedSNPCalls, aggressive=False):
        calls.minimize_memory_footprint()
        molecule2probs = {}
        for cb, ub, p_group_misaligned in calls.molecules:
            cbub = cb, ub
            molecule2probs.setdefault(cbub, []).append(p_group_misaligned)

        if aggressive:
            molecule2probs = {molecule: -1 if sum(x == min(probs) for x in probs) > 1 else min(probs) for
                              molecule, probs in molecule2probs.items()}
        else:
            molecule2probs = {molecule: min(probs) for molecule, probs in molecule2probs.items()}

        mask = []
        n_new_molecules = 0
        old2new_molecule_id = {}
        for cb, ub, p_group_misaligned in calls.molecules:
            cbub = cb, ub
            passed = p_group_misaligned <= molecule2probs[cbub]
            mask.append(passed)
            if passed:
                old2new_molecule_id[len(mask) - 1] = n_new_molecules
                n_new_molecules += 1
        print(np.mean(mask), len(mask))

        new_molecules = calls.molecules[np.asarray(mask)]
        new_snp_calls = calls.snp_calls[np.isin(calls.snp_calls['molecule_index'], list(old2new_molecule_id))]
        new_snp_calls['molecule_index'] = [old2new_molecule_id[mi] for mi in new_snp_calls['molecule_index']]

        result = CompressedSNPCalls()
        result.molecules = new_molecules
        result.n_molecules = len(result.molecules)
        result.snp_calls = new_snp_calls
        result.n_snp_calls = len(result.snp_calls)
        return result
