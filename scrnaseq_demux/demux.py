from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import pysam
from scipy.special import softmax

from scrnaseq_demux.snp_counter import CompressedSNPCalls
from scrnaseq_demux.utils import fast_np_add_at_1d, BarcodeHandler, compress_base


class ProbabilisticGenotypes:
    def __init__(self, genotype_names: List[str], default_prior=1.):
        """
        ProbabilisticGenotype represents our accumulated knowledge about SNPs (and SNPs only) for genotypes.
        Can aggregate information from GSA, our prior guesses and genotype information learnt from RNAseq.
        Genotype names can't be changed once object is created.
        Class doesn't handle more than one SNP at position (examples are A/T and A/C at the same position),
        so only the first received SNP for position is kept.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas (akin to coefficients in beta distribution).
        """
        self.default_prior = default_prior
        self.snp2snpid = {}  # chrom, pos, base -> index in variant_betas
        self.genotype_names = list(genotype_names)
        assert (np.sort(self.genotype_names) == self.genotype_names).all(), 'please order genotype names'

        self.n_variants = 0
        self.variant_betas = np.zeros([32768, len(self.genotype_names)], 'float32') + self.default_prior

    def __repr__(self):
        return f'<Genotypes with {len(self.genotype_names)} genotypes: {self.genotype_names} ' \
               f'and {len(self.snp2snpid)} SNVs >'

    def extend_variants(self, n_samples=1):
        while n_samples + self.n_variants > len(self.variant_betas):
            self.variant_betas = np.concatenate(
                [self.variant_betas, np.zeros_like(self.variant_betas) + self.default_prior], axis=0)

    def add_vcf(self, vcf_file_name, prior_strength=100):
        """
        Add information from parsed VCF
        :param vcf_file_name: path to VCF file. Only diploid values are accepted (0/0, 0/1, 1/1, ./.).
            Should contain all genotypes of interest. Can contain additional genotypes, but those will be ignored.
        """
        n_skipped_snps = 0
        for snp in pysam.VariantFile(vcf_file_name).fetch():
            if any(len(option) != 1 for option in snp.alleles):
                print('skipping non-snp, alleles = ', snp.alleles, snp.chrom, snp.pos)
                continue
            self.extend_variants(len(snp.alleles))

            snp_ids = []
            alleles = snp.alleles
            if len(set(alleles)) != len(alleles):
                n_skipped_snps += 1
                continue
            for allele in alleles:
                # pysam enumerates starting from one, thus -1
                variant = (snp.chrom, snp.pos - 1, allele)
                if variant not in self.snp2snpid:
                    self.snp2snpid[variant] = self.n_variants
                    self.n_variants += 1
                snp_ids.append(self.snp2snpid[variant])

            assert len(set(snp_ids)) == len(snp_ids), (snp_ids, snp.chrom, snp.pos, snp.alleles)

            contribution = np.zeros([len(snp_ids), len(self.genotype_names)], dtype='float32')
            for donor_id, donor in enumerate(self.genotype_names):
                called_values = snp.samples[donor]['GT']
                for call in called_values:
                    if call is not None:
                        # contribution is split between called values
                        contribution[call, donor_id] += prior_strength / len(called_values)
            not_provided = contribution.sum(axis=0) == 0
            if np.sum(~not_provided) < 2:
                n_skipped_snps += 1
                continue
            for row in range(len(contribution)):
                contribution[row, not_provided] = contribution[row, ~not_provided].mean()
            self.variant_betas[snp_ids] += contribution
        if n_skipped_snps > 0:
            print('skipped', n_skipped_snps, 'SNVs')

    def add_prior_betas(self, prior_filename, *, prior_strength):
        """
        Betas is the way Demultiplexer stores learnt genotypes. It's csv file with posterior weights.
        :param prior_filename: path to file
        :param prior_strength: float, controls impact of learnt genotypes.
        """
        prior_knowledge = pd.read_csv(prior_filename, sep='\t')
        tech_columns = ['CHROM', 'POS', 'BASE', 'DEFAULT_PRIOR']
        for column in tech_columns:
            assert column in prior_knowledge.columns
        gt_names_in_prior = [column for column in prior_knowledge.columns if column not in tech_columns]
        print('Provided prior information about genotypes:', gt_names_in_prior)
        for genotype in self.genotype_names:
            if genotype not in gt_names_in_prior:
                print(f'no information for genotype {genotype}, filling with default')
                prior_knowledge[genotype] = prior_knowledge['DEFAULT_PRIOR']

        prior_knowledge[self.genotype_names] *= prior_strength

        variant_indices = []
        for variant in zip(prior_knowledge['CHROM'], prior_knowledge['POS'], prior_knowledge['BASE']):
            if variant not in self.snp2snpid:
                self.extend_variants(1)
                self.snp2snpid[variant] = self.n_variants
                self.n_variants += 1
            variant_indices.append(self.snp2snpid[variant])

        for donor_id, donor in enumerate(self.genotype_names):
            np.add.at(
                self.variant_betas[:, donor_id],
                variant_indices,
                prior_knowledge[donor],
            )

    def get_chromosome2positions(self):
        chromosome2positions = defaultdict(list)
        for chromosome, position, base in self.snp2snpid:
            chromosome2positions[chromosome].append(position)

        return {
            chromosome: np.unique(np.asarray(positions, dtype=int))
            for chromosome, positions
            in chromosome2positions.items()
        }

    def _generate_canonical_representation(self):
        variants_order = []
        sorted_variant2id = {}
        for (chrom, pos, base), variant_id in sorted(self.snp2snpid.items()):
            variants_order.append(variant_id)
            sorted_variant2id[chrom, pos, base] = len(variants_order)

        return sorted_variant2id, self.variant_betas[variants_order]

    def get_snp_positions_set(self) -> set:
        return {(chromosome, position) for chromosome, position, base in self.snp2snpid}

    def save_betas(self, path_or_buf, *, external_betas: np.ndarray = None):
        columns = defaultdict(list)
        betas = self.variant_betas[:self.n_variants]
        if external_betas is not None:
            assert betas.shape == external_betas.shape
            assert betas.dtype == external_betas.dtype
            betas = external_betas

        for (chrom, pos, base), variant_id in self.snp2snpid.items():
            columns['CHROM'].append(chrom)
            columns['POS'].append(pos)
            columns['BASE'].append(base)

            variant_betas = betas[variant_id] - self.default_prior
            columns['DEFAULT_PRIOR'].append(variant_betas.mean())
            for donor, beta in zip(self.genotype_names, variant_betas):
                columns[donor].append(beta)

        pd.DataFrame(columns).to_csv(path_or_buf, sep='\t', index=False)


class Demultiplexer:
    """
    Demultiplexer that can infer (learn) additional information about genotypes to achieve better quality.

    There are two ways of regularizing EM algorithm.
    - one is to compute probability for each molecule, but then
      - easier to compute posterior for different mixtures
      - hard to limit contribution of a single SNP (this was deciding after all)
    - second is to compute contributions of SNPs aggregated over all reads
      - in this case limiting contribution from a single molecule is hard, but it is limited by group size and
        number of possible modifications (AS limit in terms of cellranger/STAR alignment)

    Second one is used in this implementation.
    """
    # contribution power is regularization to descrease 
    contribution_power = 2.

    @staticmethod
    def staged_genotype_learning(chromosome2compressed_snp_calls,
                                 genotypes: ProbabilisticGenotypes,
                                 barcode_handler: BarcodeHandler,
                                 n_iterations=5,
                                 p_genotype_clip=0.01,
                                 use_doublets=False,
                                 doublet_prior=0.35,
                                 save_learnt_genotypes_to=None):
        variant_index2snp_index, variant_index2betas, _, calls = \
            Demultiplexer.pack_calls(chromosome2compressed_snp_calls, genotypes)

        n_barcodes = len(barcode_handler.ordered_barcodes)
        n_genotypes = len(genotypes.genotype_names)

        genotype_snp_posterior = variant_index2betas.copy()

        for iteration in range(n_iterations):
            genotype_prob = Demultiplexer._compute_probs_from_betas(
                variant_index2snp_index, genotype_snp_posterior, p_genotype_clip=p_genotype_clip)

            barcode_posterior_logits, columns_names = Demultiplexer.compute_barcode_logits(
                genotypes.genotype_names, calls, doublet_prior=doublet_prior, genotype_prob=genotype_prob,
                n_barcodes=n_barcodes, n_genotypes=n_genotypes, only_singlets=not use_doublets,
            )

            barcode_posterior_probs = softmax(barcode_posterior_logits, axis=-1)
            barcode_posterior_probs_df = pd.DataFrame(
                data=barcode_posterior_probs, index=barcode_handler.ordered_barcodes, columns=columns_names,
            )
            # yielding here to provide aligned posteriors for genotypes and barcodes
            debug_information = {
                'barcode_logits': barcode_posterior_logits,
                'snp_prior': variant_index2betas,
                'genotype_snp_posterior': genotype_snp_posterior
            }
            if (save_learnt_genotypes_to is not None) and (iteration == n_iterations - 1):
                assert isinstance(save_learnt_genotypes_to, str)
                genotypes.save_betas(save_learnt_genotypes_to, external_betas=genotype_snp_posterior)
            yield barcode_posterior_probs_df, debug_information

            genotype_snp_posterior = variant_index2betas.copy()
            for gindex in range(n_genotypes):
                # importantly, only singlets probabilities are used here
                contribution = (barcode_posterior_probs[calls['compressed_cb'], gindex] * (
                        1 - calls['p_base_wrong'])) ** Demultiplexer.contribution_power
                fast_np_add_at_1d(genotype_snp_posterior[:, gindex], calls['variant_id'], contribution)

    @staticmethod
    def predict_posteriors(
            chromosome2compressed_snp_calls,
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            only_singlets: bool,
            p_genotype_clip=0.01,
            doublet_prior=0.35,
    ):
        variant_index2snp_index, variant_index2betas, _, calls = \
            Demultiplexer.pack_calls(chromosome2compressed_snp_calls, genotypes)

        n_genotypes = len(genotypes.genotype_names)

        genotype_prob = Demultiplexer._compute_probs_from_betas(
            variant_index2snp_index, variant_index2betas, p_genotype_clip=p_genotype_clip)
        assert np.isfinite(genotype_prob).all()

        barcode_posterior_logits, column_names = Demultiplexer.compute_barcode_logits(
            genotypes.genotype_names, calls,
            doublet_prior, genotype_prob,
            n_barcodes=len(barcode_handler.ordered_barcodes),
            n_genotypes=n_genotypes,
            only_singlets=only_singlets
        )

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
    def compute_barcode_logits(genotype_names, calls, doublet_prior, genotype_prob, n_barcodes: int, n_genotypes: int,
                               only_singlets):
        if only_singlets:
            barcode_posterior_logits = np.zeros([n_barcodes, n_genotypes], dtype="float32")
        else:
            barcode_posterior_logits = np.zeros([n_barcodes, n_genotypes * (n_genotypes + 1) // 2])
        column_names = []
        for gindex, genotype in enumerate(genotype_names):
            p = genotype_prob[calls['variant_id'], gindex]
            log_penalties = np.log(p * (1 - calls['p_base_wrong']) + calls['p_base_wrong'].clip(1e-4))
            fast_np_add_at_1d(barcode_posterior_logits[:, gindex], calls['compressed_cb'], log_penalties)
            column_names += [genotype]
        if not only_singlets:
            # computing correction for doublet as the prior proportion of doublets will
            # otherwise depend on number of genotypes. Correction comes from
            #  n_singlet_options / singlet_prior =
            #  = n_doublet_options / doublet_prior * np.exp(doublet_logit_bonus)
            doublet_logit_bonus = np.log(n_genotypes * doublet_prior)
            doublet_logit_bonus -= np.log(n_genotypes * max(n_genotypes - 1, 0.01) / 2 * (1 - doublet_prior))

            for gindex1, genotype1 in enumerate(genotype_names):
                for gindex2, genotype2 in enumerate(genotype_names):
                    if gindex1 < gindex2:
                        p1 = genotype_prob[calls['variant_id'], gindex1]
                        p2 = genotype_prob[calls['variant_id'], gindex2]
                        p = (p1 + p2) * 0.5
                        log_penalties = np.log(p * (1 - calls['p_base_wrong']) + calls['p_base_wrong'].clip(1e-4))
                        fast_np_add_at_1d(barcode_posterior_logits[:, len(column_names)], calls['compressed_cb'],
                                          log_penalties)
                        barcode_posterior_logits[:, len(column_names)] += doublet_logit_bonus
                        column_names += [f'{genotype1}+{genotype2}']
        return barcode_posterior_logits, column_names

    @staticmethod
    def _compute_probs_from_betas(variant_index2snp_index, variant_index2betas, p_genotype_clip):
        probs = np.zeros(shape=variant_index2betas.shape, dtype='float32')
        for genotype_id in range(variant_index2betas.shape[1]):
            denom = np.bincount(variant_index2snp_index, weights=variant_index2betas[:, genotype_id])[
                variant_index2snp_index]
            probs[:, genotype_id] = variant_index2betas[:, genotype_id] / denom.clip(1e-7)
        return probs.clip(p_genotype_clip, 1 - p_genotype_clip)

    @staticmethod
    def molecule_calls2barcode_calls(molecule_calls):
        barcode_calls_without_p, indices = np.unique(molecule_calls[['variant_id', 'compressed_cb']],
                                                     return_inverse=True)
        p_base_wrong = np.ones(len(barcode_calls_without_p), dtype='float32')
        np.multiply.at(p_base_wrong, indices, molecule_calls['p_base_wrong'])
        return np.rec.fromarrays(
            [barcode_calls_without_p['variant_id'], barcode_calls_without_p['compressed_cb'], p_base_wrong],
            names=['variant_id', 'compressed_cb', 'p_base_wrong']
        )

    # @staticmethod
    # def pack_calls(chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
    #                genotypes: ProbabilisticGenotypes):
    #     with Timer('pack old'):
    #         chrom_pos_base2variant_index = genotypes.snp2snpid
    #         variant_index2betas = genotypes.variant_betas[:genotypes.n_variants]
    #         chrom_pos2snp_index = {}
    #         variant_index2snp_index = np.zeros(len(variant_index2betas), dtype='int32')
    #         for (chrom, pos, _base), variant_index in sorted(
    #                 chrom_pos_base2variant_index.items(), key=lambda cbbvi: (cbbvi[0][1], cbbvi[0][0])):
    #             if (chrom, pos) not in chrom_pos2snp_index:
    #                 chrom_pos2snp_index[chrom, pos] = len(chrom_pos2snp_index)
    #         for (chrom, pos, _base), variant_index in chrom_pos_base2variant_index.items():
    #             snp_index = chrom_pos2snp_index[chrom, pos]
    #             variant_index2snp_index[variant_index] = snp_index
    #
    #         assert np.all(variant_index2betas > 0), 'bad loaded genotypes, negative betas appeared'
    #
    #         molecule_calls = np.array(
    #             [(-1, -1, -1, -1., -1.)] * sum(calls.n_snp_calls for calls in chromosome2compressed_snp_calls.values()),
    #             dtype=[('variant_id', 'int32'), ('compressed_cb', 'int32'), ('molecule_id', 'int32'),
    #                    ('p_base_wrong', 'float32'), ('p_molecule_aligned_wrong', 'float32')]
    #         )
    #
    #         start = 0
    #         n_molecules = 0
    #         for chromosome, compressed_snp_calls in chromosome2compressed_snp_calls.items():
    #             variant_calls = compressed_snp_calls.snp_calls[:compressed_snp_calls.n_snp_calls]
    #             molecules = compressed_snp_calls.molecules[:compressed_snp_calls.n_molecules]
    #
    #             fragment = molecule_calls[start:start + compressed_snp_calls.n_snp_calls]
    #             fragment['variant_id'] = [
    #                 chrom_pos_base2variant_index.get((chromosome, pos, decompress_base(base_index)), -1)
    #                 for pos, base_index in variant_calls[['snp_position', 'base_index']]
    #             ]
    #             fragment['compressed_cb'] = molecules['compressed_cb'][variant_calls['molecule_index']]
    #             fragment['molecule_id'] = variant_calls['molecule_index'] + n_molecules
    #             fragment['p_base_wrong'] = variant_calls['p_base_wrong']
    #             fragment['p_molecule_aligned_wrong'] = molecules['p_group_misaligned'][variant_calls['molecule_index']]
    #
    #             start += compressed_snp_calls.n_snp_calls
    #             if fragment['variant_id'].max() > -1:
    #                 n_molecules += compressed_snp_calls.n_molecules
    #
    #         # filtering from those calls that did not match any snp
    #         molecule_calls = molecule_calls[molecule_calls['variant_id'] != -1]
    #
    #         barcode_calls = Demultiplexer.molecule_calls2barcode_calls(molecule_calls)
    #     with Timer('pack new'):
    #         variant_index2snp_index2, variant_index2betas2, molecule_calls2, barcode_calls2 = Demultiplexer.pack_calls_new(
    #             chromosome2compressed_snp_calls, genotypes)
    #     assert np.array_equal(variant_index2snp_index2, variant_index2snp_index)
    #     assert np.array_equal(variant_index2betas2, variant_index2betas)
    #     assert np.array_equal(molecule_calls2, molecule_calls)
    #     assert np.array_equal(barcode_calls2, barcode_calls)
    #
    #     return variant_index2snp_index, variant_index2betas, molecule_calls, barcode_calls

    @staticmethod
    def pack_calls(chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
                   genotypes: ProbabilisticGenotypes):
        chrom_pos_base2variant_index = genotypes.snp2snpid
        pos_base_chrom_variant = np.array(
            [(pos, compress_base(base), chrom, variant_index)
             for (chrom, pos, base), variant_index in chrom_pos_base2variant_index.items()],
            dtype=[('snp_position', 'int32'), ('base_index', 'uint8'), ('chrom', 'O'), ('variant_index', 'int32')]
        )
        pos_base_chrom_variant = np.sort(pos_base_chrom_variant)

        variant_index2betas = genotypes.variant_betas[:genotypes.n_variants]

        _, snp_indices = np.unique(pos_base_chrom_variant[['snp_position', 'chrom']], return_inverse=True)
        variant_index2snp_index = np.zeros(len(variant_index2betas), dtype='int32') - 1
        variant_index2snp_index[pos_base_chrom_variant['variant_index']] = snp_indices
        assert len(chrom_pos_base2variant_index) == len(variant_index2betas)
        assert np.allclose(np.sort(pos_base_chrom_variant['variant_index']), np.arange(len(pos_base_chrom_variant)))
        assert np.all(variant_index2snp_index >= 0)
        assert np.all(variant_index2betas > 0), 'bad loaded genotypes, negative betas appeared'

        molecule_calls = np.array(
            [(-1, -1, -1, -1., -1.)] * sum(calls.n_snp_calls for calls in chromosome2compressed_snp_calls.values()),
            dtype=[('variant_id', 'int32'), ('compressed_cb', 'int32'), ('molecule_id', 'int32'),
                   ('p_base_wrong', 'float32'), ('p_molecule_aligned_wrong', 'float32')]
        )

        start = 0
        n_molecules = 0
        for chromosome, compressed_snp_calls in chromosome2compressed_snp_calls.items():
            variant_calls = compressed_snp_calls.snp_calls[:compressed_snp_calls.n_snp_calls]
            molecules = compressed_snp_calls.molecules[:compressed_snp_calls.n_molecules]

            lookup = pos_base_chrom_variant[pos_base_chrom_variant['chrom'] == chromosome]
            if len(lookup) == 0:
                # no snps in genotype for this chromosome
                continue
            index = lookup[['snp_position', 'base_index']]
            searched = variant_calls[['snp_position', 'base_index']]
            # clipping fights being out of bounds (output may be len(index))
            variant_id = np.searchsorted(index[['snp_position', 'base_index']], searched).clip(0, len(index) - 1)
            variant_id = np.where(index[variant_id] == searched, lookup['variant_index'][variant_id], -1)

            fragment = molecule_calls[start:start + compressed_snp_calls.n_snp_calls]
            fragment['variant_id'] = variant_id
            fragment['compressed_cb'] = molecules['compressed_cb'][variant_calls['molecule_index']]
            # just making sure no collisions for read from different chromosomes
            fragment['molecule_id'] = variant_calls['molecule_index'] + n_molecules
            fragment['p_base_wrong'] = variant_calls['p_base_wrong']
            fragment['p_molecule_aligned_wrong'] = molecules['p_group_misaligned'][variant_calls['molecule_index']]

            start += compressed_snp_calls.n_snp_calls
            n_molecules += compressed_snp_calls.n_molecules

        # filtering from those calls that did not match any snp
        molecule_calls = molecule_calls[molecule_calls['variant_id'] != -1]

        barcode_calls = Demultiplexer.molecule_calls2barcode_calls(molecule_calls)
        return variant_index2snp_index, variant_index2betas, molecule_calls, barcode_calls
