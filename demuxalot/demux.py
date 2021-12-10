from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import pysam
from scipy.special import softmax

from demuxalot.snp_counter import CompressedSNPCalls
from demuxalot.utils import fast_np_add_at_1d, BarcodeHandler, compress_base, FeatureLookup


class ProbabilisticGenotypes:
    def __init__(self, genotype_names: List[str], default_prior=1.):
        """
        ProbabilisticGenotypes represents our accumulated knowledge about SNPs for selected genotypes.
        Can aggregate information from GSA/WGS/WES, our prior guesses and genotype information learnt from RNAseq.
        Genotype names can't be changed/extended once the object is created.
        Class can handle more than two options per genomic position.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas (parameters of Dirichlet distribution,
        akin to coefficients in beta distribution).
        """
        self.default_prior: float = default_prior
        self.var2varid: Dict[Tuple, int] = {}  # chrom, pos, base -> index in variant_betas
        self.genotype_names: List[str] = list(genotype_names)
        assert (np.sort(self.genotype_names) == self.genotype_names).all(), 'please order genotype names'
        self.variant_betas: np.ndarray = np.zeros([32768, self.n_genotypes], 'float32')

    def __repr__(self):
        chromosomes = {chromosome for chromosome, _, _ in self.var2varid}
        return f'<Genotypes with {self.n_variants} variants on {len(chromosomes)} ' \
               f'and {self.n_genotypes} genotypes: \n{self.genotype_names}'

    @property
    def n_genotypes(self):
        return len(self.genotype_names)

    @property
    def n_variants(self) -> int:
        return len(self.var2varid)

    def get_betas(self) -> np.ndarray:
        # return readonly view
        variants_view: np.ndarray = self.variant_betas[: self.n_variants]
        variants_view.flags.writeable = False
        return variants_view

    def get_snp_ids_for_variants(self) -> np.ndarray:
        snp2id = {}
        result = np.zeros(self.n_variants, dtype='int32') - 1
        for (chrom, pos, _base), variant_id in self.var2varid.items():
            snp = chrom, pos
            if snp not in snp2id:
                snp2id[snp] = len(snp2id)
            result[variant_id] = snp2id[snp]
        assert np.all(result >= 0)
        assert np.all(result < self.n_variants)
        return result

    def extend_variants(self, n_samples=1):
        # pre-allocation of space for new variants
        while n_samples + self.n_variants > len(self.variant_betas):
            self.variant_betas = np.concatenate([self.variant_betas, np.zeros_like(self.variant_betas)], axis=0)

    def add_vcf(self, vcf_file_name, prior_strength: float = 100.):
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
            if any(allele not in 'ACGT' for allele in alleles):
                n_skipped_snps += 1
                continue

            for allele in alleles:
                # pysam enumerates starting from one, thus -1
                variant = (snp.chrom, snp.pos - 1, allele)
                if variant not in self.var2varid:
                    self.var2varid[variant] = self.n_variants
                snp_ids.append(self.var2varid[variant])

            assert len(set(snp_ids)) == len(snp_ids), (snp_ids, snp.chrom, snp.pos, snp.alleles)

            contribution = np.zeros([len(snp_ids), self.n_genotypes], dtype='float32')
            for donor_id, donor in enumerate(self.genotype_names):
                called_values = snp.samples[donor]['GT']
                for call in called_values:
                    if call is not None:
                        # contribution is split between called values
                        contribution[call, donor_id] += prior_strength / len(called_values)
            not_provided = contribution.sum(axis=0) == 0
            if np.sum(~not_provided) < 2:
                # at least two genotypes should have SNP
                n_skipped_snps += 1
                continue

            confidence_for_skipped = 0.1
            contribution[:, not_provided] = contribution[:, ~not_provided].mean(axis=1, keepdims=True) \
                                            * confidence_for_skipped
            self.variant_betas[snp_ids] += contribution
        if n_skipped_snps > 0:
            print('skipped', n_skipped_snps, 'SNVs')

    def add_prior_betas(self, prior_filename, *, prior_strength: float = 1.):
        """
        Betas is the way Demultiplexer stores learnt genotypes. It is a parquet file with posterior weights.
        Parquet are convenient replacement for csv/tsv files, find examples in pandas documentation.
        :param prior_filename: path to file
        :param prior_strength: float, controls impact of learnt genotypes.
        """

        prior_knowledge: pd.DataFrame = pd.read_parquet(prior_filename) * prior_strength
        print('Provided prior information about genotypes:', [*prior_knowledge.columns])
        genotypes_not_provided = [
            genotype for genotype in self.genotype_names if genotype not in prior_knowledge.columns
        ]
        if len(genotypes_not_provided) > 0:
            print(f'No information for genotypes: {genotypes_not_provided}')

        variants = prior_knowledge.index.to_frame()
        variants = zip(variants['CHROM'], variants['POS'], variants['BASE'])

        variant_indices: List[int] = []
        for variant in variants:
            if variant not in self.var2varid:
                self.extend_variants(1)
                self.var2varid[variant] = self.n_variants
            variant_indices.append(self.var2varid[variant])

        for donor_id, donor in enumerate(self.genotype_names):
            if donor in prior_knowledge.columns:
                np.add.at(
                    self.variant_betas[:, donor_id],
                    variant_indices,
                    prior_knowledge[donor],
                )

    def get_chromosome2positions(self):
        chromosome2positions = defaultdict(list)
        for chromosome, position, base in self.var2varid:
            chromosome2positions[chromosome].append(position)

        if len(chromosome2positions) == 0:
            warn('Genotypes are empty. Did you forget to add vcf/betas?')

        return {
            chromosome: np.unique(np.asarray(positions, dtype=int))
            for chromosome, positions
            in chromosome2positions.items()
        }

    def _generate_canonical_representation(self):
        variants_order = []
        sorted_variant2id = {}
        for (chrom, pos, base), variant_id in sorted(self.var2varid.items()):
            variants_order.append(variant_id)
            sorted_variant2id[chrom, pos, base] = len(variants_order)

        return sorted_variant2id, self.variant_betas[variants_order]

    def get_snp_positions_set(self) -> set:
        return {(chromosome, position) for chromosome, position, base in self.var2varid}

    def _with_betas(self, external_betas: np.ndarray):
        """ Return copy of genotypes with updated beta weights """
        assert external_betas.shape == (self.n_variants, self.n_genotypes)
        assert external_betas.dtype == self.variant_betas.dtype
        assert np.min(external_betas) >= 0
        result: ProbabilisticGenotypes = self.clone()
        result.variant_betas = external_betas.copy()
        return result

    def save_betas(self, path_or_buf):
        """ Save learnt genotypes in the form of beta contributions, can be used later. """
        index_columns = defaultdict(list)
        old_variant_order = []

        for (chrom, pos, base), variant_id in sorted(self.var2varid.items()):
            index_columns['CHROM'].append(chrom)
            index_columns['POS'].append(pos)
            index_columns['BASE'].append(base)

            old_variant_order.append(variant_id)

        old_variant_order = np.asarray(old_variant_order)
        betas = self.variant_betas[:self.n_variants][old_variant_order]

        exported_frame = pd.DataFrame(
            data=betas,
            index=pd.MultiIndex.from_frame(pd.DataFrame(index_columns)),
            columns=self.genotype_names,
        )

        exported_frame.to_parquet(path_or_buf)

    def clone(self):
        return deepcopy(self)


"""
There are two ways of regularizing EM algorithm.
Option 1. compute probability for each molecule, but then
- easier to compute posterior for different mixtures
- hard to limit contribution of a single SNP (this was deciding after all)
Option 2. compute contributions of SNPs aggregated over all reads
- in this case limiting contribution from a single molecule is hard, but it is limited by group size and
  number of possible modifications (AS limit in terms of cellranger/STAR alignment)

Second one is used in this implementation.
"""


class Demultiplexer:
    """
    Demultiplexer that can infer (learn) additional information about genotypes to achieve better quality.
    """
    # contribution_power minimizes contribution
    # from barcodes that don't have any good candidate donor
    contribution_power = 2.
    aggregate_on_snps = False
    compensation_during_computing_barcode_logits = 0.5

    @staticmethod
    def learn_genotypes(chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
                        genotypes: ProbabilisticGenotypes,
                        barcode_handler: BarcodeHandler,
                        n_iterations=5,
                        p_genotype_clip=0.01,
                        doublet_prior=0.,
                        barcode_prior_logits: np.ndarray = None,
                        ) -> Tuple[ProbabilisticGenotypes, pd.DataFrame]:
        """
        Learn genotypes starting from initial genotype guess
        :param chromosome2compressed_snp_calls: output of snp calling utility
        :param genotypes: initial genotypes (e.g. inferred from bead array or bulk rnaseq)
        :param barcode_handler: barcode handler specifies which barcodes should be considered
        :param n_iterations: number of EM iterations
        :param p_genotype_clip: minimal probability assigned to polymorphism
        :param doublet_prior: prior expectation of fraction of doublets,
            setting to zero skips computation of doublets and helpful when number of clones is too large
        :param barcode_prior_logits: optionally, one can start from assignment to possible clones
        :return: learnt genotypes and barcode-to-donor assignments from the last iteration
        """
        *_, last_iteration_output = Demultiplexer.staged_genotype_learning(
            chromosome2compressed_snp_calls=chromosome2compressed_snp_calls,
            genotypes=genotypes,
            barcode_handler=barcode_handler,
            n_iterations=n_iterations,
            p_genotype_clip=p_genotype_clip,
            doublet_prior=doublet_prior,
            barcode_prior_logits=barcode_prior_logits,
        )
        last_iteration_barcode_probs, debug_information = last_iteration_output
        learnt_genotypes = genotypes._with_betas(genotypes.get_betas() + debug_information['genotype_addition'])
        return learnt_genotypes, last_iteration_barcode_probs

    @staticmethod
    def staged_genotype_learning(
            chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            n_iterations=5,
            p_genotype_clip=0.01,
            doublet_prior=0.,
            barcode_prior_logits: np.ndarray = None,
    ):
        assert doublet_prior >= 0
        if barcode_prior_logits is not None:
            n_options = len(Demultiplexer._doublet_penalties(genotypes.n_genotypes, doublet_prior))
            assert barcode_prior_logits.shape == (barcode_handler.n_barcodes, n_options), 'wrong shape of priors'

        variant_index2snp_index, variant_index2betas, _molecule_calls, calls = \
            Demultiplexer.pack_calls(chromosome2compressed_snp_calls, genotypes, add_data_prior=True)

        genotype_addition = np.zeros_like(variant_index2betas)

        for iteration in range(n_iterations):
            genotype_prob = Demultiplexer._compute_probs_from_betas(
                variant_index2snp_index, variant_index2betas + genotype_addition, p_genotype_clip=p_genotype_clip)

            barcode_posterior_logits, columns_names = Demultiplexer.compute_barcode_logits(
                genotypes.genotype_names, barcode_calls=calls, molecule_calls=_molecule_calls,
                doublet_prior=doublet_prior, genotype_prob=genotype_prob,
                n_barcodes=barcode_handler.n_barcodes, n_genotypes=genotypes.n_genotypes,
            )
            if iteration == 0 and barcode_prior_logits is not None:
                assert barcode_prior_logits.shape == barcode_posterior_logits.shape, 'mismatching priors passed'
                barcode_posterior_logits += barcode_prior_logits

            barcode_posterior_probs = softmax(barcode_posterior_logits, axis=-1)
            barcode_posterior_probs_df = pd.DataFrame(
                data=barcode_posterior_probs, index=barcode_handler.ordered_barcodes, columns=columns_names,
            )
            # yielding here to provide aligned posteriors for genotypes and barcodes
            debug_information = {
                'barcode_logits': barcode_posterior_logits,
                'genotype_prior': variant_index2betas,
                'genotype_addition': genotype_addition,
            }
            yield barcode_posterior_probs_df, debug_information

            genotype_addition = np.zeros_like(variant_index2betas)
            for gindex in range(genotypes.n_genotypes):
                # importantly, only singlets probabilities are used here
                contribution = barcode_posterior_probs[calls['compressed_cb'], gindex] * (1 - calls['p_base_wrong'])
                contribution **= Demultiplexer.contribution_power
                fast_np_add_at_1d(genotype_addition[:, gindex], calls['variant_id'], contribution)

    @staticmethod
    def predict_posteriors(
            chromosome2compressed_snp_calls,
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            p_genotype_clip=0.01,
            doublet_prior=0.35,
    ):
        variant_index2snp_index, variant_index2betas, _molecule_calls, calls = \
            Demultiplexer.pack_calls(chromosome2compressed_snp_calls, genotypes, add_data_prior=False)

        n_genotypes = genotypes.n_genotypes

        genotype_prob = Demultiplexer._compute_probs_from_betas(
            variant_index2snp_index, variant_index2betas, p_genotype_clip=p_genotype_clip)
        assert np.isfinite(genotype_prob).all()

        barcode_posterior_logits, column_names = Demultiplexer.compute_barcode_logits(
            genotypes.genotype_names, calls,
            doublet_prior=doublet_prior,
            genotype_prob=genotype_prob,
            molecule_calls=_molecule_calls,
            n_barcodes=barcode_handler.n_barcodes,
            n_genotypes=n_genotypes,
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
    def _doublet_penalties(n_genotypes: int, doublet_prior: float) -> np.ndarray:
        if doublet_prior == 0:
            return np.zeros(n_genotypes, dtype='float32')

        # computing correction for doublet as the prior proportion of doublets will
        # otherwise depend on number of genotypes. Correction comes from
        #  n_singlet_options / singlet_prior =
        #  = n_doublet_options / doublet_prior * np.exp(doublet_logit_bonus)
        doublet_logit_bonus = np.log(n_genotypes * doublet_prior)
        doublet_logit_bonus -= np.log(n_genotypes * max(n_genotypes - 1, 1) / 2 * (1 - doublet_prior))
        n_options = n_genotypes * (n_genotypes + 1) // 2  # singlets and doublets, singlets go first
        barcode_logit_corrections_for_doublets = np.zeros(n_options, dtype='float32')
        barcode_logit_corrections_for_doublets[n_genotypes:] = doublet_logit_bonus
        return barcode_logit_corrections_for_doublets

    @staticmethod
    def _iterate_genotypes_options(genotype_names, genotype_prob: np.ndarray, doublet_prior: float):
        """Iterates through genotypes, including potential doublets """
        pseudogenotype_index = 0
        for gindex, genotype_name in enumerate(genotype_names):
            yield pseudogenotype_index, genotype_name, genotype_prob[:, gindex]
            pseudogenotype_index += 1

        if doublet_prior != 0:
            assert doublet_prior > 0
            for gindex1, genotype1 in enumerate(genotype_names):
                for gindex2, genotype2 in enumerate(genotype_names):
                    if gindex1 < gindex2:
                        p1 = genotype_prob[:, gindex1]
                        p2 = genotype_prob[:, gindex2]
                        yield pseudogenotype_index, f'{genotype1}+{genotype2}', (p1 + p2) * 0.5
                        pseudogenotype_index += 1

    @staticmethod
    def compute_barcode_logits(
            genotype_names, barcode_calls, molecule_calls, doublet_prior: float, genotype_prob: np.ndarray,
            n_barcodes: int, n_genotypes: int
    ):
        if not Demultiplexer.aggregate_on_snps:
            return Demultiplexer.compute_barcode_logits_using_barcode_calls(
                genotype_names, barcode_calls=barcode_calls, doublet_prior=doublet_prior,
                genotype_prob=genotype_prob, n_barcodes=n_barcodes, n_genotypes=n_genotypes,
            )

        """
        One can also stop aggregating naive information about number of bases and instead try to employ 
        all probabilities of calls available for each base.
        This is substantially more expensive, but improvements in quality are minor.
        
        Reserved for future experiments, but likely this code will be removed.
        """

        barcode_doublet_penaltites = Demultiplexer._doublet_penalties(n_genotypes, doublet_prior=doublet_prior)

        snp_ids = molecule_calls['snp_id']
        bns_compressor = FeatureLookup(molecule_calls['compressed_cb'], snp_ids)
        calls_bns_id, bns_molecule_counts = bns_compressor.compress(molecule_calls['compressed_cb'], snp_ids)
        bns_id2barcode, bns_id2snp_ids = bns_compressor.lookup_for_individual_features()

        bns_logits = np.zeros([bns_compressor.nvalues, len(barcode_doublet_penaltites)], dtype='float32')

        column_names = []
        for pg_index, genotype_name, variant2prob in Demultiplexer._iterate_genotypes_options(
            genotype_names, genotype_prob, doublet_prior=doublet_prior,
        ):
            column_names += [genotype_name]
            p = variant2prob[molecule_calls['variant_id']]
            log_penalties = np.log(p + molecule_calls['p_base_wrong'])
            fast_np_add_at_1d(bns_logits[:, pg_index], calls_bns_id, log_penalties)

        from scipy.special import log_softmax
        # regularization to the number of molecules contributing to SNP
        bns_logits /= bns_molecule_counts[:, None] ** Demultiplexer.compensation_during_computing_barcode_logits
        # regularization to fight over-contribution of a single SNP
        bns_logits = log_softmax(bns_logits, axis=1)
        p_bad_snp = 0.01
        bns_logits: np.ndarray = np.logaddexp(bns_logits, np.log(p_bad_snp / len(column_names)))
        bns_logits = log_softmax(bns_logits, axis=1)

        barcode_posterior_logits = [
            np.bincount(bns_id2barcode, weights=col, minlength=n_barcodes) for col in bns_logits.T
        ]
        barcode_posterior_logits = np.stack(barcode_posterior_logits, axis=1)

        return barcode_posterior_logits , column_names

    @staticmethod
    def compute_barcode_logits_using_barcode_calls(
            genotype_names, barcode_calls, doublet_prior, genotype_prob: np.ndarray,
            n_barcodes: int, n_genotypes: int
    ):
        # n_barcodes x n_genotypes
        barcode_posterior_logits = np.zeros([n_barcodes, 1], dtype='float32') \
            + Demultiplexer._doublet_penalties(n_genotypes, doublet_prior=doublet_prior)

        column_names = []
        for pg_index, genotype_name, variant2prob in Demultiplexer._iterate_genotypes_options(
            genotype_names, genotype_prob=genotype_prob, doublet_prior=doublet_prior
        ):
            column_names += [genotype_name]
            p = variant2prob[barcode_calls['variant_id']]
            log_penalties = np.log(p * (1 - barcode_calls['p_base_wrong']) + barcode_calls['p_base_wrong'].clip(1e-4))

            fast_np_add_at_1d(barcode_posterior_logits[:, pg_index], barcode_calls['compressed_cb'], log_penalties)

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
        barcode_calls, indices, barcode_variant_counts = np.unique(
            molecule_calls[['variant_id', 'snp_id', 'compressed_cb']], return_inverse=True, return_counts=True)

        # computing probability by multiplying probs of individual calls
        p_base_wrong = np.ones(len(barcode_calls), dtype='float32')
        np.multiply.at(p_base_wrong, indices, molecule_calls['p_base_wrong'])

        snp_barcode_lookup = FeatureLookup(barcode_calls['snp_id'], barcode_calls['compressed_cb'])
        snp_barcode_id, _ = snp_barcode_lookup.compress(barcode_calls['snp_id'], barcode_calls['compressed_cb'])

        barcode_snp_count = np.bincount(snp_barcode_id, barcode_variant_counts)[snp_barcode_id]

        return np.rec.fromarrays([
            barcode_calls['variant_id'],
            barcode_calls['snp_id'],
            barcode_calls['compressed_cb'],
            p_base_wrong,
            barcode_variant_counts,
            barcode_snp_count,
        ],
            names=['variant_id', 'snp_id', 'compressed_cb', 'p_base_wrong', 'barcode_variant_count', 'barcode_snp_count']
        )

    @staticmethod
    def pack_calls(
            chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
            genotypes: ProbabilisticGenotypes,
            add_data_prior: bool,
    ):
        pos_base_chrom_variant = np.array(
            [(pos, compress_base(base), chrom, variant_index)
             for (chrom, pos, base), variant_index in genotypes.var2varid.items()],
            dtype=[('snp_position', 'int32'), ('base_index', 'uint8'), ('chrom', 'O'), ('variant_index', 'int32')]
        )
        pos_base_chrom_variant = np.sort(pos_base_chrom_variant)

        variant_index2snp_index = genotypes.get_snp_ids_for_variants()
        # checking enumeration of positions
        assert np.allclose(np.sort(pos_base_chrom_variant['variant_index']), np.arange(len(pos_base_chrom_variant)))
        assert np.all(variant_index2snp_index >= 0)

        # pre-allocate array for all molecules
        n_calls_in_total = sum(calls.n_snp_calls for calls in chromosome2compressed_snp_calls.values())
        molecule_calls = np.array(
            [(-1, -1, -1, -1, -1., -1.)],
            dtype=[('variant_id', 'int32'),
                   ('snp_id', 'int32'),
                   ('compressed_cb', 'int32'),
                   ('molecule_id', 'int32'),
                   ('p_base_wrong', 'float32'),
                   ('p_molecule_aligned_wrong', 'float32')]
        ).repeat(n_calls_in_total, axis=0)

        # merge molecule calls
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
            # clipping prevents out of bounds (output may be len(index))
            variant_id = np.searchsorted(index[['snp_position', 'base_index']], searched).clip(0, len(index) - 1)
            # mark invalid calls, those will be filter afterwards
            variant_id = np.where(index[variant_id] == searched, lookup['variant_index'][variant_id], -1)

            fragment = molecule_calls[n_molecules:n_molecules + compressed_snp_calls.n_snp_calls]
            fragment['variant_id'] = variant_id
            fragment['snp_id'] = variant_index2snp_index[variant_id]
            fragment['compressed_cb'] = molecules['compressed_cb'][variant_calls['molecule_index']]
            # just to allow some backtracking to the original molecule. Never used in practice
            fragment['molecule_id'] = variant_calls['molecule_index']
            fragment['p_base_wrong'] = variant_calls['p_base_wrong']
            fragment['p_molecule_aligned_wrong'] = molecules['p_group_misaligned'][variant_calls['molecule_index']]

            n_molecules += compressed_snp_calls.n_snp_calls
        assert n_molecules == len(molecule_calls)

        # filtering out calls that did not match any snp
        did_not_match_snp = molecule_calls['variant_id'] == -1
        molecule_calls = molecule_calls[~did_not_match_snp]

        barcode_calls = Demultiplexer.molecule_calls2barcode_calls(molecule_calls)

        def normalize_over_snp(variant_counts, regularization=1.):
            assert len(variant_counts) == len(variant_index2snp_index)
            snp_counts = np.bincount(variant_index2snp_index, weights=variant_counts)[variant_index2snp_index]
            return variant_counts / (snp_counts + regularization)

        def compute_prior_betas(molecule_calls, add_data_prior: bool):
            variant_index2betas = genotypes.get_betas()
            assert np.all(variant_index2betas >= 0), 'bad genotypes provided, negative betas appeared'

            # collecting all info about SNPs to allow better priors
            baseline_regularization = 1.
            prior_betas = baseline_regularization
            if add_data_prior:
                # not used during inference, but used during training
                variant_index2n_molecules = np.bincount(molecule_calls['variant_id'], minlength=genotypes.n_variants)
                prior_betas += normalize_over_snp(variant_index2n_molecules, regularization=100.)
            prior_betas += normalize_over_snp(variant_index2betas.sum(axis=1), regularization=100.)
            addition = prior_betas[:, np.newaxis] * genotypes.default_prior
            return variant_index2betas + addition.astype(variant_index2betas.dtype)

        # add regularization
        variant_index2betas = compute_prior_betas(molecule_calls, add_data_prior=add_data_prior)
        # make it readonly
        variant_index2betas.flags.writeable = False

        return variant_index2snp_index, variant_index2betas, molecule_calls, barcode_calls
