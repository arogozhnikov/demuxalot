from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import pysam
from scipy.special import softmax

from scrnaseq_demux.snp_counter import CompressedSNPCalls
from scrnaseq_demux.utils import fast_np_add_at_1d, BarcodeHandler, read_vcf_to_header_and_pandas, decompress_base


class ProbabilisticGenotypes_old:
    def __init__(self, genotype_names: List[str]):
        """
        ProbabilisticGenotype represents our accumulated knowledge about SNPs (and SNPs only) for genotypes.
        Can aggregate information from GSA, our prior guesses and genotype information learnt from RNAseq.
        Genotype names can't be changed once object is created.
        Class doesn't handle more than one SNP at position (examples are A/T and A/C at the same position),
        so only the first received SNP for position is kept.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas (akin to coefficients in beta distribution).
        """
        self.snips = {}
        self.genotype_names = list(genotype_names)
        assert (np.sort(self.genotype_names) == self.genotype_names).all(), 'please order genotype names'
        self.genotype_name2gindex = {gindex: genotype_name for gindex, genotype_name in enumerate(self.genotype_names)}

    def __repr__(self):
        return f'<Genotypes with {len(self.genotype_names)} genotypes: {self.genotype_names} ' \
               f'and {len(self.snips)} SNVs >'

    def add_vcf(self, vcf_file_name, prior_strength=100, verbose=False):
        """
        Add information from parsed VCF
        :param vcf_file_name: path to VCF file. Only diploid values are accepted (0/0, 0/1, 1/1, ./.).
            Should contain all genotypes of interest. Can contain additional genotypes, but those will be ignored.
        """
        type2code = {"0/0": 0, "0/1": 1, "1/1": 2, "./.": 3}
        code2prior = np.array([[0.99, 0.01], [0.50, 0.50], [0.01, 0.99], [0, 0]], dtype='float32') * prior_strength

        _header, snp_df = read_vcf_to_header_and_pandas(vcf_file_name)

        snp_df = snp_df.set_index(["CHROM", "POS", "REF", "ALT"])[self.genotype_names].replace(type2code).astype(
            "uint8")

        for (chromosome, position, ref, alt), genotype_codes in snp_df.iterrows():
            genotype_codes = genotype_codes.values
            priors = code2prior[genotype_codes]

            # filling unknown genotypes with average over other genotypes
            is_unknown = genotype_codes == type2code["./."]
            assert np.sum(~is_unknown) > 0, 'SNP passed without any prior information'
            priors[is_unknown] = priors[~is_unknown].mean(axis=0, keepdims=True)

            if (chromosome, position) in self.snips:
                existing_ref_alt = self.snips[chromosome, position][:2]
                if sorted(existing_ref_alt) != sorted([ref, alt]):
                    # conflict, leaving SNP already saved
                    continue
                if existing_ref_alt == (alt, ref):
                    # swapped ref and alt
                    self.snips[chromosome, position][2][:] += priors[:, ::-1]
                else:
                    self.snips[chromosome, position][2][:] += priors
            else:
                self.snips[chromosome, position] = (ref, alt, priors)

            if verbose and len(self.snips) % 10000 == 0:
                print("completed snps: ", len(self.snips))

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

        for (chromosome, position), snp_priors in prior_knowledge.groupby(['CHROM', 'POS']):
            if len(snp_priors) != 2:
                print('Can handle only two alternatives for the same position in genome', chromosome, position)
                continue

            bases = list(snp_priors['BASE'])
            assert bases[0] != bases[1]
            snp_priors = snp_priors[self.genotype_names].values.T

            if (chromosome, position) in self.snips:
                *ref_alt, existing_prior = self.snips[chromosome, position]
                if sorted(bases) != sorted(ref_alt):
                    # different SNP present, skipping
                    continue
                if bases == ref_alt:
                    existing_prior += snp_priors
                else:
                    # reverse order
                    existing_prior += snp_priors[:, ::-1]
            else:
                self.snips[chromosome, position] = (bases[0], bases[1], snp_priors)

    def get_chromosome2positions(self):
        chromosome2positions = defaultdict(list)
        for chromosome, position in self.snips:
            chromosome2positions[chromosome].append(position)

        chromosome2positions = {
            chromosome: np.unique(np.asarray(positions, dtype=int))
            for chromosome, positions
            in chromosome2positions.items()
        }

        return chromosome2positions

    def generate_genotype_snp_beta_prior(self):
        n_genotypes = len(self.genotype_names)
        n_snps = len(self.snips)

        snp2sindex = {}
        snp2ref_alt = {}
        genotype_snp_beta_prior = np.zeros([n_snps, n_genotypes, 2], dtype="float32")

        for sindex, ((chromosome, position), (ref, alt, priors)) in enumerate(sorted(self.snips.items())):
            snp2sindex[chromosome, position] = sindex
            snp2ref_alt[chromosome, position] = (ref, alt)
            genotype_snp_beta_prior[sindex] = priors

        return snp2sindex, snp2ref_alt, genotype_snp_beta_prior

    def contains_snp(self, chromosome, position):
        return (chromosome, position) in self.snips

    def save_betas(self, path_or_buf, *, external_betas: np.ndarray = None):
        if external_betas is not None:
            assert external_betas.shape[0] == len(self.snips)
            assert external_betas.shape[1] == len(self.genotype_names)
        snp2sindex = {}
        snp2ref_alt = {}
        result = []
        for sindex, ((chromosome, position), (ref, alt, priors)) in enumerate(sorted(self.snips.items())):
            snp2sindex[chromosome, position] = sindex
            snp2ref_alt[chromosome, position] = (ref, alt)
            if external_betas is None:
                ref_betas, alt_betas = priors.T
            else:
                ref_betas, alt_betas = external_betas[sindex].T
            result.append({
                'CHROM': chromosome,
                'POS': position,
                'BASE': ref,
                'DEFAULT_PRIOR': ref_betas.mean(),
                **dict(zip(self.genotype_names, ref_betas))
            })

            result.append({
                'CHROM': chromosome,
                'POS': position,
                'BASE': alt,
                'DEFAULT_PRIOR': alt_betas.mean(),
                **dict(zip(self.genotype_names, alt_betas))
            })
        pd.DataFrame(result).to_csv(path_or_buf, sep='\t', index=False)


class ProbabilisticGenotypes:
    def __init__(self, genotype_names: List[str]):
        """
        ProbabilisticGenotype represents our accumulated knowledge about SNPs (and SNPs only) for genotypes.
        Can aggregate information from GSA, our prior guesses and genotype information learnt from RNAseq.
        Genotype names can't be changed once object is created.
        Class doesn't handle more than one SNP at position (examples are A/T and A/C at the same position),
        so only the first received SNP for position is kept.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas (akin to coefficients in beta distribution).
        """
        self.snp2snpid = {}  # chrom, pos, base -> index
        self.genotype_names = list(genotype_names)
        assert (np.sort(self.genotype_names) == self.genotype_names).all(), 'please order genotype names'
        self.n_variants = 0
        self.variant_betas = np.ones([32768, len(self.genotype_names)], 'float32')

    def __repr__(self):
        return f'<Genotypes with {len(self.genotype_names)} genotypes: {self.genotype_names} ' \
               f'and {len(self.snp2snpid)} SNVs >'

    def extend_variants(self, n_samples=1):
        while n_samples + self.n_variants > len(self.variant_betas):
            self.variant_betas = np.concatenate([self.variant_betas, np.ones_like(self.variant_betas)], axis=0)

    def add_vcf(self, vcf_file_name, prior_strength=100, verbose=False):
        """
        Add information from parsed VCF
        :param vcf_file_name: path to VCF file. Only diploid values are accepted (0/0, 0/1, 1/1, ./.).
            Should contain all genotypes of interest. Can contain additional genotypes, but those will be ignored.
        """
        for snp in pysam.VariantFile(vcf_file_name).fetch():
            if any(len(option) != 1 for option in snp.alleles):
                print('skipping non-snp, alleles = ', snp.alleles, snp.chrom, snp.pos)
                continue
            self.extend_variants(len(snp.alleles))

            snp_ids = []
            for allele in snp.alleles:
                variant = (snp.chrom, snp.pos - 1, allele)
                if variant not in self.snp2snpid:
                    self.snp2snpid[variant] = self.n_variants
                    self.n_variants += 1
                snp_ids.append(self.snp2snpid[variant])
            assert len(set(snp_ids)) == len(snp_ids)

            contribution = np.zeros([len(snp_ids), len(self.genotype_names)], dtype='float32')
            for donor_id, donor in enumerate(self.genotype_names):
                called_values = snp.samples[donor]['GT']
                for call in called_values:
                    if call is not None:
                        # contribution is split between called values
                        contribution[call, donor_id] += prior_strength / len(called_values)
            not_provided = contribution.sum(axis=0) == 0
            assert sum(~not_provided) != 0
            for row in range(len(contribution)):
                contribution[row, not_provided] = contribution[row, ~not_provided].mean()
            self.variant_betas[snp_ids] += contribution

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

    def generate_genotype_snp_beta_prior(self):
        snps = defaultdict(list)
        for (chrom, pos, base), variant_id in self.snp2snpid.items():
            snps['CHROM'].append(chrom)
            snps['POS'].append(pos)
            snps['BASE'].append(base)
            snps['VARIANT_ID'].append(variant_id)
        snps = pd.DataFrame(snps)

        snps = snps.groupby(['CHROM', 'POS']).head(2).sort_values(['CHROM', 'POS'])
        assert np.all(snps[['CHROM', 'POS']].iloc[0::2].values == snps[['CHROM', 'POS']].iloc[1::2].values)
        # TODO rewrite this
        from einops import rearrange
        snp2sindex = {tuple(snp_position): index for index, snp_position
                      in snps[['CHROM', 'POS']].drop_duplicates().reset_index(drop=True).iterrows()}
        snp2ref_alt = snps.groupby(['CHROM', 'POS'])['BASE'].agg(tuple).to_dict()
        genotype_snp_beta_prior = rearrange(self.variant_betas[snps['VARIANT_ID']],
                                            '(snp variant) donor -> snp donor variant', variant=2).copy()
        return snp2sindex, snp2ref_alt, genotype_snp_beta_prior

    def contains_snp(self, chromosome, position):
        return (chromosome, position) in self.snips

    def save_betas(self, path_or_buf, *, external_betas: np.ndarray = None):
        # TODO scenario for saving / loading of genotypes
        columns = defaultdict(list)
        betas = self.variant_betas
        if external_betas is not None:
            assert betas.shape == external_betas.shape
            betas = external_betas

        for (chrom, pos, base), variant_id in self.snp2snpid.items():
            columns['CHROM'].append(chrom)
            columns['POS'].append(pos)
            columns['BASE'].append(base)

            # subtract default. TODO maybe add defaults only during computation?
            variant_betas = betas[variant_id] - 1.
            columns['DEFAULT_PRIOR'].append(variant_betas.mean())
            for donor, beta in zip(self.genotype_names, variant_betas):
                columns[donor].append(beta)

        pd.DataFrame(columns).to_csv(path_or_buf, sep='\t', index=False)


class Demultiplexer:
    """
    Demultiplexer that can infer (learn) additional information about genotypes to achieve better quality.

    There are two ways of running EM.
    - one is to compute probability for each cb+ub, but then
      - easier to compute posterior for different mixtures
      - hard to limit contribution of a single SNP (this was deciding after all)
    - second is to compute contributions of SNPs
      - in this case limiting contribution from a single cb+ub is hard, but it is limited by group size and
        number of possible modifications (AS limit in terms of cellranger/STAR alignment)
    Second one is used here.
    """

    @staticmethod
    def preprocess_snp_calls(chromosome2compressed_snp_calls,
                             snp2ref_alt,
                             snp2sindex):

        mindex2bindex, merged_snps = CompressedSNPCalls.merge(list(chromosome2compressed_snp_calls.items()), snp2sindex)
        preprocessed_snps = []
        sindex2ref_alt = {sindex: snp2ref_alt[chrom_pos] for chrom_pos, sindex in snp2sindex.items()}
        for molecule_index, sindex, base_index, p_base_wrong in merged_snps:
            # only handle situations with either ref or alt. skip otherwise
            ref, alt = sindex2ref_alt[sindex]
            base = decompress_base(base_index)
            if base in (ref, alt):
                is_alt = base == alt
                snp = (
                    molecule_index,
                    sindex,
                    is_alt,
                    p_base_wrong,
                )
                preprocessed_snps.append(snp)
        return mindex2bindex, preprocessed_snps

    @staticmethod
    def compute_probs_from_betas(variant_index2snp_index, variant_index2betas, p_genotype_clip):
        probs = np.zeros(shape=variant_index2betas.shape, dtype='float32')
        for genotype_id in range(variant_index2betas.shape[1]):
            denom = np.bincount(variant_index2snp_index, weights=variant_index2betas[:, genotype_id])[variant_index2snp_index]
            probs[:, genotype_id] = variant_index2betas[:, genotype_id] / denom.clip(1e-7)
        return probs.clip(p_genotype_clip, 1 - p_genotype_clip)

    @staticmethod
    def staged_genotype_learning(chromosome2compressed_snp_calls,
                                 genotypes: ProbabilisticGenotypes,
                                 barcode_handler: BarcodeHandler,
                                 n_iterations=5,
                                 power=2,
                                 p_genotype_clip=0.01,
                                 save_learnt_genotypes_to=None):
        chrom_pos_base2variant_index, variant_index2snp_index, variant_index2betas, calls = \
            Demultiplexer.compute_compressed_snps(chromosome2compressed_snp_calls, genotypes)

        # genotype_snp_prior, snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = \
        #     Demultiplexer.compute_compressed_snps(chromosome2compressed_snp_calls, genotypes)

        n_barcodes = len(barcode_handler.barcode2index)
        n_genotypes = len(genotypes.genotype_names)

        genotype_snp_posterior = variant_index2betas.copy()

        for iteration in range(n_iterations):
            genotype_prob = Demultiplexer.compute_probs_from_betas(
                variant_index2snp_index, genotype_snp_posterior, p_genotype_clip=p_genotype_clip)

            barcode_posterior_logits = np.zeros([n_barcodes, n_genotypes], dtype="float32")
            for gindex in range(n_genotypes):
                p = genotype_prob[calls['variant_id'], gindex]
                # TODO track down usage of p_wrong_read (probably none) and p_wrong_group (should be somewhere in old compressing logic).
                log_penalties = np.log(p * (1 - calls['p_base_wrong']) + calls['p_base_wrong'].clip(1e-4))
                fast_np_add_at_1d(barcode_posterior_logits[:, gindex], calls['compressed_cb'], log_penalties)

            barcode_posterior_probs = softmax(barcode_posterior_logits, axis=-1)
            barcode_posterior_probs_df = pd.DataFrame(
                data=barcode_posterior_probs, index=barcode_handler.ordered_barcodes, columns=genotypes.genotype_names,
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
                contribution = (barcode_posterior_probs[calls['compressed_cb'], gindex] * (
                        1 - calls['p_base_wrong'])) ** power
                fast_np_add_at_1d(genotype_snp_posterior[:, gindex], calls['variant_id'], contribution)

    @staticmethod
    def compress_snp_calls(mindex2bindex, snps):
        """ leaves only one copy for multiple calls from multiple molecules within the same barcode """
        bindex_sindex_alt2prob = {}
        for mindex, sindex, is_alt, p_wrong in snps:
            bindex = mindex2bindex[mindex]
            prev_prob = bindex_sindex_alt2prob.get((bindex, sindex, is_alt), 1)
            bindex_sindex_alt2prob[bindex, sindex, is_alt] = prev_prob * np.clip(p_wrong, 0, 1)
        # important: need dict to be ordered
        snp_bindices, snp_sindices, snp_is_alt = np.asarray(list(bindex_sindex_alt2prob), dtype="int32").T
        snp_p_wrong = np.asarray(list(bindex_sindex_alt2prob.values()), dtype="float32")
        return snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices

    @staticmethod
    def compute_compressed_snps(chromosome2compressed_snp_calls: Dict[str, CompressedSNPCalls],
                                genotypes: ProbabilisticGenotypes):
        chrom_pos_base2variant_index = genotypes.snp2snpid
        variant_index2betas = genotypes.variant_betas
        chrom_pos2snp_index = {}
        variant_index2snp_index = np.zeros(len(variant_index2betas), dtype='int32')
        for (chrom, pos, _base), variant_index in chrom_pos_base2variant_index.items():
            if (chrom, pos) not in chrom_pos2snp_index:
                chrom_pos2snp_index[chrom, pos] = len(chrom_pos2snp_index)
            snp_index = chrom_pos2snp_index[chrom, pos]
            variant_index2snp_index[variant_index] = snp_index

        assert np.all(variant_index2betas > 0), 'bad loaded genotypes, negative betas appeared'

        calls = np.array(
            [(-1, -1, -1, -1., -1.)] * sum(calls.n_snp_calls for calls in chromosome2compressed_snp_calls.values()),
            dtype=[('variant_id', 'int32'), ('compressed_cb', 'int32'), ('molecule_id', 'int32'),
                   ('p_base_wrong', 'float32'), ('p_molecule_aligned_wrong', 'float32')]
        )

        start = 0
        n_molecules = 0
        for chromosome, compressed_snp_calls in chromosome2compressed_snp_calls.items():
            variant_calls = compressed_snp_calls.snp_calls[:compressed_snp_calls.n_snp_calls]
            molecules = compressed_snp_calls.molecules[:compressed_snp_calls.n_molecules]

            fragment = calls[start:start + compressed_snp_calls.n_snp_calls]
            fragment['variant_id'] = [
                chrom_pos_base2variant_index.get((chromosome, pos, decompress_base(base_index)), -1)
                for pos, base_index in variant_calls[['snp_position', 'base_index']]
            ]
            fragment['compressed_cb'] = molecules['compressed_cb'][variant_calls['molecule_index']]
            fragment['molecule_id'] = variant_calls['molecule_index'] + n_molecules
            fragment['p_base_wrong'] = variant_calls['p_base_wrong']
            fragment['p_molecule_aligned_wrong'] = molecules['p_group_misaligned'][variant_calls['molecule_index']]

            start += compressed_snp_calls.n_snp_calls
            n_molecules += compressed_snp_calls.n_molecules

        # filtering from those calls that did not match any snp
        calls = calls[calls['variant_id'] != -1]
        return chrom_pos_base2variant_index, variant_index2snp_index, variant_index2betas, calls

    # @staticmethod
    # def compute_compressed_snps(chromosome2compressed_snp_calls, genotypes: ProbabilisticGenotypes):
    #     snp2sindex, snp2ref_alt, genotype_snp_beta_prior = genotypes.generate_genotype_snp_beta_prior()
    #     mindex2bindex, snps = Demultiplexer.preprocess_snp_calls(
    #         chromosome2compressed_snp_calls, snp2ref_alt=snp2ref_alt, snp2sindex=snp2sindex
    #     )
    #     snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = Demultiplexer.compress_snp_calls(mindex2bindex, snps)
    #     genotype_snp_posterior = genotype_snp_beta_prior.copy()
    #     assert np.all(snp_p_wrong >= 0)
    #     assert np.all(snp_p_wrong <= 1)
    #     assert np.all(genotype_snp_posterior > 0), 'bad loaded genotypes, negative betas appeared'
    #
    #     return genotype_snp_posterior, snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices

    @staticmethod
    def predict_posteriors(
            chromosome2compressed_snp_calls,
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            only_singlets: bool,
            p_genotype_clip=0.01,
            doublet_prior=0.35,
    ):
        chrom_pos_base2variant_index, variant_index2snp_index, variant_index2betas, calls = \
            Demultiplexer.compute_compressed_snps(chromosome2compressed_snp_calls, genotypes)

        # genotype_snp_prior, snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = \
        #     Demultiplexer.compute_compressed_snps(chromosome2compressed_snp_calls, genotypes)

        n_genotypes = len(genotypes.genotype_names)

        genotype_prob = Demultiplexer.compute_probs_from_betas(
            variant_index2snp_index, variant_index2betas, p_genotype_clip=p_genotype_clip)
        assert np.isfinite(genotype_prob).all()

        if only_singlets:
            barcode_posterior_logits = np.zeros([len(barcode_handler.barcode2index), n_genotypes], dtype="float32")
        else:
            barcode_posterior_logits = np.zeros(
                [len(barcode_handler.barcode2index), n_genotypes * (n_genotypes + 1) // 2])

        column_names = []
        for gindex, genotype in enumerate(genotypes.genotype_names):
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

            for gindex1, genotype1 in enumerate(genotypes.genotype_names):
                for gindex2, genotype2 in enumerate(genotypes.genotype_names):
                    if gindex1 < gindex2:
                        p1 = genotype_prob[calls['variant_id'], gindex1]
                        p2 = genotype_prob[calls['variant_id'], gindex2]
                        p = (p1 + p2) * 0.5
                        log_penalties = np.log(p * (1 - calls['p_base_wrong']) + calls['p_base_wrong'].clip(1e-4))
                        fast_np_add_at_1d(barcode_posterior_logits[:, len(column_names)], calls['compressed_cb'],
                                          log_penalties)
                        barcode_posterior_logits[:, len(column_names)] += doublet_logit_bonus
                        column_names += [f'{genotype1}+{genotype2}']
        logits_df = pd.DataFrame(
            data=barcode_posterior_logits,
            index=list(barcode_handler.barcode2index), columns=column_names,
        )
        logits_df.index.name = 'BARCODE'
        probs_df = pd.DataFrame(
            data=softmax(barcode_posterior_logits, axis=1),
            index=list(barcode_handler.barcode2index), columns=column_names,
        )
        probs_df.index.name = 'BARCODE'
        return logits_df, probs_df

    @staticmethod
    def predict_posteriors_old(
            chromosome2compressed_snp_calls,
            genotypes: ProbabilisticGenotypes,
            barcode_handler: BarcodeHandler,
            only_singlets: bool,
            p_genotype_clip=0.01,
            doublet_prior=0.35,
    ):
        genotype_snp_posterior, snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = \
            Demultiplexer.compute_compressed_snps(chromosome2compressed_snp_calls, genotypes)
        genotype2gindex = {genotype: gindex for gindex, genotype in enumerate(genotypes.genotype_names)}

        genotype_prob = genotype_snp_posterior / genotype_snp_posterior.sum(axis=-1, keepdims=True)
        genotype_prob = genotype_prob.clip(p_genotype_clip, 1 - p_genotype_clip)
        assert np.isfinite(genotype_prob).all()

        n_genotypes = len(genotype2gindex)
        if only_singlets:
            barcode_posterior_logits = np.zeros([len(barcode_handler.barcode2index), n_genotypes], dtype="float32")
        else:
            barcode_posterior_logits = np.zeros(
                [len(barcode_handler.barcode2index), n_genotypes * (n_genotypes + 1) // 2])

        column_names = []
        for genotype, gindex in genotype2gindex.items():
            p = genotype_prob[snp_sindices, gindex, snp_is_alt]
            log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
            fast_np_add_at_1d(barcode_posterior_logits[:, len(column_names)], snp_bindices, log_penalties)
            column_names += [genotype]

        if not only_singlets:
            # computing correction for doublet as the prior proportion of doublets will
            # otherwise depend on number of genotypes. Correction comes from
            #  n_singlet_options / singlet_prior =
            #  = n_doublet_options / doublet_prior * np.exp(doublet_logit_bonus)
            doublet_logit_bonus = np.log(n_genotypes * doublet_prior)
            doublet_logit_bonus -= np.log(n_genotypes * max(n_genotypes - 1, 0.01) / 2 * (1 - doublet_prior))

            for genotype1, gindex1 in genotype2gindex.items():
                for genotype2, gindex2 in genotype2gindex.items():
                    if gindex1 < gindex2:
                        p1 = genotype_prob[snp_sindices, gindex1, snp_is_alt]
                        p2 = genotype_prob[snp_sindices, gindex2, snp_is_alt]
                        p = (p1 + p2) * 0.5
                        log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
                        fast_np_add_at_1d(barcode_posterior_logits[:, len(column_names)], snp_bindices, log_penalties)
                        barcode_posterior_logits[:, len(column_names)] += doublet_logit_bonus
                        column_names += [f'{genotype1}+{genotype2}']

        logits_df = pd.DataFrame(
            data=barcode_posterior_logits,
            index=list(barcode_handler.barcode2index), columns=column_names,
        )
        logits_df.index.name = 'BARCODE'
        probs_df = pd.DataFrame(
            data=softmax(barcode_posterior_logits, axis=1),
            index=list(barcode_handler.barcode2index), columns=column_names,
        )
        probs_df.index.name = 'BARCODE'
        return logits_df, probs_df
