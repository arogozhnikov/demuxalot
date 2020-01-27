from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from scipy.special import softmax

from demultiplexit.utils import fast_np_add_at_1d, BarcodeHandler


# TODO rename donor_names to genotypes

class ProbabilisticGenotypes:
    def __init__(self, donor_names: List[str]):
        """
        ProbabilisticGenotype represents our accumulated knowledge about SNPs (and SNPs only) for genotypes.
        Can aggregate information from GSA, our prior guesses and genotype information learnt from RNAseq.
        Donor names can't be changed once object is created.
        Class doesn't handle more than one SNP at position (examples are A/T and A/C at the same position),
        so we keep the first SNP for position.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas.
        """
        self.snips = {}
        self.donor_names = list(donor_names)
        assert (np.sort(self.donor_names) == self.donor_names).all(), 'please order donors'

    def add_vcf(self, snp_df, prior_strength=100, verbose=False):
        """
        Add information from parsed VCF
        :param snp_df: pd.DataFrame with information about possible donors.
            Columns are donors, rows are
            Can contain more donors
        """
        # TODO parse VCF right here
        type2code = {"0/0": 0, "0/1": 1, "1/1": 2, "./.": 3}
        code2prior = np.array([[0.99, 0.01], [0.50, 0.50], [0.01, 0.99], [0, 0]], dtype='float32') * prior_strength

        snp_df = snp_df.set_index(["CHROM", "POS", "REF", "ALT"])[self.donor_names].replace(type2code).astype("uint8")

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
                    self.snips[2] += priors[:, ::-1]
                else:
                    self.snips[2] += priors
            else:
                self.snips[chromosome, position] = (ref, alt, priors)

            if verbose and len(self.snips) % 10000 == 0:
                print("completed snps: ", len(self.snips))

    def add_prior_betas(self, prior_filename, *, prior_strength):
        prior_knowledge = pd.read_csv(prior_filename, sep='\t')
        tech_columns = ['CHROM', 'POS', 'BASE', 'DEFAULT_PRIOR']
        for column in tech_columns:
            assert column in prior_knowledge.columns
        donor_names_in_prior = [column for column in prior_knowledge.columns if column not in tech_columns]
        print('Provided prior information about donors:', donor_names_in_prior)
        for donor in self.donor_names:
            if donor not in donor_names_in_prior:
                print(f'no information for donor {donor}, filling with default')
                prior_knowledge[donor] = prior_knowledge['DEFAULT_PRIOR']

        prior_knowledge[self.donor_names] *= prior_strength

        for (chromosome, position), snp_priors in prior_knowledge.groupby(['CHROM', 'POS']):
            if len(snp_priors) != 2:
                print('Can handle only two alternatives for the same position in genome', chromosome, position)
                continue

            bases = list(snp_priors['BASE'])
            assert bases[0] != bases[1]
            snp_priors = snp_priors[self.donor_names].values.T

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

    def get_positions_for_chromosome(self, chromosome_name: str):
        positions = [pos for chromosome, pos in self.snips if chromosome == chromosome_name]
        return np.unique(np.asarray(positions, dtype=int))

    def generate_genotype_snp_beta_prior(self):
        n_genotypes = len(self.donor_names)
        n_snps = len(self.snips)

        snp2sindex = {}
        snp2ref_alt = {}
        genotype_snp_beta_prior = np.zeros([n_snps, n_genotypes, 2], dtype="float32")

        for sindex, ((chromosome, position), (ref, alt, priors)) in enumerate(sorted(self.snips.items())):
            snp2sindex[chromosome, position] = sindex
            snp2ref_alt[chromosome, position] = (ref, alt)
            genotype_snp_beta_prior[sindex] = priors

        return snp2sindex, snp2ref_alt, genotype_snp_beta_prior

    def save_betas(self, path_or_buf, *, external_betas: np.ndarray = None):
        if external_betas is not None:
            assert external_betas.shape[0] == len(self.snips)
            assert external_betas.shape[1] == len(self.donor_names)
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
                **dict(zip(self.donor_names, ref_betas))
            })

            result.append({
                'CHROM': chromosome,
                'POS': position,
                'BASE': alt,
                'DEFAULT_PRIOR': alt_betas.mean(),
                **dict(zip(self.donor_names, alt_betas))
            })
        pd.DataFrame(result).to_csv(path_or_buf, sep='\t', index=False)


class TrainableDemultiplexer:
    """
    Demultiplexer that can infer (learn) additional information about genotypes to achieve better quality.

    There are two ways of running EM.
    - one is to compute probability for each cb+ub, but then
      - easier to compute posterior for different mixtures
      - hard to limit contribution of a single SNP (this was deciding after all)
    - second is to compute contributions of SNPs
      - limiting contribution from a single cb+ub is hard, but it is limited by group size and
        number of possible modifications (AS limit)
    """

    def __init__(
            self,
            chromosome2cbub2qual_and_snps,
            barcode2possible_genotypes,
            barcode_handler: BarcodeHandler,
            probabilistic_genotypes: ProbabilisticGenotypes,
    ):
        self.barcode2bindex = {barcode: position for position, barcode in enumerate(barcode2possible_genotypes.keys())}
        genotypes = list(probabilistic_genotypes.donor_names)
        assert list(sorted(genotypes)) == genotypes, 'genotypes names are not sorted'
        self.genotype2gindex = {barcode: position for position, barcode in enumerate(genotypes)}

        self.snp2sindex, self.snp2ref_alt, self.genotype_snp_beta_prior \
            = probabilistic_genotypes.generate_genotype_snp_beta_prior()

        self.mindex2bindex, self.snps = self.preprocess_snp_calls(barcode_handler, chromosome2cbub2qual_and_snps)

        self.barcode_genotype_prior_logits = self.compute_genotype2barcode_logit_prior(barcode2possible_genotypes)
        self.probabilistic_genotypes = probabilistic_genotypes

    def preprocess_snp_calls(self, barcode_handler, chromosome2cbub2qual_and_snps):
        preprocessed_snps = []  # (mindex, sindex, is_alt, p_base_wrong)
        mindex2bindex = []
        for chromosome, cbub2qual_and_snps in chromosome2cbub2qual_and_snps.items():
            for (compressed_cb, _ub), (_p_group_misaligned, snps) in cbub2qual_and_snps.items():
                if snps is None:
                    # we skip group without SNPs
                    continue
                molecule_index = len(mindex2bindex)
                mindex2bindex.append(self.barcode2bindex[barcode_handler.index2barcode[compressed_cb]])
                for snp_position, bases_probs in snps.items():
                    base2p_wrong = defaultdict(lambda: 1)
                    for base, base_qual, _p_read_misaligned in bases_probs:
                        base2p_wrong[base] *= 0.1 ** (0.1 * min(base_qual, 40))

                    if len(base2p_wrong) > 1:
                        # molecule should have only one candidate, this this is artifact
                        # of reverse transcription or amplification or sequencing
                        best_prob = min(base2p_wrong.values())
                        # drop poorly sequenced candidate(s), this resolves some obvious conflicts
                        base2p_wrong = {
                            base: p_wrong
                            for base, p_wrong in base2p_wrong.items()
                            if p_wrong * 0.01 <= best_prob or p_wrong < 0.001
                        }

                    # if #candidates is still not one, discard this sample
                    if len(base2p_wrong) != 1:
                        continue

                    # only handle situations with either ref or alt. skip otherwise
                    ref, alt = self.snp2ref_alt[chromosome, snp_position]
                    if (ref in base2p_wrong) + (alt in base2p_wrong) == 1:
                        is_alt = alt in base2p_wrong
                        p_base_wrong = base2p_wrong[alt] if is_alt else base2p_wrong[ref]
                        snp = (
                            molecule_index,
                            self.snp2sindex[chromosome, snp_position],
                            is_alt,
                            p_base_wrong,
                        )
                        preprocessed_snps.append(snp)
        return mindex2bindex, preprocessed_snps

    def compute_genotype2barcode_logit_prior(self, barcode2possible_genotypes):
        barcode_genotype_prior_logits = np.zeros([len(self.barcode2bindex), len(self.genotype2gindex)], dtype="float32")
        barcode_genotype_prior_logits -= 1000
        for barcode, possible_genotypes in barcode2possible_genotypes.items():
            for genotype in possible_genotypes:
                barcode_genotype_prior_logits[self.barcode2bindex[barcode], self.genotype2gindex[genotype]] = 0
        return barcode_genotype_prior_logits

    def staged_genotype_learning(self, n_iterations=5, power=2, p_genotype_clip=0.01,
                                 genotype_snp_prior=None, save_learnt_genotypes_to=None):
        snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = self.compress_snp_calls(self.mindex2bindex, self.snps)
        if genotype_snp_prior is None:
            genotype_snp_prior = self.genotype_snp_beta_prior
        genotype_snp_posterior = genotype_snp_prior.copy()

        for iteration in range(n_iterations):
            genotype_prob = genotype_snp_posterior / genotype_snp_posterior.sum(axis=-1, keepdims=True)
            genotype_prob = genotype_prob.clip(p_genotype_clip, 1 - p_genotype_clip)

            barcode_posterior_logits = np.zeros([len(self.barcode2bindex), len(self.genotype2gindex)], dtype="float32")
            for gindex in self.genotype2gindex.values():
                p = genotype_prob[snp_sindices, gindex, snp_is_alt]
                log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
                fast_np_add_at_1d(barcode_posterior_logits[:, gindex], snp_bindices, log_penalties)

            barcode_posterior_probs = softmax(barcode_posterior_logits, axis=-1)
            barcode_posterior_probs_df = pd.DataFrame(
                data=barcode_posterior_probs, index=list(self.barcode2bindex), columns=list(self.genotype2gindex)
            )
            # yielding here to provide aligned posteriors for genotypes and barcodes
            debug_information = {
                'barcode_logits': barcode_posterior_logits,
                'snp_prior': genotype_snp_prior,
                'genotype_snp_posterior': genotype_snp_posterior
            }
            if (save_learnt_genotypes_to is not None) and (iteration == n_iterations - 1):
                assert isinstance(save_learnt_genotypes_to, str)
                self.probabilistic_genotypes.save_betas(save_learnt_genotypes_to, external_betas=genotype_snp_posterior)
            yield barcode_posterior_probs_df, debug_information

            genotype_snp_posterior = genotype_snp_prior.copy()
            for gindex in self.genotype2gindex.values():
                contribution = (barcode_posterior_probs[snp_bindices, gindex] * (1 - snp_p_wrong)) ** power
                np.add.at(genotype_snp_posterior[:, gindex, :], (snp_sindices, snp_is_alt), contribution)

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

    def predict_posteriors(
            self,
            genotype_snp_posterior,
            chromosome2cbub2qual_and_snps,
            barcode_handler,
            only_singlets: bool,
            p_genotype_clip=0.01,
    ):
        assert isinstance(genotype_snp_posterior, np.ndarray)
        assert genotype_snp_posterior.shape[0] == len(self.snp2sindex)
        assert genotype_snp_posterior.shape[1] == len(self.genotype2gindex)

        self.mindex2bindex, self.snps = self.preprocess_snp_calls(barcode_handler, chromosome2cbub2qual_and_snps)
        snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = self.compress_snp_calls(self.mindex2bindex, self.snps)

        genotype_prob = genotype_snp_posterior / genotype_snp_posterior.sum(axis=-1, keepdims=True)
        genotype_prob = genotype_prob.clip(p_genotype_clip, 1 - p_genotype_clip)

        if only_singlets:
            barcode_posterior_logits = np.zeros([len(self.barcode2bindex), len(self.genotype2gindex)], dtype="float32")
            for gindex in self.genotype2gindex.values():
                p = genotype_prob[snp_sindices, gindex, snp_is_alt]
                log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
                fast_np_add_at_1d(barcode_posterior_logits[:, gindex], snp_bindices, log_penalties)
            return barcode_posterior_logits
        else:
            barcode_posterior_logits = np.zeros(
                [len(self.barcode2bindex), len(self.genotype2gindex), len(self.genotype2gindex)], dtype="float32"
            )
            for gindex1 in self.genotype2gindex.values():
                for gindex2 in self.genotype2gindex.values():
                    p1 = genotype_prob[snp_sindices, gindex1, snp_is_alt]
                    p2 = genotype_prob[snp_sindices, gindex2, snp_is_alt]
                    p = (p1 + p2) * 0.5
                    log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
                    fast_np_add_at_1d(barcode_posterior_logits[:, gindex1, gindex2], snp_bindices, log_penalties)
            return barcode_posterior_logits

    def run_fast_em_iterations_without_self_effect(self, n_iterations=10):
        snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = self.compress_snp_calls(self.mindex2bindex, self.snps)

        genotype_snp_posterior = self.genotype_snp_beta_prior.copy()

        for _iteration in range(n_iterations):
            barcode_posterior_logits = np.zeros([len(self.barcode2bindex), len(self.genotype2gindex)], dtype="float32")
            genotype_prob = genotype_snp_posterior / genotype_snp_posterior.sum(axis=-1, keepdims=True)
            genotype_prob = genotype_prob.clip(0.01, 0.99)

            for gindex in self.genotype2gindex.values():
                p = genotype_prob[snp_sindices, gindex, snp_is_alt]
                log_penalties = np.log(p * (1 - snp_p_wrong) + snp_p_wrong.clip(1e-4))
                np.add.at(barcode_posterior_logits[:, gindex], snp_bindices, log_penalties)

            barcode_posterior_probs_df = pd.DataFrame(
                data=softmax(barcode_posterior_logits, axis=-1),
                index=list(self.barcode2bindex), columns=list(self.genotype2gindex)
            )
            yield barcode_posterior_probs_df, barcode_posterior_logits, genotype_snp_posterior

            genotype_snp_posterior = self.genotype_snp_beta_prior.copy()
            for _i in np.arange(0, len(snp_bindices), 10000):
                sel = np.index_exp[_i: _i + 10000]
                p = genotype_snp_posterior[snp_bindices[sel], :, snp_is_alt[sel]]
                p = p / genotype_snp_posterior[snp_bindices[sel], :, :].sum(axis=-1)
                p = p.clip(0.01, 0.99)  # snp x genotype
                log_penalties = np.log(p * (1 - snp_p_wrong[sel][:, None]) + snp_p_wrong[sel][:, None].clip(1e-4))
                contribution = barcode_posterior_logits[snp_bindices[sel]] - log_penalties
                contribution = (softmax(contribution, axis=1) * (1 - snp_p_wrong[sel][:, None])) ** 2

                np.add.at(
                    genotype_snp_posterior[:, :, :].transpose(0, 2, 1),
                    (snp_sindices[sel], snp_is_alt[sel]),
                    contribution,
                )
