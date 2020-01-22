from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import pysam
from scipy.special import softmax

from demultiplexit.cellranger_specific import discard_read, compute_p_mistake
from demultiplexit.utils import hash_string, fast_np_add_at_1d, BarcodeHandler


class ChromosomeSNPLookup:
    def __init__(self, positions: np.ndarray):
        """
        Allows fast checking of intersection with SNPs, a bit memory-inefficient, but quite fast.
        Important note is that information from only one chromosome can be stored, so I care only about positions.
        :param positions: zero-based (!) positions of SNPs on chromosomes
        """
        assert isinstance(positions, np.ndarray)
        assert np.array_equal(positions, np.sort(positions))
        assert len(positions) < 2 ** 31, "can't handle that big chromosome - will consume too much memory"
        # lookup takes a lot of space, so would be better to either minimize dtype
        # or to keep only a part of it
        self.lookup = np.cumsum(np.bincount(positions + 1)).astype("uint16")
        self.positions = positions

    def snips_exist(self, start, end):
        """ end is excluded, start is included """
        if start >= len(self.lookup):
            return False
        end = min(end, len(self.lookup) - 1)
        return self.lookup[end] != self.lookup[start]

    def get_snps(self, read):
        snps = []  # list of tuples: (reference_position, nucleotide, base mapping quality)
        # fast check first
        if not self.snips_exist(read.reference_start, read.reference_end + 1):
            return snps

        seq = read.seq
        qual = read.query_qualities

        read_position = 0
        refe_position = read.pos

        for code, l in read.cigartuples:
            if code in [0, 7, 8]:
                # coincides or does not coincide
                if self.snips_exist(refe_position, refe_position + l):
                    lo = np.searchsorted(self.positions, refe_position)
                    hi = np.searchsorted(self.positions, refe_position + l)
                    assert hi != lo
                    for ref_position in self.positions[lo:hi]:
                        position_in_read = read_position + (ref_position - refe_position)
                        snps.append((ref_position, seq[position_in_read], qual[position_in_read]))

                refe_position += l
                read_position += l
            elif code in (2, 3):
                # should we include deletion as snp?
                refe_position += l
            elif code in (1, 4, 5, 6):
                read_position += l
            else:
                raise NotImplementedError(f"cigar code unknown {code}")
        return snps


def compress_reads_group_to_snips(
    reads, snp_lookup: ChromosomeSNPLookup, compute_p_read_misaligned, skip_complete_duplicates=True
) -> Tuple[float, dict]:
    p_group_misaligned = 1
    processed_positions = set()
    SNPs = {}  # position to list of pairs character, quality
    for read in reads:
        p_read_misaligned = compute_p_read_misaligned(read)
        pos = (read.reference_start, read.reference_end, read.get_tag("AS"))
        if skip_complete_duplicates and (pos in processed_positions):
            # ignoring duplicates with same begin/end and number of mismatches
            # relatively cheap way to exclude complete duplicates, as those do not contribute
            continue
        processed_positions.add(pos)
        p_group_misaligned *= p_read_misaligned

        for reference_position, base, base_qual in snp_lookup.get_snps(read):
            SNPs.setdefault(reference_position, []).append([base, base_qual, p_read_misaligned])

    return p_group_misaligned, SNPs


def compress_old_groups(
    threshold_position,
    cbub2position_and_reads,
    cbub2qual_and_snps,
    snp_lookup: ChromosomeSNPLookup,
    compute_p_read_misaligned,
):
    """
    Compresses groups of reads that already left region of our consideration. We keep snp list instead of reads.
    Compressed groups are removed from cbub2position_and_reads, compressed version written to cbub2qual_and_snps
    """
    to_remove = []
    for cbub, (position, reads) in cbub2position_and_reads.items():
        if position < threshold_position:
            to_remove.append(cbub)
            if not snp_lookup.snips_exist(
                min(read.reference_start for read in reads), max(read.reference_end for read in reads) + 1,
            ):
                # no reads in this fragment, why caring? just forget about it
                continue
            p_group_misaligned, snips = compress_reads_group_to_snips(
                reads, snp_lookup, compute_p_read_misaligned=compute_p_read_misaligned
            )
            if len(snips) == 0:
                continue
            if (cbub not in cbub2qual_and_snps) or (cbub2qual_and_snps[cbub][0] > p_group_misaligned):
                cbub2qual_and_snps[cbub] = p_group_misaligned, snips

    for cbub in to_remove:
        cbub2position_and_reads.pop(cbub)


def count_call_variants(
    bamfile_or_filename,
    chromosome,
    chromosome_snps_zero_based,
    cellbarcode_compressor,
    compute_p_read_misaligned=compute_p_mistake,
    discard_read=discard_read,
    verbose=False,
):
    prev_segment = None
    cbub2qual_and_snps = {}
    cbub2position_and_reads = {}
    snp_lookup = ChromosomeSNPLookup(chromosome_snps_zero_based)
    if isinstance(bamfile_or_filename, str):
        bamfile_or_filename = pysam.AlignmentFile(bamfile_or_filename)

    for read in bamfile_or_filename.fetch(chromosome):
        curr_segment = read.pos // 10 ** 6
        if curr_segment != prev_segment:
            if verbose:
                print(f"before clearing: {len(cbub2position_and_reads):10} {len(cbub2qual_and_snps):10}", read.pos)
            compress_old_groups(
                read.pos - 10 ** 6, cbub2position_and_reads, cbub2qual_and_snps, snp_lookup, compute_p_read_misaligned,
            )
            prev_segment = curr_segment

        if discard_read(read):
            continue
        if not read.has_tag("CB"):
            continue
        cb = cellbarcode_compressor(read.get_tag("CB"))
        if cb is None:
            continue
        if not read.has_tag("UB"):
            continue
        ub = hash_string(read.get_tag("UB"))
        cbub = cb, ub
        if cbub not in cbub2position_and_reads:
            cbub2position_and_reads[cbub] = [read.reference_end, [read]]
        else:
            cbub2position_and_reads[cbub][0] = max(read.reference_end, cbub2position_and_reads[cbub][0])
            cbub2position_and_reads[cbub][1].append(read)
    compress_old_groups(
        np.inf, cbub2position_and_reads, cbub2qual_and_snps, snp_lookup, compute_p_read_misaligned,
    )
    return cbub2qual_and_snps


class GenotypesProbComputing:
    type2code = {"0/0": 0, "0/1": 1, "1/1": 2, "./.": 3}
    code2type = {v: k for k, v in type2code.items()}
    code2prior = np.array([[99, 1], [50, 50], [1, 99], [1, 1]])

    def __init__(self, snp_df, donor_names, verbose=False):
        # TODO introduce data priors for all the SNPs
        # also code below assumes that someone dealt with too bad SNPs (that is true right now)
        # TODO keep only priors for both types of snps (initial and introduced)
        self.donor_names = list(donor_names)
        assert (np.sort(self.donor_names) == self.donor_names).all()

        snp_df = (
            snp_df.set_index(["CHROM", "POS", "REF", "ALT"])[self.donor_names].replace(self.type2code).astype("uint8")
        )

        self.snips = {}
        for (chromosome, position, ref, alt), genotype_codes in snp_df.iterrows():
            self.snips[chromosome, position] = (ref, alt, genotype_codes.values)
            if verbose and len(self.snips) % 10000 == 0:
                print("completed snps: ", len(self.snips))

        self.introduced_snps_chrompos2ref_alt_priors = {}

    def get_positions_for_chromosome(self, chromosome_name: str):
        positions = [pos for chr, pos in self.snips if chr == chromosome_name]
        positions += [pos for chr, pos in self.introduced_snps_chrompos2ref_alt_priors if chr == chromosome_name]

        return np.unique(np.asarray(positions, dtype=int))

    def generate_genotype_snp_beta_prior(
        self, gsa_prior_weight=100, data_prior_strength=100,
    ):
        snp2sindex = {
            (chromosome, position): sindex
            for sindex, (chromosome, position) in enumerate(
                list(self.snips) + list(self.introduced_snps_chrompos2ref_alt_priors)
            )
        }
        snp2ref_alt = {}
        n_genotypes = len(self.donor_names)
        n_snps = len(snp2sindex)
        genotype_snp_beta_prior = np.zeros([n_snps, n_genotypes, 2], dtype="float32")
        for (chromosome, position), (ref, alt, genotype_codes) in self.snips.items():
            snp2ref_alt[chromosome, position] = (ref, alt)
            priors = self.code2prior[genotype_codes]
            is_unknown = genotype_codes == self.type2code["./."]
            priors[is_unknown] = priors[~is_unknown].mean(axis=0, keepdims=True)
            genotype_snp_beta_prior[snp2sindex[chromosome, position]] = priors * (gsa_prior_weight / priors.mean())

        for (chromosome, position), (ref, alt, priors) in self.introduced_snps_chrompos2ref_alt_priors.items():
            snp2ref_alt[chromosome, position] = (ref, alt)
            genotype_snp_beta_prior[snp2sindex[chromosome, position]] = (
                priors / priors.sum().clip(1) * data_prior_strength
            )

        return snp2sindex, snp2ref_alt, genotype_snp_beta_prior


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
        snp_prob_genotypes: GenotypesProbComputing,
        gsa_prior_weight=100,
        data_prior_strength=100,
    ):
        self.barcode2bindex = {barcode: position for position, barcode in enumerate(barcode2possible_genotypes.keys())}
        genotypes = set()
        for g in barcode2possible_genotypes.values():
            genotypes.update(g)
        genotypes = np.unique(list(genotypes))
        self.genotype2gindex = {barcode: position for position, barcode in enumerate(genotypes)}

        (
            self.snp2sindex,
            self.snp2ref_alt,
            self.genotype_snp_beta_prior,
        ) = snp_prob_genotypes.generate_genotype_snp_beta_prior(
            gsa_prior_weight=gsa_prior_weight, data_prior_strength=data_prior_strength
        )

        self.mindex2bindex, self.snps = self.preprocess_snp_calls(barcode_handler, chromosome2cbub2qual_and_snps)

        self.barcode_genotype_prior_logits = self.compute_genotype2barcode_logit_prior(barcode2possible_genotypes)

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

    def run_fast_em_iterations(self, n_iterations=10, power=2, p_genotype_clip=0.01, genotype_snp_prior=None):
        snp_bindices, snp_is_alt, snp_p_wrong, snp_sindices = self.compress_snp_calls(self.mindex2bindex, self.snps)
        if genotype_snp_prior is None:
            genotype_snp_prior = self.genotype_snp_beta_prior
        genotype_snp_posterior = genotype_snp_prior.copy()

        for _iteration in range(n_iterations):
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
            yield barcode_posterior_probs_df, barcode_posterior_logits, genotype_snp_posterior

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
                sel = np.index_exp[_i : _i + 10000]
                p = genotype_snp_posterior[snp_bindices[sel], :, snp_is_alt[sel]] / genotype_snp_posterior[
                    snp_bindices[sel], :, :
                ].sum(axis=-1)
                p = p.clip(0.01, 0.99)  # snp x genotype
                log_penalties = np.log(p * (1 - snp_p_wrong[sel][:, None]) + snp_p_wrong[sel][:, None].clip(1e-4))
                contribution = barcode_posterior_logits[snp_bindices[sel]] - log_penalties
                contribution = (softmax(contribution, axis=1) * (1 - snp_p_wrong[sel][:, None])) ** 2

                np.add.at(
                    genotype_snp_posterior[:, :, :].transpose(0, 2, 1),
                    (snp_sindices[sel], snp_is_alt[sel]),
                    contribution,
                )

