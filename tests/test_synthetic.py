"""
These are simple end-to-end tests starting from "BAM" and VCF or assignments.
(all data is synthetic)
"""
import tempfile
from collections import defaultdict

import pysam
import numpy as np
import pandas as pd
from copy import deepcopy

from typing import List
import unittest

from demuxalot import count_snps, Demultiplexer, BarcodeHandler, ProbabilisticGenotypes


def random_array(length):
    return np.random.choice(list('ACGT'), length)


def random_str(length):
    return ''.join(random_array(length))


class Reference:
    def __init__(self, chromosome2length):
        self.chromosome2sequence = {
            chromosome: random_array(length)
            for chromosome, length in chromosome2length.items()
        }
        self.chromosome2length = chromosome2length

    def generate_header_for_bamfile(self):
        return {
            'HD': {'VN': '1.0'},
            'SQ': [dict(LN=l, SN=name) for name, l in self.chromosome2length.items()]
        }

    def generate_modification(self, mutation_prob):
        result = deepcopy(self)
        for chr, seq in result.chromosome2sequence.items():
            mask = np.random.uniform(0, 1, size=len(seq)) < mutation_prob
            result.chromosome2sequence[chr][mask] = random_array(sum(mask))
        return result

    def generate_read(self, read_length, query_name, cb, ub):
        reference_id = np.random.randint(len(self.chromosome2length))
        chromosome, chr_length = list(self.chromosome2length.items())[reference_id]
        seq = self.chromosome2sequence[chromosome]
        start = np.random.randint(0, chr_length - read_length)
        # straight mapping
        a = pysam.AlignedSegment()
        a.query_name = query_name
        a.query_sequence = ''.join(seq[start:start + read_length])
        # flag taken from pysam example, did not analyze
        a.flag = 99
        a.reference_id = reference_id
        a.reference_start = start
        a.mapping_quality = 255
        a.cigar = ((0, read_length),)
        # a.next_reference_id = reference_id
        # a.next_reference_start = 199
        a.template_length = read_length
        a.query_qualities = pysam.qualitystring_to_array("<" * read_length)
        a.tags = (
            ("NM", 1),
            ("RG", "L1"),
            ("NH", 1),
            # normally should also add number of mutations compared to reference
            ("AS", read_length - 2),
            ("CB", cb),
            ("UB", ub),
        )
        return a


def generate_genotypes(genotypes: List[Reference]) -> ProbabilisticGenotypes:
    chr_pos2donor2base = defaultdict(dict)
    for genotype_id, genotype in enumerate(genotypes):
        genotype_name = f'Donor{genotype_id + 1:02}'
        for chr, seq in genotype.chromosome2sequence.items():
            for pos, base in enumerate(seq):
                chr_pos2donor2base[chr, pos][genotype_name] = base

    result = ProbabilisticGenotypes([f'Donor{genotype_id + 1:02}' for genotype_id, _ in enumerate(genotypes)])

    chrom_pos_base2snp_id = {}
    counts = np.zeros([100_000, len(genotypes)], dtype='float32') + 0.5
    for chrpos, donor2base in chr_pos2donor2base.items():
        if set(donor2base.values()).__len__() == 1:
            continue
        for donor, base in donor2base.items():
            chrom_pos_base = tuple([*chrpos, base])
            if chrom_pos_base not in chrom_pos_base2snp_id:
                chrom_pos_base2snp_id[chrom_pos_base] = len(chrom_pos_base2snp_id)
            donor_id = result.genotype_names.index(donor)
            counts[chrom_pos_base2snp_id[chrom_pos_base], donor_id] = 100

    result.snp2snpid = chrom_pos_base2snp_id
    result.variant_betas = counts[:len(result.snp2snpid)]
    return result


def generate_bam_file(
        n_genotypes=20,
        doublets_fraction=0.2,
        mutation_prob=0.01,  # for the sake of speed, real number is 10 times smaller
        read_length=100,
        filename='/tmp/test.bam',
        n_barcodes=1000,
        n_reads_per_barcode=100,
):
    reference = Reference({'chr1': 1000, 'chr2': 1000, 'chr3': 1000})

    genotypes = [reference.generate_modification(mutation_prob) for _ in range(n_genotypes)]
    prob_genotypes = generate_genotypes(genotypes)
    # spoiled_genotypes = deepcopy(prob_genotypes)
    # # dropping some of variants
    # l = spoiled_genotypes.variant_betas.shape[0]
    # spoiled_genotypes.variant_betas[np.arange(l) % 5 != 0] = 1

    barcode2donor_ids = {}
    barcode2donor_names = {}
    for _ in range(n_barcodes):
        doublet = np.random.uniform() < doublets_fraction
        donor_ids = np.random.randint(0, n_genotypes, size=1 + doublet)
        donor_names = [f'Donor{donor_id + 1:02}' for donor_id in donor_ids]
        barcode = random_str(10) + '-1'
        barcode2donor_ids[barcode] = donor_ids
        barcode2donor_names[barcode] = donor_names

    with pysam.AlignmentFile(filename, "wb", header=reference.generate_header_for_bamfile()) as f:
        for barcode, donor_ids in barcode2donor_ids.items():
            for _ in range(n_reads_per_barcode):
                donor_id = np.random.choice(donor_ids)
                read = genotypes[donor_id].generate_read(
                    read_length=read_length,
                    query_name=random_str(20),
                    cb=barcode,
                    # TODO some overlapping in ub + add mistakes
                    ub=random_str(10),
                )
                f.write(read)

    pysam.sort("-o", filename, filename)
    pysam.index(filename)
    return filename, prob_genotypes, barcode2donor_ids, barcode2donor_names


def compute_loss(barcode2donor_names, barcode2probs):
    probs = barcode2probs * 0
    for barcode, correct_donors in barcode2donor_names.items():
        for donor in correct_donors:
            probs.loc[barcode, donor] = barcode2probs.loc[barcode, donor]
    p = probs.sum(axis=1)
    return - np.log(p.clip(1e-4)).mean()


class MyTest(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.filename, cls.prob_genotypes, cls.barcode2donor_ids, cls.barcode2donor_names = generate_bam_file()

    def test_demultiplex_start_from_genotypes(self):
        """
        Testing quality against different amount of prior informatin about genotypes
        """
        bam_filename, genotypes, barcode2correct_donor = self.filename, self.prob_genotypes, self.barcode2donor_names
        barcode_handler = BarcodeHandler(list(barcode2correct_donor))

        calls = count_snps(
            bam_filename,
            chromosome2positions=genotypes.get_chromosome2positions(),
            barcode_handler=barcode_handler,
        )

        noise_percent2loss = {}
        for noise_percent in [0.0, 0.7, 0.9, 0.95, 0.98, 1.0]:
            ng = deepcopy(genotypes)
            ng.variant_betas[np.random.random(ng.n_variants) < noise_percent, :] = 0
            noised_genotypes = ng
            _logits, barcode2donor_probs = Demultiplexer.predict_posteriors(
                calls, noised_genotypes, barcode_handler=barcode_handler, only_singlets=True)
            loss_no_learning = compute_loss(barcode2correct_donor, barcode2donor_probs)
            result = {'no learning': loss_no_learning}
            for Demultiplexer.use_call_counts in [True]:
                learnt_genotypes, barcode2donor_probs = Demultiplexer.learn_genotypes(
                    calls, noised_genotypes, barcode_handler=barcode_handler)
                loss_learning = compute_loss(barcode2correct_donor, barcode2donor_probs)
                result[f'call_counts={Demultiplexer.use_call_counts}'] = loss_learning

            noise_percent2loss[noise_percent] = result
        print(pd.DataFrame(noise_percent2loss))
        # very rough check
        for label in noise_percent2loss[1.0]:
            assert noise_percent2loss[1.0][label] > noise_percent2loss[0.0][label]

    def test_demultiplex_start_from_assignment(self, labeled_fractions=(0.01, 0.05, 0.1, 0.2, 0.5)):
        """
        In this test we label some of barcodes as attributed to particular donors,
        genotypes of the other barcodes should be guessed starting from this information
        """
        bam_filename, genotypes, barcode2correct_donor = self.filename, self.prob_genotypes, self.barcode2donor_names
        barcode_handler = BarcodeHandler(list(barcode2correct_donor))
        calls = count_snps(
            bam_filename,
            chromosome2positions=genotypes.get_chromosome2positions(),
            barcode_handler=barcode_handler,
        )
        empty_genotypes = genotypes.clone()
        empty_genotypes.variant_betas[:] = 0

        # dry tun to generate pd.DataFrame with correct assignments
        _learnt_genotypes, barcode2donor_probs = Demultiplexer.learn_genotypes(
            calls, empty_genotypes, barcode_handler=barcode_handler)

        labelling_p = np.random.random(size=len(barcode2correct_donor))
        barcode2donor_logits: pd.DataFrame = barcode2donor_probs * 0 + 1

        labeled_fraction2loss = {}
        for labeled_fraction in labeled_fractions:
            for (barcode, correct_donor_names), p_label in zip(barcode2correct_donor.items(), labelling_p):
                if len(correct_donor_names) == 1 and p_label < labeled_fraction:
                    [correct_donor] = correct_donor_names
                    barcode2donor_logits.loc[barcode, str(correct_donor)] += 100.

            _learnt_genotypes, barcode2donor_probs = Demultiplexer.learn_genotypes(
                calls, empty_genotypes, barcode_handler=barcode_handler,
                barcode_prior_logits=barcode2donor_logits.values)

            loss = compute_loss(barcode2correct_donor, barcode2donor_probs)
            print(f'labeled fraction of barcodes: {labeled_fraction:<5}  loss={loss:8.4f}')
            labeled_fraction2loss[labeled_fraction] = loss

        for labeled_fraction, loss in labeled_fraction2loss.items():
            if labeled_fraction > 0.15 and loss > 0.1:
                raise RuntimeError(f'Error is too high {labeled_fraction} {loss}')

    def test_genotypes_export_and_loading(self):
        genotypes: ProbabilisticGenotypes = self.prob_genotypes
        with tempfile.TemporaryDirectory() as dir:
            filename = f'{dir}/genotypes.parquet'
            genotypes.save_betas(filename)
            genotypes2 = ProbabilisticGenotypes(
                genotype_names=genotypes.genotype_names,
                default_prior=genotypes.default_prior,
            )
            genotypes2.add_prior_betas(filename)

            assert genotypes.genotype_names == genotypes2.genotype_names
            assert genotypes.default_prior == genotypes2.default_prior
            assert set(genotypes.snp2snpid) == set(genotypes2.snp2snpid)
            for snp in genotypes.snp2snpid:
                assert np.allclose(
                    genotypes.variant_betas[genotypes.snp2snpid[snp]],
                    genotypes2.variant_betas[genotypes2.snp2snpid[snp]],
                )


