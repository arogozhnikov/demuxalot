import json
import pprint
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pysam
# TODO get rid of orgalorg dependency. and of pipeline dependency
from orgalorg.omics.utils import read_vcf_to_header_and_pandas
from pipeline.Stopwatch import Stopwatch

from scrnaseq_demux import ProbabilisticGenotypes, Demultiplexer, count_snps, BarcodeHandler

here = Path(__file__).parent

with open(here / 'lane2guessed_donor.json') as f:
    lane2inferred_genotypes = json.load(f)

bamfile_location = str(here / 'composed_1_perlane_sorted.bam')
bamfile = pysam.AlignmentFile(bamfile_location)
barcodes_all_lanes = pd.read_csv(here / 'composed_barcodes_1.tsv', header=None)[0].values

with Stopwatch('barcodes'):
    barcode_handler = BarcodeHandler(barcodes_all_lanes)
    barcode2possible_genotypes = {barcode: lane2inferred_genotypes[barcode.split('_')[1]]
                                  for barcode in barcodes_all_lanes
                                  if barcode.split('_')[1] in lane2inferred_genotypes}


def filter_snps(snps, genotype_names):
    strings = snps[genotype_names].sum(axis=1)
    mask = strings.map(lambda x: '0' in x and '1' in x)
    # no more than one undetermined SNP
    mask &= (snps[genotype_names] == './.').sum(axis=1) <= 1
    return snps[mask]


with Stopwatch('read GSA vcf'):
    _, all_snps = read_vcf_to_header_and_pandas(here / 'system1_merged_v4_mini.vcf')
    all_genotypes_names = list(all_snps.columns[9:])
    all_snps = filter_snps(all_snps, genotype_names=all_genotypes_names)

with Stopwatch('construct genotypes from GSA'):
    used_genotypes_names = list(np.unique(sum(lane2inferred_genotypes.values(), [])).tolist())
    genotypes_used = ProbabilisticGenotypes(used_genotypes_names)
    genotypes_used.add_vcf(all_snps, prior_strength=100)

with Stopwatch('update genotypes with new SNPs'):
    # extend genotypes with added SNPs
    # saving learnt genotypes to a separate file
    chrom2snp_positions_and_stats = joblib.load(here / 'chrom2possible0basedpositions_based_on_donors.joblib.pkl')
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
    genotypes_used.add_prior_betas(prior_filename, prior_strength=10)

with Stopwatch('check export'):
    posterior_filename = here / '_temp_exported_prior.tsv'
    genotypes_used.save_betas(posterior_filename)

with Stopwatch('check import'):
    genotypes_used2 = ProbabilisticGenotypes(used_genotypes_names)
    genotypes_used2.add_prior_betas(posterior_filename, prior_strength=1.)

with Stopwatch('verifying agreement'):
    assert len(genotypes_used.snips) == len(genotypes_used2.snips)
    assert genotypes_used.genotype_names == genotypes_used2.genotype_names
    for (chrom, pos), (ref, alt, priors) in genotypes_used.snips.items():
        ref2, alt2, priors2 = genotypes_used2.snips[chrom, pos]
        assert alt == alt2
        assert ref == ref2
        assert np.allclose(priors, priors2)

    snp2sindex1, _, _beta_priors1 = genotypes_used.generate_genotype_snp_beta_prior()
    snp2sindex2, _, _beta_priors2 = genotypes_used2.generate_genotype_snp_beta_prior()
    assert snp2sindex1 == snp2sindex2
    assert np.allclose(_beta_priors1, _beta_priors2)

with Stopwatch('new_snp_counting'):
    chromosome2cbub2qual_and_snps = count_snps(
        bamfile_location=bamfile_location,
        chromosome2positions=genotypes_used.get_chromosome2positions(),
        barcode_handler=barcode_handler,
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

with Stopwatch(f'demux initialization'):
    trainable_demultiplexer = Demultiplexer(
        chromosome2cbub2qual_and_snps,
        barcode2possible_genotypes={barcode: used_genotypes_names for barcode in barcode_handler.barcode2index},
        probabilistic_genotypes=genotypes_used,
        barcode_handler=barcode_handler,
    )

with Stopwatch('demultiplexing'):
    for barcode_posterior_probs_df, debug_info in trainable_demultiplexer.staged_genotype_learning(n_iterations=3):
        print('one more iteration complete')
        logits2 = trainable_demultiplexer.predict_posteriors(
            debug_info['genotype_snp_posterior'],
            chromosome2cbub2qual_and_snps,
            barcode_handler, only_singlets=True)
        logits = debug_info['barcode_logits']
        assert np.allclose(logits, logits2)

print(list(np.max(logits, axis=0)))

reference = [-4.1255603, -4.2695103, -9.513682, -3.824123, -4.157081, -4.3482676, -3.9152608, -8.900769, -4.4701805,
             -3.8421426, -4.585875, -4.1683974, -3.307035, -4.4903097, -4.402192, -3.8523905, -3.9055922, -3.9764569,
             -3.7411292, -3.964514, -3.770989]

assert np.allclose(np.max(logits, axis=0), reference)

with Stopwatch('demux initialization again'):
    trainable_demultiplexer2 = Demultiplexer(
        chromosome2cbub2qual_and_snps,
        barcode2possible_genotypes={barcode: used_genotypes_names for barcode in barcode_handler.barcode2index},
        barcode_handler=barcode_handler,
        probabilistic_genotypes=genotypes_used2,
    )

with Stopwatch('demultiplexing again and exporting difference'):
    learnt_genotypes_filename = here / '_learnt_beta_contributions.csv'
    for _, debug_info2 in trainable_demultiplexer2.staged_genotype_learning(
            n_iterations=3, save_learnt_genotypes_to=str(learnt_genotypes_filename)):
        print('one more iteration complete')

assert np.allclose(debug_info['genotype_snp_posterior'], debug_info2['genotype_snp_posterior'])

with Stopwatch('importing difference'):
    genotypes_learnt = ProbabilisticGenotypes(used_genotypes_names)
    genotypes_learnt.add_prior_betas(learnt_genotypes_filename, prior_strength=1.)
    assert genotypes_learnt.generate_genotype_snp_beta_prior()[:2] == genotypes_used.generate_genotype_snp_beta_prior()[:2]
    _, _, _beta_prior = genotypes_learnt.generate_genotype_snp_beta_prior()

    assert np.allclose(_beta_prior, debug_info['genotype_snp_posterior'])

    logits3 = trainable_demultiplexer2.predict_posteriors(
        _beta_prior,
        chromosome2cbub2qual_and_snps,
        barcode_handler=barcode_handler, only_singlets=True)

    assert np.allclose(logits2, logits3)

print('Yup, all is fine')
