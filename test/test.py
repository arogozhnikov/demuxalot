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

from demultiplexit.demutiplexit import ProbabilisticGenotypes, TrainableDemultiplexer
from demultiplexit.snp_counter import count_snps
from demultiplexit.utils import BarcodeHandler

here = Path(__file__).parent

with open(here / 'lane2guessed_donor.json') as f:
    lane2inferred_donor = json.load(f)

bamfile_location = str(here / 'composed_1_perlane_sorted.bam')
bamfile = pysam.AlignmentFile(bamfile_location)
barcodes_all_lanes = pd.read_csv(here / 'composed_barcodes_1.tsv', header=None)[0].values

with Stopwatch('barcodes'):
    barcode_handler = BarcodeHandler(barcodes_all_lanes)
    barcode2possible_donors = {barcode: lane2inferred_donor[barcode.split('_')[1]] for barcode in barcodes_all_lanes
                               if barcode.split('_')[1] in lane2inferred_donor}


def filter_snps(snps, donor_names):
    strings = snps[donor_names].sum(axis=1)
    mask = strings.map(lambda x: '0' in x and '1' in x)
    # no more than one undetermined SNP
    mask &= (snps[donor_names] == './.').sum(axis=1) <= 1
    return snps[mask]


with Stopwatch('read GSA vcf'):
    _, all_snps = read_vcf_to_header_and_pandas(here / 'system1_merged_v4.vcf')
    all_donor_names = list(all_snps.columns[9:])
    all_snps = filter_snps(all_snps, donor_names=all_donor_names)

with Stopwatch('construct genotypes from GSA'):
    used_donor_names = list(np.unique(sum(lane2inferred_donor.values(), [])).tolist())
    genotypes_used = ProbabilisticGenotypes(used_donor_names)
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
    genotypes_used.add_prior_knowledge(prior_filename)

with Stopwatch('new_snp_counting'):
    # reorder so chromosomes with most reads are in the beginning
    chromosomes = [contig for contig in bamfile.get_index_statistics()[:25]]
    chromosomes = list(sorted(chromosomes, key=lambda contig: -1e20 if 'MT' in contig.contig else -contig.mapped))
    chromosomes = [contig.contig for contig in chromosomes]

    chromosome2cbub2qual_and_snps = count_snps(
        bamfile_location=bamfile_location,
        chromosome2positions={chr: genotypes_used.get_positions_for_chromosome(chr) for chr in chromosomes},
        barcode_handler=barcode_handler,
    )

counter = {
    chromosome: len(cals) for chromosome, cals in chromosome2cbub2qual_and_snps.items()
}
pprint.pprint(counter)

assert counter == {
    'GRCh38_______1': 5074,
    'GRCh38_______10': 1091,
    'GRCh38_______11': 8393,
    'GRCh38_______12': 1760,
    'GRCh38_______13': 618,
    'GRCh38_______14': 1094,
    'GRCh38_______15': 1509,
    'GRCh38_______16': 2247,
    'GRCh38_______17': 3280,
    'GRCh38_______18': 243,
    'GRCh38_______19': 5940,
    'GRCh38_______2': 3163,
    'GRCh38_______20': 1564,
    'GRCh38_______21': 326,
    'GRCh38_______22': 1927,
    'GRCh38_______3': 3488,
    'GRCh38_______4': 1303,
    'GRCh38_______5': 1601,
    'GRCh38_______6': 2817,
    'GRCh38_______7': 2701,
    'GRCh38_______8': 2582,
    'GRCh38_______9': 1251,
    'GRCh38_______MT': 26507,
    'GRCh38_______X': 657,
    'GRCh38_______Y': 0
}

for chromosome, cbub2qual_and_snps in chromosome2cbub2qual_and_snps.items():
    print(chromosome, len(cbub2qual_and_snps))

with Stopwatch(f'initialization'):
    trainable_demultiplexer = TrainableDemultiplexer(
        chromosome2cbub2qual_and_snps,
        barcode2possible_genotypes={barcode: used_donor_names for barcode in barcode_handler.barcode2index},
        probabilistic_genotypes=genotypes_used,
        barcode_handler=barcode_handler,
    )

with Stopwatch('demultiplexing'):
    for barcode_posterior_probs_df, logits, genotype_snp_posterior in trainable_demultiplexer.run_fast_em_iterations(
            n_iterations=3):
        print('one more iteration complete')
        logits2 = trainable_demultiplexer.predict_posteriors(
            genotype_snp_posterior,
            chromosome2cbub2qual_and_snps,
            barcode_handler, only_singlets=True)

        assert np.allclose(logits, logits2)

print(list(np.max(logits, axis=0)))

reference = [-14.673564, -12.325023, -21.798777, -13.038282, -12.271305, -12.648042, -4.566181, -12.732492, -7.146144,
             -12.174306, -11.428376, -12.903507, -9.499123, -14.021539, -6.112081, -4.4732895, -6.620908, -10.586274,
             -4.9894757, -14.94, -7.0786963]

assert np.allclose(np.max(logits, axis=0), reference)
