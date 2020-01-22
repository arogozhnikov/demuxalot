import json
import pprint
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pysam
# TODO get rid of orgalorg dependency. and of pipeline dependency
from orgalorg.omics.utils import read_vcf_to_header_and_pandas
from pipeline.Stopwatch import Stopwatch

from demultiplexit.demutiplexit import GenotypesProbComputing
from demultiplexit.snp_counter import count_call_variants_for_chromosome
from demultiplexit.utils import BarcodeHandler

here = Path(__file__).parent

with open(here / 'lane2guessed_donor.json') as f:
    lane2inferred_donor = json.load(f)

samfile = pysam.AlignmentFile(here / 'composed_1_perlane_sorted.bam')
barcodes_all_lanes = pd.read_csv(here / 'composed_barcodes_1.tsv', header=None)[0].values

with Stopwatch('barcodes'):
    barcode_handler = BarcodeHandler(barcodes_all_lanes)
    barcode2possible_donors = {barcode: lane2inferred_donor[barcode.split('_')[1]] for barcode in barcodes_all_lanes
                               if barcode.split('_')[1] in lane2inferred_donor}

with Stopwatch('read GSA vcf'):
    _, all_snps = read_vcf_to_header_and_pandas(here / 'system1_merged_v4.vcf')
    all_donor_names = list(all_snps.columns[9:])
    # TODO filter SNPs


def filter_snps(snps, donor_names):
    strings = snps[donor_names].sum(axis=1)
    mask = strings.map(lambda x: '0' in x and '1' in x)
    mask &= (snps[donor_names] == './.').sum(axis=1) <= 1
    return snps[mask]


with Stopwatch('construct genotypes from GSA'):
    used_donor_names = np.unique(sum(lane2inferred_donor.values(), [])).tolist()
    genotypes_used = GenotypesProbComputing(all_snps, used_donor_names)

with Stopwatch('update genotypes with new SNPs'):
    # extend genotypes with added SNPs
    chrom2snp_positions_and_stats = joblib.load(here / 'chrom2possible0basedpositions_based_on_donors.joblib.pkl')
    _n_added_snps = 0
    for chromosome, (snp_positions, snp_stats) in chrom2snp_positions_and_stats.items():
        for snp_position, snp_stat in zip(snp_positions, snp_stats):
            _, _, alt, ref = np.argsort(snp_stat)
            alt_count, ref_count = snp_stat[alt], snp_stat[ref]
            if (chromosome, snp_position) in genotypes_used.snips:
                continue
            else:
                priors = np.asarray([ref_count, alt_count]) * 1.
                genotypes_used.introduced_snps_chrompos2ref_alt_priors[chromosome, snp_position] = (
                    'ACGT'[ref], 'ACGT'[alt], priors)

                _n_added_snps += 1
    print(f'Added {_n_added_snps} new snps in total')

with Stopwatch('snp counting'):
    # reorder so chromosomes with most reads are in the beginning
    chromosomes = [contig for contig in samfile.get_index_statistics()[:25]]
    chromosomes = list(sorted(chromosomes, key=lambda contig: -1e20 if 'MT' in contig.contig else -contig.mapped))
    chromosomes = [contig.contig for contig in chromosomes]

    with joblib.Parallel(n_jobs=10, verbose=11) as parallel:
        _cbub2qual_and_snps = parallel(
            joblib.delayed(count_call_variants_for_chromosome)(
                samfile.filename.decode(), chromosome, genotypes_used.get_positions_for_chromosome(chromosome),
                cellbarcode_compressor=lambda cb: barcode_handler.barcode2index.get(cb, None),
            )
            for chromosome in chromosomes
        )
    chromosome2cbub2qual_and_snps = dict(zip(chromosomes, _cbub2qual_and_snps))

counter = {chromosome: len(cals) for chromosome, cals in chromosome2cbub2qual_and_snps.items()}

assert counter == {
    'GRCh38_______1': 6264,
    'GRCh38_______10': 1698,
    'GRCh38_______11': 10208,
    'GRCh38_______12': 3207,
    'GRCh38_______13': 782,
    'GRCh38_______14': 1513,
    'GRCh38_______15': 1867,
    'GRCh38_______16': 2572,
    'GRCh38_______17': 4081,
    'GRCh38_______18': 400,
    'GRCh38_______19': 9863,
    'GRCh38_______2': 4216,
    'GRCh38_______20': 2132,
    'GRCh38_______21': 668,
    'GRCh38_______22': 2372,
    'GRCh38_______3': 5006,
    'GRCh38_______4': 1733,
    'GRCh38_______5': 2416,
    'GRCh38_______6': 3175,
    'GRCh38_______7': 3326,
    'GRCh38_______8': 3228,
    'GRCh38_______9': 1693,
    'GRCh38_______MT': 32576,
    'GRCh38_______X': 1365,
    'GRCh38_______Y': 55
}

for chromosome, cals in chromosome2cbub2qual_and_snps.items():
    print(chromosome, len(cals))
