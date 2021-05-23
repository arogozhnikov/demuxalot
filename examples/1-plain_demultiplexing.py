"""
Simple demultiplexing
with known genotypes.
"""
from demuxalot import Demultiplexer, BarcodeHandler, ProbabilisticGenotypes, count_snps

genotypes = ProbabilisticGenotypes(genotype_names=['Donor01', 'Donor02', 'Donor03', 'Donor04'])
genotypes.add_vcf('./example_data/test_genotypes.vcf')

print(f'Loaded genotypes: {genotypes}')

barcode_handler = BarcodeHandler.from_file('./example_data/test_barcodes.csv')
print(f'Loaded barcodes: {barcode_handler}')

snps = count_snps(
    bamfile_location='./example_data/test_bamfile.bam',
    chromosome2positions=genotypes.get_chromosome2positions(),
    barcode_handler=barcode_handler,
)

print('Collected SNPs: ')
for chromosome, snps_in_chromosome in snps.items():
    print(f'Chromosome {chromosome}, {snps_in_chromosome.n_snp_calls} calls in {snps_in_chromosome.n_molecules} mols')

# returns two dataframes with likelihoods and posterior probabilities
log_likelihoods, posterior_probabilities = Demultiplexer.learn_genotypes(
    snps,
    genotypes=genotypes,
    barcode_handler=barcode_handler,
    doublet_prior=0.25,
)

print('Result:')
print(posterior_probabilities.round(3))

