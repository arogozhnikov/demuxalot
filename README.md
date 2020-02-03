# scrnaseq_demux

Reliable demultiplexing for single-cell RNA sequencing that improves genotypes.

## Background

During single-cell RNA-sequencing (scRnaSeq) we pool cells from different donors and process them together.

- Con: we don't know which cell comes from which organoid
- Pro: all cells come through the same pipeline, so there no organoid-specific batch effects

Demultiplexing solves con: it guesses which cells come from which organoid by matching reads coming from cell against genotypes.

## Known genotypes and refined genotypes

Typical approach to get genotype-specific mutations are 
 
- whole-genome sequencing (expensive, very good)
  - you have information about all (ok, almost all) the genotype, and it is unlikely that you need to refine it
  - so you just go straight to demultiplexing
  - e.g. demuxlet solves this problem
- [GSA](https://www.well.ox.ac.uk/ogc/wp-content/uploads/2017/06/GSA-inputation-design-information.pdf) (Global Screening Array, times cheaper, fits this purpose).
  - [video promo of GSA](https://www.youtube.com/watch?v=lVG04dAAyvY) by Illumina 
  - you get information about 50k to 650k most common SNVs, and that's only a fraction of useful information you could have
  - this case is covered by `scrnaseq_demux` (this package)

## Why is it worth refining genotypes? 
   
GSA provides you up to 650k positions in the genome.
Around 20-30% of them would be specific for a genotype.

- Each genotype has around 10 times more SNV (single nucleotide variations)
- However most of them will not contribute to demultiplexing
  - because scRnaSeq consists of reads that are <700 basepairs apart from poly-A tail  
  - this also implies that we only need to learn genotypic information about these positions 

## What's different between this package and others?

- much better handling of multiple reads coming from the same UMI
  - `scrnaseq_demux` can efficiently combine information
  - this comes at the cost of higher memory consumption
- default settings are CellRanger-specific. Cellranger's and STAR's flags in BAM break common conventions, 
  - but you can still efficiently use those, you just need to provide alternative filtering  
- ability to refine genotypes. without failing 
  - Vireo is a tool that was created with similar purposes. But it either diverges or does not learn better genotypes
- optimized variant calling (compared to CellSNP). It also wins demuxlet due to multiprocessing
- this is not a command-line tool. 
  - You will write python code, this gives you full control and flexibility of demultiplexing.

## Installation

Package is pip-installable. 

```bash
git clone <repo> 
pip install scrnaseq-demux
```

## Running (simple scenario)
Only using provided genotypes

```python
from scrnaseq_demux import Demultiplexer, BarcodeHandler, ProbabilisticGenotypes, count_snps

# Loading genotypes
genotypes = ProbabilisticGenotypes(genotype_names=['Donor1', 'Donor2', 'Donor3'])
genotypes.add_vcf('path/to/genotypes.vcf')

# Loading barcodes
barcode_handler = BarcodeHandler.from_file('path/to/barcodes.csv')

snps = count_snps(
    bamfile_location='path/to/sorted_alignments.bam',
    chromosome2positions=genotypes.get_chromosome2positions(),
    barcode_handler=barcode_handler, 
)

# returns two dataframes with likelihoods and posterior probabilities 
likelihoods, posterior_probabilities = Demultiplexer.predict_posteriors(
    snps,
    genotypes=genotypes,
    barcode_handler=barcode_handler,
    only_singlets=False
)
```


## Running (complex scenario)
Refinement of known genotypes

```python
from scrnaseq_demux import Demultiplexer, BarcodeHandler, ProbabilisticGenotypes, count_snps

# Loading genotypes
genotypes = ProbabilisticGenotypes(genotype_names=['Donor1', 'Donor2', 'Donor3'])
genotypes.add_vcf('path/to/genotypes.vcf')

# Loading barcodes
barcode_handler = BarcodeHandler.from_file('path/to/barcodes.csv')

snps = count_snps(
    bamfile_location='path/to/sorted_alignments.bam',
    chromosome2positions=genotypes.get_chromosome2positions(),
    barcode_handler=barcode_handler, 
)

# Train genotypes and export 
for posterior_probs, additional_info in Demultiplexer.staged_genotype_learning(
    snps,
    genotypes=genotypes,
    barcode_handler=barcode_handler,
    n_iterations=5, 
    save_learnt_genotypes_to='path/to/leant_genotypes_betas.csv',
):
    # here you can track how probabilities change during training
    print(posterior_probs.shape)

# import learnt genotypes and use those for demultiplexing
genotypes_refined = ProbabilisticGenotypes(genotype_names=['Donor1', 'Donor2', 'Donor3'])
genotypes_refined.add_prior_betas('path/to/leant_genotypes_betas.csv')

likelihoods, posterior_probabilities = Demultiplexer.predict_posteriors(
    snps,
    genotypes=genotypes_refined,
    barcode_handler=barcode_handler,
    only_singlets=False,
)
```


   
