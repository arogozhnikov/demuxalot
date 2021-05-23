<p align="center">
<img width="500" alt="demuxalot_logo_small" src="https://user-images.githubusercontent.com/6318811/118947887-a261da00-b90c-11eb-8932-a66e6d2caa1f.png">
</p>
 
[![Run tests and deploy](https://github.com/herophilus/demuxalot/actions/workflows/run_test.yml/badge.svg)](https://github.com/herophilus/demuxalot/actions/workflows/run_test.yml)
# Demuxalot 

Reliable and efficient idenfitication of genotypes for individual cells 
in RNA sequencing that refines the knowledge about genotypes from the data.

Demuxalot is fast and optimized to work with lots of genotypes.

## Background

During single-cell RNA-sequencing (scRnaSeq) we pool cells from different donors and process them together.

- Pro: all cells come through the same pipeline, so preparation/biological variation effects are cancelled out from analysis automatically. 
  Also experiments are much cheaper!
- Con: we don't know cell origin

Demultiplexing step that demuxalot completes solves the con: 
it guesses genotype of each cell by matching reads coming from cell against genotypes.

Herophilus uses scRnaSeq to study cells in organoids with multiple genetic backgrounds at scale.

## Known genotypes and refined genotypes: the tale of two scenarios

Typical approach to get genotype-specific mutations are 
 
- whole-genome sequencing (expensive, very good)
  - you have information about all (ok, almost all) the genotype, and it is unlikely that you need to refine it
  - so you just go straight to demultiplexing
  - demuxlet solves this case
- Bead arrays (aka SNP arrays aka DNA microarrays) are super cheap and practically more relevant
  - you get information about 50k to 650k most common SNPs, and that's only a small fraction, but you also pay very little
  - this case is covered by `demuxalot` (this package)
  - [Illumina's video](https://www.youtube.com/watch?v=lVG04dAAyvY) about this technology

## Why is it worth refining genotypes? 
   
SNP array provides up to ~650k (as of 2021) positions in the genome.
Around 20-30% of them would be specific for a genotype (i.e. deviate from majority).

- Each genotype has around 10 times more SNV (single nucleotide variations) 
  that are not captured by array. Some of this missing SNPs are very valuable for demultiplexing

## What's special power of demuxalot?

- much better handling of multiple reads coming from the same UMI (i.e. same transcript)
  - `demuxalot` efficiently combines information from multiple reads with same UMI and cross-checks it
- default settings are CellRanger-specific (that is - optimized for 10X pipeline). Cellranger's and STAR's flags in BAM break some common conventions, 
  but we can still efficiently use them (by using filtering callbacks)  
- ability to refine genotypes. without failing and diverging
  - Vireo is a tool that was created with similar purposes. But it either diverges or does not learn better genotypes
- optimized variant calling. It's also faster than `demuxlet` due to multiprocessing
- this is not a command-line tool, and not meant to be 
  - write python code, this gives full control and flexibility of demultiplexing

## Installation

Package is pip-installable. Requires python >= 3.6

```bash
git clone <repo> 
pip install ./demuxalot
```

## Running (simple scenario)
Only using provided genotypes

```python
from demuxalot import Demultiplexer, BarcodeHandler, ProbabilisticGenotypes, count_snps

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
from demuxalot import Demultiplexer, BarcodeHandler, ProbabilisticGenotypes, count_snps

# Loading genotypes
genotypes = ProbabilisticGenotypes(genotype_names=['Donor1', 'Donor2', 'Donor3'])
genotypes.add_vcf('path/to/genotypes.vcf')

# TODO add SNPs detection


# Load barcodes
barcode_handler = BarcodeHandler.from_file('path/to/barcodes.csv')

snps = count_snps(
    bamfile_location='path/to/sorted_alignments.bam',
    chromosome2positions=genotypes.get_chromosome2positions(),
    barcode_handler=barcode_handler, 
)

# Infer refined genotypes 
refined_genotypes, _posterior_probabilities = \
    Demultiplexer.learn_genotypes(snps, genotypes=genotypes, n_iterations=5)

# Use learnt genotypes for demultiplexing
likelihoods, posterior_probabilities = Demultiplexer.predict_posteriors(
    snps,
    genotypes=refined_genotypes,
    barcode_handler=barcode_handler,
    only_singlets=False,
)
```

## Saving/loading genotypes
   
```python
# You can always export learnt genotypes to be used later
refined_genotypes.save_betas('learnt_genotypes.csv')
refined_genotypes = ProbabilisticGenotypes(genotype_names= <list which genotypes to load>)
refined_genotypes.add_prior_betas('learnt_genotypes.csv')
```

## Re-saving VCF genotypes with betas

Generally makes sense to export VCF to internal format only when you plan to load it many times.
Loading of internal format is much faster 

```python
genotypes = ProbabilisticGenotypes(genotype_names=['Donor1', 'Donor2', 'Donor3'])
genotypes.add_vcf('path/to/genotypes.vcf')
genotypes.save_betas('learnt_genotypes.csv')
```
