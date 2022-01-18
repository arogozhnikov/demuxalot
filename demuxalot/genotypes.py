from __future__ import annotations
from warnings import warn
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pysam


def find_duplicates(iterable):
    counts = Counter(list(iterable))
    duplicates = [k for k, v in counts.items() if v != 1]
    return duplicates


class ProbabilisticGenotypes:
    def __init__(self, genotype_names: List[str], default_prior=1.):
        """
        ProbabilisticGenotypes represents our accumulated knowledge about SNPs for selected genotypes.
        Can aggregate information from GSA/WGS/WES, our prior guesses and genotype information learnt from RNAseq.
        Genotype names can't be changed/extended once the object is created.
        Class can handle more than two options per genomic position.
        Genotype information is always accumulated, not overwritten.
        Information is stored as betas (parameters of Dirichlet distribution,
        akin to coefficients in beta distribution).
        """
        self.var2varid: Dict[Tuple, int] = {}  # chrom, pos, base -> index in variant_betas
        self.genotype_names: List[str] = list(genotype_names)
        assert (np.sort(self.genotype_names) == self.genotype_names).all(), 'please order genotype names'
        assert len(set(genotype_names)) == len(genotype_names), f'Duplicates in genotypes: {genotype_names}'
        self.variant_betas: np.ndarray = np.zeros([32768, self.n_genotypes], 'float32')
        # prior will be used in demuxing algorithm
        self.default_prior: float = default_prior

    def __repr__(self):
        chromosomes = {chromosome for chromosome, _, _ in self.var2varid}
        return f'<Genotypes with {self.n_variants} variants on {len(chromosomes)} contigs ("chromosomes") ' \
               f'and {self.n_genotypes} genotypes: \n{self.genotype_names}'

    @property
    def n_genotypes(self):
        return len(self.genotype_names)

    @property
    def n_variants(self) -> int:
        return len(self.var2varid)

    def get_betas(self) -> np.ndarray:
        # return readonly view
        variants_view: np.ndarray = self.variant_betas[: self.n_variants]
        variants_view.flags.writeable = False
        return variants_view

    def get_snp_ids_for_variants(self) -> np.ndarray:
        snp2id = {}
        result = np.zeros(self.n_variants, dtype='int32') - 1
        for (chrom, pos, _base), variant_id in self.var2varid.items():
            snp = chrom, pos
            if snp not in snp2id:
                snp2id[snp] = len(snp2id)
            result[variant_id] = snp2id[snp]
        assert np.all(result >= 0)
        assert np.all(result < self.n_variants)
        return result

    def get_variant_id(self, chrom, pos, base):
        variant = chrom, pos, base
        if variant not in self.var2varid:
            self.var2varid[variant] = self.n_variants
            self.extend_variants(1)
        return self.var2varid[variant]

    def extend_variants(self, n_samples=1):
        # pre-allocation of space for new variants
        while n_samples + self.n_variants > len(self.variant_betas):
            self.variant_betas = np.concatenate([self.variant_betas, np.zeros_like(self.variant_betas)], axis=0)

    def _check_imported_genotypes(
            self,
            imported_genotypes: List[str],
            allow_duplicates=False,
    ) -> Dict[str, int]:
        duplicate_genotypes = find_duplicates(imported_genotypes)
        if duplicate_genotypes:
            if allow_duplicates:
                warn(f"Duplicate genotypes found will be imported: {duplicate_genotypes}")
            else:
                raise RuntimeError(f'Duplicate genotypes found in imported data: {duplicate_genotypes}')

        imported_genotypes = set(imported_genotypes)
        existing_genotypes = set(self.genotype_names)

        common_genotypes = set.intersection(imported_genotypes, existing_genotypes)
        if not common_genotypes:
            raise RuntimeError(f'No genotypes to import, expected {existing_genotypes}, got {imported_genotypes}')

        new_genotypes = set.difference(imported_genotypes, existing_genotypes)
        if new_genotypes:
            warn(f'Genotypes will not be imported: {new_genotypes}')

        missed_genotypes = set.difference(existing_genotypes, imported_genotypes)
        if missed_genotypes:
            print(f'Some of genotypes are not provided during import: {missed_genotypes}')

        common_genotypes = list(common_genotypes)
        return {
            g: self.genotype_names.index(g) for g in common_genotypes
        }

    def add_vcf(self, vcf_file_name, prior_strength: float = 100.):
        """
        Add information from parsed VCF
        :param vcf_file_name: path to VCF file. Only diploid values are accepted (0/0, 0/1, 1/1, ./.).
            Should contain all genotypes of interest. Can contain additional genotypes, but those will be ignored.
        """
        n_skipped_snps = 0
        donor2donor_id = None

        n_snps_in_file = 0
        n_original_variants = self.n_variants
        for snp in pysam.VariantFile(vcf_file_name).fetch():
            n_snps_in_file += 1
            if any(len(option) != 1 for option in snp.alleles):
                print('skipping non-snp, alleles = ', snp.alleles, snp.chrom, snp.pos)
                continue

            if donor2donor_id is None:
                donor2donor_id = self._check_imported_genotypes(imported_genotypes=list(snp.samples))

            snp_ids = []
            alleles = snp.alleles
            if len(set(alleles)) != len(alleles):
                n_skipped_snps += 1
                continue
            if any(allele not in 'ACGT' for allele in alleles):
                n_skipped_snps += 1
                continue

            for allele in alleles:
                # pysam enumerates starting from one, thus -1
                snp_ids.append(self.get_variant_id(snp.chrom, snp.pos - 1, allele))

            assert len(set(snp_ids)) == len(snp_ids), (snp_ids, snp.chrom, snp.pos, snp.alleles)

            contribution = np.zeros([len(snp_ids), self.n_genotypes], dtype='float32')

            for donor, donor_id in donor2donor_id.items():
                called_values = snp.samples[donor]['GT']
                for call in called_values:
                    if call is not None:
                        # contribution is split between called values
                        contribution[call, donor_id] += prior_strength / len(called_values)
            not_provided = contribution.sum(axis=0) == 0
            if np.sum(~not_provided) < 2:
                # at least two genotypes should have SNP
                n_skipped_snps += 1
                continue

            confidence_for_skipped = 0.1
            contribution[:, not_provided] = contribution[:, ~not_provided].mean(axis=1, keepdims=True) \
                                            * confidence_for_skipped
            self.variant_betas[snp_ids] += contribution

        if n_skipped_snps > 0:
            print('skipped', n_skipped_snps, 'SNVs')
        print(f'Parsed {n_snps_in_file} SNPs, got {self.n_variants - n_original_variants} novel variants')

    def add_assignment_dataframe(self, assignment: pd.DataFrame, *, prior_stength: float = 100.):
        """
        Specific format to add just assignments. Format is dataframe
            columns : donor
            index: 'CHROM', 'POS1BASED', 'REF', 'ALT'
            values: './.', '0/0', '0/1', '1/1' and Nones
        """
        assignment = assignment.fillna('./.')
        assignment.index = pd.MultiIndex.from_frame(
            assignment.index.to_frame().loc[:, ['CHROM', 'POS1BASED', 'REF', 'ALT']]
        )
        genotype2genotype_id = self._check_imported_genotypes(list(assignment.columns))
        assignment = assignment.loc[:, list(genotype2genotype_id)]

        n_variants_before = self.n_variants

        for (chrom, pos_1based, ref, alt), row in assignment.iterrows():
            ref_snp_id = self.get_variant_id(chrom, pos_1based - 1, ref)
            alt_snp_id = self.get_variant_id(chrom, pos_1based - 1, alt)

            betas = self.variant_betas

            for genotype, value in row.items():
                genotype_id = genotype2genotype_id[genotype]
                if value == '0/0':
                    betas[ref_snp_id, genotype_id] += prior_stength
                elif value == '0/1':
                    betas[ref_snp_id, genotype_id] += prior_stength * 0.5
                    betas[alt_snp_id, genotype_id] += prior_stength * 0.5
                elif value == '1/1':
                    betas[alt_snp_id, genotype_id] += prior_stength
                else:
                    assert value == './.' or value is None, \
                        f"Unknown value: {value} of type {type(value)} at {chrom} {pos_1based} {ref} {alt}"

        print(f'Parsed {len(assignment) * 2} variants, of them  {self.n_variants - n_variants_before} are novel')

    def add_raw_values_from_bead_array(
            self,
            logrratio: pd.DataFrame,
            b_allele_freq: pd.DataFrame,
            *,
            prior_strength=100.,
    ):
        """
        Allows importing raw values of bead arrays.
        Illumina provides two values for each SNP:
        - allele ratio (something like a general configuration)
        - logrratio (relative fluorescence, should reflect total amount of material)
        These two should be provided as separate dataframes with the same structure:
        columns are donor names, index should contain ['chromosome', 'position1based', 'alleleA', 'alleleB']
        You'll still need to align those.
        """
        assert np.array_equal(logrratio.columns, b_allele_freq.columns)
        assert np.array_equal(logrratio.index, b_allele_freq.index)
        import_genotypes = list(logrratio.columns)
        self._check_imported_genotypes(import_genotypes, allow_duplicates=True)

        index = logrratio.index.to_frame()[['chromosome', 'position1based', 'alleleA', 'alleleB']]
        index = index.reset_index(drop=True)
        index['chromosome'] = index['chromosome'].map(str)
        index['position0based'] = index['position1based'] - 1

        n_variants_before = self.n_variants
        variants = [
            (self.get_variant_id(chr, pos0, allA), self.get_variant_id(chr, pos0, allB))
            for _, (chr, pos0, allA, allB) in
            index[['chromosome', 'position0based', 'alleleA', 'alleleB']].iterrows()
        ]
        variantsA, variantsB = np.asarray(variants).T

        logrratio = logrratio.values.clip(-10, 0)
        b_allele_freq = b_allele_freq.values

        undefined = ~(np.isfinite(logrratio) & np.isfinite(b_allele_freq))

        logrratio[undefined] = -20
        b_allele_freq[undefined] = 0

        assert np.isfinite(logrratio).all()
        assert np.isfinite(b_allele_freq).all()

        assert np.min(b_allele_freq) >= 0.
        assert np.max(b_allele_freq) <= 1.
        assert np.max(logrratio) == 0.

        for genotype, b_allele_freq_col, logrratio_col in zip(import_genotypes, b_allele_freq.T, logrratio.T):
            if genotype not in self.genotype_names:
                continue
            genotype_id = self.genotype_names.index(genotype)
            contribution = prior_strength * 2 ** logrratio_col

            self.variant_betas[variantsA, genotype_id] += (1 - b_allele_freq_col) * contribution
            self.variant_betas[variantsB, genotype_id] += (0 + b_allele_freq_col) * contribution

        print(f'Parsed {len(logrratio) * 2} variants, of them  {self.n_variants - n_variants_before} are novel')

    def add_prior_betas(self, prior_filename, *, prior_strength: float = 1.):
        """
        Betas is the way Demultiplexer stores learnt genotypes. It is a parquet file with posterior weights.
        Parquet are convenient replacement for csv/tsv files, find examples in pandas documentation.
        :param prior_filename: path to file
        :param prior_strength: float, controls impact of learnt genotypes.
        """

        prior_knowledge: pd.DataFrame = pd.read_parquet(prior_filename) * prior_strength
        print('Provided prior information about genotypes:', [*prior_knowledge.columns])
        genotypes_not_provided = [
            genotype for genotype in self.genotype_names if genotype not in prior_knowledge.columns
        ]
        if len(genotypes_not_provided) > 0:
            print(f'No information for genotypes: {genotypes_not_provided}')

        variants = prior_knowledge.index.to_frame()
        variants = zip(variants['CHROM'], variants['POS'], variants['BASE'])

        variant_indices: List[int] = []
        for variant in variants:
            if variant not in self.var2varid:
                self.extend_variants(1)
                self.var2varid[variant] = self.n_variants
            variant_indices.append(self.var2varid[variant])

        for donor_id, donor in enumerate(self.genotype_names):
            if donor in prior_knowledge.columns:
                np.add.at(
                    self.variant_betas[:, donor_id],
                    variant_indices,
                    prior_knowledge[donor],
                )

    def get_chromosome2positions(self):
        chromosome2positions = defaultdict(list)
        for chromosome, position, base in self.var2varid:
            chromosome2positions[chromosome].append(position)

        if len(chromosome2positions) == 0:
            warn('Genotypes are empty. Did you forget to add vcf/betas?')

        return {
            chromosome: np.unique(np.asarray(positions, dtype=int))
            for chromosome, positions
            in chromosome2positions.items()
        }

    def _generate_canonical_representation(self):
        variants_order = []
        sorted_variant2id = {}
        for (chrom, pos, base), variant_id in sorted(self.var2varid.items()):
            variants_order.append(variant_id)
            sorted_variant2id[chrom, pos, base] = len(variants_order)

        return sorted_variant2id, self.variant_betas[variants_order]

    def get_snp_positions_set(self) -> set:
        return {(chromosome, position) for chromosome, position, base in self.var2varid}

    def _with_betas(self, external_betas: np.ndarray) -> ProbabilisticGenotypes:
        """ Return copy of genotypes with updated beta weights """
        assert external_betas.shape == (self.n_variants, self.n_genotypes)
        assert external_betas.dtype == self.variant_betas.dtype
        assert np.min(external_betas) >= 0
        result: ProbabilisticGenotypes = self.clone()
        result.variant_betas = external_betas.copy()
        return result

    def as_pandas_dataframe(self) -> pd.DataFrame:
        index_columns = defaultdict(list)
        old_variant_order = []

        for (chrom, pos, base), variant_id in sorted(self.var2varid.items()):
            index_columns['CHROM'].append(chrom)
            index_columns['POS'].append(pos)
            index_columns['BASE'].append(base)

            old_variant_order.append(variant_id)

        old_variant_order = np.asarray(old_variant_order)
        betas = self.variant_betas[:self.n_variants][old_variant_order]

        return pd.DataFrame(
            data=betas,
            index=pd.MultiIndex.from_frame(pd.DataFrame(index_columns)),
            columns=self.genotype_names,
        )

    def save_betas(self, path_or_buf):
        """ Save learnt genotypes in the form of beta contributions, can be used later. """
        self.as_pandas_dataframe().to_parquet(path_or_buf)

    def clone(self):
        return deepcopy(self)
