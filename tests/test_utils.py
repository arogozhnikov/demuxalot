"""
Testing some fragments of the pipeline
"""

import unittest

import numpy as np

from demuxalot import Demultiplexer
from demuxalot.utils import FeatureLookup
from scipy.special import softmax


def test_feature_lookup(n_combinations=1000, n_samples=100000):
    feat1_comb = np.random.randint(0, 100, n_combinations)
    feat2_comb = np.random.randint(0, 1000, n_combinations)
    feat3_comb = np.random.randint(0, 100_000, n_combinations)

    samples_id = np.random.randint(0, n_combinations, n_samples)
    feat1 = feat1_comb[samples_id]
    feat2 = feat2_comb[samples_id]
    feat3 = feat3_comb[samples_id]

    lookup = FeatureLookup(feat1, feat2, feat3)
    compressed, counts = lookup.compress(feat1, feat2, feat3)
    assert compressed.max() < n_combinations
    assert np.allclose(counts, np.bincount(compressed, minlength=len(counts)))
    feat1_new, feat2_new, feat3_new = lookup.uncompress(compressed)
    assert np.allclose(feat1, feat1_new)
    assert np.allclose(feat2, feat2_new)
    assert np.allclose(feat3, feat3_new)


def test_doublet_penalties():
    for n_genotypes in [2, 3, 10]:
        for doublet_prob in [0., 0.25, 0.5]:
            doublet_logits = Demultiplexer._doublet_penalties(n_genotypes=n_genotypes, doublet_prior=doublet_prob)
            prior_probs = softmax(doublet_logits)

            assert np.allclose(prior_probs[:n_genotypes].sum(), (1 - doublet_prob))