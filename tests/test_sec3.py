#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import unittest
from experiments.Sec3_FeatureExtraction._base import compute_overlap


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Friday 7 July  15:11:59 AEST 2017'
__license__ = 'CC-BY-4.0'


class TestIBUG(unittest.TestCase):

    def test_square_overlap(self):
        """Test for 25% overlap over corners"""
        overlap = compute_overlap(
            [0, 0, 10, 10], [5, 5, 10, 10])
        self.assertEqual(overlap, 0.25)

    def test_no_overlap(self):
        """Test for no overlap"""
        overlap = compute_overlap(
            [0, 0, 5, 5], [5, 5, 10, 10])
        self.assertEqual(overlap, 0)

    def test_half_overlap(self):
        """Test for half overlap over edge"""
        overlap = compute_overlap(
            [0, 0, 10, 10], [5, 0, 10, 10])
        self.assertEqual(overlap, 0.5)

    def test_half_overlap_large(self):
        """Test for half overlap - completely cover edge"""
        overlap = compute_overlap(
            [5, 0, 10, 10], [0, 1, 10, 5])
        self.assertEqual(overlap, 0.5)
