#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Test Section 4



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import unittest
import numpy as np
from experiments.Sec4_ModelDefinition.asm._base import Base
import matplotlib.pyplot as plt


class TestBase(unittest.TestCase):

    def test_compute_normals(self):
        """Test computation of normals"""

        base = Base(None)

        x = np.array([
            [0, 0],
            [3, 0],
            [3, 4],
            ])

        normals = base._contour_normals(x)

        plt.scatter(x[:,0], x[:,1], c='r')

        for i in range(x.shape[0] - 1):
            plt.plot(
                [x[i, 0], x[i+1, 0]],
                [x[i, 1], x[i+1, 1]],
                c='b')

        print(normals)
        for norm, pt in zip(normals, x):
            plt.plot([pt[0], norm[0]], [pt[1], norm[1]], c='c')

        plt.show()
