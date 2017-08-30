#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Module to plot the 
point-to-point normalised error curves
for facial landmarking predictions 



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt


class PlotResults(object):

    def __init__(self, result_files=[], labels=[], title=""):

        self.result_files = result_files
        self.labels = labels
        self.title = title

    def plot(self, savename=None, dpi=300):#, figsize=():
        """ Plot the results """

        for result, label in zip(self.result_files, self.labels):

            data = np.loadtxt(result)

            # Normalise the proportion values
            data[1] /= np.max(data[1])

            plt.plot(data[0], data[1], label=label)

        plt.legend(loc='lower right')
        plt.grid()
        plt.title(self.title)
        plt.xlabel('Point-to-point error normalised by intra-ocular distance')
        plt.ylabel('Proportion of images')

        if savename is None:
            plt.show()
        else:
            plt.savefig(savename, dpi=dpi, bbox_inches='tight')

if __name__ == "__main__":
    p = PlotResults(
        ['300W_aam.txt', '300W_patch.txt', '300W_clm.txt'],
        ['Holistic AAM', 'Patch AAM', 'CLM'],
        title='300W dataset',
        )
    p.plot('300W.png')
