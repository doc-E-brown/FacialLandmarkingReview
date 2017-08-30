#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Constrained Local Model



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from helenAAM import HelenAAM 
from menpofit.clm import CLM, GradientDescentCLMFitter


class CLMModel(HelenAAM):

    def __init__(self, path_to_data, filename='helen_clm.txt', verbose=True):
        """ init """

        super(CLMModel, self).__init__(path_to_data, filename=filename, verbose=verbose)
        self.model_type = CLM
        self.model_fitter = GradientDescentCLMFitter

    def train_model(self, scales=(0.5, 1)):
        """ train model """

        self.model = self.model_type(
            self.train_set,
            scales=scales,
            batch_size=50,
            verbose=self.verbose)

        if self.verbose:
            print(self.model)

if __name__ == "__main__":

    a = HelenAAM('~/datasets/HELEN')
    a.load_data()
    a.train_model()
    a.fit_model()
    a.predict_test_set()
    a.generate_cdf()