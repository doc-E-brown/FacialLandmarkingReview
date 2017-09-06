#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Constrained Local Model



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from menpoAAM import MenpoAAM 
from menpofit.clm import CLM, GradientDescentCLMFitter


class MenpoCLM(MenpoAAM):

    def __init__(self, path_to_data, filename='menpo_clm.txt', verbose=True, profile=False):
        """ init """

        super(MenpoCLM, self).__init__(path_to_data, filename=filename, verbose=verbose, profile=profile)
        self.model_type = CLM
        self.model_fitter = GradientDescentCLMFitter

    def train_model(self, batch_size=None, scales=(0.5, 1)):
        """ train model """

        self.model = self.model_type(
            self.train_set,
            scales=scales,
            patch_shape=(10, 10),
            batch_size=batch_size,
            verbose=self.verbose)

        if self.verbose:
            print(self.model)

if __name__ == "__main__":

    # batch size 256
    a = MenpoCLM('/home/bjoh3944/predPap-ben/datasets/menpo_2017_trainset', filename='menpo_clm.txt', profile=False)
    a.load_data(max_images=200)
    a.train_model(batch_size=128)
    a.fit_model()
    a.predict_test_set()
    a.generate_cdf()
