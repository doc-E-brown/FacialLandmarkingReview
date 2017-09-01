#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""CLM test for MUCT


:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from muctAAM import MuctAAM
from menpofit.clm import CLM, GradientDescentCLMFitter

class MuctCLM(MuctAAM):

    def __init__(self, path_to_data, filename='muct_clm.txt', verbose=True):
        """ init """

        super(MuctCLM, self).__init__(path_to_data, filename=filename, verbose=verbose)
        self.model_type = CLM
        self.model_fitter = GradientDescentCLMFitter
        
    def train_model(self,batch_size=50, scales=(0.5, 1)):
        """ train model """

        self.model = self.model_type(
            self.train_set,
            scales=scales,
            batch_size=batch_size,
            verbose=self.verbose)

        if self.verbose:
            print(self.model)

if __name__ == "__main__":

    a = MuctCLM('~/datasets/muct/muct-images/')
    a.load_data()
    a.train_model(batch_size=256)
    a.fit_model()
    a.predict_test_set()
    a.generate_cdf()
