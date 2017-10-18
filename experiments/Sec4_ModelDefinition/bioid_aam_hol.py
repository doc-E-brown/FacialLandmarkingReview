#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
BioID AAM holistic



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import aam
import os
from aam import AAM
from functools import partial

BIOID_DATA_FOLDER = os.getenv('BIOID_DATA', '~/datasets/BioID')

# Change the compute errors function to use BioID eye coords
aam.compute_errors = partial(aam.compute_errors, pt1=9, pt2=12)
model = AAM(BIOID_DATA_FOLDER, basename='bioid_aam_hol')
model.load_data()
model.train_model(batch_size=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
