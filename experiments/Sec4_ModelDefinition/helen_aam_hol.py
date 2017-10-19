#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
HELEN holistic AAM



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from helenAAM import HelenAAM

model = HelenAAM('~/predPap-ben/datasets/HELEN', basename='helen_aam_hol')
model.load_data()
model.train_model()
model.fit_model()
model.predict_test_set()
model.generate_cdf()
