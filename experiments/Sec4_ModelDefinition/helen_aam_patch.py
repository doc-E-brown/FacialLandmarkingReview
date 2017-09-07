#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
HELEN patch AAM 



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from helenAAM import HelenAAM
from menpofit.aam import PatchAAM

model = HelenAAM('~/predPap-ben/datasets/HELEN', model_type=PatchAAM, basename='helen_patch')
model.load_data()
model.train_model()
model.fit_model()
model.predict_test_set()
model.generate_cdf()
