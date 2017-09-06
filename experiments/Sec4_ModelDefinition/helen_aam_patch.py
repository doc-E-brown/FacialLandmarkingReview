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

model = HelenAAM('~/datasets/HELEN', PatchAAM, 'helen_patch.txt')
model.load_data()
model.train_model()
#model.train_model(diagonal=200, max_shape_components=None, max_appearance_components=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
