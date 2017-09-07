#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
300W dataset holistic AAM





:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from aam import AAM

model = AAM('/home/bjoh3944/predPap-ben/datasets/ibug/300W/combined',
            basename='300W_aam_hol')
model.load_data()
model.train_model(batch_size=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
