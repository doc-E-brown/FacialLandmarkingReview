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
from menpofit.aam import PatchAAM

model = AAM('/home/bjoh3944/predPap-ben/datasets/ibug/300W/02_Outdoor',
            model_type=PatchAAM,
            basename='300W_aam_patch_outdoor')
model.load_data()
model.train_model(batch_size=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
