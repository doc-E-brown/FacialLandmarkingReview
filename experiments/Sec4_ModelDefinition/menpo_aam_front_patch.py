#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Menpo Patch active appearance model


:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from menpoAAM import MenpoAAM
from menpofit.aam import PatchAAM

model = MenpoAAM(
    filename='menpo_aam_front_patch',
    model_type=PatchAAM)
model.load_data()

# Frontal
model.train_model()
model.fit_model()
model.predict_test_set()
model.generate_cdf()
