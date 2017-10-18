#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Menpo Holistic active appearance model


:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import aam
from menpoAAM import MenpoAAM
from functools import partial

# Correct normalisation using face diagonal
aam.compute_errors = partial(aam.compute_errors, pt1=19, pt2=28)
model = MenpoAAM(
    filename='menpo_aam_profile_hol', profile=True)
model.load_data()

# Frontal
model.train_model(batch_size=128)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
