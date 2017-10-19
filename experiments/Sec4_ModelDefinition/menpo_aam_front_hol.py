#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Menpo Holistic active appearance model


:author: Ben Johnston
:license: 3-Clause BSD

"""

from menpoAAM import MenpoAAM

model = MenpoAAM(
    filename='menpo_aam_front_hol')
model.load_data()

# Frontal
model.train_model()
model.fit_model()
model.predict_test_set()
model.generate_cdf()
