#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
300W dataset holistic AAM



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
from aam import AAM

IBUG_DATA_FOLDER = os.getenv('IBUG_DATA', '~/datasets/ibug/300W')

model = AAM(os.path.join(IBUG_DATA_FOLDER, '01_Indoor'),
            basename='300W_aam_hol_indoor')
model.load_data()
model.train_model(batch_size=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
