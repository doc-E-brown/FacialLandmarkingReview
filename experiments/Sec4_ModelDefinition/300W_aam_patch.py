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
from menpofit.aam import PatchAAM

DATA_FOLDER = os.getenv('IBUG_DATA', '~/datasets/ibug/300W')

model = AAM(os.path.join(DATA_FOLDER, 'combined'),
            model_type=PatchAAM,
            basename='300W_patch')
model.load_data()
model.train_model(batch_size=None)
model.fit_model()
model.predict_test_set()
model.generate_cdf()
