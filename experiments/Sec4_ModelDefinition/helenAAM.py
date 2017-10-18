#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""AAM test for HELEN dataset



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import menpo.io as mio
from aam import AAM 
from menpofit.aam import HolisticAAM, PatchAAM

HELEN_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/ibug/HELEN')

class HelenAAM(AAM):
    """ Helen AAM class """


    def __init__(self, path_to_data=HELEN_DATA_FOLDER, model_type=HolisticAAM, basename='helen_aam', verbose=True):

        super(HelenAAM, self).__init__(
            path_to_data, model_type, basename, verbose)

    def _crop_grayscale_images(self, filepath, crop_percentage, max_images=None):

        images = []

        for i in mio.import_images(filepath, max_images=max_images, verbose=self.verbose):
            i = i.crop_to_landmarks_proportion(crop_percentage)

            # Convert to grayscale if required
            if i.n_channels == 3:
                i = i.as_greyscale() # Default to luminosity

            # Due to large training set size use generators for better memory 
            # efficiency
            yield i


    def load_data(self, crop_percentage=0.1, max_images=None):
        """ Load the images and landmarks in an menpo.io
        format and crop the images using the specified
        landmarks as a guide
        
        Parameters
        ---------
        
        """

        train_path = os.path.join(self.filepath, 'trainset')
        self.train_set = self._crop_grayscale_images(train_path, crop_percentage,
            max_images=max_images)

        test_path = os.path.join(self.filepath, 'testset')
        self.test_set = self._crop_grayscale_images(test_path, crop_percentage,
            max_images=max_images)
