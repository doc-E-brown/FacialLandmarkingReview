#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""AAM test for MUCT dataset


:author: Ben Johnston
:license: 3-Clause BSD

"""
# Imports
import os
import menpo.io as mio
from aam import AAM 
from menpofit.aam import HolisticAAM, PatchAAM
from sklearn.model_selection import train_test_split

MUCT_DATA_FOLDER = os.getenv('MUCT_DATA', '~/datasets/muct')


class MuctAAM(AAM):
    """ MUCT AAM class """

    def __init__(self, path_to_data=MUCT_DATA_FOLDER, model_type=HolisticAAM, basename='muct_aam', verbose=True):
        super(MuctAAM, self).__init__(
            path_to_data, model_type, basename, verbose)

    def load_data(self, crop_percentage=0.1, test_set_ratio=0.3, max_images=None):
        """ Load the images and landmarks in an menpo.io
        format and crop the images using the specified
        landmarks as a guide
        
        Parameters
        ---------
        
        """

        images = []

        for i in mio.import_images(self.filepath, max_images=max_images, verbose=self.verbose):

            if i.landmarks['PTS'].lms.points.shape[0] != 76:
                continue

            i = i.crop_to_landmarks_proportion(crop_percentage)

            # Convert to grayscale if required
            if i.n_channels == 3:
                i = i.as_greyscale() # Default to luminosity

            images.append(i)

        # Split into training and test sets
        self.train_set, self.test_set =\
            train_test_split(images, test_size=test_set_ratio, random_state=42)

    def _crop_grayscale_images(self, filepath, crop_percentage):

        images = []

        for i in mio.import_images(filepath, max_images=None, verbose=self.verbose):
            i = i.crop_to_landmarks_proportion(crop_percentage)

            # Convert to grayscale if required
            if i.n_channels == 3:
                i = i.as_greyscale() # Default to luminosity

            # Due to large training set size use generators for better memory 
            # efficiency
            yield i
