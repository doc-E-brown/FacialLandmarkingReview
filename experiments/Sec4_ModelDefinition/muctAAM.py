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


class MuctAAM(AAM):
    """ MUCT AAM class """

    def __init__(self, path_to_data, model_type=HolisticAAM, filename='muct_aam.txt', verbose=True):
        super(MuctAAM, self).__init__(
            path_to_data, model_type, filename, verbose)

    def load_data(self, crop_percentage=0.1, test_set_ratio=0.3):
        """ Load the images and landmarks in an menpo.io
        format and crop the images using the specified
        landmarks as a guide
        
        Parameters
        ---------
        
        """

        images = []

        for i in mio.import_images(self.filepath, max_images=None, verbose=self.verbose):

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

if __name__ == "__main__":

    #a = MuctAAM('~/datasets/muct/muct-images')
    a = MuctAAM('~/datasets/muct/muct-images', PatchAAM, 'muct_patch.txt')
    a.load_data()
    a.train_model(diagonal=None,batch_size=256, max_shape_components=None, max_appearance_components=None)
    #a.train_model(diagonal=None,max_shape_components=None, max_appearance_components=None)
    a.fit_model()
    a.predict_test_set()
    a.generate_cdf()
