#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""AAM test for MENPO dataset



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import menpo.io as mio
from aam import AAM 
from menpofit.aam import HolisticAAM, PatchAAM
from sklearn.model_selection import train_test_split


class MenpoAAM(AAM):
    """ Menpo AAM class """


    def __init__(self, path_to_data, model_type=HolisticAAM, filename='menpo_aam.txt', verbose=True, profile=False):

        super(MenpoAAM, self).__init__(
            path_to_data, model_type, filename, verbose)
        self.profile=profile

    def load_data(self, crop_percentage=0.1, test_set_ratio=0.3, max_images=None):
        """ Load the images and landmarks in an menpo.io
        format and crop the images using the specified
        landmarks as a guide
        
        Parameters
        ---------
        
        """

        images = []

        for i in mio.import_images(self.filepath, max_images=max_images, verbose=self.verbose):

            # Check if profile or frontal selected
            # Frontal has 68 landmarks, profile 39 
            if self.profile and (i.landmarks['PTS'].lms.points.shape[0] == 68):
                continue
            elif not self.profile and (i.landmarks['PTS'].lms.points.shape[0] == 39):
                continue

            i = i.crop_to_landmarks_proportion(crop_percentage)

            # Convert to grayscale if required
            if i.n_channels == 3:
                i = i.as_greyscale() # Default to luminosity

            images.append(i)

        # Split into training and test sets
        if self.verbose:
            print("%d images being used" % len(images))
        self.train_set, self.test_set =\
            train_test_split(images, test_size=test_set_ratio, random_state=42)

if __name__ == "__main__":

    from functools import partial
    import aam
    # Profile
    #a = MenpoAAM('~/predPap-ben/datasets/ibug/menpo_2017_trainset', filename='menpo_aam_profile.txt',
    #        profile=True)
    #aam.compute_errors = partial(aam.compute_errors, left=19, right=28)

    # Frontal
    #a = MenpoAAM('~/predPap-ben/datasets/ibug/menpo_2017_trainset', filename='menpo_aam_patch.txt',
    #        profile=False, model_type=PatchAAM)
    a = MenpoAAM('~/predPap-ben/datasets/ibug/menpo_2017_trainset', filename='menpo_aam2.txt',
            profile=False)
    a.load_data()

    # Profile 
    a.train_model(batch_size=None)

    # Frontal
    a.train_model()
    a.fit_model()
    a.predict_test_set()
    a.generate_cdf()

