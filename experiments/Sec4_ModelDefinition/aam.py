#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Python module for loading / exporting
data in Section 4 Model Definition



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import numpy as np
import menpo.io as mio
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
from menpofit.fitter import noisy_shape_from_bounding_box
from scipy.spatial.distance import euclidean as norm
from sklearn.model_selection import train_test_split

def euclidean_2d(x, y):
    """ Euclidian distance of pair of points """
    diff = (x - y)**2
    diff = np.sum(diff, axis=1)
    return np.sqrt(diff)

def compute_errors(predictions, pt1=36, pt2=45):
    """ Compute the error given an array of predictions and the
    corresponding ground truth (stored with predictions)"""

    image_error = []

    for pred in predictions:

        truth = pred.gt_shape.points
        pred = pred.final_shape.points

        # Landmarks 37 and 46 are the outer corners of the eyes
        inter_ocular_dist = norm(truth[pt1], truth[pt2])

        err = np.sum(euclidean_2d(pred, truth))
        err /= (pred.shape[0] * inter_ocular_dist)

        image_error.append(err)

    return image_error


class AAM(object):
    """ PDM class """

    def __init__(self, path_to_data, model_type=HolisticAAM, basename='aam', verbose=True):
        """ """
        self.filepath = path_to_data
        self.basename = basename
        self.verbose = verbose
        self.model_type = model_type 
        self.model_fitter = LucasKanadeAAMFitter

    def load_data(self, crop_percentage=0.1, test_set_ratio=0.3, max_images=None):
        """ Load the images and landmarks in an menpo.io
        format and crop the images using the specified
        landmarks as a guide
        
        Parameters
        ---------
        
        """

        images = []

        count = 0
        for i in mio.import_images(self.filepath, max_images=max_images, verbose=self.verbose):
            i = i.crop_to_landmarks_proportion(crop_percentage)

            # Convert to grayscale if required
            if i.n_channels == 3:
                i = i.as_greyscale() # Default to luminosity

            images.append(i)
            count += 1

        # Split into training and test sets
        self.train_set, self.test_set =\
            train_test_split(images, test_size=test_set_ratio, random_state=42)

    # batch size 128
    def train_model(self, diagonal=None, batch_size=256, max_shape_components=None, max_appearance_components=None, scales=(0.5, 1)):
        """ Train the model """

        self.model = self.model_type(self.train_set,
            diagonal=diagonal,
            batch_size=batch_size,
            max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            scales=scales,
            verbose=self.verbose)

        if self.verbose:
            print(self.model)

    def fit_model(self):
        """ Fit model using the a noisy shape constructed from the bounding box"""

        self.model_fit = self.model_fitter(
            self.model,
            )

    def predict_test_set(self):

        fit_results = []
        image_paths = []
        predicted_values = []

        for idx, i in enumerate(self.test_set):
            gt_s = i.landmarks['PTS'].lms
            image_paths.append(str(i.path))

            # Starting shape
            # An estimate from the bounding box
            initial_shape = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())

            # Predict the result
            fit_result = self.model_fit.fit_from_shape(i, initial_shape, gt_shape=gt_s)
            fit_results.append(fit_result)
            predicted_values.append(fit_result.final_shape.points)

        self.predictions = fit_results

        # Store predictions
        np.save("%s_predictions" % self.basename,
                   np.array(predicted_values),
                   )
        with open("%s_testset.txt" % self.basename, 'w') as f:
            f.write("\n".join(image_paths))

    def generate_cdf(self):
        """ Generate cumulative distribution function """

        errs = compute_errors(self.predictions)
        errs = np.sort(errs)
        self.cumsum = np.stack((errs, np.cumsum(errs)))
        np.save(self.basename, self.cumsum)
