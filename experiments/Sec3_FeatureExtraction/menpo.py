#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
menpo face detection



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import numpy as np 
import pandas as pd
from scipy.misc import imread
from _base import Base

class Menpo(Base):
    """Class definition for Menpo dataset"""

    def __init__(self,
        data_folder,
        pts_ext='.pts',
        photo_ext='.jpg',
        results_file='menpo_detection.csv',
        write_photos=False,
        cascade='haarcascade_frontalface_default.xml',
        profile_photo=False,
        ):
        """Constructor"""

        super().__init__(data_folder, pts_ext, photo_ext, results_file, write_photos, cascade)

        self.data_dirs = [data_folder]
        self.profile_photo = profile_photo

    def load_sample_names(self):
        """A generator which yields the basenames of the samples
        within the dataset.

        """

        for filename in os.listdir(self.data_dirs[0]):

            basename, ext = os.path.splitext(filename)

            # Only yield for photo extensions to avoid duplicates
            if ext != self.photo_ext:
                continue

            basename = os.path.join(self.data_dirs[0], basename)

            # Check the number of landmarks
            # Facial has 68, profile 29
            pts = self.load_pts("%s%s" % (basename, self.pts_ext)) 
            if (not self.profile_photo and pts.shape[0] == 68):
                yield basename
            elif (self.profile_photo and pts.shape[0] == 39):
                yield basename

    def load_images(self):
        """A generator for images in the dataset"""

        for sample_name in self.load_sample_names():
            basename = os.path.basename(sample_name)
            yield (basename, imread(sample_name + self.photo_ext))


if __name__ == "__main__":

    results = []
    detector = Menpo('/home/ben/datasets/ibug/menpo_2017_trainset', write_photos=True, profile_photo=False, cascade=None)
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes)
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    if False:
        detector = Menpo('/home/ben/datasets/ibug/menpo_2017_trainset',
            cascade="haarcascade_frontalface_alt.xml",
            results_file="menpo_alt.csv")
        result = detector.detect_faces(bboxes)
        results.append((detector.cascade, result))

        detector = Menpo('/home/ben/datasets/ibug/menpo_2017_trainset',
            cascade="haarcascade_frontalface_alt2.xml",
            results_file="menpo_alt2.csv")
        result = detector.detect_faces(bboxes)
        results.append((detector.cascade, result))

        detector = Menpo('/home/ben/datasets/ibug/menpo_2017_trainset',
            cascade="haarcascade_profileface.xml",
            results_file="menpo_profileface.csv")
        result = detector.detect_faces(bboxes)
        results.append((detector.cascade, result))

    for result in results:
        print("%s:%d\t%0.2f\t%d" % (result[0], result[1][0], result[1][1], result[1][2]))
