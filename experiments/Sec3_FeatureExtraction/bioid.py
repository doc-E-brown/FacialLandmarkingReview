#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
BioID face detection



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import numpy as np 
from ._base import Base

DATA_FOLDER = os.getenv("BIOID_DATA", '~/datasets/BioID')

class BioId(Base):
    """Class definition for BioId dataset"""

    def __init__(self,
        data_folder=DATA_FOLDER,
        pts_ext='.pts',
        photo_ext='.pgm',
        results_file='bioid_default.csv',
        write_photos=False,
        cascade='haarcascade_frontalface_default.xml'):
        """Constructor"""

        super().__init__(data_folder, pts_ext, photo_ext, results_file, write_photos, cascade)

        self.data_dirs = \
            [os.path.join(data_folder, folder) for folder in ['faces', 'points_20']]

    def load_sample_names(self):
        """A generator which yields the basenames of the samples
        within the dataset.
        """

        for filename in os.listdir(self.data_dirs[0]):

            basename, ext = os.path.splitext(filename)

            # Only yield for photo extensions to avoid duplicates
            if ext != self.photo_ext:
                continue
            
            yield os.path.join(self.data_dirs[1], basename)

    def load_pts(self, filename):
        """Load landmark points from a .pts file 
        and return a numpy array of points"""

        folder = os.path.dirname(filename)
        filename = os.path.basename(filename).lower()

        filename = os.path.join(folder, filename)

        with open(filename, 'r') as f:
            data = f.read()

        # Cut out coordinates
        data = data[data.find('{')+2:-3]

        # Put data into a 2D numpy array and return
        data = np.fromstring(data, sep='\n').reshape((-1, 2))

        data = np.asarray(data, dtype='int')

        return data

if __name__ == "__main__":

    results = []
    detector = BioId()
    bboxes = detector.get_bounding_boxes()
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = BioId(cascade="haarcascade_frontalface_alt.xml",
        results_file="bioid_alt.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = BioId(cascade="haarcascade_frontalface_alt2.xml",
        results_file="bioid_alt2.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = BioId(cascade="haarcascade_profileface.xml",
        results_file="bioid_profile.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    for result in results:
        print("%s:%d\t%0.2f\t%d" % (result[0], result[1][0], result[1][1], result[1][2]))
