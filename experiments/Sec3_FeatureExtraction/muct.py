#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
MUCT face detection



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import numpy as np 
import pandas as pd
from _base import Base

MUCT_DATA_FOLDER = os.getenv('MUCT_DATA', '~/datasets/MUCT')

class MUCT(Base):
    """Class definition for MUCT dataset"""

    def __init__(self,
        data_folder=MUCT_DATA_FOLDER,
        pts_ext='.csv',
        photo_ext='.jpg',
        results_file='muct_detection.csv',
        write_photos=False,
        cascade='haarcascade_frontalface_default.xml'):
        """Constructor"""

        super().__init__(data_folder, pts_ext, photo_ext, results_file, write_photos, cascade)

        self.data_dirs = \
            [os.path.join(data_folder, folder) for folder in ['muct-images', 'muct-landmarks']]
        
        # Read the coords early
        self._read_coords_csv()

    def _read_coords_csv(self):
        """The MUCT database stores all of the coordinates in a
        single csv file.  Read the file first to get image basenames"""

        self.coords = pd.read_csv(
            os.path.join(self.data_dirs[1], 'muct76-opencv.csv'))

        del self.coords['tag']

    def load_sample_names(self):
        """A generator which yields the basenames of the samples
        within the dataset.

        For MUCT all of the basenames and coordinates are stored in a single csv file.
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

        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]


        data = self.coords[self.coords.name == basename]
        del data['name']

        data = data.values.reshape((-1, 2))
        data = np.asarray(data, dtype='int')

        # Some images do not have all coordinates, so remove those from 
        # array
        data = np.delete(data, np.where(data ==0)[0], axis=0)

        return data


if __name__ == "__main__":

    results = []
    detector = MUCT()
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes, bbox_file='muct.pts')
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = MUCT(
        cascade="haarcascade_frontalface_alt.xml",
        results_file="muct_alt.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = MUCT(,
        cascade="haarcascade_frontalface_alt2.xml",
        results_file="muct_alt2.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))


    detector = MUCT(,
        cascade="haarcascade_profileface.xml",
        results_file="muct_profile.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = MUCT(,
        cascade=None,
        results_file="muct_hog.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    print("{:<40}{:^10}{:^40}{:^10}".format(
        'Face Detector', '# images',
        'Detection rate (%)', 'False pos'))
    for result in results:
        print("{:<40}{:^10}{:^40.2f}{:^10}".format(result[0], 
            result[1][0], result[1][1], result[1][2]))
