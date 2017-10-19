#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Extract Faces from the HELEN dataset



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import numpy as np
from _base import Base

HELEN_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/HELEN')


class HELEN(Base):
    """Class definition for HELEN dataset"""

    def __init__(self,
                 data_folder=HELEN_DATA_FOLDER,
                 pts_ext='.pts',
                 photo_ext='.jpg',
                 results_file='helen_detection.csv',
                 write_photos=False,
                 cascade='haarcascade_frontalface_default.xml'):
        """Constructor"""

        super().__init__(data_folder, pts_ext, photo_ext,
                         results_file, write_photos, cascade)

        self.data_dirs = \
            [os.path.join(data_folder, folder)
             for folder in ['trainset', 'testset']]

    def get_bounding_boxes(self):
        """Compute the face bounding boxes for each of the samples in the
        dataset"""

        bboxes = {}
        for filename in self.load_sample_names():
            basename, pts = self.load_pts("%s%s" % (filename, self.pts_ext))
            bbox = self.extract_bbox(pts)
            basename = os.path.basename(basename)
            bboxes[basename] = bbox

        return bboxes

    def load_pts(self, filename):
        """Load landmark points from a .pts file
        and return a numpy array of points"""

        with open(filename, 'r') as f:
            pts = f.read()

        pts = pts[pts.find("{") + 2:-2]
        pts = pts.split('\n')
        pts = [pt.split(' ') for pt in pts]
        pts = [[float(pt[0]), float(pt[1])] for pt in pts]

        pts = np.array(pts, dtype='int')
        basename, _ = os.path.splitext(os.path.basename(filename))

        return basename, pts


if __name__ == "__main__":

    results = []
    detector = HELEN(write_photos=True)
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes, bbox_file='helen.pts')
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = HELEN(
        cascade="haarcascade_frontalface_alt.xml",
        results_file="helen_alt.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = HELEN(
        cascade="haarcascade_frontalface_alt2.xml",
        results_file="helen_alt2.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = HELEN(
        cascade="haarcascade_profileface.xml",
        results_file="helen_profile.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = HELEN(
        cascade=None,
        results_file="helen_hog.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    print("{:<40}{:^10}{:^40}{:^10}".format(
        'Face Detector', '# images',
        'Detection rate (%)', 'False pos'))
    for result in results:
        print("{:<40}{:^10}{:^40.2f}{:^10}".format(result[0],
                                                   result[1][0],
                                                   result[1][1],
                                                   result[1][2]))
