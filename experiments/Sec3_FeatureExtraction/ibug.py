#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
300W face detector class



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
from _base import Base

IBUG_DATA_FOLDER = os.getenv('IBUG_DATA', '~/datasets/IBUG/300W')


class ibug(Base):
    """ibug 300W face detector class"""

    def __init__(self,
                 data_folder=IBUG_DATA_FOLDER,
                 pts_ext='.pts',
                 photo_ext='.png',
                 results_file='ibug_detection.csv',
                 write_photos=False,
                 cascade='haarcascade_frontalface_default.xml',
                 ):
        """__init__"""

        super().__init__(data_folder, pts_ext, photo_ext,
                         results_file, write_photos, cascade)

        self.data_dirs = \
            [os.path.join(data_folder, folder)
             for folder in ['01_Indoor', '02_Outdoor']]


if __name__ == "__main__":
    results = []
    detector = ibug()
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes)
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = ibug(
        cascade="haarcascade_frontalface_alt.xml",
        results_file="ibug_alt.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = ibug(
        cascade="haarcascade_frontalface_alt2.xml",
        results_file="ibug_alt2.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = ibug(
        cascade="haarcascade_profileface.xml",
        results_file="ibug_profile.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = ibug(
        cascade=None,
        results_file="ibug_hog.csv")
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
