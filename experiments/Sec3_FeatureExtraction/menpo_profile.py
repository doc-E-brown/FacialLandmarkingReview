#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
menpo profile face detection



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from menpo_front import Menpo

if __name__ == "__main__":

    results = []
    detector = Menpo(profile_photo=True)
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes)
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = Menpo(profile_photo=True,
        cascade="haarcascade_frontalface_alt.xml",
        results_file="menpo_profile_alt.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = Menpo(profile_photo=True,
        cascade="haarcascade_frontalface_alt2.xml",
        results_file="menpo_profile_alt2.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = Menpo(profile_photo=True,
        cascade="haarcascade_profileface.xml",
        results_file="menpo_profile_profileface.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    detector = Menpo(profile_photo=True,
        cascade=None,
        results_file="menpo_profile_hog.csv")
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    print("{:<40}{:^10}{:^40}{:^10}".format(
        'Face Detector', '# images',
        'Detection rate (%)', 'False pos'))
    for result in results:
        print("{:<40}{:^10}{:^40.2f}{:^10}".format(result[0], 
            result[1][0], result[1][1], result[1][2]))
