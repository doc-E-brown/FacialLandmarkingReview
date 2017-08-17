#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Base face detector class



:author: Ben Johnston
:license: 3-Clause BSD

"""


# Imports
import os
import cv2
import dlib
import numpy as np
from scipy.misc import imread
from PIL import Image, ImageDraw
from functools import partial

def compute_overlap(rect1, ref):
    """Compute the percentage overlap of rect1 to ref"""
    x1, y1, w1, h1 = rect1
    x2, y2 = x1 + w1, y1 + h1
    x3, y3, w2, h2 = ref 
    x4, y4 = x3 + w2, y3 + h2

    # Generate array of zeros for overlap
    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)
    temp = np.zeros((max_x - min_x, max_y - min_y))

    # Increment the ref array
    temp[y3 - min_y:y4 - min_y, x3 - min_x:x4 - min_x] += 1
    temp[y1 - min_y:y2 - min_y, x1 - min_x:x2 - min_x] += 2

    overlap = temp[np.where(temp==3)]
    return overlap.size / (w2 * h2) 

def face_detect_hog(image=None):
    """Detect a face in an image using the dlib implementation of
    Dalal & Triggs HoG detector
    """

    detector = dlib.get_frontal_face_detector()
    detected = detector(image, 1)

    # Modify the detected objects to be in the same format as
    # Viola-Jones
    rects = []
    for det in detected:
        rects.append([det.left(), det.top(), 
                      det.right() - det.left(),
                      det.bottom() - det.top()])

    return rects

def face_detect_viola(cascade=None, image=None):
    """Detect a face in an image using opencv implementation of
    viola jones"""

    classifier = cv2.CascadeClassifier(cascade)
    return classifier.detectMultiScale(image, 1.1, 3, 0, (0, 0))


class IBUG_300W(object):
    """Class definition for IBUG dataset"""

    def __init__(self,
        data_folder,
        pts_ext='.pts',
        photo_ext='.png',
        results_file='ibug_detection.csv',
        write_photos=True,
        cascade='haarcascade_frontalface_default.xml',
        ):
        """Constructor"""

        self.data_dirs = \
            [os.path.join(data_folder, folder)
            for folder in ['01_Indoor', '02_Outdoor']]

        self.pts_ext = pts_ext 
        self.photo_ext = photo_ext
        self.results_file = results_file
        self.cascade = cascade 
        self.write_photos = write_photos

        if cascade is not None:
            self.detector = partial(face_detect_viola, cascade=self.cascade) 
        else:
            self.detector = face_detect_hog 
            self.cascade = "hog"

    def load_pts(self, filename):
        """Load landmark points from a .pts file 
        and return a numpy array of points"""

        with open(filename, 'r') as f:
            data = f.read()

        # Extract only the coordinates_lines 
        data = data[data.find('{') + 1: data.find('}')]

        # Put data into a 2D numpy array and return
        data = np.fromstring(data, sep='\n').reshape((-1, 2))

        data = np.asarray(data, dtype='int')

        return data

    def load_sample_names(self):
        """A generator which yields the basenames of the samples
        within the dataset.
        """

        for folder in self.data_dirs:
            for filename in os.listdir(folder):

                basename, ext = os.path.splitext(filename)

                # Only yield for coords files to avoid duplicates
                if ext != self.pts_ext:
                    continue
                
                yield os.path.join(folder, basename)

    def extract_bbox(self, pts):
        """Determine the bounding box for the face within the image
        The function takes the MULTI-PIE coordinates for the face within
        the image and returns the lower and upper bounding box
        (x0, y0), (w, h) of the face within the image
        """

        upper = np.max(pts, axis=0)
        lower = np.min(pts, axis=0)

        return [lower[0], lower[1],
                upper[0] - lower[0], upper[1] - lower[1]]

    def get_bounding_boxes(self):
        """Compute the face bounding boxes for each of the samples in the
        dataset"""

        bboxes = {}
        for basename in self.load_sample_names():
            pts = self.load_pts("%s%s" % (basename, self.pts_ext)) 
            bbox = self.extract_bbox(pts)
            basename = os.path.basename(basename)
            #line = [basename] + bbox
            bboxes[basename] = bbox

        return bboxes

    def store_bounding_boxes(self, bboxes, bbox_file='IBUG_300W_boxes.pts'):
        """Store the bounding boxes in the specified file"""

        # Write the header to the file
        with open(bbox_file, 'w') as f:
            f.write("Basename, x0, y0, x1, y1\n")

            for box in bboxes:
                name, box = box, bboxes[box]
                f.write("%s,%0.2f,%0.2f,%0.2f,%0.2f\n" %
                    (name, box[0], box[1], box[2], box[3]))

    def load_images(self):
        """Open an image in the dataset"""

        for folder in self.data_dirs:
            for filename in os.listdir(folder):

                basename, ext = os.path.splitext(filename)

                # Only yield for image files 
                if ext != self.photo_ext:
                    continue

                yield (basename, imread(os.path.join(folder, filename)))

    def detect_faces(self, bboxes):

        detection_dict = {}
        false_pos_dict = {}
        sample_size = 0
        detection = 0
        false_pos = 0

        for basename, image in self.load_images():
            print(basename)
            rects = self.detector(image=image)
            sample_size += 1

            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            for rect in rects:
                cv2.rectangle(image,
                    (rect[0], rect[1]),
                    (rect[0] + rect[2], rect[1] + rect[3]),
                    (0, 255, 0),
                    4
                    )

            gtruth = bboxes[basename]
            cv2.rectangle(image,
                (gtruth[0], gtruth[1]),
                (gtruth[0] + gtruth[2], gtruth[1] + gtruth[3]),
                (0, 0, 255),
                4
                )

            if self.write_photos:
                cv2.imwrite("%s.jpg" % basename, image)

            detection_dict[basename], false_pos_dict[basename] = \
                self.is_face_detected(rects, bboxes[basename])
            detection += 1 if detection_dict[basename] else 0
            false_pos += false_pos_dict[basename]

        # Write results to file
        with open(self.results_file, "w") as f:
            for basename in detection_dict:
                f.write("%s, %d, %d\n" % (
                    basename,
                    detection_dict[basename], 
                    false_pos_dict[basename]))

        detection_rate = 100 * (float(detection) / float(sample_size))
        #print("Detection Rate: %0.2f %%" % detection_rate)
        return sample_size, detection_rate, false_pos

    def is_face_detected(self, rects, bbox):
        """Determine the rate of successful detection / false detection"""

        for rect in rects:
            # What is the percentage overlap between the rect and bbox
            if compute_overlap(rect, bbox) >= 0.5:
                return True, len(rects) - 1

        return False, len(rects)

if __name__ == "__main__":

    results = []
    detector = IBUG_300W('/home/ben/datasets/ibug/300W', write_photos=True, cascade=None)
    bboxes = detector.get_bounding_boxes()
    detector.store_bounding_boxes(bboxes)
    result = detector.detect_faces(bboxes)
    results.append((detector.cascade, result))

    if False:
        detector = IBUG_300W('/home/ben/datasets/ibug/300W',
            cascade="haarcascade_frontalface_alt.xml",
            results_file="ibug_alt.csv")
        result = detector.face_detect_viola(bboxes)
        results.append((detector.cascade, result))

        detector = IBUG_300W('/home/ben/datasets/ibug/300W',
            cascade="haarcascade_frontalface_alt2.xml",
            results_file="ibug_alt2.csv")
        result = detector.face_detect_viola(bboxes)
        results.append((detector.cascade, result))

        detector = IBUG_300W('/home/ben/datasets/ibug/300W',
            cascade="haarcascade_profileface.xml",
            results_file="ibug_profile.csv")
        result = detector.face_detect_viola(bboxes)
        results.append((detector.cascade, result))

    for result in results:
        print("%s:%d\t%0.2f\t%d" % (result[0], result[1][0], result[1][1], result[1][2]))
