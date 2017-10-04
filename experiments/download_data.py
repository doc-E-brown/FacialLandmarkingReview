#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Download datasets

This Python 2 module downloads and extracts availables datasets into the correct
locations for use within Sec3_FeatureExtraction and Sec4_ModelDefinition.

Some datasets such as Menpo and ibug cannot be automatically downloaded
as they require additional information to be provided to the ibug group.
These datasets will need to be downloaded manually.



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
import os
import shutil
from urllib import urlretrieve
from zipfile import ZipFile

BIOID_DATA_FOLDER = os.getenv('BIOID_DATA', '~/datasets/BioID')

def getBioID():
    """ Function to download and prepare the BioID data set """
    print("Get BioID dataset")
    print("The BioID dataset is provided by the BioID company")
    print("https://www.bioid.com/About/BioID-Face-Database")
    print("=" * 30)

    # Create BIOID_DATA_FOLDER if it doesnt exist
    if not os.path.exists(BIOID_DATA_FOLDER):
        os.mkdir(BIOID_DATA_FOLDER)

    # Create the faces and points_20 folder
    faces_folder = os.path.join(BIOID_DATA_FOLDER, 'faces')

    # Create the faces folder
    if not os.path.exists(BIOID_DATA_FOLDER):
        os.mkdir(faces_folder)

    # Download the datasets
    # Download face images
    print("Downloading faces data")
    filename, headers = urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip')

    print("Extracting faces data")
    with ZipFile(filename, 'r') as f:
        f.extractall(faces_folder)

    # Download the 20 point configuration landmarks 
    print("Downloading landmarks data")
    filename, headers = urlretrieve('https://ftp.uni-erlangen.de/pub/facedb/bioid_pts.zip')

    print("Extracting landmarks data")
    with ZipFile(filename, 'r') as f:
        f.extractall(BIOID_DATA_FOLDER)

if __name__ == "__main__":
    getBioID()
