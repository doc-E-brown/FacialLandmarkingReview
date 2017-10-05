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
import tarfile
from urllib import urlretrieve
from zipfile import ZipFile

BIOID_DATA_FOLDER = os.getenv('BIOID_DATA', '~/datasets/BioID')
MUCT_DATA_FOLDER = os.getenv('MUCT_DATA', '~/datasets/MUCT')
HELEN_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/HELEN')
IBUG_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/IBUG/300W')
MENPO_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/IBUG/menpo_2017_trainset')

def getBioID():
    """ Function to download and prepare the BioID data set """
    print("Get the BioID dataset")
    print("The BioID dataset is published by the BioID company")
    print("https://www.bioid.com/About/BioID-Face-Database")
    print("=" * 60)

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

def getMUCT():
    """ Download the muct dataset """

    print("Get the MUCT dataset")
    print("The MUCT dataset is published by Stephen Milborrow")
    print("https://github.com/StephenMilborrow/muct")
    print("=" * 60)

    # Create MUCT_DATA_FOLDER if it doesnt exist
    if not os.path.exists(MUCT_DATA_FOLDER):
        os.mkdir(MUCT_DATA_FOLDER)

    # Get the images data
    for alpha in ['a', 'b', 'c', 'd', 'e']:
        print("Downloading images data: {}".format(alpha))
        filename, headers = urlretrieve(
            'https://github.com/StephenMilborrow/muct/raw/master/muct-{}-jpg-v1.tar.gz'.\
                format(alpha))

        print("Extracting images: {}".format(alpha))
        with tarfile.open(filename, 'r|gz') as f:
            f.extractall(MUCT_DATA_FOLDER)

    # Rename the images folder
    shutil.move(
        os.path.join(MUCT_DATA_FOLDER, 'jpg'),
        os.path.join(MUCT_DATA_FOLDER, 'muct-images'),
    )
    
    # Get the landmarks data
    print("Downloading landmarks")
    filename, headers = urlretrieve(
        'https://github.com/StephenMilborrow/muct/raw/master/muct-landmarks-v1.tar.gz'.\
            format(alpha))

    print("Extracting landmarks")
    with tarfile.open(filename, 'r|gz') as f:
        f.extractall(MUCT_DATA_FOLDER)

def getHELEN():
    """ Download the HELEN dataset"""

    print("Get the ibug annotated HELEN dataset")
    print("The original HELEN dataset is published by Vuong Le")
    print("http://www.ifp.illinois.edu/~vuongle2/helen/")
    print("=" * 60)
    print("Due to licensing constraints you will need to download the ibug annotated "\
    "HELEN dataset manually from:\n"\
    "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip/\n"\
    "and extract the zip file into\n"\
    "{}".format(HELEN_DATA_FOLDER))

    # Create MUCT_DATA_FOLDER if it doesnt exist
    if not os.path.exists(HELEN_DATA_FOLDER):
        os.mkdir(HELEN_DATA_FOLDER)

def get300W():
    """ Download the 300W dataset"""

    print("Get the ibug annotated HELEN dataset")
    print("The original HELEN dataset is published by Vuong Le")
    print("http://www.ifp.illinois.edu/~vuongle2/helen/")
    print("=" * 60)
    print("Due to licensing constraints you will need to download the ibug annotated "\
    "HELEN dataset manually from:\n"\
    "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip/\n"\
    "and extract the zip file into\n"\
    "{}".format(HELEN_DATA_FOLDER))

    # Create MUCT_DATA_FOLDER if it doesnt exist
    if not os.path.exists(HELEN_DATA_FOLDER):
        os.mkdir(HELEN_DATA_FOLDER)



if __name__ == "__main__":
    #getBioID()
    #getMUCT()
    getHELEN()
