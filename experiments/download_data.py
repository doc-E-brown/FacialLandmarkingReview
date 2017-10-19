#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Download datasets

This Python 2 module downloads and extracts availables
datasets into the correct locations for use within
Sec3_FeatureExtraction and Sec4_ModelDefinition.

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
import argparse
from urllib import urlretrieve
from zipfile import ZipFile

BIOID_DATA_FOLDER = os.getenv('BIOID_DATA', '~/datasets/BioID')
MUCT_DATA_FOLDER = os.getenv('MUCT_DATA', '~/datasets/muct')
HELEN_DATA_FOLDER = os.getenv('HELEN_DATA', '~/datasets/HELEN')
IBUG_DATA_FOLDER = os.getenv('IBUG_DATA', '~/datasets/ibug/300W')
MENPO_DATA_FOLDER = os.getenv('MENPO_DATA',
                              '~/datasets/ibug/menpo_2017_trainset')


def _parse_config(*args, **kwargs):
    """ Manage the bash script configuration """

    parser = argparse.ArgumentParser(description="Get the datasets")
    parser.add_argument('-b', '--bioid', dest='bioid', action='store_const',
                        const='getBioID',
                        help="Download the BioID dataset")
    parser.add_argument('-e', '--helen', dest='helen', action='store_const',
                        const='getHELEN',
                        help="Print the instructions to download the "
                        "ibug annotated HELEN dataset")
    parser.add_argument('-i', '--ibug', dest='ibug', action='store_const',
                        const='get300W',
                        help="Print the instructions to download the "
                        "300W dataset and organise folder structure")
    parser.add_argument('-m', '--muct', dest='muct', action='store_const',
                        const='getMUCT',
                        help="Dwnload the MUCT dataset")
    parser.add_argument('-p', '--menpo', dest='menpo', action='store_const',
                        const='getMenpo',
                        help="Print the instructions to download the "
                        "menpo dataset")

    return parser.parse_args()


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
    filename, headers = urlretrieve(
        'https://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip')

    print("Extracting faces data")
    with ZipFile(filename, 'r') as f:
        f.extractall(faces_folder)

    # Download the 20 point configuration landmarks
    print("Downloading landmarks data")
    filename, headers = urlretrieve(
        'https://ftp.uni-erlangen.de/pub/facedb/bioid_pts.zip')

    print("Extracting landmarks data")
    with ZipFile(filename, 'r') as f:
        f.extractall(BIOID_DATA_FOLDER)

    # Extract the points and images into the main folder
    for folder in ['faces', 'points_20']:
        folder = os.path.join(BIOID_DATA_FOLDER, folder)

        for filename in os.listdir(folder):
            dest_filename = filename.replace('bioid', 'BioID')
            shutil.copy(os.path.join(folder, filename),
                        os.path.join(BIOID_DATA_FOLDER, dest_filename))


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
            'https://github.com/StephenMilborrow/muct'
            '/raw/master/muct-{}-jpg-v1.tar.gz'.
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
        'https://github.com/StephenMilborrow/muct/raw'
        '/master/muct-landmarks-v1.tar.gz'.
        format(alpha))

    print("Extracting landmarks")
    with tarfile.open(filename, 'r|gz') as f:
        f.extractall(MUCT_DATA_FOLDER)


def getHELEN():
    """ Download the HELEN dataset"""

    print("The original HELEN dataset is published by Vuong Le")
    print("http://www.ifp.illinois.edu/~vuongle2/helen/")
    print("=" * 60)
    print("Due to licensing constraints you will need to download "
          "the ibug annotated "
          "HELEN dataset manually from:\n"
          "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip/\n"
          "and extract the zip file into\n"
          "{}".format(HELEN_DATA_FOLDER))

    # Create MUCT_DATA_FOLDER if it doesnt exist
    if not os.path.exists(HELEN_DATA_FOLDER):
        os.mkdir(HELEN_DATA_FOLDER)


def get300W():
    """ Download the 300W dataset"""

    print("The 300W dataset is published by the ibug group")
    print("https://ibug.doc.ic.ac.uk/")
    print("=" * 60)
    print("Due to licensing constraints you will need to "
          "download the 300W annotated "
          "dataset manually from:\n"
          "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip/\n"
          "and extract the zip file into\n"
          "{}".format(IBUG_DATA_FOLDER))

    # Create IBUG_DATA_FOLDER if it doesnt exist
    if not os.path.exists(IBUG_DATA_FOLDER):
        os.mkdir(IBUG_DATA_FOLDER)

    # If the 01_Indoor and 02_Outdoor folders dont exist do nothing
    if not os.path.exists(os.path.join(IBUG_DATA_FOLDER, '01_Indoor')) and \
            not os.path.exists(os.path.join(IBUG_DATA_FOLDER, '02_Outdoor')):
        return

    # Move the data to the combined folder
    if not os.path.exists(os.path.join(IBUG_DATA_FOLDER, 'combined')):
        os.mkdir(os.path.join(IBUG_DATA_FOLDER, 'combined'))

    # Copy the files over
    dest = os.path.join(IBUG_DATA_FOLDER, 'combined')
    for folder in ['01_Indoor', '02_Outdoor']:
        folder = os.path.join(IBUG_DATA_FOLDER, folder)

        for filename in os.listdir(folder):
            shutil.copy(os.path.join(folder, filename),
                        dest)


def getMenpo():
    """ Download the Menpo dataset"""

    print("The Menpo dataset is published by the ibug group")
    print("https://ibug.doc.ic.ac.uk/")
    print("=" * 60)
    print("Due to licensing constraints you will need to download the menpo "
          "dataset manually from:\n"
          "https://ibug.doc.ic.ac.uk/download/annotations/helen.zip/\n"
          "and extract the zip file into\n"
          "{}".format(MENPO_DATA_FOLDER))

    # Create MENPO_DATA_FOLDER if it doesnt exist
    if not os.path.exists(MENPO_DATA_FOLDER):
        os.mkdir(MENPO_DATA_FOLDER)


if __name__ == "__main__":
    args = _parse_config()

    # Execute the functions specified at the command line
    for arg in args.__dict__:
        attr = getattr(args, arg)
        if attr:
            locals()[attr]()
