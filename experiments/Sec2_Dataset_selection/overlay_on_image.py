#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Overlay results onto image"""

# Imports
import os
import numpy as np
from PIL import Image, ImageDraw

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 June  09:06:02 AEST 2017'
__license__ = 'CC-BY-4.0'

DISPLAY_IMAGE = "display_image.jpg"
RESULTS_FOLDER = "raw_data"
SAVE_IMAGE_NAME = "image_with_all_marks.jpg"
RADIUS = 1
MEAN_RADIUS = 2 * RADIUS
COLOUR="#e50000"
MEAN_COLOUR = "#eeff00"
STD_COLOUR = "#18ff00"
SCALE = 5

def plot_individual():
    """Plot the individual submissions to the template and save the results"""

    for filename in os.listdir(RESULTS_FOLDER):
        if os.path.splitext(filename)[1] == '.csv':
            image = Image.open(DISPLAY_IMAGE)
            draw = ImageDraw.Draw(image)
            all_mrks = []

            lmrks = np.genfromtxt(
                os.path.join(RESULTS_FOLDER, filename),
                delimiter=',')
            all_mrks.append(lmrks)

            for pts in lmrks:
                draw.ellipse((
                    pts[0] - (RADIUS / 2), pts[1] - (RADIUS / 2),
                    pts[0] + RADIUS, pts[1] + RADIUS),
                    fill=COLOUR)

            # Increase the size of the image
            img_w, img_h = image.size
            image = image.resize((SCALE * img_w, SCALE * img_h), Image.ANTIALIAS)

            image.save(os.path.join(RESULTS_FOLDER,
                filename.replace('.csv', '.jpg')))

def overlay_all(plot_mean=True, plot_std_dev=True):
    """Apply all landmarks to template image and save the results"""

    image = Image.open(DISPLAY_IMAGE)
    draw = ImageDraw.Draw(image)
    all_mrks = []

    for filename in os.listdir(RESULTS_FOLDER):
        if os.path.splitext(filename)[1] != '.csv':
            continue
        lmrks = np.genfromtxt(
            os.path.join(RESULTS_FOLDER, filename),
            delimiter=',')
        all_mrks.append(lmrks)

        for pts in lmrks:
            draw.ellipse((
                pts[0] - (RADIUS / 2), pts[1] - (RADIUS / 2),
                pts[0] + RADIUS, pts[1] + RADIUS),
                fill=COLOUR)


    all_mrks = np.array(all_mrks)
    # Determine the mean for each point
    mean_mrks = np.mean(all_mrks, axis=0)
    # Determine the standard deviation in x and y directions for each point
    std_marks = np.std(all_mrks, axis=0)

    if plot_mean:

        # Plot mean points
        for pts in mean_mrks:
            draw.ellipse((
                pts[0] - (MEAN_RADIUS / 2), pts[1] - (MEAN_RADIUS / 2),
                pts[0] + MEAN_RADIUS, pts[1] + MEAN_RADIUS),
                fill=MEAN_COLOUR)

    if plot_std_dev:

        # Plot std ellipses 
        for mean, std in zip(mean_mrks, std_marks):
            x_mean = mean[0]
            y_mean = mean[1]
            x_std = std[0]
            y_std = std[1]
            draw.ellipse((
                x_mean - (x_std / 2), y_mean - (y_std / 2),
                x_mean + (x_std / 2), y_mean + (y_std / 2)),
                outline=STD_COLOUR)


    # Increase the size of the image
    img_w, img_h = image.size
    image = image.resize((SCALE * img_w, SCALE * img_h), Image.ANTIALIAS)

    image.save(SAVE_IMAGE_NAME)


if __name__ == "__main__":
    #plot_individual()
    overlay_all()
