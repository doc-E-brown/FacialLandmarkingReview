#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Plot landmarks separately to image"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from overlay_on_image import RESULTS_FOLDER

MEAN_RADIUS = 5 
MEAN_COLOUR = "#C4403B"
STD_COLOUR = "#3BBFC4"
STD_LINE_WIDTH = 1
SAVE_NAME = "landmarks_plot.tiff"


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Thursday 15 June  10:34:21 AEST 2017'
__license__ = 'CC-BY-4.0'

def _main():
    """Plot the landmarks"""

    # Read in all the data
    all_mrks = []
    for filename in os.listdir(RESULTS_FOLDER):
        lmrks = np.genfromtxt(
            os.path.join(RESULTS_FOLDER, filename),
            delimiter=',')
        all_mrks.append(lmrks)

    all_mrks = np.array(all_mrks)

    # Determine the mean and stdev landmarks
    mean_mrks = np.mean(all_mrks, axis=0)
    std_axes = np.std(all_mrks, axis=0)

    # Add the mean scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean_plot = plt.scatter(mean_mrks[:,0], mean_mrks[:,1],
        c=MEAN_COLOUR, s=MEAN_RADIUS, label='Mean Landmarks')

    # Add the standard deviation plot
    for mean_pt, std_pt in zip(mean_mrks, std_axes):
        ax.add_patch(
            Ellipse((mean_pt[0], mean_pt[1]),
                    std_pt[0], std_pt[1],
                    fill=None,
                    linewidth=STD_LINE_WIDTH,
                    edgecolor=STD_COLOUR,
                    ))

    # Construct ellipse legend
    std_line  = Line2D([], [], color=STD_COLOUR,
        label='Standard Deviations')
    plt.gca().invert_yaxis()
    plt.legend(handles=[mean_plot, std_line],
        loc='lower left',
        bbox_to_anchor=(0, -0.2, 1, -0.2),
        ncol=2,
        mode="expand")
    plt.title('Ground Truth Landmark Position Variation')
    plt.savefig(SAVE_NAME, bbox_inches='tight',
        dpi=100, format='tiff')

if __name__ == "__main__":
    _main()
