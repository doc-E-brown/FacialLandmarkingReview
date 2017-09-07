# For BioID, copy *.pts files from points_20 to faces folder then
"""
for filename in os.listdir(os.getcwd()):
    basename, ext = os.path.splitext(filename)
    if ext == '.pts':
        basename = basename.split('_')[-1]
        shutil.move(filename, 'BioID_%s.pts' % basename)  
"""

# Generate *.pts files for muct database
# Store them on a website as a zip somewhere for others to use

import os
import pandas as pd
import numpy as np
MUCT_FOLDER = '/home/bjoh3944/predPap-ben/datasets/muct/'
MUCT_IMAGES = os.path.join(MUCT_FOLDER, 'muct-images')
VERSION = 1
df = pd.read_csv(os.path.join(MUCT_FOLDER, 'muct-landmarks/muct76-opencv.csv'))

def export_pts(filename, coords):
    """
    version: 1
    n_points: xx
    {
    xxx.yyy aaa.bbb
    xxx.yyy aaa.bbb
    xxx.yyy aaa.bbb
    xxx.yyy aaa.bbb
    xxx.yyy aaa.bbb
    }
    """
    content = "version %d\n" % VERSION
    content += "n_points: %d\n{\n" % coords.shape[0]

    for row in coords:
        content += "%i %i\n" % (row[0], row[1])

    content += "}\n"
    
    with open(filename, 'w') as f:
        f.write(content)
        

for idx, row in df.iterrows():
    basename = row['name']
    print(basename)
    del row['tag']
    del row['name']

    coords = row.values.reshape((-1, 2))
    coords = np.asarray(coords, dtype='int')

    # Some images do not have all coordinates, so remove those from 
    # array
    coords = np.delete(coords, np.where(coords ==0)[0], axis=0)

    filename = os.path.join(MUCT_IMAGES, '%s.pts' % basename)
    export_pts(filename, coords)
