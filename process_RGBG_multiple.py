"""
Simple data processing for images from the RGBG polarisation camera.
Demosaicking is done by splitting the image into its components.

Call signature:
    python process_RGBG_simple.py my_file.raw
"""
import numpy as np
from matplotlib import pyplot as plt
from spectacle import plot, symmetric_percentiles
from pathlib import Path
from sys import argv

import dofp

# Get the filename from the command line
folder = Path(argv[1])
filenames = folder.glob("*.raw")

# Loop over the filenames
for filename in filenames:
    print(filename)
    label = filename.stem

    # Load the RAW file as an array
    img = dofp.io.load_image_blackfly(filename)

    # Demosaicking
    img_demosaicked = dofp.stokes.demosaick_RGB(img)  # Dimensions: [x, y, RGB, Polarisers]
    # img_demosaicked = np.moveaxis(img_demosaicked, (0, 1), (-2, -1))  # Move image axes to the end

    # Calculate the Stokes vector
    img_stokes = dofp.stokes.convert_demosaicked_image_to_stokes(img_demosaicked)  # Dimensions: [x, y, RGB, IQU]
    img_intensity, img_dolp, img_aolp = dofp.stokes.convert_stokes_to_lp(img_stokes)

    # Show the result
    dofp.plot.show_intensity_dolp_aolp_RGB(img_intensity, img_dolp, img_aolp, title=label, saveto=f"E:/processed/{label}.png")
