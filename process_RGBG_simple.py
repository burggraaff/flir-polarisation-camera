"""
Simple data processing for images from the RGBG polarisation camera.
Demosaicking is done by splitting the image into its components.

Call signature:
    python process_RGBG_simple.py my_file.raw
"""
from sys import argv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import fpc

# Get the filename from the command line
filename = Path(argv[1])
label = filename.stem

# Load the RAW file as an array
img = fpc.io.load_image_blackfly(filename, mask_saturated=True)

# Demosaicking
img_demosaicked = fpc.stokes.demosaick_RGB(img)  # Dimensions: [x, y, RGB, Polarisers]
# img_demosaicked = np.moveaxis(img_demosaicked, (0, 1), (-2, -1))  # Move image axes to the end

# Calculate the Stokes vector
img_stokes = fpc.stokes.convert_demosaicked_image_to_stokes(img_demosaicked)  # Dimensions: [x, y, RGB, IQU]
img_intensity, img_dolp, img_aolp = fpc.stokes.convert_stokes_to_lp(img_stokes)

# Show the result
fpc.plot.show_intensity_dolp_aolp_RGB_separate(img_intensity, img_dolp, img_aolp, title=label, saveto=f"results/{label}.png")
fpc.plot.show_intensity_dolp_aolp_RGB(img_intensity, img_dolp, img_aolp, title=label, saveto=f"results/{label}_RGB.png")
