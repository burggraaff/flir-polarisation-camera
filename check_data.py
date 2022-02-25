"""
Process a single image and plot its raw values in an image and histogram, to check for saturation or underexposure.

Call signature:
    python check_data.py my_file.raw
"""
from sys import argv
from pathlib import Path
import numpy as np
import fpc

# Get the filename from the command line
filename = Path(argv[1])
label = filename.stem

# Load the RAW file as an array
img = fpc.io.load_image_blackfly(filename)

# Show the RAW file and its histogram
fpc.plot.show_testplot(img, bins=np.linspace(0, 65536, 250), label="RAW Pixel value")
