"""
Analyse Pearson r (linearity) maps generated using the calibration scripts.
This script generates map images and histograms.

Script taken from https://github.com/monocle-h2020/camera_calibration/blob/master/analysis/linearity_characterise.py

Command line arguments:
    * `file_raw`: the file containing the Pearson r map to be analysed. This r
    map should be an NPY stack generated using linearity_raw.py.
    Optional:
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, analyse, plot
from pathlib import Path

# Get the data folder from the command line
# Use JPEG data if these are provided
file_raw = io.path_from_input(argv)

# # Load Camera object
# camera = io.load_camera(root)
# print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = Path("results")
save_to_maps = savefolder/f"map_raw.pdf"
save_to_histogram_RGB = savefolder/f"histogram_raw.pdf"

# Load the data
r = np.load(file_raw)
print("Loaded RAW Pearson r map")

# Make Gaussian maps of the RAW data
r_gauss = analyse.gauss_filter_multidimensional(r, sigma=5)
plot.show_image(r_gauss-1, colorbar_label="Pearson $r - 1$", saveto=save_to_maps)
# camera.plot_gauss_maps(r_raw, colorbar_label="Pearson $r$", saveto=save_to_maps)
print(f"Saved maps of RAW Pearson r to '{save_to_maps}'")

# Make an RGB histogram of the RAW r values
xmax = 1.
plt.figure(figsize=(3, 3))
plt.hist(r.ravel(), bins=250)
plt.xlabel("Pearson $r$")
plt.ylabel("Frequency")
plt.title("Linearity per pixel")
plot._saveshow(save_to_histogram_RGB)
# camera.plot_histogram_RGB(r_raw, xmax=xmax, xlabel="Pearson $r$", saveto=save_to_histogram_RGB)
print(f"Saved RGB histogram to '{save_to_histogram_RGB}'")
