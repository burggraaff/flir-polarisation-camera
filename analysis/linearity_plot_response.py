"""
Plot the camera response at various incoming intensities for the central pixels
in a stack of images.

Script from https://github.com/monocle-h2020/camera_calibration/blob/master/analysis/linearity_plot_response.py

Command line arguments:
    * `folder`: the folder containing linearity data stacks. These should be
    NPY stacks taken at different exposure conditions, with the same ISO speed.
"""

import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin
from pathlib import Path
from matplotlib import pyplot as plt

# Get the data folder from the command line
folder = io.path_from_input(argv)

# # Load Camera object
# camera = io.load_camera(root)
# print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = Path("results")

# Load the data
intensities_with_errors, means = io.load_means(folder, retrieve_value=lin.filename_to_intensity)
intensities_with_errors, stds = io.load_stds(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Loaded data")

# Plot the data for one pixel
m = means[:, 700, 700]
s = stds[:, 700, 700]
best_fit = np.polyfit(intensities, m, 1)
xfit = [0, 5000]
best_fit_line = np.polyval(best_fit, xfit)
plt.errorbar(intensities, m, yerr=s, fmt="o", c='k')
plt.plot(xfit, best_fit_line, c='k')
plt.xlim(0, intensities.max()*1.05)
plt.ylim(0, m.max()*1.05)
plt.grid(ls="--")
plt.xlabel("Exposure time [$\mu$s]")
plt.ylabel("Camera reponse [ADU]")
plt.title("Linearity")
plot._saveshow(savefolder/"camera_response.pdf")
