"""
Analyse linearity data stacks.
Script taken from https://github.com/monocle-h2020/camera_calibration/blob/master/analysis/linearity_raw.py

Call signature:
    python linearity.py E:/linearity/stacks/
"""
import numpy as np
from matplotlib import pyplot as plt
from spectacle import plot, linearity as lin, io, symmetric_percentiles
from pathlib import Path
from sys import argv

# Temporary
saturation = 0.95*(2**16 - 1)

# Get the filename from the command line
folder = Path(argv[1])

# Save locations
savefolder = Path("results")
save_to_result = savefolder/"linearity_raw.npy"

# Load the data
intensities_with_errors, means = io.load_means(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Loaded data")

# Calculate the Pearson r value for each pixel
print("Calculating Pearson r...", end=" ", flush=True)
r, saturated = lin.calculate_pearson_r_values(intensities, means, saturate=saturation)
print("... Done!")

# Save the results
np.save(save_to_result, r)
print(f"Saved results to '{save_to_result}'")
