"""
Walk through a folder and create NPY stacks based on the images found. This
script will walk through all the subfolders of a given folder and generate
NPY stacks one level above the lowest level found. For example, in a given file
structure `level1/level2/level3/image1.raw`, stacks will be generated at
`level1/level2/level3_mean.npy` and `level1/level2/level3_stds.npy`.

By default, the save folder is the same as the data folder, but with `images`
replaced with `stacks`.

Command line arguments:
    * `folder`: folder containing data. Any RAW images in this folder and any
        of its subfolders will be stacked, as described above.
"""

import numpy as np
from sys import argv
from spectacle import io
from os import walk, makedirs
import dofp

# Get the data folder from the command line
folder = io.path_from_input(argv)

# Pattern for the raw files
raw_pattern = "*.raw"

# Walk through the folder and all its subfolders
for tup in walk(folder):
    # The current folder
    folder_here = io.Path(tup[0])

    # The folder to save stacks to
    goal = io.replace_word_in_path(folder_here, "images", "stacks")

    # Find all RAW files in this folder
    raw_files = list(folder_here.glob(raw_pattern))
    if len(raw_files) == 0:
        # If there are no RAW files in this folder, move on to the next
        continue

    # Create the goal folder if it does not exist yet
    makedirs(goal.parent, exist_ok=True)

    # Load all RAW files
    arrs = dofp.io.load_image_blackfly_multi(raw_files)

    # Calculate and save the mean per pixel
    mean = arrs.mean(axis=0, dtype=np.float32)
    np.save(f"{goal}_mean.npy", mean)
    del mean

    # Calculate and save the standard deviation per pixel
    stds = arrs.std(axis=0, dtype=np.float32)
    np.save(f"{goal}_stds.npy", stds)
    del stds, arrs

    # Print the input and output folder as confirmation
    print(f"{folder_here}  -->  {goal}_x.npy")
