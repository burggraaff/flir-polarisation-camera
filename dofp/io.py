"""
File input/output for polarisation cameras.
"""
import numpy as np
import spectacle


def load_image_blackfly(filename, dtype=np.uint16, shape=(2048, 2448)):
    """
    Load an image from one of the blackfly polarisation cameras.
    Returns a numpy array containing the image data.
    """
    img = np.fromfile(filename, dtype=dtype).reshape(shape)
    return img


def load_image_blackfly_multi(filenames, dtype=np.uint16, shape=(2048, 2448)):
    """
    Load multiple images from one of the blackfly polarisation cameras.
    Simply loops over the filenames, loads each, and stacks them in an array.
    """
    # First, create an empty array with the appropriate shape
    data = np.empty((len(filenames), *shape), dtype=dtype)

    # Now load each image and put it into the array
    for j, filename in enumerate(filenames):
        data[j] = load_image_blackfly(filename, dtype=dtype, shape=shape)

    return data
