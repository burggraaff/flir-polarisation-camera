"""
File input/output for polarisation cameras.
"""
import numpy as np
import spectacle


def generate_mask(image, saturation_threshold=65000):
    """
    Generate a mask to hide saturated pixels.
    """
    where_saturated = (image > saturation_threshold)
    return where_saturated


def load_image_blackfly(filename, dtype=np.uint16, shape=(2048, 2448), mask_saturated=False, saturation_threshold=65000):
    """
    Load an image from one of the blackfly polarisation cameras.
    Returns a numpy array containing the image data.
    Saturated pixels are masked if `mask_saturated` is True.
    """
    # Load the image
    img = np.fromfile(filename, dtype=dtype).reshape(shape)

    # Mask the image if desired
    if mask_saturated:
        mask = generate_mask(img, saturation_threshold=saturation_threshold)
        img = np.ma.MaskedArray(data=img, mask=mask)
    return img


def load_image_blackfly_multi(filenames, dtype=np.uint16, shape=(2048, 2448), mask_saturated=False):
    """
    Load multiple images from one of the blackfly polarisation cameras.
    Simply loops over the filenames, loads each, and stacks them in an array.
    Masking to be implemented.
    """
    # First, create an empty array with the appropriate shape
    data = np.empty((len(filenames), *shape), dtype=dtype)

    # Now load each image and put it into the array
    for j, filename in enumerate(filenames):
        data[j] = load_image_blackfly(filename, dtype=dtype, shape=shape)

    return data
