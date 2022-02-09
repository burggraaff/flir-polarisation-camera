"""
Stokes/Mueller calculus.
"""
import numpy as np
import polanalyser as pa  # https://github.com/elerac/polanalyser
from spectacle.linearity import sRGB

# Order of the polariser filters on the Blackfly camera
filter_angles = np.array([0, 45, 90, 135])  # Degrees
filter_angles_rad = np.deg2rad(filter_angles)  # Radians


def demosaick_RGB(img, **kwargs):
    """
    Demosaick an RGB polarised image.
    """
    img_demosaicked = pa.demosaicing(img, code="COLOR_PolarRGB")
    return img_demosaicked


def convert_demosaicked_image_to_stokes(img_demosaicked, filters=filter_angles_rad, **kwargs):
    """
    Calculate the linear Stokes parameters (IQU, not normalised) for each pixel in a demosaicked image.
    """
    img_stokes = pa.calcStokes(img_demosaicked, filters)
    return img_stokes


def convert_stokes_to_lp(img_stokes, **kwargs):
    """
    Calculate the intensity (I), degree of linear polarisation (DoLP), and angle of linear polarisation (AoLP) for each pixel in a Stokes vector image.
    """
    img_intensity = pa.cvtStokesToIntensity(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes)
    img_AoLP = pa.cvtStokesToAoLP(img_stokes)
    img_AoLP = np.rad2deg(img_AoLP)

    return img_intensity, img_DoLP, img_AoLP


def convert_to_RGB_image(img, normalization=65535, gamma=2.4):
    """
    Convert RGB data (any of the Stokes parameters, DoLP/AoLP, etc.) to an RGB image.
    Applies a gamma correction for better visibility.
    """
    img_gamma = sRGB(img, normalization=normalization, gamma=gamma)
    return img_gamma
