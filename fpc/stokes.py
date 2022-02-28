"""
Stokes/Mueller calculus.
"""
import numpy as np
import polanalyser as pa  # https://github.com/elerac/polanalyser

# Order of the polariser filters on the Blackfly camera
filter_angles = np.array([0, 45, 90, 135])  # Degrees
filter_angles_rad = np.deg2rad(filter_angles)  # Radians


def demosaick_RGB(img):
    """
    Demosaick an RGB polarised image.
    Data masks are propagated.
    """
    # Demosaic the data
    img_demosaicked = pa.demosaicing(img, code="COLOR_PolarRGB")

    # If the image was masked, extend its mask to the new dimensions and apply it to the demosaicked data
    if isinstance(img, np.ma.MaskedArray):
        mask_extended = img.mask[...,np.newaxis,np.newaxis].repeat(3, axis=2).repeat(4, axis=3)
        img_demosaicked = np.ma.MaskedArray(data=img_demosaicked, mask=mask_extended)

    return img_demosaicked


def convert_demosaicked_image_to_stokes(img_demosaicked, filters=filter_angles_rad, **kwargs):
    """
    Calculate the linear Stokes parameters (IQU, not normalised) for each pixel in a demosaicked image.
    Data masks are propagated.
    """
    img_stokes = pa.calcStokes(img_demosaicked, filters)

    # If the demosaicked image was masked, re-shape its mask to the new dimensions and apply it to the demosaicked data
    if isinstance(img_demosaicked, np.ma.MaskedArray):
        mask_extended = img_demosaicked.mask[...,:3]
        img_stokes = np.ma.MaskedArray(data=img_stokes, mask=mask_extended)
    return img_stokes


def convert_stokes_to_lp(img_stokes, **kwargs):
    """
    Calculate the intensity (I), degree of linear polarisation (DoLP), and angle of linear polarisation (AoLP) for each pixel in a Stokes vector image.
    Data masks are propagated.
    """
    img_intensity = pa.cvtStokesToIntensity(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes)
    img_AoLP = pa.cvtStokesToAoLP(img_stokes)
    img_AoLP = np.rad2deg(img_AoLP)
    # These functions all propagate masks already, so we need not do anything else

    return img_intensity, img_DoLP, img_AoLP
