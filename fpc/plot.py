"""
Plotting functions for polarisation cameras.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import spectacle
from spectacle.plot import _saveshow
from cmcrameri import cm as colourmaps
from . import stokes

plt.rcParams['figure.dpi'] = 300


def convert_to_RGB_image(img, normalization=65535, gamma=2.4):
    """
    Convert RGB data (any of the Stokes parameters, DoLP/AoLP, etc.) to an RGB image.
    Applies a gamma correction for better visibility.
    Data masks are propagated.
    """
    img_gamma = spectacle.linearity.sRGB(img, normalization=normalization, gamma=gamma)
    return img_gamma



def show_image(data, lims=None, label="RAW Pixel value", ax=None, saveto=None, **kwargs):
    """
    Plot a RAW image from one of the polarisation cameras.
    `data` is the input array.
    `lims` contains the min/max values of the colour bar. If None, these are estimated from percentiles.
    `saveto` is the destination. If None is given, only show the plot.
    **kwargs are passed to `plt.imshow`.
    """
    # Make plot
    newfig = (ax is None)
    if newfig:
        plt.figure(figsize=(6,6), tight_layout=True)
        ax = plt.gca()

    # Get limits for the colour bar
    if lims is None:
        vmin, vmax = spectacle.symmetric_percentiles(data.ravel(), percent=0.5)
    else:
        vmin, vmax = lims

    # Make plot
    im = ax.imshow(data, cmap=plt.cm.cividis, vmin=vmin, vmax=vmax)
    spectacle.plot.colorbar(im, label=label, location="right")

    # Save/show the plot
    if newfig:
        _saveshow(saveto)


def show_histogram(data, bins=250, xlabel="", ax=None, saveto=None, **kwargs):
    """
    Plot a histogram for a data set.
    `data` is the input array.
    `bins` is the number of bins, or the bin edges, as passed to plt.hist.
    `saveto` is the destination. If None is given, only show the plot.
    **kwargs are passed to `plt.hist`
    """
    # Make plot
    newfig = (ax is None)
    if newfig:
        plt.figure(figsize=(3,3), tight_layout=True)
        ax = plt.gca()

    # Plot histogram
    _, bin_edges, _ = ax.hist(data.ravel(), bins=bins, **kwargs)

    # Plot settings
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(xlabel)

    # Save/show the plot
    if newfig:
        _saveshow(saveto)


def show_testplot(data, lims=None, bins=250, label="RAW Pixel value", saveto=None):
    """
    Create a plot to show test results for a data set.
    This includes the image and its histogram.
    """
    # Make plot
    fig, axs = plt.subplots(ncols=2, figsize=(10,3), tight_layout=True)

    # Plot the image
    show_image(data, lims=lims, label=label, ax=axs[0])

    # Plot the histogram
    show_histogram(data, bins=bins, xlabel=label, ax=axs[1])

    # Save/show the plot
    _saveshow(saveto)


def show_intensity_dolp_aolp(img_intensity, img_dolp, img_aolp, axs=None, intensity_lims=None, dolp_lims=(0, 0.2), aolp_lims=(0, 360), colorbar_location="bottom", saveto=None, **kwargs):
    """
    Plot the intensity, DoLP, and AoLP in a column of images.
    If `axs` are supplied, use those. Otherwise, create a new figure.
    """
    # Create a new figure if necessary
    if axs is None:
        newfigure = True
        fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, tight_layout=True, figsize=(3,6))
    else:
        newfigure = False

    # Plot the intensity
    if intensity_lims is None:
        if isinstance(img_intensity, np.ma.MaskedArray):  # np.nanpercentile does not work well with masked arrays, they need to be compressed first
            intensity_lims = spectacle.symmetric_percentiles(img_intensity.compressed(), percent=0.5)
        else:
            intensity_lims = spectacle.symmetric_percentiles(img_intensity, percent=0.5)
    vmin, vmax = intensity_lims
    im = axs[0].imshow(img_intensity, cmap=plt.cm.cividis, vmin=vmin, vmax=vmax, **kwargs)
    spectacle.plot.colorbar(im, label="Intensity [ADU]", location=colorbar_location)

    # Plot the DoLP
    vmin, vmax = dolp_lims
    im = axs[1].imshow(img_dolp, cmap=plt.cm.cividis, vmin=vmin, vmax=vmax, **kwargs)
    spectacle.plot.colorbar(im, label="DoLP", location=colorbar_location)

    # Plot the AoLP
    vmin, vmax = aolp_lims
    im = axs[2].imshow(img_aolp, cmap=colourmaps.romaO, vmin=vmin, vmax=vmax, **kwargs)
    spectacle.plot.colorbar(im, label="AoLP [degrees]", location=colorbar_location)

    # Remove ticks and labels on the x and y axes (since these are images)
    for ax in axs:
        ax.tick_params(axis="both", left=False, labelleft=False, bottom=False, labelbottom=False)

    # If this was a new figure, save/show it
    if newfigure:
        _saveshow(saveto)


def show_intensity_dolp_aolp_RGB_separate(img_intensity_RGB, img_dolp_RGB, img_aolp_RGB, title="", saveto=None, **kwargs):
    """
    Plot the intensity, DoLP, and AoLP for an RGB image in three columns.
    """
    # Create a figure
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, tight_layout=True, figsize=(9,9))

    # Plot each channel into one column
    img_intensity_RGB, img_dolp_RGB, img_aolp_RGB = [np.moveaxis(imgs_RGB, -1, 0) for imgs_RGB in (img_intensity_RGB, img_dolp_RGB, img_aolp_RGB)]  # Make the RGB channel the first dimension
    for axs_column, img_intensity, img_dolp, img_aolp, label in zip(axs.T, img_intensity_RGB, img_dolp_RGB, img_aolp_RGB, "RGB"):
        show_intensity_dolp_aolp(img_intensity, img_dolp, img_aolp, axs=axs_column, **kwargs)
        axs_column[0].set_title(label)

    # Add a title if one was provided
    if title:
        fig.suptitle(title)

    # Save/show the result
    _saveshow(saveto)


def show_intensity_dolp_aolp_RGB(img_intensity_RGB, img_dolp_RGB, img_aolp_RGB, title="", saveto=None, **kwargs):
    """
    Plot the intensity, DoLP, and AoLP for an RGB images in three colour panels.
    """
    # Create a figure
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, tight_layout=True, figsize=(10,5))

    # Normalise and show RGB image
    img_intensity = convert_to_RGB_image(img_intensity_RGB).astype(np.uint8)
    img_dolp = convert_to_RGB_image(img_dolp_RGB, normalization=1).astype(np.uint8)
    img_aolp = convert_to_RGB_image(img_aolp_RGB, normalization=360).astype(np.uint8)

    # Show each image
    for ax, img, subtitle in zip(axs, [img_intensity, img_dolp, img_aolp], ["Radiance", "DoLP", "AoLP"]):
        ax.imshow(img)
        ax.set_title(subtitle)

    # Add a title if one was provided
    if title:
        fig.suptitle(title)

    # Save/show the result
    _saveshow(saveto)
