"""
Simple data processing for images from the RGBG polarisation camera.
Demosaicking is done by splitting the image into its components.

Call signature:
    python process_RGBG_multiple.py my_folder/
"""
from sys import argv
from pathlib import Path
from os import mkdir
import numpy as np
from matplotlib import pyplot as plt
import spectacle
import fpc

# Get the filename from the command line
folder = Path(argv[1])
filenames = list(folder.glob("*.raw"))

# Where to save the results
saveto = Path("E:/processed/") / folder.stem
try:
    mkdir(saveto)
except FileExistsError:
    pass

# Loop over the filenames
for filename in filenames[::100]:
    print(filename)
    label = filename.stem

    # Load the RAW file as an array
    img = fpc.io.load_image_blackfly(filename, mask_saturated=True)

    # Demosaicking
    img_demosaicked = fpc.stokes.demosaick_RGB(img)  # Dimensions: [x, y, RGB, Polarisers]
    # img_demosaicked = np.moveaxis(img_demosaicked, (0, 1), (-2, -1))  # Move image axes to the end

    # Calculate the Stokes vector
    img_stokes = fpc.stokes.convert_demosaicked_image_to_stokes(img_demosaicked)  # Dimensions: [x, y, RGB, IQU]
    img_intensity, img_dolp, img_aolp = fpc.stokes.convert_stokes_to_lp(img_stokes)

    # Separate the G images out
    G_intensity, G_dolp, G_aolp = img_intensity[..., 1], img_dolp[..., 1], img_aolp[..., 1]

    # Show the result
    fpc.plot.show_intensity_dolp_aolp(G_intensity, G_dolp, G_aolp, saveto=saveto/f"{label}_G.png")
    # fpc.plot.show_intensity_dolp_aolp_RGB_separate(img_intensity, img_dolp, img_aolp, title=label, saveto=saveto/f"{label}.png")
    # fpc.plot.show_intensity_dolp_aolp_RGB(img_intensity, img_dolp, img_aolp, title=label, saveto=saveto/f"{label}_RGB.png")


# # Extra plot
# img_intensity_gamma = fpc.plot.convert_to_RGB_image(img_intensity).astype(np.uint8)

# # Crop images
# # s = np.s_[:, :, :]
# s = np.s_[550:, :1700]
# sigma = 3

# # Create a figure
# fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, tight_layout=True, figsize=(5.3, 5.3))

# # Plot RGB image
# axs[0,0].imshow(img_intensity_gamma[s])
# axs[0,0].set_title("Protective foam")

# # Show G-band radiance, dolp, aolp
# intensity = spectacle.gauss_filter_multidimensional(img_intensity[...,1][s], sigma=sigma)
# im = axs[0,1].imshow(intensity, cmap="cividis", vmin=0, vmax=np.nanmax(intensity))
# fpc.plot.spectacle.plot.colorbar(im, label="Radiance [a.u.]", location="top")

# dolp = spectacle.gauss_filter_multidimensional(img_dolp[...,1][s], sigma=sigma)
# im = axs[1,0].imshow(dolp, cmap="cividis", vmin=0, vmax=0.2)
# fpc.plot.spectacle.plot.colorbar(im, label="$P_L$", location="bottom")

# aolp = spectacle.gauss_filter_multidimensional(img_aolp[...,1][s], sigma=sigma)
# im = axs[1,1].imshow(aolp, cmap=fpc.plot.colourmaps.romaO, vmin=0, vmax=180)
# cbar = fpc.plot.spectacle.plot.colorbar(im, label=r"$\phi_L$ [$\degree$]", location="bottom")
# cbar.set_ticks(np.arange(0,181,45))

# # Finalise
# for ax in axs.ravel():
#     ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

# plt.savefig("BlackFly.pdf", dpi=400)
# plt.show()
# plt.close()
