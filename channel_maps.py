"""
Channel maps from data cubes.

Created: August 8, 2023
Updated: September 27, 2023 to make it more general
"""
__author__ = "Ramsey Karim"

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.convolution import Box1DKernel
from spectral_cube import SpectralCube


# file paths
# Data directory
default_data_file_directory = "/home/ramsey/Downloads"
# Wherever the figures go
default_figure_save_path = "/home/ramsey/Downloads" # 2023-08-10

data_filepaths = { # Original data
    'rcw120': os.path.join(default_data_file_directory, "rcw120-cii-20arcsec-0-5kms.fits"),
    'rcw49': os.path.join(default_data_file_directory, "rcw49-cii-25arcsec-1kms.fits"),
    'ngc1977': os.path.join(default_data_file_directory, "ngc1977-data.fits"),
}

default_data_rebinned_filepaths = { # Rebinned data (after running rebin_spectra)
    'rcw120': os.path.join(default_data_file_directory, "rcw120-cii-20arcsec-0-5kms.rebin2kms.fits"),
    'rcw49': os.path.join(default_data_file_directory, "rcw49-cii-25arcsec-1kms.rebin2kms.fits"),
    'ngc1977': os.path.join(default_data_file_directory, "ngc1977-data.rebin1kms.fits"),
}

default_velocity_ranges = {
    # (low, high, step)
    # low, high are low and high velocity limits in km/s
    # These are inclusive limits
    # step = channel width in km/s
    'ngc1977': (5, 16, 1), # 12 channels
    'rcw120': (-30, 10, 2), # 21 channels
    'rcw49': (-30, 30, 2), # 31 channels
}

default_grid_shape = {
    'ngc1977': (3, 4), # 12 channels
    'rcw120': (4, 6), # 21 channels
    'rcw49': (4, 8), # 31 channels
}
default_figsize = {
    # in inches; fine tune these to remove gaps between the channel maps
    'ngc1977': (12, 8),
    'rcw120': (16, 10),
    'rcw49': (17, 11.5)
}
default_vlims = {
    'ngc1977': dict(vmin=0, vmax=40),
    'rcw120': dict(vmin=0, vmax=25),
    'rcw49': dict(vmin=0, vmax=17)
}


"""
Text/label config for the test regions:
text_x = 0.05 if source=='rcw49' else 0.5
tick_labelrotation = 50 if source=='rcw49' else 25
tick_labelpad = 26 if source=='rcw49' else 13
ha = 'left' if source=='rcw49' else 'center'
bottom=(0.08 if source=='rcw49' else 0.06)
"""


def rebin_spectra(source):
    """
    August 10, 2023
    Smooth and rebin the spectra so that channels are sampled more coarsely.
    This step takes a while, so best to just do it once and save the result.

    ** This needs debugging, as I've renamed a bunch of things to make this more general and useful in my own code **

    :param source: the string label of the region (e.g. 'ngc1977', etc)
    """
    # Unpack arguments using the 'source' label; all are assumed to be in km/s
    v_lo, v_hi, new_dv = default_velocity_ranges[source]
    # Load data cube
    data_filename = os.path.abspath(data_filepaths[source])
    cube = SpectralCube.read(data_filename)
    # Check units
    try:
        # See if the cube can be converted to Kelvins easily
        cube = cube.to(u.K)
    except:
        # Check if it looks like a temperature
        old_bunit = cube.header['BUNIT']
        if "K (Ta*)" in old_bunit:
            cube._unit = u.K
            print(f"Data unit {cube.unit} assigned, based on the header BUNIT {old_bunit}.")
        else:
            # Don't bother trying to fix it, leave it alone
            print(f"Data units <{cube._unit}> aren't equivalent to Kelvins, leaving them alone")
    # Get current channel width (np.diff should return an array of all the same value, so np.mean is overkill but it doesn't  matter)
    old_dv = np.mean(np.diff(cube.spectral_axis))
    # Construct a box filter to average the channels
    # Filter width is number of channels; if rebinning from 0.1 km/s to 1 km/s, filter is 10 channels
    # Need to add km/s units to new_dv (old_dv already has units)
    filter_width = np.abs(((new_dv*u.km/u.s) / old_dv).decompose().to_value())
    # Round to nearest integer
    filter_width = np.around(filter_width, 0)
    # Make filter using astropy.convolution.Box1DKernel
    filter = Box1DKernel(filter_width)
    # Define the new spectral axis using the inclusive limits and the new channel width
    new_spectral_axis = np.arange(v_lo, v_hi+new_dv, new_dv) * u.km/u.s

    # Do the computationally intensive work
    print("Starting spectral smooth")
    cube = cube.spectral_smooth(filter)
    print("Finished spectral smooth. Starting spectral rebin.")
    cube = cube.spectral_interpolate(new_spectral_axis)
    print("Finished spectral rebin.")
    # Create savename with "rebin" and the channel width inserted before the filetype suffix
    save_filename = data_filename.replace(".fits", f".rebin{new_dv:d}kms.fits")
    cube.write(save_filename, format='fits')



def channel_maps_figure(source,
    grid_shape=None, figsize=None, vlims=None, panel_offset=0,
    **kwargs):
    """
    August 8, 2023
    Create a publication-ready channel maps figure.
    Updated Sept 27, 2023 to make this more general and useful for my code.
    :param panel_offset: how many empty subplots before the channel maps start.
        If you don't want the top left corner to have a subplot, like you want
        a sort of "indent" look to the figure, set panel_offset > 0.
        Default is 0, start in the top left. No negative numbers.
    """
    # Load data cube
    if source in default_data_rebinned_filepaths:
        # Check defaults dict
        data_filename = default_data_rebinned_filepaths[source]
        cube = SpectralCube.read(data_filename)
    elif isinstance(source, SpectralCube):
        cube = source
        if "name" not in kwargs:
            raise RuntimeError("Need to give \'name\' kwarg if source is a SpectralCube object.")
        source = kwargs["name"]
    else:
        try:
            os.path.exists(source)
            # Filepath
            data_filename = source
            cube = SpectralCube.read(data_filename)
            if "name" not in kwargs:
                raise RuntimeError("Need to give \'name\' kwarg if source is a filepath.")
            source = kwargs["name"]
        except TypeError as e:
            raise RuntimeError(f"Cannot find a source for data <{source}>") from e
    # Switch units to km/s (optional)
    cube = cube.with_spectral_unit(u.km/u.s)
    # Get WCS object for 2D spatial image
    wcs_flat = cube[0, :, :].wcs
    # Calculate pixel scale for the spatial image
    pixel_scale = wcs_flat.proj_plane_pixel_scales()[0] # gives RA, Dec scales; pick one. They're almost certainly equal, so doesn't matter
    # Get velocity limits
    if "velocity_limits" in kwargs:
        velocity_limits = kwargs["velocity_limits"]
        v_lo, v_hi = (v*u.km/u.s for v in velocity_limits)
        first_channel_idx, last_channel_idx = (cube.closest_spectral_channel(v) for v in (v_lo, v_hi))
    else:
        # Use all the channels in the saved FITS file
        v_lo, v_hi = cube.spectral_axis[0], cube.spectral_axis[-1]
        first_channel_idx = 0 # Using all channels
        last_channel_idx = cube.shape[0] - 1 # Using all channels
    print("First and last channels ", v_lo, v_hi, " at indices ", first_channel_idx, last_channel_idx)

    if figsize is None and source in figsize:
        figsize = default_figsize[source]
    fig = plt.figure(figsize=figsize)
    # Matplotlib gridspec setup so that we can have a big colorbar on the side
    # mega_gridspec will contain all the channel maps and the Axes created within it serves as an anchor for the colorbar
    gridspec_kwargs = {key: kwargs.get(key, default_val) for key, default_val in zip(("right", "left", "top", "bottom"), (0.9, 0.06, 0.98, 0.06))}
    mega_gridspec = fig.add_gridspec(**gridspec_kwargs)
    # Create a single Axes object from mega_gridspec; this will anchor the colorbar
    mega_axis = mega_gridspec.subplots()
    # Hide the bounding box for this large Axes object
    mega_axis.set_axis_off()
    # Create the channel map gridspec within the large gridspec
    if grid_shape is None:
        if source in default_grid_shape:
            grid_shape = default_grid_shape[source]
        else:
            raise RuntimeError("grid_shape required")
    gs = mega_gridspec[0,0].subgridspec(*grid_shape, hspace=0, wspace=0)
    # Memoize axes
    axes = {}
    def get_axis(index):
        # Index is 1D index of channel counting from first_channel_idx as 0.
        # In other words, index of the panel in the figure.
        # (if first_channel_idx == 0 then axis index == channel index)
        if index not in axes:
            axes[index] = fig.add_subplot(gs[np.unravel_index(index-first_channel_idx+panel_offset, grid_shape)], projection=wcs_flat)
        return axes[index]

    # Text defaults
    text_x = kwargs.get("text_x", 0.5)
    text_y = kwargs.get("text_y", 0.94)
    # ha/va are horizontal and vertical alignment
    ha = kwargs.get('ha', 'center')
    # the color I use there is from Marc's collection of colorblind-friendly colors and works well against "plasma"
    default_text_kwargs = dict(fontsize=14, color='#ff7f00', ha=ha, va='center')
    tick_labelsize = kwargs.get("tick_labelsize", 14)
    tick_labelrotation = kwargs.get("tick_labelrotation", 25)
    tick_labelpad = kwargs.get("tick_labelpad", 13)

    # Colors
    cmap = kwargs.get("cmap", "plasma") # Image colormap
    beam_patch_ec = "grey" # edge color
    beam_patch_fc = "white" # face color
    # vlims for images (min and max for image colorscales in data units)

    # Loop through channels and plot
    for channel_idx in range(first_channel_idx, last_channel_idx+1):
        velocity = cube.spectral_axis[channel_idx]
        channel_data = cube[channel_idx].to_value()

        print(first_channel_idx, channel_idx, last_channel_idx)
        ### print the [min, mean, median, max] for each panel so that we can find the best vlims (min, max) for all of them
        # print([f(channel_data) for f in (np.nanmin, np.nanmean, np.nanmedian, np.nanmax)])


        # Setup Axes
        ax = get_axis(channel_idx)
        # Remove x and y labels on individual panels (use the "super" titles)
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ss = ax.get_subplotspec()
        # Coordinate labels
        if ss.is_last_row() and ss.is_first_col():
            # Coordinates only on bottom left corner panel
            # Mess around with the rotation, position, and size of coordinate labels
            ax.coords[0].set_ticklabel(rotation=tick_labelrotation, rotation_mode='anchor', pad=tick_labelpad, fontsize=tick_labelsize, ha='right', va='top')
            ax.coords[1].set_ticklabel(fontsize=tick_labelsize)
        else:
            # If not the bottom left panel, no coordinates (panels have no space in between)
            # Hide coordinates
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
        # Plot
        vlims = {key: kwargs[key] for key in ("vmin", "vmax") if key in kwargs}
        # for key in ("vmin", "vmax"):
        #     if key in kwargs:
        #         vlims[key] = kwargs[key]
        ### For now, ignore the default dictionary; I don't think I'm using it again
        im = ax.imshow(channel_data, origin='lower', cmap=cmap, **vlims)
        # Label velocity on each panel
        ax.text(text_x, text_y, f"{velocity.to_value():.0f} {velocity.unit.to_string('latex_inline')}", transform=ax.transAxes, **default_text_kwargs)
        # Beam on every panel
        beam_patch = cube.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), pixel_scale)
        beam_patch.set(alpha=0.9, facecolor=beam_patch_fc, edgecolor=beam_patch_ec)
        ax.add_artist(beam_patch)

    # Colorbar
    # Create a space to the right of the panels using the height/location of the mega_axis as an anchor
    cbar_ax = mega_axis.inset_axes([1.03, 0, 0.03, 1])
    cbar = fig.colorbar(im, cax=cbar_ax, label='T$_{\\rm MB}$ (K)')
    ticks = {
        # 'rcw120'
    }
    # cbar.set_ticks(ticks[source])
    # Titles
    fig.supxlabel("Right Ascension")
    fig.supylabel("Declination")

    dpi = 100
    dpi_stub = "" if dpi==100 else f"_dpi{dpi}"

    fig_save_name = f"channel_maps_{source}{dpi_stub}.png"
    savefig_kwargs = {"dpi": dpi}
    if "metadata" in kwargs:
        savefig_kwargs['metadata'] = kwargs['metadata']

    figure_save_path = kwargs.get("figure_save_path", default_figure_save_path)

    fig.savefig(
        os.path.join(figure_save_path, fig_save_name),
        **savefig_kwargs
    )
    print(f"Figure saved to {os.path.join(figure_save_path, fig_save_name)}")


if __name__ == "__main__":
    # Things to run when the file is run by name (but not run when imported by another file)

    # rebin_spectra('ngc1977')
    # channel_maps_figure('ngc1977')
    channel_maps_figure('rcw49')
