"""
A designated place for M16 entire region / bubble related procedures and images
Descended from m16_investigation and m16_pictures and m16_deepdive


Created: April 25, 2023
Just circulated the M16 pillars paper and jumping into the next project before
my keyboard gets cold.
Starting with some moment 0 and PV diagrams of the entire region in CII and
CO(3-2). Maybe even the CO(1-0) that I found online
"""
__author__ = "Ramsey Karim"

# All imports dumped from the last file, m16_deepdive
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 13}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import sys
import os
import glob
import datetime
import time
import warnings

# from math import ceil
# from scipy import signal
# from scipy.interpolate import UnivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
# from astropy.table import Table, QTable
from astropy import constants as const

import pandas as pd
# from io import StringIO

# Lord forgive me
from . import crosscut
pvdiagrams = crosscut.pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils
reproject_interp = pvdiagrams.reproject_interp
pvextractor = pvdiagrams.pvextractor
regions = pvdiagrams.regions
SpectralCube = pvdiagrams.SpectralCube
Cutout2D = pvdiagrams.Cutout2D

from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

from .mantipython.physics import greybody, dust, instrument


# vel_stub is for plotting (contains spaces and special characters)
make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit}"
# simple_vel_stub is for filenames (no spaces or special characters)
make_simple_vel_stub = lambda x : ".".join(f"{y.to_value():.1f}" for y in x)
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

"""
File i/o and other useful, reusable things.
Dictionaries/lists of constants/names and helper functions that use them.
"""

large_map_filenames = {
        'cii': "sofia/m16_CII_final_15_5_0p5_clean.fits",
        '12co10': "purplemountain/G17co12.fits", '13co10': "purplemountain/G17co13.fits",
        'c18o10': "purplemountain/G17c18o.fits",
}
herschel_path = "herschel/anonymous1603389167/1342218995/level2_5"
photometry_filenames = {
    '8um': "spitzer/SPITZER_I4_mosaic_ALIGNED.fits", '250um': "extdPSW/hspirepsw696_25pxmp_1823_m1335_1342218995_1342218996_1462565962570.fits.gz",
    '350um': "extdPMW/hspirepmw696_25pxmp_1823_m1335_1342218995_1342218996_1462565960474.fits.gz",
    '500um': "extdPLW/hspireplw696_25pxmp_1823_m1335_1342218995_1342218996_1462565958982.fits.gz",
    '70um': "HPPJSMAPB/hpacs_25HPPJSMAPB_blue_1822_m1337_00_v1.0_1471714532334.fits.gz",
    '160um': "HPPJSMAPR/hpacs_25HPPJSMAPR_1822_m1337_00_v1.0_1471714553094.fits.gz",
}
photometry_beams = {
    '8um': (1.98, 1.98, 0), '70um': (9.0, 5.75, 62),
    '160um': (13.32, 11.31, 40.9), '250um': (18.4, None, 0),
    '350um': (25.2, None, 0), '500um': (36.7, None, 0),
}
cutout_box_filenames = {
    'N19': "N19_cutout_box.reg", # Small, only N19
    'med-large': "m16_cutout_box_medium-large.reg", # sort of like the Hill 2012 footprint, includes some of the filament. Goes well outside CII
    'med': "m16_cutout_box_medium.reg", # CII and CO (3-2) footprint, aligned in equatorial (so there will be NaN gaps)
}
vlim_memo = { # hash things somehow and put them here
    '8um': (40, 320),
    'cii.levels': np.concatenate([np.arange(2.5, 61, 5), np.arange(65, 126, 15)]), 'cii.generic': (0, 65),
    '12co32.levels': np.arange(2.5, 51, 5), '12co32.generic': (0, 40),
    '13co32.levels': np.arange(1, 27, 2.5), '13co32.generic': (0, 18),

    # '13co10.levels': np.arange(0.25, 5, 0.5), '13co10.generic': (0, 4), # these are good for large velocity intervals
    '13co10.levels': np.arange(1, 22, 2), '13co10.generic': (0, 20), # small velocity intervals (like 1 km/s)

    '12co10.levels': np.arange(5, 55, 5), '12co10.generic': (0, 40),
    '250um': (140, 4500), 'irac1': (1, 30), '70um': (-0.06, 3.5), # 70um can also do vmax=1.5 for greater sensitivity to low emission
    '160um': (-0.1, 2.5), '500um': (50, 500), '500um.levels': np.arange(200, 2001, 100), '500um.generic': (150, 500),
    'irac2': (1.5, 15), 'irac3': (10, 130),
}
default_reg_filename_list = [ # commonly used region filenames
    "catalogs/N19_points_along_path_1.reg", "catalogs/m16_up_points_along_path.reg",
    "catalogs/m16_across_pillars_points_along_path.reg", "catalogs/spire_up_points_along_path.reg",
    "catalogs/m16_across_points_along_path.reg",
]

def vlim_hash(data_stub, velocity_limits=None, generic=False):
    """
    May 3, 2023
    Make keys for vlim_memo. This is to make image creation very easy
    Ignore 'APEX' or 'CONV' suffix; this rarely makes a difference.
    """
    data_stub = data_stub.replace('CONV', '').replace('APEX', '')
    if velocity_limits is None:
        return data_stub
    elif generic:
        # Special case for requesting the "generic" hash (separate from no velocity hash)
        return data_stub + ".generic"
    else:
        unit_clean = lambda i : f"{velocity_limits[i].to(kms).to_value():0.2f}"
        return f"{data_stub}.{unit_clean(0)}.{unit_clean(1)}"

def get_vlim(key):
    """
    May 3, 2023
    Retrieve vlims based on a hashed key
    Only returns the vmin/vmax if it's not None in the memo dict
    Returns a dict(vmin=x, vmax=y) (with no, either, or both keys)
    Empty dict if we haven't memoized
    Ignore 'APEX' or 'CONV' suffix; this rarely makes a difference.
    """
    key = key.replace('CONV', '').replace('APEX', '')
    vlims = {}
    vlims_keys = ['vmin', 'vmax']
    if key in vlim_memo:
        val = vlim_memo[key]
        for k, v in zip(vlims_keys, val):
            if v:
                vlims[k] = v
    # special rule (can remove later): if no vmin but velocity specified (e.g. moment map), vmin=0
    if 'vmin' not in vlims and '.' in key.replace('.generic', ''):
        # since the '.generic' tag has a '.' in it, filter that out
        vlims['vmin'] = 0
    return vlims

def get_levels(data_stub):
    """
    May 4, 2023
    Try to find memoized contour levels
    """
    data_stub = data_stub.replace('CONV', '').replace('APEX', '')
    key = data_stub + ".levels"
    return vlim_memo.get(key, None)

def get_generic_vlim(data_stub):
    """
    May 4, 2023
    Check for general use catch-all vlims in the memoization list
    Uses get_vlim but only uses data_stub without the velocity hashed in
    Returns dictionary with possible keys (vmin, vmax), same as get_vlim
    """
    key = data_stub + ".generic"
    return get_vlim(key)

def get_map_filename(data_stub, beam=None):
    """
    May 2, 2023
    Shortcut function to get cube filenames and stuff

    Adding onto this to handle photometry since some of it is easier to do programmatically rather than with a dictionary.
    Photometry will be returned as a single-element tuple (str(full path),) to signal that it is photometry and not a cube
    """
    if data_stub in photometry_filenames:
        if 'um' in data_stub and (70 <= int(data_stub.replace('um', '')) <= 500):
            return (os.path.join(herschel_path, photometry_filenames[data_stub]),)
        else:
            return (photometry_filenames[data_stub],)
    elif data_stub[:4] == 'irac':
        # irac1, irac2, irac3. Stick to '8um' instead of irac4, though it will work either way
        return (f"spitzer/SPITZER_I{data_stub[-1]}_mosaic_ALIGNED.fits",)
    # At this point, must be a cube
    elif data_stub in large_map_filenames:
        fn = large_map_filenames[data_stub]
        if beam == 'APEX':
            return fn.replace('.fits', '.APEXbeam.fits')
        else:
            return fn
    elif 'APEX' in data_stub and data_stub.replace('APEX', '') in large_map_filenames:
        # Special case to avoid 'beam' keyword and select via data_stub
        return large_map_filenames[data_stub.replace('APEX', '')].replace('.fits', '.APEXbeam.fits')
    else:
        # Use the cube_utils default
        return data_stub

def get_data_name(data_stub):
    if data_stub in cube_utils.cubenames:
        return cube_utils.cubenames[data_stub]
    elif data_stub[:4] == 'irac':
        return str([3.6, 4.5, 5.8, 8][int(data_stub[-1]) - 1]) + " $\mu$m"
    else:
        warnings.warn(f"unrecognized data_stub <{data_stub}>, submitting empty string \"\" as name")
        return ""

def get_2d_map(data_stub, velocity_limits=None, average_Tmb=False, data_memo=None):
    """
    May 3, 2023
    Helper function for overlay_moment: create maps.
    data_stub can refer to either an image or a cube. If it's a cube, velocity_limits
    should be supplied and will be used to make a moment. If they aren't,
    entire range is used.
    Returns a tuple (data_array, info_dict)
    data_array will be a 2d numpy array (not quantity)
    info_dict will contain all the necessary metadata, like WCS, header (if applicable), Unit, etc.
    (this way I can add functionality as I go and not have to rewrite things constantly)
    :param average_Tmb: if False, moment 0s are returned in K km/s. If True,
        the width of the velocity interval is divided out to get an average.
        This is useful for stabilizing color limits across moments from different velocity ranges.
    :param data_memo: memoization dictionary to cut down on reloads of the same data.
        keys are the vlim_hash keys.
    """
    info_dict = {}
    data_filename = get_map_filename(data_stub)
    if isinstance(data_filename, tuple):
        # Image
        # First check for memoization
        info_dict['vlim_hash'] = vlim_hash(data_stub)
        if data_memo and info_dict['vlim_hash'] in data_memo:
            img, info_dict = data_memo[info_dict['vlim_hash']]
            return img, info_dict
        # Not memoized, continue loading
        data_filename, = data_filename
        img, hdr = fits.getdata(catalog.utils.search_for_file(data_filename), header=True)
        info_dict['wcs'] = WCS(hdr)
        info_dict['hdr'] = hdr
        if 'BUNIT' in hdr:
            # If this throws an error I'll cross that bridge when I get to it
            info_dict['unit'] = u.Unit(hdr['BUNIT'])

        # Beam lookup (the images never have beams in headers...)
        if data_stub[:4] == 'irac':
            beam_key = '8um' # they're all the same
        else:
            beam_key = data_stub
        if beam_key in photometry_beams:
            major, minor, pa = photometry_beams[beam_key]
            beam_params = dict(major=major*u.arcsec, pa=pa*u.deg)
            if minor is not None:
                beam_params['minor'] = minor*u.arcsec
            info_dict['beam'] = cube_utils.Beam(**beam_params)
        else:
            # No beam!
            pass

        info_dict['obs_type'] = 'image'
        return img, info_dict
    else:
        # Cube
        # First check for memoization
        info_dict['vlim_hash'] = vlim_hash(data_stub, velocity_limits=velocity_limits)
        if data_memo and info_dict['vlim_hash'] in data_memo:
            img, info_dict = data_memo[info_dict['vlim_hash']]
            return img, info_dict
        # Not memoized, continue loading, need to make moment
        cube_obj = cube_utils.CubeData(data_filename).convert_to_K().convert_to_kms()
        if velocity_limits is not None:
            cube = cube_obj.data.spectral_slab(*velocity_limits)
        else:
            cube = cube_obj.data
        mom0 = cube.moment0()
        if average_Tmb:
            # Divide out the velocity interval to get average temperature
            if velocity_limits is not None:
                dv = velocity_limits[1] - velocity_limits[0]
            else:
                dv = np.abs(cube_obj.data.spectral_axis[-1] - cube_obj.data.spectral_axis[0])
            mom0 = (mom0 / dv).decompose()
        info_dict['wcs'] = cube_obj.wcs_flat
        info_dict['unit'] = mom0.unit
        info_dict['beam'] = cube_obj.data.beam
        info_dict['obs_type'] = 'cube'
        return mom0.to_value(), info_dict

def trim_values_to_vlims(data, vmin=None, vmax=None):
    """
    May 4, 2023
    Trim data values to vmin, vmax. Skips them if they aren't present.
    Makes a copy of data (unless vmin, vmax are both None)
    """
    if vmin is None and vmax is None:
        # Short circuit to avoid copying
        return data
    # Do not modify the original (just in case)
    data = data.copy()
    if vmin is not None:
        data[data < vmin] = vmin
    if vmax is not None:
        data[data > vmax] = vmax
    return data

def get_cutout_box_filename(cutout_box_stub):
    """
    May 9, 2023
    Getter function for the cutout box region filenames.
    Returns the absolute file path.
    """
    return catalog.utils.search_for_file("catalogs/" + cutout_box_filenames[cutout_box_stub])


def get_pv_and_regions(reg_filename_or_idx):
    """
    May 11, 2023
    Load up to 1 vector as a PV path and 0 or more non-vector sky regions.
    Returns (pv_path, reg_list, reg_fn_stub)
    If no vector, pv_path is None.
    If no regions, reg_list is empty list.
    If no list at all (i.e. reg_filename_or_idx is None), then both of the above
    and reg_fn_stub is empty string. reg_fn_stub is for record keeping, goes in
    png metadata or image save filenames.

    path_idx will apply only to vectors; has no effect if no vectors

    """
    if reg_filename_or_idx is None:
        # Short circuit and return the default empty values
        return (None, [], "")
    # Some regions have been specified
    if isinstance(reg_filename_or_idx, int):
        # Index into existing list; all these only have 1 vector
        reg_filename_short = default_reg_filename_list[reg_filename_or_idx]
        # First (only) vector
        path_idx = 0
    elif isinstance(reg_filename_or_idx, tuple):
        # Tuple of filename, path_idx
        reg_filename_short, path_idx = reg_filename_or_idx
    elif isinstance(reg_filename_or_idx, str):
        # Filename, assume first (or only) path in file
        reg_filename_short = reg_filename_or_idx
        path_idx = 0
    else:
        raise RuntimeError(f"Unknown path specifier {reg_filename_or_idx}")
    try:
        reg_filename = catalog.utils.search_for_file(reg_filename_short)
    except Exception as e:
        raise RuntimeError(f"Issue with path specifier {reg_filename_or_idx}") from e
    # Regions.read will return an empty list if no plottable regions (i.e. vectors only)
    reg_list = regions.Regions.read(reg_filename)
    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=path_idx)
    reg_fn_stub = os.path.basename(reg_filename_short).replace('.reg', '')
    if path_idx != 0:
        # Indicate path index if it's not 0
        reg_fn_stub += f"-p{path_idx}"
    return (pv_path, reg_list, reg_fn_stub)


"""
Image creation functions below here
"""

def manual_pv_slice_series():
    """
    April 25, 2023
    I wanna see if I can just plot array slices for PV diagrams since pvextractor takes a while on large/many slices
    Ok yes this works, WCS likes to throw lots of warnings about it but it is ultimately fine

    I can expand this out to PV movies along RA and Dec (non-aligned paths will need to be done with pvextractor still)
    but this will take more time than I have tonight before the meeting tomorrow

    April 26, 2023: trying to expand this into movies
    """

    """
    PV cut orientation, vertical or horizontal
    Vertical means slice at a single RA and plot velocity vs Dec
    Horizontal means slice at a single Dec and plot velocity vs RA
    """
    orientation = 'horizontal'
    start_idx, step_idx = 25, 50

    # Load cube
    line_stub = 'cii'
    if line_stub in large_map_filenames:
        # Use the custom filename rather than the default
        filename = large_map_filenames[line_stub]
    else:
        # Use default filename from cube_utils (many of these are centered around Pillars)
        filename = line_stub
    cube_obj = cube_utils.CubeData(filename).convert_to_K().convert_to_kms()
    dimension_size = (cube_obj.data.shape[2] if orientation=='vertical' else cube_obj.data.shape[1])

    # Make image
    ref_vel_lims = (10*kms, 35*kms)
    ref_mom0 = cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
    ref_img = ref_mom0.to_value()

    # Set colors
    pv_cmap = 'plasma'
    img_cmap = 'Greys_r'
    line_color = marcs_colors[1]

    # Loop thru slice index
    for slice_idx in range(start_idx, dimension_size, step_idx):

        if orientation == 'vertical':
            # Cube index order is V,Y,X = Velocity,Dec,RA = V,I,J
            cube_slices = (slice(None), slice(None), slice_idx)
        else:
            cube_slices = (slice(None), slice_idx, slice(None))

        pv_slice = cube_obj.data[cube_slices]

        # First try to remake fig/axes each time. Try persistent if slow
        fig = plt.figure(figsize=(8, 10))
        gs = fig.add_gridspec(2, 1)
        ax_img = fig.add_subplot(gs[0,0], projection=cube_obj.wcs_flat)
        ax_pv = fig.add_subplot(gs[1,0], projection=pv_slice.wcs)

        im = ax_img.imshow(ref_img, origin='lower', vmin=0, cmap=img_cmap)
        fig.colorbar(im, ax=ax_img, label=ref_mom0.unit.to_string('latex_inline'))

        im = ax_pv.imshow(pv_slice.to_value(), origin='lower', vmin=0, cmap=pv_cmap)
        fig.colorbar(im, ax=ax_pv, label=pv_slice.unit.to_string('latex_inline'), orientation='horizontal')

        # Plot line
        if orientation == 'vertical':
            plot_line = ax_img.axvline
        else:
            plot_line = ax_img.axhline
        plot_line(slice_idx, color=line_color, linewidth=2)
        # Reference image velocity interval stamp
        ax_img.text(0.1, 0.9, make_vel_stub(ref_vel_lims), color=line_color, ha='left', va='bottom')

        # Clean up axes labels
        # ax_img.set_xlabel("RA")
        # ax_img.set_ylabel("Dec")
        ax_pv.coords[1].set_format_unit(kms)

        savename = f"/home/ramsey/Pictures/2023-04-26/m16_pv_{orientation}_{slice_idx:03d}.png"
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {filename}', file=__file__, func='manual_pv_slice_series'))

def m16_large_moment0():
    """
    August 25, 2023
    Make some big moment0 maps of CII and CO3-2 to showcase the interesting
    features and places for further study
    """
    # Use the big CII map. Needs some cleaning but generally much larger than the original I was using before.

    # line_stub = 'cii'
    # line_stub = '12co32'
    line_stub = 'c18o10'

    if line_stub in large_map_filenames:
        # Use the custom filename rather than the default
        filename = large_map_filenames[line_stub]
    else:
        # Use default filename from cube_utils (many of these are centered around Pillars)
        filename = line_stub

    velocity_intervals = [ # what I have for now.. will add later
        (10, 17), (17, 21), (23.5, 27), (27, 33), (30, 35),
    ]

    cube_obj = cube_utils.CubeData(filename).convert_to_K().convert_to_kms()

    # Can always remove this loop
    for i in range(len(velocity_intervals)):

        vel_lims = tuple(x*kms for x in velocity_intervals[i])
        mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0()

        fig = plt.figure()
        ax = plt.subplot(111, projection=cube_obj.wcs_flat)
        im = ax.imshow(mom0.to_value(), origin='lower', vmin=0, cmap='plasma')
        fig.colorbar(im, ax=ax, label=f"{cube_utils.cubenames[line_stub]} {mom0.unit.to_string('latex_inline')}")
        vel_stub_simple = make_simple_vel_stub(vel_lims)
        savename = f"/home/ramsey/Pictures/2023-04-25/mom0_{line_stub}_{vel_stub_simple}.png"
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))
            print("Created", os.path.dirname(savename))
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=make_vel_stub(vel_lims),
            file=__file__, func='m16_large_moment0'))

def real_easy_pv(reg_idx=0):
    """
    May 1, 2023
    Trying to stay true to the name this time. Put in a file, and index, and
    make the PV. No funny business. I'll try to dress it up nicely.
    I think m16_deepdive.easy_pv_2 is a good model.
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = None

    # line_stub = 'cii'; vmax = None # 20
    line_stub = '12co32'; vmax = None # 30
    if line_stub in large_map_filenames:
        cube_filename_short = large_map_filenames[line_stub]
    else:
        cube_filename_short = line_stub

    # reg_filename_short = "catalogs/N19_pv_paths.reg"
    reg_filename_short = "catalogs/m16_small_bubble_pvs.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    # reg_idx = 1

    vel_lims = (9*kms, 41*kms)

    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=reg_idx)
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)

    # Reload the region using regions to get its name
    pv_name = regions.Regions.read(reg_filename)[reg_idx].meta['text']

    if reference_filename_short is None:
        mom0 = cube.spectral_slab(*vel_lims).moment0()
        ref_img_wcs = mom0.wcs
        ref_img = mom0.to_value()
        ref_colorbar_label = mom0.unit.to_string('latex_inline')
    else:
        raise NotImplementedError

    pv_cmap = 'plasma'
    img_cmap = 'Greys_r'
    line_color = marcs_colors[1]

    fig = plt.figure(figsize=(8,12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_img = fig.add_subplot(gs[0,0], projection=ref_img_wcs)
    ax_pv = fig.add_subplot(gs[1,0], projection=WCS(sl.header))

    im = ax_img.imshow(ref_img, origin='lower', vmin=0, cmap=img_cmap)
    fig.colorbar(im, ax=ax_img, label=ref_colorbar_label)

    im = ax_pv.imshow(sl.data, origin='lower', vmin=0, vmax=vmax, cmap=pv_cmap, aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv, label=cube.unit.to_string('latex_inline'), orientation='horizontal')

    ax_pv.coords[1].set_format_unit(kms)
    ax_img.text(0.1, 0.9, make_vel_stub(vel_lims), color=line_color, ha='left', va='bottom')

    ax_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=line_color, linestyle='-', lw=3, transform=ax_img.get_transform('world'))
    ax_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=line_color, fontsize=10, va='bottom', ha='center', transform=ax_img.get_transform('world'))

    # 2023-05-01,02
    savename = f"/home/ramsey/Pictures/2023-05-02/m16_pv_{pv_name}_{line_stub}.png"
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_easy_pv"))

def real_easy_spectra(reg_idx=0):
    """
    May 1, 2023
    Quick look spectra for CO3-2 and CII. Keeping it real simple
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = "spitzer/SPITZER_I4_mosaic_ALIGNED.fits"

    line_stub_list = ['cii', '12co32']

    reg_filename_short = "catalogs/N19_points.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    reg = regions.Regions.read(reg_filename)[reg_idx]
    reg_name = reg.meta['text']

    ref_img, ref_hdr = fits.getdata(catalog.utils.search_for_file(reference_filename_short), header=True)
    ref_wcs = WCS(ref_hdr)

    fig = plt.figure(figsize=(8,12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_img = fig.add_subplot(gs[0,0], projection=ref_wcs)
    ax_spec = fig.add_subplot(gs[1,0])

    ax_img.imshow(ref_img, origin='lower', vmin=45, vmax=290)

    for i, line_stub in enumerate(line_stub_list):
        # Load cube
        if line_stub in large_map_filenames:
            cube_filename_short = large_map_filenames[line_stub]
        else:
            cube_filename_short = line_stub
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms()
        pixreg = reg.to_pixel(cube.wcs_flat)
        j, i = [int(round(c)) for c in pixreg.center.xy]
        spectrum = cube.data[:, i, j]

        ax_spec.plot(cube.data.spectral_axis.to_value(), spectrum.to_value(), label=f"{cube_utils.cubenames[line_stub]}")

        pixreg = reg.to_pixel(ref_wcs)
        x, y = pixreg.center.xy
        ax_img.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

    ax_spec.legend()

    # 2023-05-01
    savename = f"/home/ramsey/Pictures/2023-05-01/spectrum_{reg_name}.png"
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_easy_spectra"))

def convolve_cii_to_co32():
    """
    May 2, 2023
    Quick convolve CII to CO(3-2)
    """
    reference_cube_fn = '12co32'
    reference_cube = cube_utils.CubeData(reference_cube_fn).convert_to_K().data

    target_cube_fn = large_map_filenames['cii']
    target_cube = cube_utils.CubeData(target_cube_fn).convert_to_K().data

    savename = catalog.utils.search_for_file(target_cube_fn).replace('.fits', '.APEXbeam.fits')
    print("writing to ", savename)
    print("Convolving...")

    convolved_cube = target_cube.convolve_to(reference_cube.beam)
    convolved_cube.write(savename, format='fits')
    print("done")

def real_medium_spectra(velocity_index=0):
    """
    May 2, 2023
    Follow-up to m16_bubble.real_easy_spectra, a slightly more fine-tuned figure
    with moment images and velocity intervals and stuff like that.
    Didn't want to work it into real_easy_spectra because that should be kept
    simple and reusable.

    Going to go full on with this image. I want 3 rows: moments/images up top,
    PV in the middle, and spectra on the bottom.
    Three images up top: Spitzer for reference, then CII and CO.
    Middle, just one long PV with bars showing velocity intervals.
    Bottom, two to three spectra taken along the PV diagram. Maybe mark their
    approximate locations along the PV with vertical bars (have to approx this in DS9).
    """
    # Leave None to use the moment image, or set to an image filename
    reference_filename_short = "spitzer/SPITZER_I4_mosaic_ALIGNED.fits"

    line_stub_list = ['12co32', 'cii', '13co32']

    # global vmaxes
    pv_vmaxes = {'cii': 20, '12co32': 30}
    img_vmaxes = {'cii': [10, 14, 19, 30, 30, 25, 12],
        '12co32': [6, 21, 25, 25, 25, 19, 3]}

    # each (i, i+1) is an interval
    velocity_intervals = (10, 18, 21.5, 23.5, 26, 28, 31, 34)
    # velocity_index = 0 # current interval
    velocity_interval = tuple(velocity_intervals[velocity_index+delta_i]*kms for delta_i in (0, 1))
    full_velocity_limits = (8*kms, 35*kms)

    """ Hillenbrand Stars """
    star_df = pd.read_csv(catalog.utils.search_for_file("catalogs/hillenbrand_stars_1.csv"), index_col='ID')
    star_ra = star_df['RA'].values
    star_dec = star_df['DE'].values
    del star_df

    """
    Let the region file have 1 vector and 3 points.
    Regions will not read the vector, and pvextractor will not read the points. Perfect!
    """
    # reg_filename_short = "catalogs/N19_points_along_path_1.reg"
    # reg_filename_short = "catalogs/m16_up_points_along_path.reg"
    # reg_filename_short = "catalogs/m16_across_pillars_points_along_path.reg"
    # reg_filename_short = "catalogs/spire_up_points_along_path.reg"
    reg_filename_short = "catalogs/m16_across_points_along_path.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    point_reg_list = regions.Regions.read(reg_filename)
    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=0)

    ref_img, ref_hdr = fits.getdata(catalog.utils.search_for_file(reference_filename_short), header=True)
    ref_wcs = WCS(ref_hdr)

    fig = plt.figure(figsize=(20,16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])


    """ Image row: Spitzer, (line_1), (line_2) """
    ax_ref_img = fig.add_subplot(gs[0,0], projection=ref_wcs)
    axes_moment_imgs = {line_stub: (0, i+1) for i, line_stub in enumerate(line_stub_list[:2])}

    # ax_pv will use gs[1, :] but needs its WCS first
    ax_pv = (1, slice(None))
    pv_ref_hdr = None
    pv_ref_shape = None
    pv_cbar = None

    """ Make room for 3 spectrum locations """
    axes_spec = [fig.add_subplot(gs[2,i]) for i in range(3)]

    # Reference
    ax_ref_img.imshow(ref_img, origin='lower', vmin=45, vmax=290)

    for line_idx, line_stub in enumerate(line_stub_list):
        # Load cube
        cube_filename_short = get_map_filename(line_stub)
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K()

        # Spectrum for each location
        for reg_idx, reg in enumerate(point_reg_list):
            pixreg = reg.to_pixel(cube.wcs_flat)
            j, i = [int(round(c)) for c in pixreg.center.xy]
            spectrum = cube.data[:, i, j]
            # Only label first spectrum plot
            spectrum_kwargs = (dict(label=f"{cube_utils.cubenames[line_stub]}") if reg_idx == 0 else {})
            axes_spec[reg_idx].plot(cube.data.spectral_axis.to(kms).to_value(), spectrum.to_value(), **spectrum_kwargs)

            # Do a few things only once
            if line_idx == 0:
                # Plot the PV path on the Spitzer image
                ax_ref_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=marcs_colors[1], linestyle='-', lw=1, transform=ax_ref_img.get_transform('world'))
                ax_ref_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=marcs_colors[1], fontsize=10, va='center', ha='right', transform=ax_ref_img.get_transform('world'))

                # Plot stars
                ref_xlim = ax_ref_img.get_xlim()
                ref_ylim = ax_ref_img.get_ylim()
                ax_ref_img.scatter(star_ra, star_dec, color=marcs_colors[3], marker='*', transform=ax_ref_img.get_transform('world'))
                ax_ref_img.set_xlim(ref_xlim)
                ax_ref_img.set_ylim(ref_ylim)

                # Plot the points on the Spitzer image
                pixreg = reg.to_pixel(ref_wcs)
                x, y = pixreg.center.xy
                ax_ref_img.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

                # Plot velocity interval bars on the spectra
                for v in velocity_intervals:
                    axes_spec[reg_idx].axvline(v, color='grey', alpha=0.7, linestyle='--')
                # Highlight current interval
                for vkms in velocity_interval:
                    axes_spec[reg_idx].axvline(vkms.to_value(), color='k', alpha=0.9, linestyle='-', linewidth=2)
                axes_spec[reg_idx].axvspan(*(vkms.to_value() for vkms in velocity_interval), color='grey', alpha=0.25)
                # Plot 0
                axes_spec[reg_idx].axhline(0, color='grey', alpha=0.7, linestyle='--')
                # Set xlims
                axes_spec[reg_idx].set_xlim([vkms.to_value() for vkms in full_velocity_limits])



        # Stop this loop for 13co since we only want spectra
        if line_stub == '13co32':
            # No PV, no moment, all done
            continue
        # Keep going for cii and 12co

        """ Moment 0 """
        mom0 = cube.data.spectral_slab(*velocity_interval).moment0().to(u.K*kms)
        avg_intensity = (mom0 / (velocity_interval[1] - velocity_interval[0])).decompose()
        ax_i, ax_j = axes_moment_imgs[line_stub]
        ax_mom = fig.add_subplot(gs[ax_i, ax_j], projection=mom0.wcs)
        axes_moment_imgs[line_stub] = ax_mom
        im = ax_mom.imshow(avg_intensity.to_value(), origin='lower', vmin=0, vmax=img_vmaxes[line_stub][velocity_index])
        fig.colorbar(im, ax=ax_mom, label="Avg Tmb ("+avg_intensity.unit.to_string('latex_inline')+")")
        ax_mom.text(0.05, 0.9, cube_utils.cubenames[line_stub], fontsize=15, color='k', transform=ax_mom.transAxes)
        # Plot the PV path on the Spitzer image
        ax_mom.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=marcs_colors[1], linestyle='-', lw=1, transform=ax_mom.get_transform('world'))
        ax_mom.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=marcs_colors[1], fontsize=10, va='center', ha='right', transform=ax_mom.get_transform('world'))
        for reg in point_reg_list:
            # Plot the points on the Spitzer image
            pixreg = reg.to_pixel(mom0.wcs)
            x, y = pixreg.center.xy
            ax_mom.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

        xlim = ax_mom.get_xlim()
        ylim = ax_mom.get_ylim()
        ax_mom.scatter(star_ra, star_dec, ec=marcs_colors[3], fc='None', alpha=0.2, marker='o', transform=ax_mom.get_transform('world'))
        ax_mom.set_xlim(xlim)
        ax_mom.set_ylim(ylim)


        """ PV """
        sl_raw = pvextractor.extract_pv_slice(cube.data.spectral_slab(*full_velocity_limits), pv_path)
        if pv_ref_hdr is None:
            # Plot this PV as the image and save its header
            pv_ref_hdr = sl_raw.header
            pv_ref_shape = sl_raw.data.shape
            pv_ref_hdr['CTYPE2'] = 'VRAD'
            ax_pv = fig.add_subplot(gs[ax_pv], projection=WCS(pv_ref_hdr))
            im = ax_pv.imshow(sl_raw.data, origin='lower', cmap='plasma', vmin=0, vmax=pv_vmaxes[line_stub], aspect=(sl_raw.data.shape[1]/(3*sl_raw.data.shape[0])))
            cs = ax_pv.contour(sl_raw.data, colors='k', linewidths=1, levels=np.arange(5, 41, 5))
            pv_cbar = fig.colorbar(im, ax=ax_pv, label=cube.data.unit.to_string('latex_inline'))
            for l in cs.levels:
                pv_cbar.ax.axhline(l, color='grey')
            ax_pv.text(0.05, 0.95, cube_utils.cubenames[line_stub], fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)

            ax_pv.coords[1].set_format_unit(u.km/u.s)
            ax_pv.coords[1].set_major_formatter('x.xx')
            ax_pv.coords[0].set_format_unit(u.deg)
            ax_pv.coords[0].set_major_formatter('x.xx')

        else:
            """ Stupid bug fix! I don't think this step should be necessary but it appears to be necessary """
            sl_raw.header['RESTFRQ'] = pv_ref_hdr['RESTFRQ']
            sl_reproj = reproject_interp((sl_raw.data, sl_raw.header), WCS(pv_ref_hdr), shape_out=pv_ref_shape, return_footprint=False)
            cs = ax_pv.contour(sl_reproj, cmap='viridis_r', linewidths=1.5, levels=np.arange(3, 37, 4), vmax=26)
            for l in cs.levels:
                pv_cbar.ax.axhline(l, color='white', linewidth=2)
            ax_pv.text(0.05, 0.90, cube_utils.cubenames[line_stub] + ' (c)', fontsize=10, color='white', va='top', ha='left', transform=ax_pv.transAxes)

            # Plot velocity intervals
            x_length = pv_path._coords[0].separation(pv_path._coords[1]).deg
            for v in velocity_intervals:
                ax_pv.plot([0, x_length], [v*1e3]*2, color='grey', alpha=0.7, linestyle='--', transform=ax_pv.get_transform('world'))
            for vkms in velocity_interval:
                ax_pv.plot([0, x_length], [vkms.to_value()*1e3]*2, color='k', alpha=0.9, linestyle='-', linewidth=3, transform=ax_pv.get_transform('world'))

            # Plot region positions
            # Assume their position in arcseconds is in the reg.meta['text'] after '/'
            for reg in point_reg_list:
                offset = float(reg.meta['text'].split('/')[1]) * u.arcsec
                ax_pv.plot([offset.to(u.deg).to_value()]*2, [v.to_value()*1e3 for v in full_velocity_limits], color='k', alpha=1, linestyle=':', linewidth=4, transform=ax_pv.get_transform('world'))


    axes_spec[0].legend()
    plt.tight_layout()

    # 2023-05-02
    savename = f"/home/ramsey/Pictures/2023-05-02/mega_spectrum_{os.path.basename(reg_filename_short).replace('.reg', '')}_vidx{velocity_index:02d}.png"
    print(savename)
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {cube_filename_short}', file=__file__, func="real_medium_spectra"))

def pv_regrid_debug():
    """
    May 2, 2023
    There is some issue with PV diagram reprojecting, I need to isolate it and sort it out
    """

    reg_filename_short = "catalogs/N19_pv_paths.reg"
    reg_filename = catalog.utils.search_for_file(reg_filename_short)
    reg_idx = 1

    vel_lims = (9*kms, 41*kms)

    pv_path = pvdiagrams.path_from_ds9(reg_filename, index=reg_idx)

    # Reload the region using regions to get its name
    pv_name = regions.Regions.read(reg_filename)[reg_idx].meta['text']

    fig = plt.figure(figsize=(8,12))

    line_stub = 'cii'; vmax = 20
    cube_filename_short = line_stub#get_map_filename(line_stub)
    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    sl = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)

    ax_pv = fig.add_subplot(211, projection=WCS(sl.header))
    im = ax_pv.imshow(sl.data, origin='lower', vmin=0, vmax=vmax, cmap='plasma', aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv, label=cube.unit.to_string('latex_inline'), orientation='horizontal')



    line_stub = '12co32'; vmax = 30
    cube_filename_short = line_stub #get_map_filename(line_stub)
    cube = cube_utils.CubeData(cube_filename_short).convert_to_K().convert_to_kms().data
    sl2 = pvextractor.extract_pv_slice(cube.spectral_slab(*vel_lims), pv_path)
    sl2_header = sl2.header
    """ THIS IS THE FIX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! this is new... and not good. """
    sl2_header['RESTFRQ'] = sl.header['RESTFRQ']

    sl2_regrid = reproject_interp((sl2.data, sl2.header), WCS(sl.header), shape_out=sl.data.shape, return_footprint=False)


    ax_pv2 = fig.add_subplot(212, projection=WCS(sl.header))
    im = ax_pv2.imshow(sl2_regrid, origin='lower', vmin=0, vmax=vmax, cmap='plasma', aspect=(sl.data.shape[1]/(1.5*sl.data.shape[0])))
    fig.colorbar(im, ax=ax_pv2, label=cube.unit.to_string('latex_inline'), orientation='horizontal')


    # ax_pv.coords[1].set_format_unit(kms)

    savename = f"/home/ramsey/Pictures/2023-05-02/pv_debug.png"
    plt.show()
    # fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='debug', file=__file__, func="pv_regrid_debug"))

def overlay_moment(background='8um', overlay='cii', velocity_limits=None, data_memo=None, data_memo_rules='background', cutout_reg_stub='N19', reg_filename_or_idx=None, plot_stars=False):
    """
    May 3, 2023
    Ambitious code project: make flexible image-contour overlays.
    Anticipate backgrounds of 8um and 250um but can keep flexible to use optical maps or something later.
    Apply cutout, if requested, and regrid the overlay data to that WCS.
    Make moment using velocity_limits is overlay data is a cube.

    :param data_memo_rules: either 'background', 'overlay', or 'both'.
        Only used if data_memo is not None. If data_memo is None, data_memo_rules is effectively "neither".
        Chooses which data to memoized; only the user knows which data is staying constant and which is cycling in the loop.
        Default: 'background', that is normal behavior.
    """
    # Redefine the conveniently-named input keywords
    background_stub = background
    overlay_stub = overlay
    del background, overlay # they will be reused, but I want to be clear here
    """
    Set up data_memo_rules
    Check for background memo with dmr%2==0
    Check for overlay memo with drm>0
    """
    if data_memo is None:
        dmr = -1
    else:
        dmr = ['background', 'overlay', 'both'].index(data_memo_rules)

    """
    Load in regions to plot, if any
    This can be up to one path and any number of astropy regions-recognized regions
    """
    pv_path, reg_list, reg_fn_stub = get_pv_and_regions(reg_filename_or_idx)

    """ Hillenbrand Stars """
    if plot_stars:
        star_df = pd.read_csv(catalog.utils.search_for_file("catalogs/hillenbrand_stars_1.csv"), index_col='ID')
        star_ra = star_df['RA'].values
        star_dec = star_df['DE'].values
        del star_df

    # Load in background
    img, img_info = get_2d_map(background_stub, velocity_limits=velocity_limits, average_Tmb=True, data_memo=data_memo)
    # Cutout background (and update WCS)
    if cutout_reg_stub is not None:
        img_info['cutout'] = misc_utils.cutout2d_from_region(img, img_info['wcs'], get_cutout_box_filename(cutout_reg_stub))
        img = img_info['cutout'].data
        img_info['wcs'] = img_info['cutout'].wcs
    # Memoize it if it's not already there
    if dmr%2 == 0 and img_info['vlim_hash'] not in data_memo:
        data_memo[img_info['vlim_hash']] = (img, img_info)

    # Load in overlay
    overlay, overlay_info = get_2d_map(overlay_stub, velocity_limits=velocity_limits, average_Tmb=True, data_memo=data_memo)
    # Regrid overlay to background
    overlay_regrid = reproject_interp((overlay, overlay_info['wcs']), img_info['wcs'], shape_out=img.shape, return_footprint=False)
    # From here on out, don't use the overlay wcs, use the img wcs
    # Memoize it if it's not already there
    if dmr > 0 and overlay_info['vlim_hash'] not in data_memo:
        data_memo[overlay_info['vlim_hash']] = (overlay_regrid, overlay_info)

    # print("img hash", img_info['vlim_hash'])
    # print("overlay hash", overlay_info['vlim_hash'])

    # Figure
    figsizes = {'n19': (10, 10), 'med-large': (13, 9)}
    fig = plt.figure(figsize=figsizes.get(cutout_reg_stub, (10, 10)))
    ax = fig.add_subplot(111, projection=img_info['wcs'])
    # Plot image
    # Use generic vlims as a backup for specific vlims
    img_vlim = get_generic_vlim(background_stub)
    img_vlim.update(get_vlim(img_info['vlim_hash']))
    im = ax.imshow(img, origin='lower', **img_vlim, cmap='Greys_r')
    # Plot contours
    # Check for levels or use vlims
    memo_levels = get_levels(overlay_stub)
    if memo_levels is None:
        contour_args = (trim_values_to_vlims(overlay_regrid, **get_vlim(overlay_info['vlim_hash'])),)
        contour_kwargs = {}
    else:
        contour_args = (overlay_regrid,)
        contour_kwargs = dict(levels=memo_levels, **get_generic_vlim(overlay_stub))
    cs = ax.contour(*contour_args, **contour_kwargs, cmap='magma_r')

    # Plot the footprint of the overlay if it would be visible at all
    overlay_nan_map = np.isnan(overlay_regrid)
    if np.any(overlay_nan_map):
        ax.contour(overlay_nan_map.astype(float), levels=[0.5], colors='SlateGray', linestyles=':', linewidths=1)
    del overlay_nan_map

    # Image colorbar (right)
    cbar_ax = ax.inset_axes([1, 0, 0.05, 1])
    velocity_stub = " "+make_vel_stub(velocity_limits) if (img_info['obs_type'] == 'cube' and velocity_limits) else ""
    cbar = fig.colorbar(im, cax=cbar_ax, label=f"{get_data_name(background_stub)}{velocity_stub} ({img_info['unit'].to_string('latex_inline')})")
    # Contour colorbar (top)
    cbar_ax2 = ax.inset_axes([0, 1, 1, 0.05])
    velocity_stub = " "+make_vel_stub(velocity_limits) if (overlay_info['obs_type'] == 'cube' and velocity_limits) else ""
    cbar2 = fig.colorbar(cs, cax=cbar_ax2, location='top', spacing='proportional', label=f"{get_data_name(overlay_stub)}{velocity_stub} ({overlay_info['unit'].to_string('latex_inline')})")

    # Beams
    beam_patch_kwargs = dict(alpha=0.9, hatch='////')
    beam_x, beam_y = 0.93, 0.1
    beam_ecs = [['white', 'grey'], [cs.cmap(cs.norm(cs.levels[j])) for j in [0, 2]]]
    for i, data_info_dict in enumerate((img_info, overlay_info)):
        if 'beam' not in data_info_dict:
            continue
        # Beam is known, plot it
        patch = data_info_dict['beam'].ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(img_info['wcs']))
        patch.set(**beam_patch_kwargs, facecolor=beam_ecs[i][0], edgecolor=beam_ecs[i][1])
        ax.add_artist(patch)
        beam_x -= 0.03

    """ Plot paths and regions """
    # Before starting this, save the current x and y limits in case a region goes out of bounds
    ref_xlim = ax.get_xlim()
    ref_ylim = ax.get_ylim()
    # Check for PV paths, plot if found
    reg_color = "LimeGreen"
    if pv_path:
        ax.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=reg_color, linestyle='-', lw=1, transform=ax.get_transform('world'))
        ax.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=reg_color, fontsize=10, va='center', ha='right', transform=ax.get_transform('world'))
    if reg_list:
        for reg in reg_list:
            if isinstance(reg, regions.LineSkyRegion):
                # Skip Line regions
                continue
            reg_patch = reg.to_pixel(img_info['wcs']).as_artist()
            # Gotta mess around with the lines vs other patches
            if isinstance(reg_patch, Line2D):
                # Point!!!
                reg_patch.set(mec=reg_color)
                ax.add_artist(reg_patch)
            else:
                # Anything besides Point
                reg_patch.set(ec=reg_color)
                ax.add_patch(reg_patch)
    if plot_stars:
        # Plot stars if requested; if plot_stars is True, then stars_ra and stars_dec exist
        ax.scatter(star_ra, star_dec, mec=marcs_colors[3], mfc='None', marker='o', transform=ax.get_transform('world'))
    ax.set_xlim(ref_xlim)
    ax.set_ylim(ref_ylim)



    velocity_stub = "" if not velocity_limits else "_"+make_simple_vel_stub(velocity_limits)
    if overlay_info['obs_type'] != 'cube' and img_info['obs_type'] != 'cube':
        # Override velocity stub with empty string if neither image is a cube
        velocity_stub = ""
    cutout_stub = "" if cutout_reg_stub is None else f"cutout {cutout_reg_stub} from {os.path.basename(get_cutout_box_filename(cutout_reg_stub))}"
    if reg_fn_stub:
        reg_fn_stub = "_"+reg_fn_stub
    if plot_stars:
        reg_fn_stub += '_with-stars'
    # 2023-05-03,04,05,09,10,11
    fig.savefig(f"/home/ramsey/Pictures/2023-05-11/overlay_{overlay_stub}_on_{background_stub}{velocity_stub}{reg_fn_stub}.png",
        metadata=catalog.utils.create_png_metadata(title=cutout_stub, file=__file__, func="overlay_moment"))
    # Some cleanup since things seem to pile up
    plt.close(fig)

def fast_pv(reg_filename_or_idx=0, line_stub_list=['12co32', 'ciiAPEX']):
    """
    May 5, 2023
    Re-create the PV diagrams from real_medium_spectra but without the other
    stuff on the plots.
    Formerly real_medium_pv, but planning to use it live so gave it a better name.
    """
    # Moved line_stub_list to argument
    # Use CII and either 12 or 13CO 3-2

    # Reference image
    reference_filename_short = "spitzer/SPITZER_I4_mosaic_ALIGNED.fits"
    ref_img, ref_hdr = fits.getdata(catalog.utils.search_for_file(reference_filename_short), header=True)
    ref_wcs = WCS(ref_hdr)
    # Hillenbrand stars
    star_df = pd.read_csv(catalog.utils.search_for_file("catalogs/hillenbrand_stars_1.csv"), index_col='ID')
    star_ra = star_df['RA'].values
    star_dec = star_df['DE'].values
    del star_df


    pv_vmaxes = {'ciiAPEX': 20, '12co32': 30, '13co32': 15}
    pv_levels = {'ciiAPEX': (3, 37, 4), '12co32': (5, 41, 5), '13co32': (1, 27, 2.5)}
    def _get_levels(line_stub):
        """
        Get levels from the above dictionary. Return None if not present.
        """
        if line_stub in pv_levels:
            return np.arange(*pv_levels[line_stub])
        else:
            return None
    velocity_limits = (8*kms, 35*kms)
    velocity_intervals = np.arange(20, 35, 2)

    # Most of these files have 3 points and a vector
    # In theory, it can have any number of points which will be handled correctly
    # If it doesn't have points, ignore the points and use the vector.
    pv_path, point_reg_list, reg_fn_stub = get_pv_and_regions(reg_filename_or_idx)

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
    ax_ref_img = fig.add_subplot(gs[0,0], projection=ref_wcs)

    # Plot reference image
    ax_ref_img.imshow(ref_img, origin='lower', vmin=45, vmax=290, cmap='Greys_r')

    pv_slices = []
    for line_stub in line_stub_list:
        cube_filename_short = get_map_filename(line_stub)
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K()
        pv_slices.append(pvextractor.extract_pv_slice(cube.data.spectral_slab(*velocity_limits), pv_path))

    # Regrid [1] to [0]; clean up the headers to allow reprojection
    sl0, sl1_raw = pv_slices
    sl0.header['CTYPE2'] = sl1_raw.header['CTYPE2'] = "VRAD"
    sl1_raw.header['RESTFRQ'] = sl0.header['RESTFRQ']
    sl1_reproj = reproject_interp((sl1_raw.data, sl1_raw.header), sl0.header, return_footprint=False)

    # Plot
    ax_pv = fig.add_subplot(gs[0, 1], projection=WCS(sl0.header))
    ls0, ls1 = line_stub_list
    # Background [0]; can either use Greys(_r) to match wtih plasma(_r) contours, or plasma to match with cool contours
    im = ax_pv.imshow(sl0.data, origin='lower', cmap='Greys_r', vmin=0, vmax=pv_vmaxes.get(ls0, None), aspect=(sl0.data.shape[1]/(2.5*sl0.data.shape[0])))
    cs = ax_pv.contour(sl0.data, colors='k', linewidths=1, linestyles=':', levels=_get_levels(ls0))
    pv_cbar = fig.colorbar(im, ax=ax_pv, location='right', label=cube.data.unit.to_string('latex_inline'))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='k')
    ax_pv.text(0.05, 0.95, cube_utils.cubenames[ls0], fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)
    # Overlay [1]
    cs = ax_pv.contour(sl1_reproj, cmap='plasma_r', linewidths=1.5, levels=_get_levels(ls1), vmax=pv_vmaxes.get(ls1, None))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='w', linewidth=2)
    ax_pv.text(0.05, 0.90, cube_utils.cubenames[ls1], fontsize=13, color='w', va='top', ha='left', transform=ax_pv.transAxes)

    # Plot velocity intervals
    x_length = pv_path._coords[0].separation(pv_path._coords[1]).deg
    for v in velocity_intervals:
        ax_pv.plot([0, x_length], [v*1e3]*2, color='grey', alpha=0.7, linestyle='--', transform=ax_pv.get_transform('world'))
    # Plot region positions, if there are point regions
    for reg in point_reg_list:
        # Plot vertical bars on the PV diagram
        offset = float(reg.meta['text'].split('/')[1]) * u.arcsec
        ax_pv.plot([offset.to(u.deg).to_value()]*2, [v.to_value()*1e3 for v in velocity_limits], color='k', alpha=1, linestyle=':', linewidth=4, transform=ax_pv.get_transform('world'))
        # Plot the points on the Spitzer image
        pixreg = reg.to_pixel(ref_wcs)
        x, y = pixreg.center.xy
        ax_ref_img.plot([x], [y], 'o', mfc=marcs_colors[1], mec='k')

    ax_pv.coords[1].set_format_unit(u.km/u.s)
    ax_pv.coords[1].set_major_formatter('x.xx')
    ax_pv.coords[0].set_format_unit(u.arcsec)
    ax_pv.coords[0].set_major_formatter('x.xx')
    ax_ref_img.set_xlabel("RA")
    ax_ref_img.set_ylabel("Dec")
    ax_ref_img.plot([c.ra.deg for c in pv_path._coords], [c.dec.deg for c in pv_path._coords], color=marcs_colors[1], linestyle='-', lw=1, transform=ax_ref_img.get_transform('world'))
    ax_ref_img.text(pv_path._coords[0].ra.deg, pv_path._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=marcs_colors[1], fontsize=10, va='center', ha='right', transform=ax_ref_img.get_transform('world'))

    plt.tight_layout()
    # 2023-05-06,11
    savename = f"/home/ramsey/Pictures/2023-05-11/pv_{ls1}_on_{ls0}_along_{reg_fn_stub}.png"
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='pv',
        file=__file__, func="fast_pv"))


if __name__ == "__main__":
    """
    Moment 0 examples
    """
    # data_memo = {}
    # vel_lims_list = [(16, 20), (21, 23), (25, 28)]
    # # vel_lims_list = [(x, x+1) for x in range(10, 15)]
    # f_vel_lims = lambda j : tuple(x*kms for x in vel_lims_list[j])
    # for line_stub in ['13co10', '13co32', '12co10', '12co32', 'ciiAPEX']:
    #     for i in range(len(vel_lims_list)):
    #         overlay_moment(background='250um', overlay=line_stub, velocity_limits=f_vel_lims(i), cutout_reg_stub='med-large', data_memo=data_memo)
    overlay_moment(background='250um', overlay='ciiAPEX', velocity_limits=(21*kms, 22*kms), cutout_reg_stub='med', reg_filename_or_idx="catalogs/N19_pv_2.reg", plot_stars=False)

    """
    PV examples
    """
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['13co32', '13co10'])
    # fast_pv(reg_filename_or_idx=0)
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['ciiAPEX', '12co32'])
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['12co10', 'ciiAPEX'])
