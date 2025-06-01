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
import cmocean
import sys
import os
import glob
import datetime
import time
import warnings
import math
import itertools
import bisect

# from math import ceil
# from scipy import signal
from scipy.interpolate import UnivariateSpline
from scipy import signal

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
# from astropy.table import Table, QTable
from astropy import constants as const
from astropy.convolution import convolve, convolve_fft

from reproject import reproject_adaptive

import pandas as pd
# from io import StringIO

from spectralradex import radex
from multiprocessing import Pool

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
from . import channel_maps as channel_maps_utils

mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors
mpl_transforms = pvdiagrams.mpl_transforms
mpatches = pvdiagrams.mpatches

from .mantipython.physics import greybody, dust, instrument


# vel_stub is for plotting (contains spaces and special characters)
make_vel_stub = lambda x : f"[{x[0].to_value():.1f}, {x[1].to_value():.1f}] {x[0].unit.to_string('latex_inline')}"
# simple_vel_stub is for filenames (no spaces or special characters)
make_simple_vel_stub = lambda x : ".".join(f"{y.to_value():.1f}" for y in x)
kms = u.km/u.s
marcs_colors = ['#377eb8', '#ff7f00','#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


ratio_12co_to_H2 = 8.5e-5 # Tielens book number
Cp_H_ratio = 1.6e-4 # Sofia et al 2004, what Tiwari et al 2021 used in the N(C+)->N(H) section

ratio_12co_to_13co = 44.65 # call it 45 in paper, the difference will be miniscule
ratio_12co_to_c18o = 417 # From Wilson and Rood 1994 using 6.46

los_distance_M16 = 1740 * u.pc # Kuhn et al 2019
err_los_distance_M16 = 130 * u.pc # Kuhn et al 2019; it was +130, -120, so for a symmetric error bar I'll take +/- 130

H_mass_amu = 1.00794
Hmass = const.u * H_mass_amu
# H2_mass is 1/1e4 larger than 2x Hmass. All of these ignore He
# no longer using this: # H2_mass_permole = 2.016 * u.g / u.mol
# no longer using this: H2_mass =  (H2_mass_permole / const.N_A).decompose()
# Fix all instances of H2_mass to be 2*Hmass (easier to find them thru NameErrors)

# Adopting Y = 0.25, Z=0
mean_molecular_weight_neutral = 1.33

"""
File i/o and other useful, reusable things.
Dictionaries/lists of constants/names and helper functions that use them.
"""

large_map_filenames = {
        'cii': "sofia/m16_CII_final_15_5_0p5_clean.fits",
        'cii-30': "sofia/m16_CII_final_30_15_2_clean.fits", # larger beam and wider channel CII (lower noise)
        '12co10-pmo': "purplemountain/G17co12.fits", '13co10-pmo': "purplemountain/G17co13.fits",
        'c18o10-pmo': "purplemountain/G17c18o.fits",
        '12co10-nob': "nobeyama/FGN_01700+0000_2x2_12CO_v1.00_cube.fits", '13co10-nob': "nobeyama/FGN_01700+0000_2x2_13CO_v1.00_cube.fits",
        'c18o10-nob': "nobeyama/FGN_01700+0000_2x2_C18O_v1.00_cube.fits",
        '12co32-pmo': "apex/M16_12CO3-2_truncated.PMObeam.fits", '13co32-pmo': "apex/M16_13CO3-2_truncated.PMObeam.fits", # convolved APEX CO 3-2 to PMO 1-0 beam
        '12co32-2kms': "apex/M16_12CO3-2_truncated.rebin2kms.fits", # 2kms rebin for channel maps
        'rrl': "thor_vla/RRL_stacked_L17.50_deg.m16cutout.fits",
}
descriptor_names = {'pmo': "PMO", 'nob': "NRO", '30': "30beam", '2kms': "2km/s rebin"}
herschel_path = "herschel/anonymous1603389167/1342218995/level2_5"
photometry_filenames = {
    '8um': "spitzer/SPITZER_I4_mosaic_ALIGNED.fits", '250um': "extdPSW/hspirepsw696_25pxmp_1823_m1335_1342218995_1342218996_1462565962570.fits.gz",
    '350um': "extdPMW/hspirepmw696_25pxmp_1823_m1335_1342218995_1342218996_1462565960474.fits.gz",
    '500um': "extdPLW/hspireplw696_25pxmp_1823_m1335_1342218995_1342218996_1462565958982.fits.gz",
    '70um': "HPPJSMAPB/hpacs_25HPPJSMAPB_blue_1822_m1337_00_v1.0_1471714532334.fits.gz",
    '160um': "HPPJSMAPR/hpacs_25HPPJSMAPR_1822_m1337_00_v1.0_1471714553094.fits.gz",
    '850um': "apex/atlasgal-m16-large.fits", '24um': "spitzer/m16_MIPS24um_mosaic.fits.fz",
    'NB_659': "vlt/mosaic_NB_659_small.fits.fz", 'g_SDSS': "vlt/mosaic_g_SDSS.fits.fz",
    '90cm': "magpis_vla/G17.000000+0.750000_90cm.goodheader.fits",

}
photometry_beams = {
    '8um': (1.98, 1.98, 0),
    '70um': (9.0, 5.75, 62), '160um': (13.32, 11.31, 40.9), # http://herschel.esac.esa.int/Docs/PACS/html/ch03.html#sec-characteristics-photometer
    '250um': (18.4, None, 0),
    '350um': (25.2, None, 0), '500um': (36.7, None, 0),
}
cutout_box_filenames = {
    'N19': "N19_cutout_box.reg", # Small, only N19
    'N19-small': "N19_cutout_box_small.reg", # even smaller, only the top part of the bubble, nothing to the south
    'N19-med': "N19_cutout_box_med.reg", # smaller than "N19" but includes the south part of the bubble, so bigger than "N19-small"
    'north-cloud': "northerncloud_cutout_box.reg", # the full northern cloud in CII between 10-21 km/s
    'med-large': "m16_cutout_box_medium-large.reg", # sort of like the Hill 2012 footprint, includes some of the filament. Goes well outside CII
    'med': "m16_cutout_box_medium.reg", # CII and CO (3-2) footprint, aligned in equatorial (so there will be NaN gaps)
    'blueclump': 'm16_cutout_box_blueshifted_clump.reg', # The IRAC clump that shows up in CII from 6-11 km/s
    'blueclump-large': 'm16_cutout_box_blueshifted_clump_larger.reg', # Same clump, larger field
    'blueclump-large2': 'm16_cutout_box_blueshifted_clump_larger2.reg', # Same clump, even larger field
    'blueclump-zoom': 'm16_cutout_box_blueshifted_clump_zoom.reg', # Same clump, but zoomed in so clump fills image
}
vlim_memo = { # hash things somehow and put them here
    '8um': (40, 320), 'irac4': (40, 320),

    # 'cii.levels': np.concatenate([np.arange(2.5, 61, 5), np.arange(65, 126, 20)]),
    'cii.levels': np.concatenate([np.arange(1.5, 61, 3), np.arange(65, 126, 15)]),

    'cii.generic': (0, 25), #25 some, 65 is better for most, 14 or 10 good for some
    # 'cii-30.levels': np.arange(0.2, 10, 0.8), # this cii map is diluted, so lower emission,
    'cii-30.levels': np.arange(0.5, 20, 2), # for short intervals, high emission
    'cii-30.generic': (0, 20), # 10 is generally good, 1 can be good for very low emission
    '12co32.levels': np.arange(2.5, 51, 5), '12co32.generic': (0, 25), # or 40
    '13co32.levels': np.arange(1, 27, 2.5), '13co32.generic': (0, 7),
    'rrl.levels': np.arange(3, 30, 6), # RRLs should be "sliced" into a 1 km/s channel
    '90cm.levels': np.arange(0, 0.61, 0.06), '850um': (0, 0.5), '850um.levels': np.concatenate([np.arange(.15, 2.1, 0.3), np.arange(3, 6, 1)]), '850um.generic': (0.15, 2),

    # '13co10.levels': np.arange(0.25, 5, 0.5), '13co10.generic': (0, 4), # these are good for large velocity intervals
    '13co10.levels': np.arange(1, 22, 2),
    # '13co10.levels': np.arange(2, 22, 4),
    '13co10.generic': (0, 8), # small velocity intervals (like 1 km/s)

    '12co10.levels': np.arange(2.5, 26, 2.5), # 5,55,5 good for most
    '12co10.generic': (0, 20), # 0,40 good for most
    '250um': (140, 4500), 'irac1': (1, 30), '70um': (-0.06, 3.5), # 70um can also do vmax=1.5 for greater sensitivity to low emission
    '160um': (-0.1, 2.5), '500um': (50, 500), '500um.levels': np.arange(200, 2001, 100), '500um.generic': (150, 500),
    'irac2': (1.5, 15), 'irac3': (10, 130),
}
default_reg_filename_list = [ # commonly used region filenames
    "catalogs/N19_points_along_path_1.reg", "catalogs/m16_up_points_along_path.reg",
    "catalogs/m16_across_pillars_points_along_path.reg", "catalogs/spire_up_points_along_path.reg",
    "catalogs/m16_across_points_along_path.reg",
]
onesigmas = {
    '12co10-pmo': 0.415, '13co10-pmo': 0.200, 'c18o10-pmo': 0.202,
    '12co32-pmo': 0.1, '13co32-pmo': 0.1, # checked 2023-10-19 by eye (0.07, 0.09 for 12 and 13); updated 2023-12-28 to 0.1 for both (not a big difference)
}

def vlim_hash(data_stub, velocity_limits=None, generic=False):
    """
    May 3, 2023
    Make keys for vlim_memo. This is to make image creation very easy
    Ignore 'APEX' or 'CONV' suffix; this rarely makes a difference.
    """
    data_stub = data_stub.replace('CONV', '').replace('APEX', '')
    if '-' in data_stub:
        # Get rid of the observatory tag
        data_stub = data_stub.split('-')[0]
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
    hyphen_exceptions = ['cii-30']
    if '-' in key:
        # Get rid of the observatory tag (without getting rid of "generic")
        key = ".".join([(s.split('-')[0] if (i==0 and s not in hyphen_exceptions) else s) for i, s in enumerate(key.split('.'))])
    vlims = {}
    vlims_keys = ['vmin', 'vmax']
    if key in vlim_memo:
        vlim_memo_value = vlim_memo[key]
        for k, v in zip(vlims_keys, vlim_memo_value):
            if v is not None:
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
    # Specific carveout for cii-30 since it's diluted emission
    hyphen_exceptions = ['cii-30']
    if '-' in data_stub and data_stub not in hyphen_exceptions:
        # Get rid of the observatory tag
        data_stub = data_stub.split('-')[0]
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
    elif data_stub[:4] == 'irac' and '-' not in data_stub:
        # irac1, irac2, irac3. Stick to '8um' instead of irac4, though it will work either way
        return (f"spitzer/SPITZER_I{data_stub[-1]}_mosaic_ALIGNED.fits",)
    elif 'irac' in data_stub and '-large' in data_stub:
        # Large IRAC mosaic (like, "irac1-large")
        return (f"spitzer/m16_IRAC{data_stub.replace('-large', '')[-1]}_mosaic.fits.fz",)
    # At this point, must be a cube
    elif data_stub in large_map_filenames:
        # Works for CII and the observatory-tagged CO
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
    elif '-' in data_stub and data_stub.split('-')[0] in cube_utils.cubenames:
        return cube_utils.cubenames[data_stub.split('-')[0]] + " " + descriptor_names[data_stub.split('-')[1]]
    elif data_stub[:4] == 'irac':
        if '-' in data_stub:
            data_stub = data_stub.split('-')[0]
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
        data_full_filename = catalog.utils.search_for_file(data_filename)
        info_dict['full_path'] = data_full_filename
        img, hdr = fits.getdata(data_full_filename, header=True)
        info_dict['wcs'] = WCS(hdr)
        info_dict['hdr'] = hdr
        if 'BUNIT' in hdr:
            # If this throws an error I'll cross that bridge when I get to it
            unit_str = hdr['BUNIT']
            if 'BEAM' in unit_str:
                # Probably Jy/beam
                unit_str = unit_str.lower().replace('jy', 'Jy')
            info_dict['unit'] = u.Unit(unit_str)
            # heads up: 850 micron is in Jy/beam

        # Beam lookup (the images almost never have beams in headers...)
        if data_stub[:4] == 'irac':
            beam_key = '8um' # they're all the same
        else:
            beam_key = data_stub
        try:
            # See if the beam is in the header
            info_dict['beam'] = cube_utils.Beam.from_fits_header(hdr)
        except cube_utils.NoBeamException:
            # Look for the beam in our dictionaries
            if beam_key in photometry_beams:
                major, minor, pa = photometry_beams[beam_key]
                beam_params = dict(major=major*u.arcsec, pa=pa*u.deg)
                if minor is not None:
                    beam_params['minor'] = minor*u.arcsec
                info_dict['beam'] = cube_utils.Beam(**beam_params)
            else:
                # No beam!
                # Notify me about this
                warnings.warn(f"No beam found for {data_stub}")
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
        # Put filepath info in info_dict just in case
        info_dict['original_cube_basename'] = cube_obj.basename
        info_dict['original_cube_full_path'] = cube_obj.full_path
        info_dict['original_cube_directory'] = cube_obj.directory
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

def get_onesigma(data_stub):
    if data_stub in onesigmas:
        return onesigmas[data_stub]
    else:
        return cube_utils.onesigmas[data_stub]

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
CO Column density
"""
def find_co10_noise():
    """
    May 16, 2023
    Find the PMO CO 1-0 noise.
    Use the following 0-indexed boxes:
        X 8-37, Y 69-90, channel 0-78
        X 8-54, Y 69-90, channel 0-65
    I think it goes (channel, Y, X) but it always surprises me. Yes it does!
    The shape is (200, 97, 97) so I can't even tell by dimensions.

    The larger box has a little bit of signal left in it, but both boxes are
    relatively clear after channel 150 so I can try that too.

    Worked well!
    PMO observations:
    12CO10 is 0.415 K
    13CO10 is 0.200 K
    C18O10 is 0.202 K

    Each box produces a very similar answer such that the variation from box to
    box is minimal (~few percent).

    May 24, 2023
    Reusing this for the Nobeyama CO noise. I'll find new boxes for this.
    Nobeyama observations:
    12CO10 is 1.4 K
    13CO10 is 0.7 K
    C18O10 is 0.6 K
    """
    data_stub = 'c18o10-nob'
    cube_fn = get_map_filename(data_stub)
    cube = cube_utils.CubeData(cube_fn)
    print("Shape of ", data_stub, cube.data.shape)
    # Get a signal-free subcube to test the RMS
    if data_stub.split('-')[1] == 'pmo':
        clean_slab_params = [ # PMO
            ((0, 78), (69, 90), (8, 37)),
            ((0, 65), (69, 90), (8, 54)),
            ((150, None), (69, 90), (8, 37)),
            ((150, None), (69, 90), (8, 54)),
            ]
    elif data_stub.split('-')[1] == 'nob':
        clean_slab_params = [ # NRO Nobeyama
            ((0, 127), (430, None), (0, None)), # low V, top two tiles
            ((0, 127), (0, 430), (427, None)), # low V, bottom right tile
            ((384, None), (430, None), (0, None)), # high V, top two tiles
            ((384, None), (0, 430), (427, None)), # high V, bottom right tile
        ]
    for i in range(len(clean_slab_params)):
        clean_slab = cube.data[tuple(slice(*x) for x in clean_slab_params[i])]
        plt.figure(f"Slab {i}")
        plt.subplot(121)
        plt.imshow(clean_slab.moment0().to_value(), origin='lower')
        plt.subplot(122)
        plt.plot(clean_slab.mean(axis=(1, 2)))
        clean_slab_rms = clean_slab.std()
        print(f"box {i}: {clean_slab_rms:0.2f}")
    plt.show()


def co_column_manage_inputs(line='10', isotope='13', velocity_limits=None, cutout_reg_stub=None):
    """
    May 16, 2023
    Wrapper function for the more general function below.
    This function is specific to the M16 CO 1-0 and 3-2 maps we have.
    It does the work of m16_deepdive.integrate_13_and_18_co_for_column_density
    and m16_deepdive.get_excitation_temperature_12co.

    The only user input is whether to use 1-0 or 3-2.
    Can consider adding C18O 1-0 later.
    And write in the capability (to be refined later) to use specific velocity limits.
    :param line: str CO transition descriptor. Should be the two J levels, in
        decreasing order. J=1-0 would be specified with the string "10",
        J=3-2 with "32", and so on.
        The transition must be supported by the COColumnDensity class.
    :param velocity_limits: tuple(Quantity, Quantity) low, high velocity limits.
        Quantities must be velocities, like km/s.
    :param cutout_reg_stub: (optional) str label for a defined cutout box.
        If given, a spatial cutout will be made of the integrated intensity
        prior to calculation of the column density.
    """
    if line == '32':
        thick_cube_stub = '12co32-pmo'
        thin_cube_stub = '13co32-pmo'
        thick_channel_uncertainty = get_onesigma(thick_cube_stub) * u.K
        thin_channel_uncertainty = get_onesigma(thin_cube_stub) * u.K
    elif line == '10':
        # Could add in support for Nobeyama data too, if we think it's useful
        thick_cube_stub = '12co10-pmo'
        if isotope == '13':
            thin_cube_stub = '13co10-pmo'
        elif isotope == '18':
            thin_cube_stub = 'c18o10-pmo'
        thick_channel_uncertainty = get_onesigma(thick_cube_stub) * u.K
        thin_channel_uncertainty = get_onesigma(thin_cube_stub) * u.K

    # Placeholder for velocity limits argument
    # Expose the if logic as the top layer, since we only need to check once!
    if velocity_limits is None:
        def _apply_velocity_limits(cube):
            return cube
    else:
        def _apply_velocity_limits(cube):
            return cube.spectral_slab(*velocity_limits)

    # Load 12 (optically thick) and 13 or 18 (optically thin) maps.
    # Get excitation temperature from optically thick cube
    """
    Potential edit: should we be applying the velocity interval to the thick cube? Kind of depends on whether we think excitation conditions are the same, or if the lines are separated and so forth.
    """
    thick_cube_fn = get_map_filename(thick_cube_stub)
    thick_cube_obj = cube_utils.CubeData(thick_cube_fn)
    thick_cube = _apply_velocity_limits(thick_cube_obj.data)
    thick_cube_wcs = thick_cube_obj.wcs_flat
    # Save this in final cube
    peak_temperature = thick_cube.max(axis=0).to(u.K)
    del thick_cube_obj, thick_cube

    # Trim peak temperature to be >3 K, avoid no-emission regions
    peak_temperature[peak_temperature < 6*u.K] = np.nan*peak_temperature.unit

    # Get integrated optically thin emission
    thin_cube_fn = get_map_filename(thin_cube_stub)
    thin_cube_obj = cube_utils.CubeData(thin_cube_fn).convert_to_kms()
    thin_cube = _apply_velocity_limits(thin_cube_obj.data)
    # Grab a couple other things from the CubeData object
    beam = thin_cube_obj.data.beam
    save_path = os.path.dirname(thin_cube_obj.full_path)
    # Make a cleaner moment image by trimming out the emission-free parts
    # Part 1: trim the full cube (full spectra)
    thin_cube_trimmed = thin_cube.with_mask(thin_cube > (thin_channel_uncertainty*3))
    mom0 = thin_cube_trimmed.moment0()
    # Noise
    cube_dv = np.abs(np.diff(thin_cube.spectral_axis[:2]))[0].to(kms)
    n_channels = thin_cube.shape[0]
    moment0_noise = thin_channel_uncertainty * cube_dv * np.sqrt(n_channels)
    # Part 2: Trim moment 0 again, this time on the integrated intensity uncertainty
    mom0[mom0 < 1*moment0_noise] = np.nan*moment0_noise.unit
    # Save the moment 0 in final cube
    del thin_cube_obj, thin_cube
    # Apply region cutout if necessary
    if cutout_reg_stub is not None:
        cutout = misc_utils.cutout2d_from_region(mom0.to(u.K*kms).to_value(), mom0.wcs, get_cutout_box_filename(cutout_reg_stub), align_with_frame='icrs')
        thin_mom0 = cutout.data * u.K*kms
        out_wcs = cutout.wcs
    else:
        thin_mom0 = mom0
        out_wcs = mom0.wcs

    # plt.imshow(mom0.to(u.K*kms).to_value(), origin='lower')
    # plt.show()

    col_dens_calculator = COColumnDensity(thin_cube_stub)
    col_dens_calculator.set_data((peak_temperature, thick_cube_wcs), (thin_mom0.to(u.K*kms), out_wcs))
    if False:
        ff = 0.5
        col_dens_calculator.set_filling_factor(ff)
    else:
        ff = None
    col_dens_calculator.set_uncertainty(thick_channel_uncertainty, moment0_noise)
    if isotope == '13':
        col_dens_calculator.set_abundance_ratios(ratio_12co_to_13co, ratio_12co_to_H2)
    elif isotope == '18':
        col_dens_calculator.set_abundance_ratios(ratio_12co_to_c18o, ratio_12co_to_H2)
    col_dens_calculator.calculate_column_density()
    col_dens_calculator.calculate_mass_per_pixel(out_wcs, los_distance=los_distance_M16, e_los_distance=err_los_distance_M16, thin_line_beam=beam)

    # plt.subplot(121)
    # plt.imshow(col_dens_calculator.H2_column_density.to_value(), origin='lower')
    # plt.subplot(122)
    # plt.imshow(col_dens_calculator.mass_per_pixel.to_value(), origin='lower')
    # plt.show()

    """ Write results to FITS """
    cutout_reg_filename_stub = "_"+cutout_reg_stub if cutout_reg_stub is not None else ""
    velocity_stub = "_"+make_simple_vel_stub(velocity_limits) if velocity_limits is not None else ""
    ff_stub = f"ff{ff:.1f}" if ff is not None else ""
    savename = os.path.join(save_path, f"column_density_v3_{ff_stub}_{thin_cube_stub}{cutout_reg_filename_stub}{velocity_stub}.fits")

    header_pairs = col_dens_calculator.create_header_comments()

    def make_and_fill_header():
        # fill header with stuff, make it from WCS
        hdr = out_wcs.to_header()
        for k, v in header_pairs:
            hdr[k] = v
        return hdr

    # common messages
    h2_msg = "This is MOLECULAR hydrogen (H2)"
    mass_msg = "mass is per pixel on this image"

    if isotope == '13':
        thin_line_extname_stub = '13CO'
    elif isotope == '18':
        thin_line_extname_stub = 'C18O'

    extensions = [
        ('H2coldens', col_dens_calculator.H2_column_density, h2_msg),
        ('err_H2coldens', col_dens_calculator.e_H2_column_density, h2_msg),
        ('12COcoldens', col_dens_calculator.thick_line_column_density),
        ('err_12COcoldens', col_dens_calculator.e_thick_line_column_density),
        (f'{thin_line_extname_stub}coldens', col_dens_calculator.thin_line_column_density),
        (f'err_{thin_line_extname_stub}coldens', col_dens_calculator.e_thin_line_column_density),
        ('mass', col_dens_calculator.mass_per_pixel, mass_msg),
        ('err_mass', col_dens_calculator.e_mass_per_pixel, mass_msg),
        ('Tex', col_dens_calculator.Tex, f"Excitation temperature - err {thick_channel_uncertainty:.3f}"),
        (f'{thin_line_extname_stub}mom0', thin_mom0.to(u.K*kms), f"{thin_line_extname_stub} Integrated intensity - err {moment0_noise:.3f}")
    ]

    # Going to try just copying the header, I think it's fine

    def make_extension(ext_tup, header_template):
        # make an HDU from a tuple of info and (shared) header starter
        if len(ext_tup) == 2:
            extname, data = ext_tup
            msg = None
        else:
            extname, data, msg = ext_tup
        ext_header = header_template.copy()
        ext_header['EXTNAME'] = extname
        ext_header['BUNIT'] = str(data.unit)
        if msg:
            ext_header['COMMENT'] = msg
        return fits.ImageHDU(data=data.to_value(), header=ext_header)

    phdu = fits.PrimaryHDU()
    header_template = make_and_fill_header()
    list_of_hdus = [phdu] + [make_extension(ext_info, header_template) for ext_info in extensions]
    hdul = fits.HDUList(list_of_hdus)
    hdul.writeto(savename)
    print("Done, wrote to ", savename)


def calculate_co_column_density_detection_threshold():
    """
    January 22, 2023
    Similar to calculate_cii_column_density_detection_threshold
    Use 3 km/s wide line, but try 2 and 4 to see if anything changes.
    """
    # Velocity array
    v_arr = np.arange(-5, 5.01, 0.1) * kms
    # linewidth
    fwhm = 3 * kms
    # Gaussian model
    g = cps2.models.Gaussian1D(amplitude=1, mean=0, stddev=(fwhm.to_value() / 2.355))
    t_arr = g(v_arr.to_value())

    if False:
        plt.plot(v_arr.to_value(), t_arr)
        plt.show()

    thin_line_stub = "13co10-pmo"
    thin_channel_uncertainty = get_onesigma(thin_line_stub) * u.K
    # in the CO column density function we trim 12co by 6 K (not related to error)
    # then we trim 13co by 3sigma (all channels)
    # then we trim again by 1sigma on the 13co moment 0 map which was created with the trimmed 13co > 3sigma
    # So for our test here, let's see what those errors look like.
    n_channels = v_arr.size
    dv = v_arr[1]-v_arr[0]
    newline_tab = "\n" + "\t"*3
    print(f"{thin_line_stub} 1 sigma channel noise{newline_tab}{thin_channel_uncertainty}")
    print(f"N channels{newline_tab}{n_channels}")
    print(f"channel width (dV){newline_tab}{dv:.3f}")
    mom0_1sigma = thin_channel_uncertainty * dv * np.sqrt(n_channels)
    print(f"Moment 0 1sigma{newline_tab}{mom0_1sigma:.3f}")
    t_arr = t_arr*thin_channel_uncertainty
    mom0 = np.trapz(t_arr, x=v_arr)
    print(f"Max, linewidth of test spectrum{newline_tab}{np.max(t_arr):.3f}, {fwhm:.3f}")
    print(f"Moment 0 of test spectrum{newline_tab}{mom0:.3f}")

    co_line = COColumnDensity(thin_line_stub)
    Tex = 30 * u.K
    ff = 1

    # Calculate thin line column density, from COColumnDensity._calculate_thin_line_column_density
    # Rotational partition function Qrot
    Qrot = (const.k_B * Tex / (const.h * co_line.B0)).decompose() + (1./3)
    # Exponential term
    exp_term = np.exp(co_line.Eu / Tex)
    # Some constants
    g = 2*co_line.Ju + 1
    S = co_line.Ju / g
    # Prefactor (after cancelling 4pi from top and bottom)
    prefactor_numerator = const.eps0 * 3 * const.k_B
    prefactor_denominator = 2 * np.pi**2 * co_line.nu * S * co_line.mu**2
    prefactor = prefactor_numerator / prefactor_denominator
    # All together
    thin_line_column_density = (prefactor * (Qrot/g) * exp_term * mom0/ff).to(u.cm**-2)
    h2_column_density = thin_line_column_density * ratio_12co_to_13co / ratio_12co_to_H2
    print(f"For Tex = {Tex}")
    print(f"Column density detection limit{newline_tab}{h2_column_density:.2E}")




class COColumnDensity:
    """
    May 17, 2023
    General use class to calculate CO column density and propagate uncertainty.
    Instances are specific to the optically thin species (isotope and transition)
    i.e. 13co10 or 13co32 or c18o10.
    The actual data is fed in outside of this class, which does no file I/O of
    its own.
    Making it a class instead of a function to have more fun with it.
    """

    # Constants, in order:
    # B_0 (MHz), E_upper (K), mu (Debye), nu (GHz), J_upper
    # From the above, we construct g and S
    # Looks like B_0 is constant to the molecule, and does not change with transtion
    _constants = {
        '13co10': (55101.01, 5.28880, 0.11046, 110.20135400, 1),
        '13co32': (55101.01, 31.73179, 0.11046, 330.587965300, 3), # see 2023-09-11 notes, still re-researching this
        'c18o10': (55101.01, 5.26868, 0.11046, 109.78217340, 1),
        # For the 12co lines, only the frequency is needed
        '12co10': 115.27120180,
        '12co32': 345.79598990,
    }

    _thick_line = {
        '13co10': '12co10',
        'c18o10': '12co10',
        '13co32': '12co32',
    }

    def __init__(self, optically_thin_line_stub):
        """
        Initialize instance using a line stub like '13co10', whatever the
        optically thin species is.
        The function will then obtain molecular constants from internal
        dictionaries.
        :param optically_thin_line_stub: str, line stub describing optically
            thin CO line
        """
        # Take out anything on the other side of a hyphen and ignore it
        if '-' in optically_thin_line_stub:
            optically_thin_line_stub = optically_thin_line_stub.split('-')[0]
        if optically_thin_line_stub not in self._constants:
            raise RuntimeError(f"Line {optically_thin_line_stub} not supported.")
        B0, Eu, mu, nu, self.Ju = self._constants[optically_thin_line_stub]
        self.B0 = B0 * u.MHz
        self.Eu = Eu * u.K
        self.mu = mu * u.Debye
        # relying on the units to correct bugs from mixing up "mu" and "nu"
        self.nu = nu * u.GHz
        # Get the frequency of the optically thick line used for Tex
        self.thick_nu = self._constants[self._thick_line[optically_thin_line_stub]] * u.GHz
        # Default filling factor is 1.0
        self.ff = 1.0 # can set a different one in .set_filling_factor()

    def set_data(self, peak_temperature, integrated_intensity, peak_thin_temperature=None):
        """
        Provide data to the COColumnDensity instance.
        Both inputs should be Quantity arrays of matching dimension and share
        WCS, so that a given pixel references the same sky position
        in both images. WCS info does not need to be specified at this stage,
        just needs to be the same for both.
        :param peak_temperature: Quantity, peak temperature of optically thick
            line to be used as excitation temperature Tex
            Can also be tuple(Quantity, WCS) where the WCS facilitates
            reprojection if the peak_temperature grid does not align perfectly
            with the integrated intensity grid. peak_temperature will be
            reprojected to integrated intensity.
        :param integrated_intensity: Quantity, integrated intensity of optically
            thin line to be used for column density calculation
            If WCS was given for peak_temperature, then this should also be a
            tuple(Quantity, WCS). This integrated intensity WCS will be the
            output map projection.
        :param peak_thin_temperature: (optional) Quantity, peak temperature of
            optically thin line to be used for calculating more precise
            optical depth and excitation temperature
            No WCS is necessary; WCS must be the same for integrated_intensity
            since they are from the same line map
        """
        if isinstance(peak_temperature, tuple):
            # WCS given; check shape and reproject the maps onto each other
            peak_T, peak_T_wcs = peak_temperature
            int_intens, int_intens_wcs = integrated_intensity
            # WCS set to the integrated intensity WCS
            self.wcs = int_intens_wcs
            # Check if reprojection needed; use shapes as a proxy, even though this isn't totally infallible (it's fine for these maps)
            if peak_T.shape != int_intens.shape:
                # Reproject peak_T to integrated intensity grid
                peak_T_reproj, fp = reproject_interp((peak_T.to_value(), peak_T_wcs), int_intens_wcs, shape_out=int_intens.shape, return_footprint=True)
                # Find anywhere where the footprint is 0, i.e. places where int_intens has coverage but peak_T doesn't
                not_overlapping_pixels = fp==0
                if np.any(not_overlapping_pixels):
                    # Not perfect overlap
                    # Just assign NaNs to the fp==0 stuff for now
                    int_intens[not_overlapping_pixels] = np.nan
                self.peak_temperature = peak_T_reproj * peak_T.unit
                self.integrated_intensity = int_intens
                assert hasattr(self.peak_temperature, "unit")
                assert hasattr(self.integrated_intensity, "unit")
            else:
                # No reprojection
                self.peak_temperature = peak_T
                self.integrated_intensity = int_intens
        else:
            assert not isinstance(integrated_intensity, tuple)
            self.wcs = None
            self.peak_temperature = peak_temperature
            self.integrated_intensity = integrated_intensity

        self.peak_thin_temperature = peak_thin_temperature
        if self.peak_thin_temperature is not None:
            assert self.peak_thin_temperature.shape == self.integrated_intensity.shape
            print("Warning: The optical depth estimate is not yet implemented. See some of the 2023-10-18 notes on root finders and other possible solutions.")

    def set_filling_factor(self, ff):
        """
        Set a beam filling factor. Measured beam temperatures will be divided by
        this.
        """
        assert 0 < ff <= 1
        self.ff = ff

    def set_uncertainty(self, e_pt, e_ii):
        """
        Set the uncertainties for both data values.
        Argument order is the same as for self.set_data():
            peak_temperature, integrated_intensity
        Both arguments should be scalar Quantity objects, but could adapt this
        in the future to accept maps if uncertainty ever varies.
        :param e_pt: scalar (0-D, single value) Quantity, uncertainty in
            peak_temperature. This is just channel noise of the optically thick
            line.
        :param e_ii: scalar Quantity, uncertainty in integrated_intensity.
            This must be calculated from the channel noise, the channel width,
            and the number of integrated channels for the optically thin line.
        """
        self.e_pt = e_pt
        self.e_ii = e_ii

    def set_abundance_ratios(self, thick_to_thin, co_to_h2):
        """
        Give abundance ratios for, for example, 12CO/13CO and 12CO/H2.
        :param thick_to_thin: float, scalar abundance ratio expressed as
            12CO/13CO or something similar, so that the number is > 1.
            This is an argument rather than an internally stored variable because
            the abundance may depend on galactocentric radius (in the Galactic context).
        :param co_to_h2: float, scalar abundance ratio expressed as
            12CO/H2. This is opposite to the ratio above for 12CO/13CO) since
            this its more common appearance in the literature. This means the
            number should be < 1, typically ~8e-5.
            Leaving this as an argument just in case, though I will probably
            be using 8.5e-5 in any Galactic context in the near future,
            from Xander's 2021(?) book.
        """
        self.ratio_thick_to_thin = thick_to_thin
        self.ratio_co_to_h2 = co_to_h2

    @staticmethod
    def calculate_Tex(t_b, nu):
        """
        Calculate Tex from peak TB without the RJ approximation
        RJ approximation h*nu << kT implies TB = Tex. Not applicable for
        CO 3-2, and only somewhat for CO 1-0. Better to correct properly, which
        is what we're doing here.
        See the CO column density notes.

        Equation to implement is the basic definition of brightness temperature
        T_ex = (h*nu / k) * [ln{ h*nu/(k*TB) + 1 }]^-1

        staticmethod so that we can use this outside the function, it's really
        useful. It does not depend on any instance attributes.

        :param t_b: Quantity, measured brightness temperature T_B
        :returns: Quantity excitation temperature T_ex
        """
        hnu_kB = const.h * nu / const.k_B
        return (hnu_kB / np.log((hnu_kB/t_b) + 1)).to(u.K)

    def _calculate_thin_line_column_density(self):
        """
        Calculate the column densities and propagate uncertainty.
        Makes use of astropy.constants (const)
        """
        # Let Tex be equal to peak MB temperature in the opt. thick line.
        # This is an assumption, so I'll keep them as separate variables.
        self.Tex = self.calculate_Tex(self.peak_temperature/self.ff, self.thick_nu)
        self.e_Tex = self.e_pt # # TODO: need to error propagate this!! I'll leave it as it is now because it won't be a ton of error.
        # Rotational partition function Qrot
        Qrot = (const.k_B * self.Tex / (const.h * self.B0)).decompose() + (1./3)
        e_Qrot = (const.k_B * self.e_Tex / (const.h * self.B0)).decompose() # constant falls of in derivative
        # Exponential term
        exp_term = np.exp(self.Eu / self.Tex)
        e_exp_term = self.e_Tex * exp_term * self.Eu/(self.Tex**2) # d(e^(a/x)) = (a dx / x^2) e^(a/x)
        # Some constants
        g = 2*self.Ju + 1
        S = self.Ju / g
        # Prefactor (after cancelling 4pi from top and bottom)
        prefactor_numerator = const.eps0 * 3 * const.k_B
        prefactor_denominator = 2 * np.pi**2 * self.nu * S * self.mu**2
        prefactor = prefactor_numerator / prefactor_denominator
        # All together
        self.thin_line_column_density = (prefactor * (Qrot/g) * exp_term * self.integrated_intensity/self.ff).to(u.cm**-2)
        # Uncertainty! d(cxyz) = cyz dx + cxz dy + cxy dz. But you gotta do quadrature sum instead of regular sum
        # Collected all constants (prefactor_numerator/prefactor_denominator and 1/g) at the end, outside the derivatives and quad sum
        helper_1 = (Qrot * exp_term * self.e_ii)**2
        helper_2 = (Qrot * e_exp_term * self.integrated_intensity)**2
        helper_3 = (e_Qrot * exp_term * self.integrated_intensity)**2
        self.e_thin_line_column_density = (np.sqrt(helper_1 + helper_2 + helper_3) * prefactor/g).to(u.cm**-2)

    def _calculate_thick_line_column_density(self):
        """
        Make the much simpler jump from optically thin line column to the 12CO column
        For now, I don't have a 12CO/C18O abundance, so that one will have to wait.
        Only 12/13 ratio.
        """
        self.thick_line_column_density = self.ratio_thick_to_thin * self.thin_line_column_density
        self.e_thick_line_column_density = self.ratio_thick_to_thin * self.e_thin_line_column_density

    def _calculate_H2_column_density(self):
        """
        Make the simple jump from 12CO column density to H2 column density
        """
        self.H2_column_density = self.thick_line_column_density / self.ratio_co_to_h2
        self.e_H2_column_density = self.e_thick_line_column_density / self.ratio_co_to_h2

    def calculate_column_density(self):
        """
        Calculate all column densities. Call this method of an instance.
        """
        self._calculate_thin_line_column_density()
        self._calculate_thick_line_column_density()
        self._calculate_H2_column_density()

    def calculate_mass_per_pixel(self, wcs_obj, los_distance, e_los_distance=0., thin_line_beam=None):
        """
        Calculate the H2 gas mass in each pixel.
        Mean molecular weight of 1.33 is assumed, implying Y = 0.25 and Z = 0.
        :param wcs_obj: WCS object describing the image grid used for the input data.
        :param los_distance: scalar Quantity, heliocentric distance to the target.
        :param e_los_distance: scalar Quantity, uncertainty on the heliocentric
            distance to the target los_distance. Default is 0, implying no
            uncertainty in this value.
        :param thin_line_beam: Beam, optional, the Beam object for the optically
            thin CO line observation. Used to correct the mass/pixel error map
            for correlated pixels so that the mass errors from summing over
            these pixels is not unrealistically small.
            The correction only affects the error map, not the value map.
            If this argument is left alone or set to None, this correction will
            be skipped.
        """
        self.los_distance = los_distance
        self.e_los_distance = e_los_distance
        # Pixel area
        self.pixel_scale = misc_utils.get_pixel_scale(wcs_obj) # angular unit Quantity
        self.pixel_area = (self.pixel_scale * self.los_distance / u.radian)**2 # physical area
        e_pixel_area = 2 * (self.pixel_scale/u.radian)**2 * self.los_distance * self.e_los_distance
        # Particle mass
        # H2 mass = 2 * H mass
        self.particle_mass = 2 * mean_molecular_weight_neutral * Hmass
        self.mass_per_pixel = (self.pixel_area * self.H2_column_density * self.particle_mass).to(u.solMass)
        # Uncertainty from both column density and distance
        e_from_pixel = e_pixel_area * self.H2_column_density * self.particle_mass
        e_from_coldens = self.pixel_area * self.e_H2_column_density * self.particle_mass
        self.e_mass_per_pixel = np.sqrt(e_from_pixel**2 + e_from_coldens**2).to(u.solMass)
        # Correlated pixel correction
        if thin_line_beam is not None:
            self.pixels_per_beam = (thin_line_beam.sr / self.pixel_scale**2).decompose()
            # Multiply the uncertainties by the sqrt of pixels/beam oversampling factor
            self.e_mass_per_pixel *= np.sqrt(self.pixels_per_beam)
        else:
            # Set pixels_per_beam to None as a signal that the correction wasn't made
            self.pixels_per_beam = None


    def create_header_comments(self):
        """
        Create and return a list of header items that describe the
        calculation process. These include relevant scalar quantities that we
        might want to reference later.
        :returns: list, FITS Header-standard tuples of (key, value) with str elements.
        Does not override WCS information at all, only adds DATE, CREATOR, HISTORY, and COMMENTS.
        """
        def process_text_into_phrases(text):
            """ Turn a block of newline-separated string text into individual strings """
            # Remove trailing and leading whitespace from entire block
            # Split by newline and remove leading/trailing whitespace from phrases
            return [line.strip() for line in text.strip().split("\n")]

        header_text = f"""
            12CO/H2 = {self.ratio_co_to_h2:.2E}
            12C/13C = {self.ratio_thick_to_thin:.2f}
            Hmass = {Hmass:.3E}
            mean mol weight (atomic) = {mean_molecular_weight_neutral:.2f}
            adopted particle mass = {self.particle_mass:.2E}
            pixel scale = {self.pixel_scale.to(u.arcsec):.3E}
            pixel area = {self.pixel_area.to(u.pc**2):.3E}
            LOS distance = {self.los_distance.to(u.pc):.2f}
            err LOS distance = {self.e_los_distance.to(u.pc):.2f}
        """
        header_phrases = process_text_into_phrases(header_text)
        if self.pixels_per_beam is not None:
            optional_text = f"""
                sqrt(pixels/beam) multiplied into the uncertainties to account for oversampling
                sqrt(pixels/beam) = {np.sqrt(self.pixels_per_beam):.2f}
            """
            header_phrases += process_text_into_phrases(optional_text)
        date_text = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        creator_text = f"Ramsey, {__file__}.{type(self).__name__}.create_header_comments"

        header_kws = [('DATE', date_text), ('CREATOR', creator_text)]
        header_kws += [('HISTORY', phrase) for phrase in header_phrases]
        return header_kws


def get_co_spectra_for_radex():
    """
    October 13, 2023 (Friday the 13th I just realized)
    Grab some CO spectra for spectralradex (working in spectralradex_column_fit.py)
    """
    # selected_coord = SkyCoord("18:18:14.9371 -13:34:26.988", unit=(u.hourangle, u.deg), frame='fk5') # first test coord, sort of bright west side of N19
    # selected_coord = SkyCoord("18:18:44.6122 -13:33:40.152", unit=(u.hourangle, u.deg), frame='fk5') # second test coord, smooth part of east N19 where Tex is same for CO 1-0 and 3-2

    selected_coord = SkyCoord("18:18:17.8904 -13:33:04.941", unit=(u.hourangle, u.deg), frame='fk5') # north slightly west on the CII ring
    # selected_coord = SkyCoord("18:18:40.6282 -13:34:27.298", unit=(u.hourangle, u.deg), frame='fk5') # north slightly west on the CII ring

    print("THIS IS FOR CII!")

    line_stub = "cii"
    cube_co = cube_utils.CubeData(get_map_filename(line_stub)).convert_to_K().convert_to_kms()
    x, y = cube_co.wcs_flat.world_to_pixel(selected_coord)
    plt.subplot(121)
    vel_lims = (11*kms, 21*kms)
    plt.imshow(cube_co.data.spectral_slab(*vel_lims).moment0().to_value(), origin='lower')
    plt.plot([x], [y], marker='o', color='red')
    plt.title(make_vel_stub(vel_lims))
    plt.subplot(122)
    # print("CUBE SHAPE", cube_co.data.shape)
    spectrum = cube_co.data[:, int(round(float(y))), int(round(float(x)))]
    plt.plot(cube_co.data.spectral_axis.to_value(), spectrum.to_value())

    # Use a few different methods of finding half-max of the line centered near 20 km/s
    # Trim the spectrum at <21 (anything >21 is not N19; can see that in a 21-23 moment)
    trim = cube_co.data.spectral_axis < 21*kms
    trimmed_x_axis = cube_co.data.spectral_axis.to_value()[trim]
    trimmed_spectrum = spectrum[trim]

    # Check if increasing x axis (13CO sometimes is flipped)
    # Flip if so; UnivariateSpline needs x axis increasing
    if trimmed_x_axis[0] > trimmed_x_axis[-1]:
        trimmed_x_axis = trimmed_x_axis[::-1]
        trimmed_spectrum = trimmed_spectrum[::-1]

    plt.axvline(21, color='k', linestyle='--', alpha=0.6)
    plt.fill_between(trimmed_x_axis, trimmed_spectrum.to_value(), where=(trimmed_spectrum.to_value() > 0), alpha=0.3)

    # Find half-max the simplest way: draw a line
    T_R_max = np.nanmax(trimmed_spectrum.to(u.K).to_value())
    peak_vel_max = trimmed_x_axis[trimmed_spectrum.argmax()]
    plt.axhline(T_R_max/2)
    # for the point at "18:18:14.9371 -13:34:26.988" in 12co32, half-max looks by-eye to be 18-21 so 3 kms
    linewidth_eye = 3*kms

    # print(type(spectrum)) ## doesn't seem to work like cube[:, y, x].linewidth_fwhm(), since OneDSpectrum doesn't have linewidth methods
    linewidth_sc_fwhm = cube_co.data.spectral_slab(11*kms, 21*kms).linewidth_fwhm()[int(round(float(y))), int(round(float(x)))]
    # linewidth_fwhm gives 2.7 km/s for that test point, pretty close to 3 by eye

    # import down here since I don't need this for other functions (and I only run ~1 function per script execution with these files)
    from scipy.interpolate import UnivariateSpline

    spline_fit = UnivariateSpline(trimmed_x_axis, trimmed_spectrum.to_value() - T_R_max/2, s=trimmed_x_axis.size//2) # can try len or len//2, but 0 is probbaly not ideal
    plt.plot(trimmed_x_axis, spline_fit(trimmed_x_axis) + T_R_max/2, linestyle=':', label='Spline')
    # print(spline_fit.roots()) # if the spectrum doesn't actually hit half-max (like this one, background component), only 1 root; but, that root is right!

    # Fit with astropy.modeling, which is already imported into cps2 (models and fitting)
    gauss_0 = cps2.models.Gaussian1D(amplitude=T_R_max, mean=peak_vel_max, stddev=linewidth_sc_fwhm.to_value()/2.355,
        bounds={'mean': (peak_vel_max-(linewidth_sc_fwhm.to_value()/2), peak_vel_max+(linewidth_sc_fwhm.to_value()/2)),
                'amplitude': (T_R_max*0.6, T_R_max*1.1)})
    fitter = cps2.fitting.LevMarLSQFitter()
    gauss_fit = fitter(gauss_0, trimmed_x_axis, trimmed_spectrum.to_value())
    analog_results = [peak_vel_max, T_R_max, linewidth_sc_fwhm.to_value()]
    analog_colnames = ["Peak V", "Peak T", "FWHM sc"]
    gauss_results = [gauss_fit.mean.value, gauss_fit.amplitude.value, gauss_fit.stddev.value*2.355]
    gauss_colnames = ["Mean", "Amplitude", "FWHM G"]
    results = list(itertools.chain.from_iterable(zip(analog_results, gauss_results)))
    colnames = list(itertools.chain.from_iterable(zip(analog_colnames, gauss_colnames)))
    # results = analog_results + gauss_results ## alternate column ordering

    results_str = ["", line_stub] + [f"{x:.2f}" for x in results] + [""] # empty string elements so that .join() adds at beginning and end
    colnames = ["", "Line Stub"] + colnames + [""]

    print(" | ".join(colnames).strip())
    print("|".join(("---" if s else s) for s in colnames))
    print(" | ".join(results_str).strip())
    # print("peak vel", peak_vel_max)
    # print("peak T", T_R_max)
    # print("SC FWHM", linewidth_sc_fwhm)
    # print("A", gauss_fit.amplitude.value)
    # print("M", gauss_fit.mean.value)
    # print("FWHM", gauss_fit.stddev*2.355)

    plt.plot(trimmed_x_axis, gauss_fit(trimmed_x_axis), linestyle=':', label='Gauss')

    plt.legend()

    plt.show()


def get_co32_to_10_ratio_for_density(velocity_limits=None, isotope10='13', noise_cutoff=None):
    """
    December 9, 2023
    Use the CO 3-2 convolved to purple mountain.
    Do 11-20 km/s and 21-27 km/s bins, take max of each 13CO line.

    Dec 14: added in the uncertainty; ratios are easy using fractional uncertainty

    Dec 16: switched the order of the peak T frames so that they are saved with
    numerator 3rd and denominator 4th/last. Matches the order of the 13CO / C18O ratio files
    """
    if velocity_limits is None:
        if False:
            velocity_limits = (11*kms, 21*kms)
        else:
            velocity_limits = (21*kms, 27*kms)
    results = {}
    frac_errors = {}
    savepath = None

    if noise_cutoff is None:
        noise_cutoff = 5

    if isotope10 == '13':
        stub10 = '13co10-pmo'
    elif isotope10 == '18':
        stub10 = 'c18o10-pmo'
    for stub in ['13co32-pmo', stub10]:
        fn = get_map_filename(stub)
        cube = cube_utils.CubeData(fn).convert_to_K()
        if stub == '13co32-pmo':
            savepath = cube.directory
        subcube = cube.data.spectral_slab(*velocity_limits)
        peak_T_map = subcube.max(axis=0).to(u.K).to_value()
        # Filter maps by a few times noise
        peak_T_map[peak_T_map < (get_onesigma(stub) * noise_cutoff)] = np.nan
        results[stub] = (peak_T_map, cube.wcs_flat)
    peak_T_32, wcs_32 = results['13co32-pmo']
    peak_T_10, wcs_10 = results[stub10]
    peak_T_10 = reproject_interp((peak_T_10, wcs_10), wcs_32, shape_out=peak_T_32.shape, return_footprint=False)
    ratio_32_10 = peak_T_32 / peak_T_10

    # Calculate uncertainty outside of the loop since we had to reproject
    frac_err_32 = get_onesigma('13co32-pmo') / peak_T_32
    frac_err_10 = get_onesigma(stub10) / peak_T_10
    ratio_err_frac = np.sqrt(frac_err_32**2 + frac_err_10**2)
    ratio_err = ratio_32_10 * ratio_err_frac

    data_list = [
        (ratio_32_10, "ratio_32_to_10", ''), # ratio
        (ratio_err, "err_ratio_32_to_10", ''), # error on ratio
        (peak_T_32, "peak_32", 'K'), # numerator (used to be the last frame, moved on Dec 16 2023)
        (peak_T_10, "peak_10", 'K'), # denominator
    ]
    hdu_list = [fits.PrimaryHDU()]
    header_template = wcs_32.to_header()
    header_template['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    header_template['CREATOR'] = f"Ramsey, {__file__}.get_co32_to_10_ratio_for_density"
    header_template['COMMENT'] = f"Using 13co32 (pmo resolution) and {stub10}"
    header_template['HISTORY'] = f"Cutoff each peak T map by > {noise_cutoff} sigma"
    header_template['HISTORY'] = f"Error calculated using the flat RMS uncertainties for each cube."
    for data, extname, unit_str in data_list:
        hdr = header_template.copy()
        hdr['EXTNAME'] = extname
        hdr['BUNIT'] = unit_str
        hdu = fits.ImageHDU(data=data, header=hdr)
        hdu_list.append(hdu)
    hdul = fits.HDUList(hdu_list)
    if isotope10 == '13':
        isotope_stub = ""
    elif isotope10 == '18':
        isotope_stub = "c18o_"
    savename = os.path.join(savepath, f"ratio_v2_13co_32_to_{isotope_stub}10_pmo_{make_simple_vel_stub(velocity_limits)}.fits")
    hdul.writeto(savename, overwrite=True)


def get_13co10_to_c18o10_ratio_for_opticaldepth(velocity_limits=None):
    """
    December 14, 2023
    Lee said this is a common diagnostic to find out if 13CO is optically thick.
    Uses the same transition, in our case 1-0
    """
    if velocity_limits is None:
        velocity_limits = (21*kms, 27*kms)
    savepath = None
    wcs_obj = None
    results = {}
    frac_errors = {}
    for stub in ['13co10-pmo', 'c18o10-pmo']:
        fn = get_map_filename(stub)
        cube = cube_utils.CubeData(fn).convert_to_K()
        # This runs twice but it doesn't matter, they are in the same directory
        savepath = cube.directory
        wcs_obj = cube.wcs_flat
        subcube = cube.data.spectral_slab(*velocity_limits)
        peak_T_map = subcube.max(axis=0).to(u.K).to_value()
        # Skip filtering by noise, just leave it alone
        results[stub] = peak_T_map
        # Save the fractional error
        frac_errors[stub] = get_onesigma(stub) / peak_T_map
        if 'pmo' not in stub:
            # PMO data are on the same exact grid, so no need to regrid; I checked Dec 14, 2023
            raise RuntimeError("At least one of these isn't PMO; you should probably regrid or check and make the explicit exception.")
    ratio_13_18 = results['13co10-pmo'] / results['c18o10-pmo']

    # Calculate uncertainty
    # Fractional uncertainty is fairly easy in this simple geometric function
    # (sigma_f / f)^2 = (sigma_a / a)^2 + (sigma_b / b)^2
    ratio_err_frac = np.sqrt(frac_errors['13co10-pmo']**2 + frac_errors['c18o10-pmo']**2)
    # Get the absolute error on the ratio by multiplying the fractional error by the ratio
    ratio_err = ratio_err_frac * ratio_13_18

    data_list = [
        (ratio_13_18, "ratio_13_to_18", ''), # ratio
        (ratio_err, "err_ratio_13_to_18", ''), # error on ratio
        (results['13co10-pmo'], "peak_13co10", 'K'), # numerator
        (results['c18o10-pmo'], "peak_c18o10", 'K'), # denominator
    ]
    hdu_list = [fits.PrimaryHDU()]
    header_template = wcs_obj.to_header()
    header_template['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    header_template['CREATOR'] = f"Ramsey, {__file__}.get_13co10_to_c18o10_ratio_for_opticaldepth"
    header_template['COMMENT'] = f"Using 13co10-pmo and c18o10-pmo"
    header_template['HISTORY'] = f"Error calculated using the flat RMS uncertainties for each cube."
    for data, extname, unit_str in data_list:
        hdr = header_template.copy()
        hdr['EXTNAME'] = extname
        hdr['BUNIT'] = unit_str
        hdu = fits.ImageHDU(data=data, header=hdr)
        hdu_list.append(hdu)
    hdul = fits.HDUList(hdu_list)
    savename = os.path.join(savepath, f"ratio_13co_to_c18o_10_pmo_{make_simple_vel_stub(velocity_limits)}.fits")
    hdul.writeto(savename, overwrite=True)


class COData:
    """
    December 19, 2023
    Inspired by the code in sample_multiple_maps(), need a common format for
    these multi-extension column density and ratio maps.
    This class will hold a lookup dictionary for filenames and will also know
    the extensions and data names.

    This class deals with observed data.

    I'll throw in the useful loading/sampling/unpacking functions into here too.
    """

    def __init__(self, velocity_limits):
        # All the data sources we will use and keys to describe them
        self.velocity_limits = velocity_limits
        self.vel_stub_simple = make_simple_vel_stub(self.velocity_limits)
        self._map_filenames = {
            "column_70-160": "herschel/coldens_70-160_colorsolution_70zeroedat160.fits",
            "column_160-500": "herschel/m16_coldens_high.fits",
            "column_13co10": f"purplemountain/column_density_v3__13co10-pmo_{self.vel_stub_simple}.fits",
            "column_c18o10": f"purplemountain/column_density_v3__c18o10-pmo_{self.vel_stub_simple}.fits",
            "ratio_32_to_10": f"apex/ratio_v2_13co_32_to_10_pmo_{self.vel_stub_simple}.fits", # 13co32 to 13co10
            "ratio_13_to_18": f"purplemountain/ratio_13co_to_c18o_10_pmo_{self.vel_stub_simple}.fits", # 13co10 to c18o10
            "12co10": "12co10-pmo",
            "13co10": "13co10-pmo",
            "c18o10": "c18o10-pmo",
            "12co32": "12co32-pmo",
            "13co32": "13co32-pmo",
        }
        """
        The extensions to each FITS file
        Top level keys match _map_filenames keys (KEY)
        Their values are tuples. The first element indicates what kind of data it is (DATA TYPE)
        The second element is either a tuple or a string (DATA INFO)
        If DATA INFO a tuple, the elements represent a single extension / map in a multi-extension file
        That tuple's elements are 2-tuples which are EXTNAME, DATA NAME
        data_name is the descriptive name (DATA NAME) of the extension that should be used in the DataFrame
        If DATA INFO is a string, then it's a single extension. string is the descriptive name DATA NAMAE

        All together, that means that
        self._key_lookup = {
            KEY: (DATA TYPE, DATA INFO)
        }
        DATA TYPE:
            single, multi, or cube
        DATA INFO: depends on DATA TYPE.
            single: DATA NAME
            multi: (EXTNAME, DATA NAME)
            cube: DATA NAME
        """
        self._key_lookup = {
            "column_70-160": ("single", "column_density_70-160"),
            "column_160-500": ("single", "column_density_160-500"),
            "column_13co10": ("multi", (("H2coldens", "column_density_13co10"), ("err_H2coldens", "err_column_density_13co10"))),
            "column_c18o10": ("multi", (("H2coldens", "column_density_c18o10"), ("err_H2coldens", "err_column_density_c18o10"))),
            "ratio_32_to_10": ("multi", ((1, "ratio_32_10"), (2, "err_ratio_32_10"))), # (3, "peak_13co32"), (4, "peak_13co10"))),
            "ratio_13_to_18": ("multi", ((1, "ratio_13_18"), (2, "err_ratio_13_18"))), # (3, "peak_13co10"), (4, "peak_c18o10"))),
            "12co10": ("cube", "peak_12co10"),
            "13co10": ("cube", "peak_13co10"),
            "c18o10": ("cube", "peak_c18o10"),
            "12co32": ("cube", "peak_12co32"),
            "13co32": ("cube", "peak_13co32"),
        }
        self.sample_type_setting = None
        self.sample_framework_setting = None
        self.diagnostic_plot = False

    def get_load_information(self, data_key):
        """
        Create an "extnames_to_extract" dict using the data_key
        or return the data_name of the single cube or extension
        :returns: tuple, 2 elements
            1) either the extnames_to_extract dict or the string data_name
            2) data_type string: single, cube, or multi
        """
        # Check for some short-circuit cases
        if data_key not in self._key_lookup:
            raise RuntimeError(f"{data_key} not in {__class__} lookup tables")
        # data_type is a string describing the format of the data, not type(data)
        data_type, data_info = self._key_lookup[data_key]
        if data_type in ['single', 'cube']:
            data_name = data_info
            return data_name, data_type
        elif data_type == 'multi':
            extnames_to_extract = {}
            for extname, data_name in data_info:
                extnames_to_extract[data_name] = extname
            return extnames_to_extract, data_type
        else:
            raise RuntimeError(f"Unknown format <{data_type}>")

    def sample_all_data(self):
        """
        December 20, 2023
        Convenience function for looping self.sample_data() through all keys
        in self._key_lookup
        Builds a single dictionary and returns it.
        """
        return_dict = {}
        for k in self._key_lookup:
            return_dict.update(self.sample_data(k))
        return return_dict

    def sample_data(self, data_key, sample_framework=None, sample_type=None):
        """
        Load multi- or single-extension or cube FITS data and return the result
        of sampling it somehow.
        sample_framework and sample_type can be None and will be passed through
        so that defaults can be checked later.

        This could probably be combined with self.get_load_information but it's
        not worth the time to refactor.

        This is a user-facing function.
        """
        name_or_extnames, data_type = self.get_load_information(data_key)
        if data_type == 'multi':
            return self.load_multi_extension_data(data_key, name_or_extnames, sample_framework, sample_type)
        elif data_type == 'single':
            return self.load_single_extension_data(data_key, name_or_extnames, sample_framework, sample_type)
        elif data_type == 'cube':
            return self.load_cube_and_find_peak(data_key, name_or_extnames, sample_framework, sample_type)


    def load_multi_extension_data(self, data_key, extnames_to_extract, sample_framework, sample_type):
        """
        Load a multi-extension FITS file and extract values from several
        extensions
        :param data_key: a key to self._map_filenames
        :param extnames_to_extract: should be a dict with "data names" as keys;
            these should be short descriptive strings that will be attached to
            the values after extraction.
            The dict values should be the EXTNAME in the FITS file.
        :returns: dict, the keys are the same as extnames_to_extract
            and the values are lists of float values.

        Historical
        :param short_filename: a relative path which can be solved with
            catalog.utils.search_for_file
        """
        short_filename = self._map_filenames[data_key]
        return_dict = {}
        with fits.open(catalog.utils.search_for_file(short_filename)) as hdul:
            for data_name in extnames_to_extract:
                extname = extnames_to_extract[data_name]
                values = self.extract_values_from_image(hdul[extname].data, WCS(hdul[extname].header), sample_framework=sample_framework, sample_type=sample_type)
                return_dict[data_name] = values
        return return_dict

    def load_single_extension_data(self, data_key, data_name, sample_framework, sample_type):
        """
        Simpler than multi, somewhat. This time, pass the data name as an
        argument directly.
        I want to use getdata for this because I don't want to guess if the
        data extension is 0 or 1.
        :returns: dict with only one entry. data_name as the key, the value list
            as the value
        """
        short_filename = self._map_filenames[data_key]
        data, header = fits.getdata(catalog.utils.search_for_file(short_filename), header=True)
        values = self.extract_values_from_image(data, WCS(header), sample_framework=sample_framework, sample_type=sample_type)
        return {data_name: values}

    def load_cube_and_find_peak(self, data_key, data_name, sample_framework, sample_type):
        """
        Dec 26 2023
        Same goal as load_single and load_multi, but different type of data.
        """
        line_stub = self._map_filenames[data_key]
        cube_obj = cube_utils.CubeData(get_map_filename(line_stub)).convert_to_K()
        peak_T_map = cube_obj.data.spectral_slab(*self.velocity_limits).max(axis=0).to(u.K).to_value()
        values = self.extract_values_from_image(peak_T_map, cube_obj.wcs_flat, sample_framework=sample_framework, sample_type=sample_type)
        # Kinda awkward way to get errors but it's what we gotta do
        error_map = np.ones_like(peak_T_map) * get_onesigma(line_stub)
        errors = self.extract_values_from_image(error_map, cube_obj.wcs_flat, sample_framework=sample_framework, sample_type=sample_type)
        return {data_name: values, "err_"+data_name: errors}


    def extract_values_from_image(self, data, wcs_obj, sample_framework=None, sample_type=None):
        """
        Given an array and WCS, grab values in a way defied by the sample_type

        This is a user-facing function

        :param sample_framework: some information for sampling the data.
            Maybe a region list, maybe a mask.
        :param sample_framework: str, type of sample to take.
            ["regions", "mask"] options supported at the moment.
        """
        # Load in sample information configuration if not set in function args
        if sample_type is None:
            sample_type = self.sample_type_setting
        if sample_framework is None:
            sample_framework = self.sample_framework_setting
        # Sample
        if sample_type == "regions":
            return self.extract_values_from_image_regions(data, wcs_obj, sample_framework)
        elif sample_type == "mask":
            # sample_framework is tuple(mask, mask_wcs)
            # "mask" implies getting the mean and standard deviation, etc
            return self.extract_values_from_image_mask(data, wcs_obj, *sample_framework)
        elif sample_type == "mask_vals":
            # sample_framework same as for "mask"
            # "mask_vals" implies getting the full list of values under the mask
            return self.extract_values_from_image_mask(data, wcs_obj, *sample_framework, return_all_values=True)
        else:
            raise RuntimeError(f"Sample type {sample_type} not supported.")

    def extract_values_from_image_regions(self, data, wcs_obj, reg_list):
        """
        Grab values using the region list.
        Return a list of values.
        """
        # Iterate regions and grab values at those pixels
        values_list = []
        for reg in reg_list:
            j, i = [int(round(c)) for c in reg.to_pixel(wcs_obj).center.xy]
            values_list.append(data[i, j])
        return values_list

    def extract_values_from_image_mask(self, data, wcs_obj, mask, mask_wcs, return_all_values=False):
        """
        Grab values using the mask.
        Return a tuple (value, error) where error is the standard deviation
        under the mask and value is the mean.
        :param mask: bool array
            Will be converted to array of float 0-1. 1 is True, 0 is False.
            Float means that we can reproject it! Can't reproject bool array.
            Will have to do (mask > 0.5) to make it a real bool array.
        :param return_all_values: bool
            Whether or not to return the list of values under the mask.
            If False, then return (mean, low, high, standard deviation).
            (low, high) are the 16th and 84th quantiles.
            If True, return the full list of values.

        At some point need to add error to this function, though the large avg
        over pixels will probably render a pre-existing pixel error uselessly
        small. The stddev error under the mask is probably more useful.
        Although... some of the PACS, SPIRE errors are absolute, not just
        statistical/relative. Like the background error. So that's probably
        worth dealing with in a smarter way.
        See 2023-12-19 notes for more info on how to do this
        """
        data_cut, wcs_cut, cutout = COData.cutout_to_footprint(data, wcs_obj, mask_wcs, mask.shape, return_cutout=True)
        if return_all_values:
            # Reproject the data to the mask so that the lists are all the same length
            data_reproj = reproject_interp((data_cut, wcs_cut), mask_wcs, shape_out=mask.shape, return_footprint=False)
            return data_reproj[mask]
        else:
            # Reproject the mask to the data so that we get faithful stats
            mask_reproj_float = reproject_interp((mask.astype(float), mask_wcs), wcs_cut, shape_out=data_cut.shape, return_footprint=False)
            mask_reproj = mask_reproj_float > 0.5
            values_under_mask = data_cut[mask_reproj]
            if self.diagnostic_plot:
                data_copy = data_cut.copy()
                data_copy[~mask_reproj] = np.nan
                big_data_copy = data*0 + 1
                big_data_copy[cutout.slices_original] = np.nan
                plt.figure()
                plt.subplot(231)
                plt.imshow(mask, origin='lower')
                plt.subplot(232)
                plt.imshow(mask_reproj, origin='lower')
                plt.subplot(233)
                plt.imshow(data_copy, origin='lower')
                plt.subplot(234)
                plt.imshow(data, origin='lower')
                plt.subplot(235)
                plt.imshow(big_data_copy, origin='lower')
            # Return mean, stddev
            clean_values = values_under_mask[np.isfinite(values_under_mask) & (values_under_mask > 0)]
            normalized = False
            norm_val = 1e19
            if np.any(clean_values > norm_val):
                # Some issues with very large numbers in the np.std() function; this addresses them.
                clean_values = clean_values / norm_val
                normalized = True
            mean = np.mean(clean_values)
            stddev = np.std(clean_values)
            lo, hi = misc_utils.flquantiles(clean_values, 6) # 6 approximates 16, 84 %iles for -?+ 1sigma
            if normalized:
                mean = mean * norm_val
                stddev = stddev * norm_val
                lo = lo * norm_val
                hi = hi * norm_val
            return {'value': mean, 'lo': lo, 'hi': hi, 'stddev': stddev}

    @staticmethod
    def cutout_to_footprint(target_data, target_wcs, reference_wcs, reference_shape, return_cutout=False):
        """
        So that we don't have to reproject the tiny mask to the huge Herschel image
        Target data, wcs is the stuff that should be cut down to size.
        reference wcs, shape describes the smaller footprint which should define
        the cutout.
        """

        # Okay, WCS.calc__footprint() is a nearly-useless function without some
        # serious assumptions. See 2023-12-28 notes. Have to find footprint manually
        footprint_coords = []
        for i in [0, reference_shape[0]-1]:
            for j in [0, reference_shape[1]-1]:
                footprint_coords.append(reference_wcs.array_index_to_world(i, j))

        ra = [x.fk5.ra.deg for x in footprint_coords]
        de = [x.fk5.dec.deg for x in footprint_coords]
        min_ra, max_ra = np.min(ra), np.max(ra)
        min_de, max_de = np.min(de), np.max(de)

        """
        An older version of this function used this code, but it relies on
        knowing the frame (FK5, Galactic, etc) a priori. There is not a good
        way to get the frame from the WCS object, so I changed the code to
        use WCS.array_index_to_world, which gives SkyCoords that can be
        converted easily.

        reference_footprint = reference_wcs.calc_footprint(axes=reference_shape)
        ra, de = reference_footprint[:, 0], reference_footprint[:, 1]
        """

        min_ra, max_ra = np.min(ra), np.max(ra)
        min_de, max_de = np.min(de), np.max(de)
        center_ra, center_de = (min_ra + max_ra)/2, (min_de + max_de)/2
        size_ra, size_de = (max_ra - min_ra), (max_de - min_de)

        # size will be flipped (y, x) = (de, ra) because that's Cutout2D's call signature
        cutout = Cutout2D(target_data, SkyCoord(center_ra*u.deg, center_de*u.deg, frame="fk5"), wcs=target_wcs, size=(size_de*u.deg, size_ra*u.deg))
        if return_cutout:
            return cutout.data, cutout.wcs, cutout
        else:
            return cutout.data, cutout.wcs



def sample_multiple_maps_regions(velocity_limits=None):
    """
    December 15, 2023
    Sample the column density and line ratio maps using a region file and print
    out the results.

    The first region file is m16_column_sample_points_21-27.reg
    the second one is 11-21
    """
    # Eventually this will be a selection of different reg files
    if velocity_limits[0] == 21*kms:
        reg_filename_short = "catalogs/m16_column_sample_points_21-27.reg"
    elif velocity_limits[0] == 11*kms:
        reg_filename_short = "catalogs/m16_column_sample_points_11-21.reg"
    else:
        raise RuntimeError
    # All the data sources we will use and keys to describe them
    vel_stub_simple = make_simple_vel_stub(velocity_limits)

    # We will also get the peak 12CO 3-2 line. The peak 13CO 3-2 line is already in the ratio_32_to_10 map

    # The values and errors are scattered throughout different extensions of these files, so we will have some functions to help
    # Moved functions to COData

    """
    Get all the extension names or numbers lined up with data name keys

    The ratios are organized the same way:
    0: empty (primary)
    1: ratio
    2: error on ratio
    3: numerator (peak T)
    4: denominator (peak T)

    I confirmed on Dec 16 23 that this is true for both sets of ratios.

    The column density maps store column as follows:
    H2coldens
    err_H2coldens

    The Herschel maps are both single-extension

    And the 12 CO 3-2 peak line brightness has to be extracted from the cube
    """

    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    result_dict = {"reg_name": [reg.meta['text'] for reg in reg_list]}
    lookup_obj = COData(velocity_limits)
    lookup_obj.sample_type_setting = "regions"
    lookup_obj.sample_framework_setting = reg_list
    result_dict.update(lookup_obj.sample_all_data())
    result_df = pd.DataFrame(result_dict)


    save_df_path = os.path.join(catalog.utils.m16_data_path, "misc_regrids")
    assert os.path.exists(save_df_path)
    save_df_name = f"sample_points_test_1_{vel_stub_simple}.csv"
    save_df_full_path = os.path.join(save_df_path, save_df_name)

    """
    Divide the Herschel 70-160 columns by 2 because they are N_H and everything
    else is N(H2); N_H = N(H) + 2*N(H2) and N(H) is 0 by assumption.
    """

    result_df["column_density_70-160"] = result_df["column_density_70-160"] / 2
    print(result_df)

    result_df.to_csv(save_df_full_path)


def sample_masked_map(region_name='N19'):
    """
    December 17, 2023
    Try using a mask-based approach to sampling. Take averages of each quantity
    and output just one value + error per velocity interval / area

    First mask will be N19 molecular shell

    :param region_name: either 'N19' or 'BNR' (Bright Northern Ridge)
    """
    # Construct and test the  mask
    mask_data_stub = "12co32-pmo"
    cube = cube_utils.CubeData(get_map_filename(mask_data_stub)).convert_to_K().convert_to_kms()


    if region_name == "N19":
        mask_velocity_limits = (15*kms, 21*kms)
        mom0 = cube.data.spectral_slab(*mask_velocity_limits).moment0()
        mask_base = mom0.to_value()
        mask_wcs = cube.wcs_flat

        mask = mask_base > 40 #; 40 for 12co32
        # Blank out a square around the long-tail CO source at the MYSO
        islice = slice(106, 130)
        jslice = slice(161, 193)
        mask[islice, jslice] = False
        # Blank out the other emission, the sort of part-ring around NGC 6611
        islice = slice(112, 143)
        jslice = slice(89, 134)
        mask[islice, jslice] = False
        # Blank out the part of the northern cloud that isn't illuminated
        islice = slice(159, 206)
        jslice = slice(187, 218)
        mask[islice, jslice] = False
        # Blank out a patch on the east side of the shell
        islice = slice(180, 207)
        jslice = slice(80, 104)
        mask[islice, jslice] = False
        # Blank out a few stray spots
        islice = slice(100, 167)
        jslice = slice(37, 70)
        mask[islice, jslice] = False

        # Configure velocity limits for later
        velocity_limits = (11*kms, 21*kms)
        vel_stub_simple = make_simple_vel_stub(velocity_limits)


    elif region_name == "BNR":
        mask_velocity_limits = (23*kms, 27*kms)
        mom0 = cube.data.spectral_slab(*mask_velocity_limits).moment0()
        mask_base = mom0.to_value()
        mask_wcs = cube.wcs_flat
        mask = mask_base > 30 # for 12co32, 40 also looks good. see image in 2024-01-13 folder and notes from that day.
        # Only allow values within this box
        islice = slice(119, 182)
        jslice = slice(145, 213)
        # Create an all-False array
        box_mask = np.zeros_like(mask_base).astype(bool)
        # True the values within the square
        box_mask[islice, jslice] = True
        # AND together the mask and the square
        mask = mask & box_mask

        # Velocity
        velocity_limits = mask_velocity_limits # same as for mask. we made ratios for this (23-27) already
        vel_stub_simple = make_simple_vel_stub(velocity_limits)


    if True:
        # Regrid to the PMO grid so that we don't have a billion reduntant pixels
        pmo10 = cube_utils.CubeData(get_map_filename("12co10-pmo"))
        wcs_pmo10 = pmo10.wcs_flat
        shape_pmo10 = pmo10.data.shape[1:]
        mask_base = reproject_interp((mask_base, mask_wcs), wcs_pmo10, shape_out=shape_pmo10, return_footprint=False)
        mask = reproject_interp((mask.astype(float), mask_wcs), wcs_pmo10, shape_out=shape_pmo10, return_footprint=False) > 0.5
        mask_wcs = wcs_pmo10

    n_pixels_mask = np.sum(mask)
    print("Pixels under mask:", n_pixels_mask)


    if True:
        # Find area under map
        pixel_scale = misc_utils.get_pixel_scale(mask_wcs)
        pixel_area_physical = ((pixel_scale * los_distance_M16 / u.radian)**2).to(u.pc**2)
        print(f"Pixel area: {pixel_area_physical:.8f}")
        print(f"Mask area: {(pixel_area_physical*n_pixels_mask):.8f}")
        # Check mass using an average column density
        coldens = [0.5, 2] * u.cm**-2 * 1e22 # N(H2)
        mass = (coldens * mean_molecular_weight_neutral * 2 * Hmass * pixel_area_physical * n_pixels_mask).to(u.solMass)
        print(mass)


    save_df_path = os.path.join(catalog.utils.m16_data_path, "misc_regrids")

    if False:
        # Save mask to FITS for later reference
        mask_float = mask.astype(float)
        hdr = mask_wcs.to_header()
        hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        hdr['CREATOR'] = f"Ramsey, {__file__}.sample_masked_map"
        hdr['COMMENT'] = f"Using PMO grid and beam"
        hdu = fits.PrimaryHDU(data=mask_float, header=hdr)
        save_mask_name = f"sample_mask_{region_name}_{vel_stub_simple}_regrid_mask.fits"
        hdu.writeto(os.path.join(save_df_path, save_mask_name), overwrite=False)
        print("Wrote to ", save_mask_name)



    if True:
        # Debug plot to check mask
        test_img = mask_base.copy()
        test_img[~mask] = np.nan
        plt.imshow(test_img, origin='lower')
        plt.colorbar()
        plt.show()
        # plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"mask_{mask_data_stub}_{make_simple_vel_stub(mask_velocity_limits)}_regrid.png"),
        #     metadata=catalog.utils.create_png_metadata(title="mask from co32", file=__file__, func="sample_masked_map"))

    # Done with this mask!


    if False:
        # Try reprojecting it to some sample data
        lookup_obj = COData(velocity_limits)
        lookup_obj.sample_type_setting = "mask_vals"
        lookup_obj.sample_framework_setting = (mask, mask_wcs)
        lookup_obj.diagnostic_plot = False

        result_df = pd.DataFrame(lookup_obj.sample_all_data())

        assert os.path.exists(save_df_path)
        save_df_name = f"sample_mask_{region_name}_{vel_stub_simple}_vals_regrid.csv"
        save_df_full_path = os.path.join(save_df_path, save_df_name)

        """
        Divide the Herschel 70-160 columns by 2 because they are N_H and everything
        else is N(H2); N_H = N(H) + 2*N(H2) and N(H) is 0 by assumption.
        """

        result_df["column_density_70-160"] = result_df["column_density_70-160"] / 2
        print(result_df)

        result_df.to_csv(save_df_full_path)
        # plt.show()

class CORadexGridCreate:
    """
    December 22, 2023
    Handle Radex grid production. Uses the framework laid out in
    co_column_grid.ipynb. Uses multiprocessing.Pool to speed things up.

    This is the theoretical/model side, whereas COData is the observational
    side. They are complimentary and will be used together for maximum
    flexibility.

    Seamlessly swap between different grid axes for ultimate flexibility.
    (Rejected plan) Predefine your output values for ultimate customization potential.
    """

    allowed_keys = ['n', 'NH2', 'Tk']

    default_range = {
        'n': (1, 6, 1),
        'NH2': (19, 24, 1),
        'Tk': (28, 40, 1),
    }

    default_scaling = {
        'n': 'log',
        'NH2': 'log',
        'Tk': 'linear',
    }

    radex_parameter_aliases = {
        # what spectralradex calls each parameter
        'n': 'h2',
        'NH2': 'cdmol', # these aren't the same, but that is handled elsewhere
        'Tk': 'tkin',
    }

    # Radex names for each CO isotope
    co_isotopes = ('co', '13co', 'c18o')

    def __init__(self, axis_names, **kwargs):
        """
        Going to play the kwargs dict game with this one.
        Axis names is the only required argument so it's the only one explicitly
        named.
        :param axis_names: iterable of string labels.
            As few as 1 or as many as you want is ok. String labels are the
            predetermined parameter keys that I decided on. They are as follows:
            NH2, n, Tk
            Any or all of these may be placed in any order. Each can only appear
            once (so you have a practical limit of 3 elements to this iterable
            unless I were to support more parameters, which I don't plan to do).
            Order is (x, y, z, ...). If "n" is the first element, then n is the
            x axis, and so on.
        optional kwargs
        :param range: dict with keys in axis_names and values of 3-tuples.
            3-tuples are (start, stop, step) range descriptions for each axis's
            parameter.
            Just like any Python indexing, or Python's "range" function or
            numpy's "arange", stop is not inclusive but start is.
            Default ranges will be used for each parameter for which range is
            not set.
        Upon initialization, the grid will be set up, but the Radex calls won't
        start until another method is called. This way, grid configuration can
        be checked before too much time is invested in grid population.
        """
        self.axis_keys = tuple(axis_names)
        # Fill ranges with defaults
        self.axis_ranges = [CORadexGridCreate.default_range[k] for k in self.axis_keys]
        # Check if any ranges were manually set
        manually_set_ranges = kwargs.get("range", None)
        if manually_set_ranges is not None:
            # Update self.axis_ranges with the manually set ranges
            # Loop thru axis keys
            for i, k in enumerate(self.axis_keys):
                # Check if a range was set for that key
                k_range = manually_set_ranges.get(k, None)
                if k_range is not None:
                    # Put that range into the axis_ranges list
                    self.axis_ranges[i] = k_range
        # Set up the axes. log axis are still in log here
        self.axis_arrays = [np.arange(*r) for r in self.axis_ranges]
        # The "xyz" here is important: ijk (the array shape) is this reversed
        self.grid_shape_xyz = tuple(arr.size for arr in self.axis_arrays)
        # This is the array shape
        self.grid_shape_ijk = self.grid_shape_xyz[::-1]
        # Create the default parameters dictionary
        # This could be modified by the user before running the grid
        self.params = radex.get_default_parameters()
        # Modify some of the parameters
        self.params['fmin'] = 100.
        self.params['fmax'] = 400.
        self.params['linewidth'] = 2.
        # Figure out what the leftover parameters are and flag them so that they
        # are manually set before the grid is run.
        # This is important for NH2, which cannot be set in the params_dict because
        # cdmol depends on the CO isotope
        unused_params = set(self.allowed_keys) - set(self.axis_keys)
        # Create dict that tracks the value of fixed parameters. May be empty if grid is 3D
        self.fixed_params = {k: None for k in unused_params}

        # Check kwargs for any additional text that need to be added to the savename
        if kwargs.get("stub", None) is not None:
            # Add leading underscore to separate from rest of filename
            self.save_stub = "_" + kwargs.get("stub")
        else:
            self.save_stub = ""

        # Disallow file overwriting unless we expressly permit it
        self.allow_overwrite = False

    def manually_set_param(self, key, value):
        """
        Dec 23 2023
        Manually set one of the three available axis parameters which is not
        being varied in the grid.
        i.e., if the grid is (n, NH2), this method would be used to set Tk.

        This function MUST be run manually on all parameters in self.fixed_params.
        If not, self.run_grid will raise an error.

        :param key: string, one of the allowed_keys
        :param value: float value. will be appropriately transformed
        """
        if key in self.fixed_params:
            self.fixed_params[key] = value
        else:
            # Warn, but do not stop the code.
            print(f"Warning: manually setting a value for a variable parameter <{key}>. This value will be ignored. The grid will still run.")

    def run_grid(self, n_procs=4):
        """
        Execute the Radex calls to make the grid.
        :param n_procs: int number of workers in the multiprocessing.Pool.
            Default is 4, the number of physical cores on my laptop.
        """
        # Make sure n_procs is something reasonable. Cap it at 8, I'll fix the code if it's necessary to go over that.
        n_procs_cap = 8
        if n_procs > n_procs_cap:
            raise RuntimeError(f"I'm preventing you from running a Pool with > {n_procs_cap} processes. If you know it's okay to do so, go into the code and increase n_procs_cap.")

        # Check if there are unfixed parameters which need to be set.
        # Stop the code with an Error if so.
        # It's important and warrants a full crash; a printed warning will just cause some other crash down the line.
        unfixed_params = self.unfixed_params_remain()
        if unfixed_params:
            raise RuntimeError(f"The parameter(s) {list(unfixed_params)} have not been set. Please use CORadexGridCreate.manually_set_param(key, value) to set them.")

        # Set up grid iteration tuple
        self.axis_tup_list = itertools.product(*self.axis_arrays)
        # Mirror that grid iteration list with a list of index tuples
        # This will give easy access into the 2 or 3D arrays
        self.axis_index_list = itertools.product(*[range(x) for x in self.grid_shape_xyz])

        # Print some stuff about the grid we're about to run
        print(f"Grid shape {self.grid_shape_ijk} and size {math.prod(self.grid_shape_ijk)}.")

        # Task function. This function will be sent to the Pool
        process_task = CORadexGridCreate.process_task_run_radex
        # Iterable from generator function to emit the constant arguments with every element of the iterable
        # The zip() combines two lists into an arg list for the Pool
        # the list elements are tuples(param_args, indices)
        # param_args are tuple(*float) and indices are tuple(*int). Those tuples are equal length
        all_args_iterable = CORadexGridCreate.generate_args(zip(self.axis_tup_list, self.axis_index_list), self.axis_keys, self.params, self.fixed_params)
        if n_procs > 1:
            print(f"Starting pool of {n_procs} workers...", flush=True)
            # Timekeeping
            t0 = time.perf_counter()
            with Pool(n_procs) as pool:
                entire_result_list = list(pool.imap_unordered(process_task, all_args_iterable))
            t1 = time.perf_counter()
            print(f"Finished, {t1-t0:.3g} seconds elapsed.")
        else:
            print(f"Starting in serial...", flush=True)
            t0 = time.perf_counter()
            entire_result_list = [process_task(x) for x in all_args_iterable]
            t1 = time.perf_counter()
            print(f"Finished, {t1-t0:.3g} seconds elapsed.")

        output_keys, output_arrays = self.unpack_and_rearrange_radex_outputs(entire_result_list)

        # Determine file save location
        self.savename = self.create_savename()
        print(f"Writing to {self.savename} ... ", end="", flush=True)
        self.save(output_keys, output_arrays)
        print("Done")

    @staticmethod
    def process_task_run_radex(args):
        """
        TASK FOR TOMORROW:
        pickle cannot serialize local (inner) functions. The function must be
        available in the global namespace. I will have to restructure this
        a bit, but luckily only this.

        December 23, 2023
        Run Radex for one gridpoint. Designed to be farmed out to a Pool.
        :param args: (tuple(*float), tuple(*int))
            The first tuple, of floats, gives the parameter values.
            The second tuple, of ints, gives the grid indices.
            They are equal length; length is the number of variable
            parameters (len(self.axis_keys)).
            The indices aren't exactly used in this function; they are
            returned by the function so that it's much easier to look up
            the correct grid cell later.
        :returns: tuple(tuple(*int), tuple(*float))
            The first tuple, of ints, is the tuple of grid indices.
            This is identical to the second tuple in args.
            The second tuple, of floats, holds output values extracted
            from the Radex runs.
        """
        # Unpack the arguments. The two elements are also tuples.
        (grid_param_values, grid_indices), grid_param_keys, params_dict, fixed_params = args
        # Copy everything (again) to make sure we aren't modifying
        params_dict_copy = params_dict.copy()
        fixed_params_copy = fixed_params.copy()
        # Loop through isotopes and run Radex once for each
        interim_result_dict = {}
        for isotope in CORadexGridCreate.co_isotopes:
            # Fill out params
            params_dict_copy['molfile'] = isotope + ".dat"
            # Set the axis parameters
            for k, v in zip(grid_param_keys, grid_param_values):
                CORadexGridCreate.set_parameter(isotope, params_dict_copy, k, v)
            # Set the fixed parameters
            for k, v in fixed_params_copy.items():
                CORadexGridCreate.set_parameter(isotope, params_dict_copy, k, v)
            interim_result_dict[isotope] = radex.run(params_dict_copy)
        output_values = CORadexGridCreate.extract_and_pack_radex_outputs(interim_result_dict)
        return (grid_indices, output_values)

    @staticmethod
    def set_parameter(isotope, params_dict, key, value):
        """
        Dec 23 2023
        Set the correct parameter in the dictionary
        Static so that it can be used within a process in the Pool
        """
        value = CORadexGridCreate.parameter_transform(isotope, key, value)
        params_dict[CORadexGridCreate.radex_parameter_aliases[key]] = value

    @staticmethod
    def parameter_transform(isotope, key, value):
        """
        Dec 23 2023
        Apply required transforms to the parameter value before it is sent to
        Radex. These include scaling and/or column density transforms for a
        given CO isotope.
        This function essentially coordinates/wraps a couple other static methods
        :param isotope: string co, 13co, or c18o
        :param key: grid parameter key, must be in CORadexGridCreate.allowed_keys
        :param value: float parameter value
        :returns: float transformed parameter value
        """
        # scaling
        value = CORadexGridCreate.scale_to_param(value, key)
        # check if column density, apply correct factors
        if key == "NH2":
            value = CORadexGridCreate.convert_nh2_to_co_isotope(value, isotope)
        return value

    @staticmethod
    def scale_to_param(x, key):
        """
        December 23, 2023
        Scale value(s) to their Radex parameter formats
        i.e., if scaling is "log", then the input x=4 will become 10,000
        If scaling is "linear", then input x=4 will remain 4.
        :param x: value
        :param key: name of parameter
        :returns: scaled value
        """
        scaling = CORadexGridCreate.default_scaling[key]
        if scaling == 'log':
            return 10.**x
        elif scaling == 'linear':
            return x
        else:
            raise RuntimeError(f"Scaling type <{scaling}> unknown")

    @staticmethod
    def scale_from_param(x, key):
        """
        December 23, 2023
        Inverse operation of scale_to_param
        """
        scaling = CORadexGridCreate.default_scaling[key]
        if scaling == 'log':
            return np.log10(x)
        elif scaling == 'linear':
            return x
        else:
            raise RuntimeError(f"Scaling type <{scaling}> unknown")

    @staticmethod
    def convert_nh2_to_co_isotope(nh2, isotope):
        """
        Dec 23 2023
        convert N(H2) to N(13CO) or N(C18O)
        """
        n12co = nh2 * ratio_12co_to_H2 # 8.5e-5, Tielens book number
        if isotope == 'co':
            return n12co
        elif isotope == '13co':
            n13co = n12co / ratio_12co_to_13co # 44.65, Karim et al. number
            return n13co
        elif isotope == 'c18o':
            nc18o = n12co / ratio_12co_to_c18o # 417, Wilson and Rood 1994
            return nc18o
        else:
            raise RuntimeError(f"isotope? {isotope}")

    @staticmethod
    def extract_and_pack_radex_outputs(interim_result_dict):
        """
        Dec 23 2023
        Extract the relevant information from interim_result_dict and repackage
        it as a tuple in a predetermined order so that it can be unpacked later.
        This function is meant to be called from within a Pool.
        :param interim_result_dict: dict with string keys co, 13co, c18o.
            Each value is the DataFrame generated by spectralradex upon running
            Radex for that isotope (and the parameter configuration, which is
            not of concern for this function)
        :returns: tuple(*float) in the order which would be generated by
            for property in (TR, tau):
                for isotope in (co, 13co, c18o):
                    for line in (1-0, 3-2):
                        skip c18o 3-2
                        append value for property of line of isotope
            This results in a tuple of length 10.
            Why a tuple a not a list? Lists overallocate, particularly when you
            are building them with list.append, which I will be doing. I don't
            need to save an empty allocated list buffer for every single pixel
            in the grid; that could waste memory on the order of the size of the
            grid. The grids of course aren't very big so this isn't a huge deal.
            But philosophically, I think it's reasonable to convert the list to
            an immutable tuple when I send it back to the calling function.
        """
        result_list = []
        for property_colname in ["T_R (K)", "tau"]:
            for isotope in CORadexGridCreate.co_isotopes:
                # Get the DataFrame for this isotope
                isotope_df = interim_result_dict[isotope]
                for line_iloc in [0, 2]:
                    # [0, 2] are the index locations of the 1-0 and 3-2 transitions in the DataFrame
                    if (isotope == 'c18o') and (line_iloc == 2):
                        # No need for the C18O 3-2 line
                        continue
                    result_list.append(isotope_df[property_colname].iloc[line_iloc])
        return tuple(result_list)

    def unpack_and_rearrange_radex_outputs(self, result_tuples):
        """
        Dec 23 2023
        Outside-Pool counterpart to extract_and_pack_radex_outputs.
        extract_and_pack_radex_outputs worked inside each process in the Pool
        to arrange the outputs in a sensible order.
        unpack_and_rearrange_radex_outputs takes an iterable made up of many
        return values from extract_and_pack_radex_outputs and arranges them in
        their own 1, 2, or 3D arrays.
        :param result_tuples: iterable of tuples.
            Each element is a tuple returned by extract_and_pack_radex_outputs.
        :returns: (list, dict)
            list has strings in good order for saving. The strings match the
            keys in the dict.
            dict has string keys and array values.
            The key describes the array; "TR_13co_10" would indicate that the
            array holds the radiation temperature of 13CO 1-0.
        """
        # Create the empty arrays. We know in advance we need 10 for the outputs.
        output_labels = []
        # Generate output value keys using the same nested for loop used to
        # pack up the values
        for property_name in ["TR", "tau"]:
            for isotope in CORadexGridCreate.co_isotopes:
                for line in ["10", "32"]:
                    if (isotope == 'c18o') and (line == '32'):
                        continue
                    output_labels.append(f"{property_name}_{isotope}_{line}")
        # Initialize a bunch of arrays
        output_arrays = {k: np.zeros(self.grid_shape_ijk) for k in output_labels}
        # Just loop-fill the arrays. No fancy indexing, it'll take longer to write
        # that it will run.
        for indices_xyz, output_values in result_tuples:
            indices_ijk = indices_xyz[::-1]
            for k, v in zip(output_labels, output_values):
                output_arrays[k][indices_ijk] = v
        # Add another len(self.axis_keys) arrays to hold the input parameters
        # Have to flip around the axis_arrays list so that ij indexing works.
        # Can't use xy indexing because in 3D it does (z, x, y) shape
        # Flip it back around afterwards so that we can match it up with axis_keys easily
        parameter_arrays = np.meshgrid(*self.axis_arrays[::-1], indexing='ij')[::-1]
        for k, arr in zip(self.axis_keys, parameter_arrays):
            output_arrays[k] = arr
        output_labels = list(self.axis_keys) + output_labels
        return output_labels, output_arrays

    def create_savename(self):
        """
        Dec 23 2023
        Make a unique filename using the axis_keys, ranges, and fixed_params
        Add ".fits" to the end.
        Use {misc_data_path}/co_grids/ directory
        """
        key_strings = []
        for k, r in zip(self.axis_keys, self.axis_ranges):
            s = ".".join(str(x) for x in r)
            key_strings.append(f"{k}.{s}")
        # Loop through axis_keys (list) to preserve order in fixed_params (dict)
        for k in self.allowed_keys:
            if k in self.fixed_params:
                key_strings.append(f"fixed.{k}.{self.fixed_params[k]:.2f}")
        filename = "_".join(key_strings) + self.save_stub + ".fits"
        filepath = os.path.join(catalog.utils.misc_data_path, "co_grids")
        return os.path.join(filepath, filename)

    def save(self, output_keys, output_arrays):
        """
        Dec 24, 2023
        Save the completed grid to a FITS file. Use the output_keys as extnames
        :param output_keys: iterable(*string), order of the iterable determines
            the order of the FITS extensions in the saved file.
        :param output_arrays: dict(string -> array), strings are the same keys
            in output_keys, arrays are the data to save to FITS
        """
        # Set up HDU list and Header template
        hdu_list = [fits.PrimaryHDU()]
        header_template = fits.Header({
            'CREATOR': f"Ramsey Karim via {__file__}.{type(self).__name__}.save",
            'DATE': f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}",
            'COMMENT': "Grids created using the spectralradex wrapper of Radex",
        })
        # Use the CTYPE Header keys to indicate the quantities on each axis
        for i, k in enumerate(self.axis_keys):
            header_template[f"CTYPE{i+1:d}"] = k
        # Use the keyword 'FIXED{i}' to list fixed parameters
        i = 0
        for k in self.allowed_keys:
            if k in self.fixed_params:
                header_template[f"FIXED{i+1:d}"] = f"{k},{self.fixed_params[k]:.2f}"
        # Use keys as extnames
        for k in output_keys:
            arr = output_arrays[k]
            hdr = header_template.copy()
            hdr['BUNIT'] = self.decide_units(k)
            hdr['EXTNAME'] = k
            hdu_list.append(fits.ImageHDU(data=arr, header=hdr))
        # Write
        fits.HDUList(hdu_list).writeto(self.savename, overwrite=self.allow_overwrite)

    def decide_units(self, key):
        """
        Dec 24 2023
        Choose the units for a given extension
        Copied and edited from co_column_grid.ipynb
        :param key: string name of FITS extension
        :returns: string unit description. Compatible with u.Unit() constructor
        """
        easy_ones = {"NH2": str(u.cm**-2), "n": str(u.cm**-3), "Tk": "K"}
        if key in easy_ones:
            return easy_ones[key]
        elif key[:2] == 'TR':
            return 'K'
        elif key[:3] == 'tau':
            return ''

    def unfixed_params_remain(self):
        """
        Dec 24 2023
        Check for parameters that are not one of the grid axes and do not have
        a value set by manually_set_param.
        These need to be set in the class. Something like Tk can theoretically
        be set directly into the params_dict, but the main issue with this
        approach is that the class won't know the value, or at least whether it
        was intentional or not, so it can't be used in a save filename.
        The greater issue is that of NH2. NH2 cannot be set in the params_dict
        because cdmol is the parameter, and that depends on the isotope and must
        be varied. So the class has to know about NH2.
        Therefore, we enforce that all unused axis parameters are set through
        the class using manually_set_param
        :returns: list(*string keys) if there are unfixed parameters, or empty
            list if not.
            bool(return value) is True if there are unfixed parameters and False
            otherwise.
            "False" is the desired return behavior if we want the code to keep
            running.
        """
        ## This is a nice looking solution but I need to get the keys
        # return any(fixed_value is None for fixed_value in self.fixed_params.values())
        return [k for k, v in self.fixed_params.items() if v is None]

    @staticmethod
    def generate_args(iterable_arg, *constant_args):
        """
        From one iterable argument and some number of "constant arguments",
        create an iterable that pairs each element of iterable_arg with all of the
        constant args.
        This is to help Pool.imap work and to avoid holding too many copies of
        the constant_args in memory at once
        :param iterable_arg: iterable
        :param *constant_args: any number of other arguments, any type(s).
        """
        for element in iterable_arg:
            yield (element, *constant_args)


class CORadexGridRead:
    """
    December 22, 2023
    Handle Radex grid reading. Uses the framework laid out in
    co_column_grid.ipynb.

    Should feel similar to CORadexGridCreate but doesn't do any grid creation.
    """
    def __init__(self, filename):
        """
        Dec 25, 2023
        :param filename: string name of a file saved by CORadexGridCreate
            I'm using the same directory, so I'll have this function find that
            automatically
        """
        # Get filename
        self.full_filename = self.find_full_path(filename)
        self.data = {}
        self.units = {}
        self.axis_keys = []
        self.axis_arrays = []
        self.fixed_params = {}
        self.grid_shape = None # will be tuple
        # Populate the above instance variables
        self.load_grid(self.full_filename)


    def find_full_path(self, filename):
        """
        Dec 25 2023
        Knows about the directory where the grids are saved
        :param filename: the filename relative to the grid directory (so just
            the filename)
        :returns: string full path
        """
        filepath = os.path.join(catalog.utils.misc_data_path, "co_grids")
        return os.path.join(filepath, filename)

    def load_grid(self, filename):
        """
        Dec 25
        Loads in the file. Makes a dictionary from the extnames to the data.
        Gathers some useful information from the Headers.
        Does not return; populates instance variables:
            self.data, self.units, self.axis_keys, self.axis_arrays,
            self.fixed_params
        :param filename: absolute filename of the grid
        """
        header = None
        with fits.open(filename) as hdul:
            for i, hdu in enumerate(hdul):
                if i == 0:
                    # PrimaryHDU
                    continue
                elif i == 1:
                    # Save one of the headers
                    header = hdu.header
                k = hdu.header['EXTNAME']
                self.data[k] = hdu.data
                self.units[k] = hdu.header['BUNIT']
        # Extract additional information from the saved header
        naxis = header['NAXIS']
        for i in range(naxis):
            self.axis_keys.append(header[f"CTYPE{i+1:d}"])
        for k in list(header.keys()):
            if "FIXED" in k:
                name, val = header[k].split(',')
                self.fixed_params[name] = float(val)
        # Found a fun trick to get the right slices
        slices = ((0,)*(naxis-1) + (slice(None),))*2
        # Get the axes. Things are in XYZ order.
        for i, k in enumerate(self.axis_keys):
            # full_arr might be 2 or 3D; will still work if 1D
            full_arr = self.data[k]
            # First axis is x, so that's the last index
            axis_array = full_arr[slices[0+i:naxis+i]]
            self.axis_arrays.append(axis_array)
        # Get the shape of the array; any array will work, use a parameter array
        self.grid_shape = self.data[self.axis_keys[0]].shape

    def get(self, key):
        """
        Dec 26 2023
        General method to get a data array from this grid.
        Can also handle a ratio of two arrays.
        :param key: string key to the self.data dict
        :returns: array from the grid
        """
        if '/' in key:
            # Ratio
            numerator, denominator = [self._get(x.strip()) for x in key.split('/')]
            return numerator / denominator
        else:
            return self._get(key)

    def _get(self, key):
        """
        Dec 26 2023
        Simple getter method for one data array. Will raise KeyError if
        not found in dict.
        :param key: string key
        :returns: array from the grid
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not available in this grid.")
        return self.data[key]

    # Create function for this so it's easier
    def plot_two_grids(self, xgrid_name, ygrid_name, param_names, ax=None, lims={}):
        """
        Convenience function for plotting the constant parameter contours on a
        plot of two observables.
        xgrid will be the x axis value and so on
        cd_lims and n_lims should each be lists of 2 Quantities, [lo, hi] limits for plotting
        Specify two parameter names; these must be in self.axis_keys

        This is only going to work in two dimensions easily, so I'm banning all others
        """
        assert len(self.axis_keys) <= 2
        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        # Within limits function for ease
        def within_lims(x, limits):
            return limits[0] <= x <= limits[1]
        # Set up for the slice trick
        slices_template = ((0,)*(len(self.axis_keys)-1) + (slice(None),))*2
        # Set up a linestyle list
        ls = ['-', '--', ':']
        # Loop through the parameter names
        for i_key, key in enumerate(param_names):
            if key not in self.axis_keys:
                raise RuntimeError(f"Parameter {key} not available in this grid ({', '.join(self.axis_keys)})")
            if key not in lims:
                # Make permissive limit lists if they're not specified
                lims[key] = [-np.inf, np.inf]
            # Find out which parameter this is so that we know how to index grids
            key_idx = self.axis_keys.index(key)
            # Reset the color cycle since the later colors are tricky
            ax.set_prop_cycle(None)
            # Iterate through the constant values of this parameter
            for i in range(0, self.grid_shape[1-key_idx], 1):
                # The start=2, step=4 is so we don't plot a ton of contours. just plot the whole number log ones
                # Do the slice trick! Modified so that it's i instead of 0
                # And flip the slices around because we're getting the "variation" across the *other* parameter
                slices = tuple(i if x==0 else x for x in slices_template[0+key_idx:len(self.axis_keys)+key_idx])[::-1]
                param_val = self._get(key)[slices][0]
                if within_lims(param_val, lims[key]):
                    xarr = self.get(xgrid_name)[slices]
                    yarr = self.get(ygrid_name)[slices]
                    ax.plot(xarr, yarr, linestyle=ls[i_key], label=f"{key}={param_val:.2f}")
        ax.set_xlabel(xgrid_name)
        ax.set_ylabel(ygrid_name)
        ax.legend()


def test_radex_grid():
    """
    December 23, 2023
    test the CORadexGridCreate and CORadexGridRead classes
    """
    if True:
        # grid_creator = CORadexGridCreate(["n", "NH2"], range={"NH2": (19, 24, 0.25), "n": (1, 6, 0.25)})
        # grid_creator.manually_set_param("Tk", 30)
        grid_creator = CORadexGridCreate(["n", "Tk"], range={"Tk": (29, 41, 1), "n": (1, 6, 0.25)})
        grid_creator.manually_set_param("NH2", 22)
        # grid_creator.allow_overwrite = False
        grid_creator.run_grid(n_procs=4)
        savename = os.path.basename(grid_creator.savename)
        print(savename)
    else:
        savename = "n.1.6.1_NH2.21.23.1_fixed.Tk.30.00.fits"
    if True:
        grid_reader = CORadexGridRead(savename)
        print(grid_reader.axis_keys)

def make_more_radex_grids():
    """
    Dec 26 2023
    Make more Radex grids
    First job:
    Using the point NC-5, which is in the low velocity bunch of points.
    It has NH2 = 22.52 and Tex(12CO10) = 20.9

    I guess Tex(12CO10) cannot == Tk according to Radex; it seems the correct
    Tk should be closer to 22 K. Close, but the difference matters.
    """
    ranges = {"NH2": (20, 24, 0.25), "Tk": (16, 40, 1), "n": (1, 5.5, 0.25)}

    # fine grid
    ranges['NH2'] = (21, 23, 0.05)
    ranges['n'] = (2.6, 4.6, 0.05)
    # Tk vs n grid
    if False:
        grid_creator = CORadexGridCreate(["n", "Tk"], range=ranges)
        grid_creator.manually_set_param("NH2", 22.)
        grid_creator.run_grid(n_procs=4)
        savename = os.path.basename(grid_creator.savename)
        print(savename)
    # NH2 vs n grid
    if True:
        grid_creator = CORadexGridCreate(["n", "NH2"], range=ranges, stub="fine0.05")
        grid_creator.manually_set_param("Tk", 30.)
        grid_creator.run_grid(n_procs=4)
        savename = os.path.basename(grid_creator.savename)
        print(savename)


def calc_extent(arr):
    """
    Dec 26 2023: Moved here (temporarily to the global namespace)
    Written a little earlier than Dec 14 in a jupyter notebook

    Calculate the extent for a given axis which makes the center of the pixels
    line up correctly with their axis values.

    This must be run for both x and y axes and the resulting lists concatenated
    before passed to imshow or contour. i.e.
    plt.imshow(..., extent=(calc_extent(x_axis) + calc_extent(y_axis)))
    """
    diff = np.diff(arr)[0]
    tmp_arr = arr - diff/2
    return [tmp_arr[0], tmp_arr[-1]+diff]


def grid_reader_filename_wrapper(select_grid=None, stub=""):
    """
    Dec 28, 2023
    Wrap the CORadexGridRead object in a function that knows which grids we
    have created and the peculiarities of our naming scheme.
    This is outside the bounds of what CORadexGridRead should know about, so
    I'm putting it in this function.
    :returns: tuple(CORadexGridRead, extra_stub)
        extra_stub is a descriptive element that should be put onto PNG
        filenames for distinction from "regular" grids.
    """
    # Add "flair" to the filename to select special cases
    extra_stub = "" if not stub else "_"+stub

    # Load grid
    ## original grids
    grid_menu = {
        # original fixed parameters, but ranges updated
        ## "t30" is in t_grids now
        "N22": "n.1.5.5.0.25_Tk.16.40.1_fixed.NH2.22.00.fits",

        ## New grids for NC-5 (row_idx = 4)
        "nc-5_t20.9": "n.1.5.5.0.25_NH2.20.24.0.25_fixed.Tk.20.90.fits",
        "nc-5_N22.52": "n.1.5.5.0.25_Tk.16.40.1_fixed.NH2.22.52.fits",
        ## New grid for NC-5 with Tk=22 (slighlty higher than before)
        ## Now checking if Tk=24 messes everything up or is OK
        ## added those to t_grids
        ## New grid for NC-5 with NH2=22.37 (slightly lower than before)
        "nc-5_N22.37": "n.1.5.5.0.25_Tk.16.40.1_fixed.NH2.22.37.fits",

    }
    ## New grids to test out NC 1, 2, 3
    t_grids = [22, 24, 26, 28, 29, 30, 31, 32] # a bunch of constant Tk grids at whole-number Tk values
    grid_menu.update({t: f"n.1.5.5.0.25_NH2.20.24.0.25_fixed.Tk.{t:d}.00{extra_stub}.fits" for t in t_grids})

    grid_menu.update({f'{t:d}fine0.1': f"n.2.6.4.6.0.1_NH2.21.23.0.1_fixed.Tk.{t:d}.00_fine0.1.fits" for t in [30]})
    grid_menu.update({f'{t:d}fine0.1-0.05': f"n.2.6.4.6.0.05_NH2.21.23.0.1_fixed.Tk.{t:d}.00_fine0.05.fits" for t in [30]})
    grid_menu.update({f'{t:d}fine0.05': f"n.2.6.4.6.0.05_NH2.21.23.0.05_fixed.Tk.{t:d}.00_fine0.05.fits" for t in [30]})

    if select_grid is None:
        select_grid = 30

    if "fine" in stub:
        select_grid = f"{select_grid:d}{stub}"

    grid_savename = grid_menu[select_grid]
    grid_reader = CORadexGridRead(grid_savename)
    return grid_reader, extra_stub

def compare_data_with_radex_grid(select_grid=None, stub=""):
    """
    December 26, 2023
    Got the grid creator and reader ready and also have the data sampler ready.
    Combine these and see what happens.
    """
    grid_reader, extra_stub = grid_reader_filename_wrapper(select_grid=select_grid, stub=stub)
    # Load data
    data_fn = "misc_regrids/sample_points_test_1_11.0.21.0.csv"
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    # print(grid_reader.data.keys())
    # print(data_df.columns)

    plot_pairs = [ # (grid, data)
        ("TR_13co_32 / TR_13co_10", "ratio_32_10"), # 32 to 10
        ("TR_13co_10 / TR_c18o_10", "ratio_13_18"), # 13 to 18
        ("TR_co_10", "peak_12co10"), # 12 CO 10
        ("TR_co_32", "peak_12co32"), # 12 CO 32
        ("TR_13co_10", "peak_13co10"), # 13 CO 10
        ("TR_13co_32", "peak_13co32"), # 13 CO 32
        ("TR_c18o_10", "peak_c18o10"), # 13 CO 10
        ("NH2", "column_density_13co10"),
        ("NH2", "column_density_c18o10"),
        ("NH2", "column_density_70-160"),
        ("NH2", "column_density_160-500"),
    ]

    fig = plt.figure(figsize=(15, 10))
    fig_gridspec_shape = (2, 3)
    gs = fig.add_gridspec(*fig_gridspec_shape)
    extent = calc_extent(grid_reader.axis_arrays[0]) + calc_extent(grid_reader.axis_arrays[1])
    legend_handles = []

    # color_list_template = list(mpl_cm.rainbow(np.linspace(0, 1, len(plot_pairs))))
    # color_list = color_list_template[0::2] + color_list_template[1::2]
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2

    row_idx = 0
    for row_idx in range(6):
        row = data_df.iloc[row_idx]
        reg_name = row['reg_name']
        print(reg_name)
        ax = fig.add_subplot(gs[np.unravel_index(row_idx, fig_gridspec_shape)])
        img_select = ["_co_10", "_co_32", "_13co_10", "_13co_32"]
        # if row_idx < len(img_select):
        #     ax.imshow(grid_reader.get("tau"+img_select[row_idx]), origin='lower', cmap='Greys', extent=extent, aspect=(img.shape[0]/img.shape[1]))
        chisq_array = np.zeros(grid_reader.grid_shape)
        obs_count = 0
        for i, (grid_key, data_key) in enumerate(plot_pairs):
            try:
                img = grid_reader.get(grid_key)
            except KeyError:
                assert grid_key == "NH2"
                continue
            val = row[data_key]
            if grid_key == "NH2":
                levels = [np.log10(val)]
            else:
                levels = [val]
            # ax.imshow(img, origin='lower', extent=extent)
            color = color_list[i]
            if "peak_12co" in data_key:
                ls = ":"
            else:
                ls = "-"
            ax.contour(img, levels=levels, colors=color, extent=extent, linestyles=ls)
            try:
                err = row["err_" + data_key]
                levels = [val-err, val+err]
                if grid_key == "NH2":
                    levels = [np.log10(x) for x in levels]
                ax.contourf(img, levels=levels, colors=color, extent=extent, alpha=0.2)
            except:
                # print(f"no available error found for {data_key}")
                ...
            # Legend creation
            if row_idx == 0:
                legend_handles.append(mpatches.Patch(color=color, label=data_key))
            # Get the chi squared for this measurement
            select_pixel = (9, 11)
            if grid_key == 'NH2':
                chisq_array += ((val/1e19 - 10**(img-19))/(err/1e19))**2
                print("nh2", f"{((val/1e19 - 10**(img[select_pixel]-19))/(err/1e19))**2:.2f}")
            else:
                print(grid_key, f"{val:.2f}", f"{img[select_pixel]:.2f}", f"{err:.2f}", f"{((val - img[select_pixel])/err)**2:.2f}")
                chisq_array += ((val - img)/err)**2
            obs_count += 1

            im = ax.imshow(np.log10(chisq_array/(obs_count-2)), origin='lower', cmap='Greys_r', extent=extent, aspect=(chisq_array.shape[0]/chisq_array.shape[1]), vmin=1, vmax=3)
        print(f"{np.log10(chisq_array[select_pixel]/(obs_count-2)):.2f}")
        fig.colorbar(im, ax=ax)

        ax.set_title(reg_name)
        get_unit = lambda ax_idx : grid_reader.units[grid_reader.axis_keys[ax_idx]]
        ax.set_xlabel(f"{grid_reader.axis_keys[0]} ({get_unit(0)})")
        ax.set_ylabel(f"{grid_reader.axis_keys[1]} ({get_unit(1)})")

        if reg_name == "NC-5" and grid_reader.axis_keys[1] == "Tk":
            ax.axhline(22, color='k', linestyle=':', alpha=0.5)

        # Check Tex for both 12CO 3-2 and 1-0; can use this to make new grids
        # Print out the 13co10 and c18o10 column density; can use this to make new grids
        if False:
            print(f"REG {reg_name}")
            for k in ["peak_12co10", "peak_12co32"]:
                line_stub = k.split("_")[1]
                line_freq = COColumnDensity._constants[line_stub] * u.GHz
                tex = COColumnDensity.calculate_Tex(row[k]*u.K, line_freq)
                print(f"{k}, {row[k]:.2f} +/- {row['err_'+k]:.2f} -> {tex:.2f}")
            for k in ["column_density_13co10", "column_density_c18o10"]:
                print(f"{k}, {row[k]:.2E} +/- {row['err_'+k]:.2E} (log10 = {np.log10(row[k]):.2f})")

    # on the last axis
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    fixed_params_stub = "".join([f"{k}{v:.2f}" for k, v in grid_reader.fixed_params.items()])
    savename = f"gridplots_{grid_reader.axis_keys[1]}vs{grid_reader.axis_keys[0]}_at_{fixed_params_stub}{extra_stub}.png"
    savename = os.path.join(catalog.utils.todays_image_folder(), savename)
    plt.show()
    # fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title="grid",
    #     file=__file__, func="compare_data_with_radex_grid"
    # ))

def scatter_radex_grid(select_grid=None, stub=""):
    """
    Dec 28, 2023
    Re-create the scatter plots that I debuted in co_radex_grid_plots.ipynb
    """
    # Load grid
    grid_reader, extra_stub = grid_reader_filename_wrapper(select_grid=select_grid, stub=stub)
    # Load data
    data_fn = "misc_regrids/sample_mask_test_1_11.0.21.0_vals_regrid.csv"
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    grid_reader.plot_two_grids("TR_c18o_10", "TR_13co_32 / TR_13co_10",
        grid_reader.axis_keys, ax=ax, lims={"NH2": (21.5, 22.5), "n": (3, 4)})

    xdata_name = "peak_c18o10"
    ydata_name = "ratio_32_10"
    color_name = "column_density_13co10"

    notnull_mask = data_df[xdata_name].notnull() & data_df[ydata_name].notnull()
    xdata_notnull = data_df[xdata_name][notnull_mask]
    ydata_notnull = data_df[ydata_name][notnull_mask]
    x_mean = np.mean(xdata_notnull)
    y_mean = np.mean(ydata_notnull)
    x_std = np.std(xdata_notnull)
    y_std = np.std(ydata_notnull)
    """
    The 16th/84th quantiles are pretty symmetric for peak_c18o10 and ratio_32_10, so I'll just use the stddev
    x_lo, x_hi = misc_utils.flquantiles(xdata_notnull, q=6)
    y_lo, y_hi = misc_utils.flquantiles(ydata_notnull, q=6)
    print(x_mean-x_lo, x_hi-x_mean, x_mean, x_std)
    print(y_mean-y_lo, y_hi-y_mean, y_mean, y_std)
    """

    ax.errorbar([x_mean], [y_mean], xerr=[x_std], yerr=[y_std], alpha=1, color='k', markersize=3, linewidth=4, capsize=7, capthick=2)


    ax.errorbar(data_df[xdata_name], data_df[ydata_name], xerr=data_df["err_"+xdata_name], yerr=data_df["err_"+ydata_name], marker='none', alpha=0.2, linestyle='none', color='k')
    sc = ax.scatter(data_df[xdata_name], data_df[ydata_name], marker='o', alpha=1, c=np.log10(data_df[color_name]), cmap=cmocean.cm.matter_r)

    fig.colorbar(sc, ax=ax, label=color_name)
    ax.set_xlim((0, 4))
    ax.set_ylim((0, 2.5))

    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"scatter_{xdata_name}_{ydata_name}.png"),
        metadata=catalog.utils.create_png_metadata(title="scatter plot. std error bars",
            file=__file__, func="scatter_radex_grid"))

def calc_mass_from_masked_data():
    """
    Dec 30, 2023
    Using the 12co32 mask, PMO-grid data, find the mass for all the column density measurements.
    0.06404582 pc2 is the pixel area
    """
    # data_fn = "misc_regrids/sample_mask_N19_11.0.21.0_vals_regrid.csv"; reg_stub = "N19"
    data_fn = "misc_regrids/sample_mask_BNR_23.0.27.0_vals_regrid.csv"; reg_stub = "BNR"
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    cd_colnames = [colname for colname in data_df.columns if "column_density" in colname]
    print("REGION:", reg_stub)
    print()
    for colname in cd_colnames:
        print(colname)
        col = data_df[colname]
        n_nan = np.sum(col.isnull())
        n_valid = np.sum(col.notnull())
        print("nan", n_nan, "not nan", n_valid)
        print("total == sum nan+notnan", len(col)==n_nan+n_valid)
        cdavg = np.mean(col) * u.cm**2
        cdstd = np.std(col) * u.cm**-2
        cdtot = np.sum(col) * u.cm**-2 # works even though there are nans!
        pixel_area_physical = 0.06404582 * u.pc**2 # see notes 2023-12-30
        mass = (cdtot * pixel_area_physical * 2 * Hmass * mean_molecular_weight_neutral).to(u.solMass)
        print(f"mean, std cd {cdavg:.1E} +/- {cdstd:.1E}")
        print(f"mass {mass:.2f}\n\t({mass:.1E})")
        print()

def sum_chisq_to_get_masked_area_errorbars():
    """
    Jan 2, 2024
    2024!!!

    Outlined a routine to get NH2 and n error bars in 2024-01-02 notes
    Use ratio_32_10 and peak_c18o10 to make overlay plot chisq.
    Average them for the masked area.
    """
    # no args gets the 30 K grid, fine for N19 ring
    tk = 30
    grid_reader, extra_stub = grid_reader_filename_wrapper(tk, 'fine0.05')
    # grid_reader, extra_stub = grid_reader_filename_wrapper(tk)
    data_fn = "misc_regrids/sample_mask_test_1_11.0.21.0_vals_regrid.csv"; region_label = "N19"
    # data_fn = "misc_regrids/sample_mask_BNR_23.0.27.0_vals_regrid.csv"; region_label = "BNR"
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    # Identify the measurements to use
    measurement_keys = ["ratio_32_10", "peak_c18o10", "peak_13co32"]
    grid_keys = ["TR_13co_32 / TR_13co_10", "TR_c18o_10", "TR_13co_32"]
    grid_arrays = [grid_reader.get(k) for k in grid_keys]
    notnull_mask = None

    # Iterate thru measurements once quickly to build the notnull_mask
    for meas_key in measurement_keys:
        mask = data_df[meas_key].notnull()
        if notnull_mask is None:
            # Establish
            notnull_mask = mask
        else:
            # Add on
            notnull_mask = notnull_mask & mask

    # notnull_mask = notnull_mask * data_df["column_density_c18o10"].isnull()


    # With the completed mask in hand, iterate first through the notnull points
    # and then through the measurement keys
    chisq_array = np.zeros(grid_reader.grid_shape)
    count = 0
    for i in data_df.index:
        if not notnull_mask[i]:
            # is null, skip
            continue
        for j, k in enumerate(measurement_keys):
            meas = data_df.loc[i, k]
            err = data_df.loc[i, "err_"+k]
            # err = 1

            # Absolute calibration error of 10%
            abscal_rel = 0.1
            if 'ratio' in k:
                # Account for the sum of 2 10%s in the ratio, sqrt(2)*10%
                abscal_err = np.sqrt(2)*abscal_rel*meas
            else:
                abscal_err = abscal_rel*meas
            # print("err/abscal_err", err/abscal_err)
            err = err + abscal_err

            chisq_array += ((meas - grid_arrays[j])/err)**2
        count += 1
    print(f"Using {count} pixel samples")

    chisq_array /= np.sum(notnull_mask)
    # Reduce chisq
    # no action needed, dof = 1 (3 data - 2 params)

    extent = calc_extent(grid_reader.axis_arrays[0]) + calc_extent(grid_reader.axis_arrays[1])

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(211)
    im = ax.imshow(np.log10(chisq_array), origin='lower', extent=extent)
    fig.colorbar(im, ax=ax)

    min_chisq = np.min(chisq_array)
    min_loc = np.where(chisq_array==min_chisq)
    print(min_chisq)
    print(min_loc)

    solution = [grid_reader.axis_arrays[i][j] for i, j in enumerate(min_loc[::-1])]
    print(solution)

    ax.contour(chisq_array, levels=[1, 2], origin='lower', extent=extent, colors='r')
    ax.contour(chisq_array, levels=[min_chisq*2], origin='lower', extent=extent, colors='white')
    ax.plot(*solution, marker='x', color='red')

    # Find the row and column indices for the solution. min_loc is ij indexed
    cd_min_loc, n_min_loc = [x.item() for x in min_loc]

    ax = plt.subplot(223)
    # Find the constant cd, varying n chisq curve
    chisq_curve = chisq_array[cd_min_loc, :]
    chisq_curve = (chisq_curve - min_chisq*2)*-1
    print(chisq_curve)
    xaxis = grid_reader.axis_arrays[0]
    plt.plot(xaxis, chisq_curve, marker='x')
    try:
        spline = UnivariateSpline(xaxis[chisq_curve>(-1*min_chisq*2)], chisq_curve[chisq_curve>(-1*min_chisq*2)], s=0)
        x0, x1 = spline.roots()
        plt.axvline(x0, color='k')
        plt.axvline(x1, color='k')
        xresamp = np.linspace(x0, x1, 50)
        plt.plot(xresamp, spline(xresamp))
        plt.ylim((0, min_chisq*1.3))
        print(grid_reader.axis_keys[0])
        xsoln = xaxis[n_min_loc]
        elo, ehi = xsoln-x0, x1-xsoln
        print(elo, ehi)
        xe = (elo + ehi)/2
        plt.errorbar([xsoln], [min_chisq*1.1], xerr=xe)
    except:
        pass


    ax = plt.subplot(224)
    # Find the constant cd, varying n chisq curve
    chisq_curve = chisq_array[:, n_min_loc]
    chisq_curve = (chisq_curve - min_chisq*2)*-1
    print(chisq_curve)
    xaxis = grid_reader.axis_arrays[1]
    plt.plot(xaxis, chisq_curve, marker='x')
    try:
        spline = UnivariateSpline(xaxis[chisq_curve>(-1*min_chisq*2)], chisq_curve[chisq_curve>(-1*min_chisq*2)], s=0)
        x0, x1 = spline.roots()
        plt.axvline(x0, color='k')
        plt.axvline(x1, color='k')
        xresamp = np.linspace(x0, x1, 50)
        plt.plot(xresamp, spline(xresamp))
        plt.ylim((0, min_chisq*1.3))
        print(grid_reader.axis_keys[1])
        xsoln = xaxis[cd_min_loc]
        elo, ehi = xsoln-x0, x1-xsoln
        print(elo, ehi)
        xe = (elo + ehi)/2
        plt.errorbar([xsoln], [min_chisq*1.1], xerr=xe)
    except:
        pass

    # plt.show()
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"chisq_grid{tk}{extra_stub}_{region_label}_v2_10pct-abs-err.png"),
        metadata=catalog.utils.create_png_metadata(title=f"+10pct abscal. {', '.join(measurement_keys)}",
            file=__file__, func="sum_chisq_to_get_masked_area_errorbars"))
    # plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"chisq_grid{tk}{extra_stub}_{region_label}.png"),
    #     metadata=catalog.utils.create_png_metadata(title=f"no abscal. {', '.join(measurement_keys)}",
    #         file=__file__, func="sum_chisq_to_get_masked_area_errorbars"))

def individual_pixel_ensemble_chisq():
    """
    Feb 9, 2024
    Getting very close to the defense here, but still doing analysis!...
    Lee suggested redoing some parts of the CO analysis.
    Add 10% absolute calibration error, for one
    Two, try fitting/solving individual pixels for N, n and taking the ensemble
    stats of the N and n values. See if they agree with the summed chisq.
    """
    # no args gets the 30 K grid, fine for N19 ring
    tk = 30
    grid_reader, extra_stub = grid_reader_filename_wrapper(tk, 'fine0.05')
    # grid_reader, extra_stub = grid_reader_filename_wrapper(tk)
    # data_fn = "misc_regrids/sample_mask_test_1_11.0.21.0_vals_regrid.csv"; region_label = "N19"
    data_fn = "misc_regrids/sample_mask_BNR_23.0.27.0_vals_regrid.csv"; region_label = "BNR"
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    # Identify the measurements to use
    measurement_keys = ["ratio_32_10", "peak_c18o10", "peak_13co32"]
    grid_keys = ["TR_13co_32 / TR_13co_10", "TR_c18o_10", "TR_13co_32"]
    grid_arrays = [grid_reader.get(k) for k in grid_keys]
    notnull_mask = None

    abscal_pct = 15

    # Iterate thru measurements once quickly to build the notnull_mask
    for meas_key in measurement_keys:
        mask = data_df[meas_key].notnull()
        if notnull_mask is None:
            # Establish
            notnull_mask = mask
        else:
            # Add on
            notnull_mask = notnull_mask & mask

    # notnull_mask = notnull_mask * data_df["column_density_c18o10"].isnull()


    # With the completed mask in hand, iterate first through the notnull points
    # and then through the measurement keys
    chisq_list = []
    count = 0
    for i in data_df.index:
        if not notnull_mask[i]:
            # is null, skip
            continue
        chisq_array = np.zeros(grid_reader.grid_shape)

        for j, k in enumerate(measurement_keys):
            meas = data_df.loc[i, k]
            err = data_df.loc[i, "err_"+k]
            # err = 1

            # Absolute calibration error of 10%
            abscal_rel = abscal_pct/100.
            if 'ratio' in k:
                # Account for the sum of 2 10%s in the ratio, sqrt(2)*10%
                abscal_err = np.sqrt(2)*abscal_rel*meas
            else:
                abscal_err = abscal_rel*meas
            # print("err/abscal_err", err/abscal_err)
            err = err + abscal_err

            chisq_array += ((meas - grid_arrays[j])/err)**2
        chisq_list.append(chisq_array)
        count += 1
    print(f"Using {count} pixel samples")
    print(f"Len of chisq list {len(chisq_list)}")

    # chisq_array /= np.sum(notnull_mask)
    # Reduce chisq
    # no action needed, dof = 1 (3 data - 2 params)

    def _describe_error_ellipse(chi_squared_array):
        """
        Given a 2d reduced chisq array, describe the chisq<1 error ellipse.
        Return (x0,x1, y0,y1) where x and y are the grid_reader.axis_array values
        """
        good_locations = np.where(chi_squared_array < 1)
        xs, ys = [grid_reader.axis_arrays[param_select][coord_idxs] for param_select, coord_idxs in enumerate(good_locations[::-1])]
        if len(xs) > 0:
            return (np.min(xs), np.max(xs), np.min(ys), np.max(ys))
        else:
            return (np.nan,)*4

    """ Fit each pixel's chisq grid """

    extent = calc_extent(grid_reader.axis_arrays[0]) + calc_extent(grid_reader.axis_arrays[1])

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(211)
    # im = ax.imshow(np.log10(np.mean(chisq_list, axis=0)), origin='lower', extent=extent)
    im = ax.imshow(np.log10(np.mean(chisq_list, axis=0)), origin='lower', extent=extent)

    soln_x_arr = []
    soln_y_arr = []
    min_chisq_arr = []
    errorbars_arr = []
    for arr in chisq_list:
        ax.contour(arr, levels=[1], colors='w', origin='lower', extent=extent, alpha=0.2)
        min_chisq = np.min(arr)
        min_chisq_arr.append(min_chisq)
        min_loc = np.where(arr==min_chisq)
        solution_x, solution_y = [grid_reader.axis_arrays[i][j] for i, j in enumerate(min_loc[::-1])]
        soln_x_arr.append(solution_x)
        soln_y_arr.append(solution_y)
        # Estimate error bars using chisq < 1
        errorbars_arr.append(_describe_error_ellipse(arr))

    # sc = ax.scatter(soln_x_arr, soln_y_arr, c=min_chisq_arr, vmin=0, vmax=3, alpha=0.7, cmap='jet')
    soln_x_arr = np.array(soln_x_arr).ravel()
    soln_y_arr = np.array(soln_y_arr).ravel()

    x0, x1, y0, y1 = np.array(list(zip(*errorbars_arr)))
    x0 = soln_x_arr - x0
    x1 = x1 - soln_x_arr
    y0 = soln_y_arr - y0
    y1 = y1 - soln_y_arr
    ax.errorbar(soln_x_arr, soln_y_arr, xerr=[x0, x1], yerr=[y0, y1], color='cyan', alpha=0.5, linestyle='none', capsize=6, marker='o')

    fig.colorbar(im, ax=ax)

    ax = plt.subplot(223)
    hvals, _, _ = ax.hist(soln_x_arr)
    ax.plot([np.mean(soln_x_arr)], [np.max(hvals)*1.1], marker='+', color='k')
    ax.plot([np.median(soln_x_arr)], [np.max(hvals)*1.1], marker='x', color='k')
    lo, hi = misc_utils.flquantiles(soln_x_arr, 6)
    ax.plot([lo, hi], [[np.max(hvals)*1.1]]*2, color='g', alpha=0.4)
    ax = plt.subplot(224)
    ax.hist(soln_y_arr)

    # plt.show()
    # return
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"test_individual_pix_chisq_grid{tk}{extra_stub}_{region_label}_abscal{abscal_pct}.png"),
        metadata=catalog.utils.create_png_metadata(title=f"+{abscal_pct}pct abscal. {', '.join(measurement_keys)}",
            file=__file__, func="individual_pixel_ensemble_chisq"))


def unified_chisq_plotting_system(region_label="N19", abscal_pct=10):
    """
    Feb 9, 2024
    Going to go a little code-heavy here to make these plots easy to reproduce
    and compare between
    The idea is to have a common data loading and plot formatting framework, but
    to separate the actual plotting and analysis into two blocks of code in an
    "if" statement or two separate functions or something.
    This is meant to combine sum_chisq_to_get_masked_area_errorbars() and
    individual_pixel_ensemble_chisq()
    Oh maybe we can just combine the two methods entirely.. let's try the separate
    plots first to see how close the answers are

    For the 1d histograms on the sides, I referenced like 7 year old code lol
    helpss.analyze_manticore.setup_scaHist()
    There, I use the plt.axes() Axes constructor which can use a `rect` box
    argument [x0 y0 dx dy] where the x0 and y0 define the bottom left corner
    Continued in today's notes 2024-02-09
    """
    # no args gets the 30 K grid, fine for N19 ring
    tk = 30
    grid_reader, extra_stub = grid_reader_filename_wrapper(tk, 'fine0.05')
    # grid_reader, extra_stub = grid_reader_filename_wrapper(tk)
    if region_label == "N19":
        data_fn = "misc_regrids/sample_mask_N19_11.0.21.0_vals_regrid.csv"
        # data_fn = "misc_regrids/sample_mask_test_1_11.0.21.0_vals_regrid.csv" # N19
    elif region_label == "BNR":
        data_fn = "misc_regrids/sample_mask_BNR_23.0.27.0_vals_regrid.csv"
        # data_fn = "misc_regrids/sample_mask_BNR_23.0.27.0_vals_regrid.csv" # BNR
    else:
        raise RuntimeError(f"unknown region label {region_label}")
    data_df = pd.read_csv(catalog.utils.search_for_file(data_fn))
    # Identify the measurements to use
    measurement_keys = ["ratio_32_10", "peak_c18o10", "peak_13co32"]
    grid_keys = ["TR_13co_32 / TR_13co_10", "TR_c18o_10", "TR_13co_32"]
    official_names = ["$\\frac{^{13}{\\rm CO}(3-2)}{ ^{13}{\\rm CO}(1-0)}$", "${\\rm C^{18}O}(1-0)$", "$^{13}{\\rm CO}(3-2)$"]
    grid_arrays = [grid_reader.get(k) for k in grid_keys]
    # Set absolute calibration percentage
    # abscal_pct = 10

    # Iterate thru measurements once quickly to build the notnull_mask
    notnull_mask = None
    for meas_key in measurement_keys:
        mask = data_df[meas_key].notnull()
        if notnull_mask is None:
            # Establish
            notnull_mask = mask
        else:
            # Add on
            notnull_mask = notnull_mask & mask

    # Make figure and define Axes
    fig = plt.figure(figsize=(10, 8))
    # Bottom and left edge of center Axes and full Axes width
    # Define the rectangle arguments for the plots
    anchor, width = 0.1, 0.62
    # Bottom or left edge of side Axes and thin side Axes width
    anchor_sideplot = 1.03
    width_sideplot = 0.3

    rect_center = [anchor]*2 + [width]*2
    ax_center = fig.add_axes(rect_center)

    # Define the hist axes using inset_axes
    colorbar_allowance = 0 # 0.2 # width allowance for colorbar
    ax_x = ax_center.inset_axes([0, anchor_sideplot, 1, width_sideplot], sharex=ax_center)
    ax_y = ax_center.inset_axes([anchor_sideplot+colorbar_allowance, 0, width_sideplot, 1], sharey=ax_center)

    ax_x.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_y.tick_params(axis='y', which='both', left=False, labelleft=False)

    # # rect_x is along the top, and shares the x axis with the center
    # rect_x = [axes_anchor, axes_anchor_sideplot, axes_width, 0.22]
    # rect_y = [axes_anchor_sideplot, axes_anchor, 0.2, axes_width]
    # ax_x = fig.add_axes(rect_x, sharex=ax_center)
    # ax_y = fig.add_axes(rect_y, sharey=ax_center)
    # For the central image
    extent = calc_extent(grid_reader.axis_arrays[0]) + calc_extent(grid_reader.axis_arrays[1])
    # Colorbar as another Axes. This rect is defined w.r.t. the center Axes
    rect_cbar = [1.005, 0, 0.05, 1]
    # cbar_ax = ax_center.inset_axes(rect_cbar)

    # Plot the average measurements and their variation
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # median_meas_chisq_array = np.zeros(grid_reader.grid_shape)
    for k_idx, k in enumerate(measurement_keys):
        meas = data_df[k][notnull_mask]
        mean_meas = np.mean(meas)
        median_meas = np.median(meas)
        lo, hi = misc_utils.flquantiles(meas, 6)
        std_err = np.std(meas)
        c = color_list[k_idx]
        img = grid_reader.get(grid_keys[k_idx])
        ax_center.contour(img, levels=[median_meas], linestyles='-', colors=c, linewidths=3, extent=extent, alpha=0.9)
        ax_center.contour(img, levels=[mean_meas], linestyles='--', colors=c, linewidths=2, extent=extent, alpha=0.9)
        # ax_center.contour(img, levels=[median_meas-std_err, median_meas+std_err], linestyles=':', colors=c, linewidths=2, extent=extent, alpha=0.9)
        ax_center.contourf(img, levels=[lo, hi], colors=c, alpha=0.2, extent=extent)

    # Some helper functions
    def _describe_error_ellipse(chi_squared_array, cutoff="1"):
        """
        Given a 2d reduced chisq array, describe the chisq<1 error ellipse.
        :param cutoff: string "1" or "2x"
            "1" means chisq < 1
            "2x" means chisq < 2*min(chisq)
        Return (x0,x1, y0,y1) where x and y are the grid_reader.axis_array values
        """
        if cutoff == "1":
            cutoff_number = 1
        elif cutoff == "2x":
            cutoff_number = 2*np.min(chi_squared_array)
        good_locations = np.where(chi_squared_array < cutoff_number)
        xs, ys = [grid_reader.axis_arrays[param_select][coord_idxs] for param_select, coord_idxs in enumerate(good_locations[::-1])]
        if len(xs) > 0:
            return (np.min(xs), np.max(xs), np.min(ys), np.max(ys))
        else:
            return (np.nan,)*4

    def _find_chisq_min(chi_squared_array):
        """
        Given 2d X2/dof, find location of minimum in data (parameter) coordinates
        Return (x, y)
        """
        min_chisq = np.min(chi_squared_array)
        min_loc = np.where(chi_squared_array == min_chisq)
        solution_x, solution_y = [grid_reader.axis_arrays[param_select][coord_idx] for param_select, coord_idx in enumerate(min_loc[::-1])]
        return solution_x, solution_y

    def _chisq_iter_over_pixels():
        """
        Generator function. Good for either of the methods.
        On each next() call, yields the chisq array for a single pixel (all measurements)
        """
        for i in data_df.index:
            if not notnull_mask[i]:
                # is null, skip
                continue
            # Make empty chisq array for this pixel
            chisq_array = np.zeros(grid_reader.grid_shape)
            for j, k in enumerate(measurement_keys):
                meas = data_df.loc[i, k]
                err = data_df.loc[i, "err_"+k]
                # err = 1

                # Absolute calibration error of some % close to 10
                abscal_rel = abscal_pct/100.
                if 'ratio' in k:
                    # Account for the sum of 2 10%s in the ratio, sqrt(2)*10%
                    abscal_err = np.sqrt(2)*abscal_rel*meas
                else:
                    abscal_err = abscal_rel*meas
                # print("err/abscal_err", err/abscal_err)
                # Add statistical and systematic error estimates
                err = err + abscal_err

                chisq_array += ((meas - grid_arrays[j])/err)**2
            yield chisq_array

    color = 'k'
    alt_color = color_list[len(measurement_keys)+3]

    """ Calculate the chisqs """
    chisq_list = [x for x in _chisq_iter_over_pixels()]
    count = len(chisq_list)

    """ Average the chisqs """
    # chisq_avg = np.mean(chisq_list, axis=0)
    # print(f"Using {count} pixels for region {region_label}")
    # solution_x, solution_y = _find_chisq_min(chisq_avg)
    # # im = ax_center.imshow(np.log10(chisq_avg), origin='lower', extent=extent, cmap='viridis')
    # # cbar = fig.colorbar(im, cax=cbar_ax, label="log$_{10}$ average $\\chi^2 / {\\rm dof}$")
    # ax_center.contour(chisq_avg, levels=[np.min(chisq_avg)*2], colors=alt_color, linewidths=4, origin='lower', extent=extent)
    # ax_center.text(0.02, 0.02, "Minimum avg $\\chi^2 / {\\rm dof} = $" + f"{np.min(chisq_avg):.0f}", color=alt_color, transform=ax_center.transAxes, ha='left', va='bottom')
    # ax_center.text(0.02, 0.07, "Red contour at 2$\\times$ min avg $\\chi^2 / {\\rm dof}$", color=alt_color, transform=ax_center.transAxes, ha='left', va='bottom')
    #
    # chisq_avg_errors = _describe_error_ellipse(chisq_avg, cutoff='2x')
    # # print("soln", solution_x, solution_y)
    # # print(chisq_avg_errors)
    # xerr = [abs(solution_x - x) for x in chisq_avg_errors[:2]]
    # yerr = [abs(solution_y - y) for y in chisq_avg_errors[2:]]
    # # print(xerr, yerr)
    # ax_center.errorbar([solution_x], [solution_y], xerr=xerr, yerr=yerr, marker='o', color=alt_color, linewidth=2)

    formatter_f = [
        lambda logn : f"{(10.**logn):.0f}",
        lambda logNH2 : f"{(10.**logNH2):.2E}"
    ]

    text_dy = [0.2, 0.05]
    text_x = [0.23, 0.99]
    text_y = 0.95
    text_kwargs = dict(ha='right', va='top')

    """ Get individual pixel solutions and errors """
    print(f"REGION: {region_label}")
    # We still have access to chisq_list from above
    soln_tups = []
    error_tups = []
    for arr in chisq_list:
        ax_center.contour(arr, levels=[1], colors=color, origin='lower', extent=extent, alpha=0.1)
        soln_tups.append(_find_chisq_min(arr))
        error_tups.append(_describe_error_ellipse(arr, cutoff="1"))
    soln_arrs = np.squeeze(np.array(soln_tups)).T
    error_arrs = np.array(error_tups).T
    median_solns_and_lims = [] # should be in (val, lo, hi) order
    ms = 10
    for i in range(2):
        # Make the error arrs into +/- instead of limits
        soln_arr = soln_arrs[i, :]
        error_arrs[i*2:(i+1)*2, :] = np.abs(soln_arr - error_arrs[i*2:(i+1)*2, :])
        ax = [ax_x, ax_y][i]
        orientation = ['vertical', 'horizontal'][i]
        hvals, _, histp = ax.hist(soln_arr, histtype='step', align='mid', color=color, orientation=orientation, label="Pixel solutions")
        hmax = [np.max(hvals)*0.7]
        arg_order = [1, -1][i]
        soln_mean, soln_median = np.mean(soln_arr), np.median(soln_arr)

        # span_f = [ax.axvspan, ax.axhspan][i]
        # line_f = [ax.axvline, ax.axhline][i]
        lo, hi = misc_utils.flquantiles(soln_arr, 6)

        median_solns_and_lims.append((soln_median, soln_median-lo, hi-soln_median)) # (val, -lo, +hi) for errorbar plotting

        p1, = ax.plot(*[[soln_median], hmax][::arg_order], marker='o', markersize=ms, color=alt_color, label=('Median: '*(1-i) + formatter_f[i](soln_median)))
        p2, = ax.plot(*[[soln_mean], hmax][::arg_order], marker='x', markersize=ms, color=alt_color, label=('Mean: '*(1-i) + formatter_f[i](soln_mean)))
        ax.plot(*[(lo, hi), hmax*2][::arg_order], color=alt_color, linewidth=2)
        ax.legend(handles=[p1, p2, histp[0]][:3-i], loc=['upper left', 'upper center'][i])

        # ax.plot(*[[soln_mean], hmax][::arg_order], marker='x', markersize=ms, color=alt_color)
        # ax.text(text_x[i], text_y-text_dy[i], "med " + formatter_f[i](soln_median), color=color, transform=ax.transAxes, **text_kwargs)
        # ax.text(text_x[i], text_y-text_dy[i]*2, "mean " + formatter_f[i](soln_mean), color=color, transform=ax.transAxes, **text_kwargs)

        # line_f(soln_mean, color=color, linestyle='--')
        # line_f(soln_median, color=color)
        # span_f(lo, hi, color=color, alpha=0.2)
        param_linear = lambda lx : 10.**lx
        param_fmt = lambda x : f"{x:.1E}"
        param_fmt_l = lambda lx : f"{lx:.2f}"
        param_linear_diff_from_log = lambda med_lx, lim_lx : param_linear(lim_lx) - param_linear(med_lx)
        print(f"\t param {i}")
        print("\t abs")
        print("\t"*2, tuple(param_fmt_l(x) for x in (soln_median, lo, hi)))
        print("\t"*2, tuple(param_fmt(param_linear(x)) for x in (soln_median, lo, hi)))
        print("\t rel")
        print("\t"*2, tuple(param_fmt_l(x) for x in (soln_median, soln_median-lo, hi-soln_median)))
        print("\t"*2, param_fmt(param_linear(soln_median)), param_fmt(param_linear_diff_from_log(soln_median, lo)), "+", param_fmt(param_linear_diff_from_log(soln_median, hi)))

        # [ax_center.axvline, ax_center.axhline][i](soln_median, color='k', alpha=0.7)
        # [ax_center.axvspan, ax_center.axhspan][i](lo, hi, color='k', alpha=0.2)

        # # Add some stuff about the average solutions
        # soln_val = [solution_x, solution_y][i]
        # ax = [ax_x, ax_y][i]
        # lims = chisq_avg_errors[i*2:(i+1)*2]
        # ax.plot(*[[soln_val], hmax][::arg_order], marker='o', markersize=ms, color=alt_color)
        # ax.plot(*[lims, hmax*2][::arg_order], color=alt_color, linewidth=2)
        # # line_f(soln_val, color='LimeGreen')
        # # span_f(*lims, color='LimeGreen', alpha=0.3)
        # ax.text(text_x[i], text_y, formatter_f[i](soln_val.item()), color=alt_color, transform=ax.transAxes, **text_kwargs)



    ax_center.errorbar(*soln_arrs, xerr=error_arrs[:2, :], yerr=error_arrs[2:, :], color=color, alpha=0.1, linestyle='none', capsize=0, marker='o') # capsize=6 looks nice
    solution_x, solution_y = [median_solns_and_lims[i][0] for i in range(2)]
    xerr, yerr = [np.array(median_solns_and_lims[i][1:]).reshape(2, 1) for i in range(2)]
    ax_center.errorbar([solution_x], [solution_y], xerr=xerr, yerr=yerr, marker='o', color=alt_color, capsize=6, capthick=2, elinewidth=2, markersize=ms, zorder=1000)

    # ax_center.text(0.01, 0.25, "Black contours at $\\chi^2 / {\\rm dof} = 1$", color=color, transform=ax_center.transAxes, ha='left', va='bottom')

    # ax_center.text(0.02, 0.12, "Solid lines: median measurement", color=color, transform=ax_center.transAxes, ha='left', va='bottom')
    # ax_center.text(0.02, 0.07, "Dashed lines: mean measurement", color=color, transform=ax_center.transAxes, ha='left', va='bottom')
    # ax_center.text(0.02, 0.02, "Shaded: uncertainty", color=color, transform=ax_center.transAxes, ha='left', va='bottom')

    bottom_left_legend = ax_center.legend(handles=[
        ax_center.errorbar([], [], xerr=[], yerr=[], color=color, alpha=0.8, linestyle='none', capsize=0, marker='o', label='Pixel solution'),
        Line2D([], [], color=color, linestyle='-', linewidth=1, alpha=0.8, marker='none', label="Pixel $\\chi^2 / {\\rm dof} = 1$"),
        Line2D([], [], color='gray', linestyle='-', linewidth=3, marker='none', label='Median'),
        Line2D([], [], color='gray', linestyle='--', linewidth=2, marker='none', label='Mean'),
        mpatches.Patch(color='gray', alpha=0.2, label="16$^{\\rm th}$"+"$-$"+"84$^{\\rm th}$ %ile")
    ], loc='lower left')
    # bbox_to_anchor=[anchor_sideplot+colorbar_allowance, anchor_sideplot, width_sideplot, width_sideplot]
    top_right_legend = ax_center.legend(loc='upper left', handles=[
        mpatches.Patch(color=color_list[i], label=official_names[i]) for i in range(len(official_names))
    ])
    ax_center.add_artist(bottom_left_legend)

    ax_center.set_xlabel("log$_{10}\ n$ [cm$^{-3}$]")
    ax_center.set_ylabel("log$_{10}\ {\\rm N}(H_2)$ [cm$^{-2}$]")

    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"chisq_grid_full_analysis{tk}{extra_stub}_{region_label}_abscal{abscal_pct}.png"),
        metadata=catalog.utils.create_png_metadata(title=f"+{abscal_pct}pct abscal. {', '.join(measurement_keys)}",
            file=__file__, func="unified_chisq_plotting_system"))

def mask_footprint_reference_plot():
    """
    Feb 12, 2024
    Plot the CO pixel mask over the spitzer for a sense of reference
    """

    # Load 8 micron for reference image
    ref_stub = "160um" # "irac4-large"
    img_large, img_info = get_2d_map(ref_stub)
    cutout = misc_utils.cutout2d_from_region(img_large, img_info['wcs'], get_cutout_box_filename('med'), align_with_frame='galactic')
    img = cutout.data
    wcs = cutout.wcs
    del img_large
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=wcs)
    # ax.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(25), vmax=np.arcsinh(400), cmap=cmocean.cm.matter_r)
    ax.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(-0.3), vmax=np.arcsinh(1.3), cmap=cmocean.cm.matter_r)
    # ax.imshow(np.arcsinh(img), origin='lower', vmin=np.arcsinh(450), vmax=np.arcsinh(5000), cmap=cmocean.cm.matter_r)
    # ax.imshow(img, origin='lower', vmax=200, cmap=cmocean.cm.matter_r)


    data_fns = ["misc_regrids/sample_mask_N19_11.0.21.0_regrid_mask.fits", "misc_regrids/sample_mask_BNR_23.0.27.0_regrid_mask.fits"]
    for i, data_fn in enumerate(data_fns):
        mask_float, mask_hdr = fits.getdata(catalog.utils.search_for_file(data_fn), header=True)
        mask_reproj = reproject_interp((mask_float, mask_hdr), wcs, shape_out=img.shape, return_footprint=False)
        ax.contour(mask_reproj, levels=[0.5], colors='kw'[i], linewidths=4)


    ax.coords[0].set_major_formatter("d.d")
    ax.coords[1].set_major_formatter("d.d")
    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")
    ax.text(0.02, 0.98, get_data_name(ref_stub), color='white', fontsize=15, transform=ax.transAxes, ha='left', va='top')
    fig.tight_layout()
    # plt.show()
    # return
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"mask_overlay_BOTH_{ref_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title="mask", file=__file__, func="mask_footprint_reference_plot"))


def print_out_particle_mass_for_rho():
    """
    January 3, 2024
    Print out the particle mass needed to turn number density n in to mass density rho
    This will account for mean molecular weight
    """
    atomic_particle_mass = Hmass * mean_molecular_weight_neutral
    print(f"Atomic Hmass * mu: {atomic_particle_mass:.2E}")
    print(f"Molecular 2 * Hmass * mu: {2*atomic_particle_mass:.2E}")


def trim_CO_mass_to_CII_grid(velocity_limits=None):
    """
    January 29, 2024
    Mask the PMO column densities using the CII image. Add them up for mass
    """
    if velocity_limits is None:
        velocity_limits = (11*kms, 21*kms)
    vel_stub_simple = make_simple_vel_stub(velocity_limits)
    pmo_fn = f"purplemountain/column_density_v3__13co10-pmo_{vel_stub_simple}.fits"
    # Make CII valid mask
    cii_mask_fn = "sofia/cii_mom0_0.0.40.0.fits"
    cii_img, cii_hdr = fits.getdata(catalog.utils.search_for_file(cii_mask_fn), header=True)
    cii_mask = np.isfinite(cii_img).astype(float)

    # Reproject CII to PMO
    pmo_mass_pixel, pmo_hdr = fits.getdata(catalog.utils.search_for_file(pmo_fn), extname='mass', header=True)
    mask_reproj = reproject_interp((cii_mask, cii_hdr), pmo_hdr, return_footprint=False) > 0.5
    pmo_mass_pixel_subset = pmo_mass_pixel.copy()
    pmo_mass_pixel_subset[~mask_reproj] = np.nan

    cii_fn = f"sofia/Cp_largeM16_coldens_ff1.0_{vel_stub_simple}.fits"
    cii_mass_pixel, cii_hdr = fits.getdata(catalog.utils.search_for_file(cii_fn), extname='mass', header=True)

    cii_mass_sum = np.nansum(cii_mass_pixel) * u.solMass
    pmo_mass_sum = np.nansum(pmo_mass_pixel) * u.solMass
    pmo_mass_sum_subset = np.nansum(pmo_mass_pixel_subset) * u.solMass

    print("velocity", make_vel_stub(velocity_limits))
    print(f"CII {cii_mass_sum:.0f}\n\t{cii_mass_sum:.1E}")
    print(f"PMO full {pmo_mass_sum:.0f}\n\t{pmo_mass_sum:.1E}")
    print(f"PMO subset {pmo_mass_sum_subset:.0f}\n\t{pmo_mass_sum_subset:.1E}")

    plt.subplot(221)
    plt.imshow(mask_reproj, origin='lower')
    plt.subplot(222)
    plt.imshow(pmo_mass_pixel, origin='lower')
    plt.subplot(223)
    plt.imshow(pmo_mass_pixel_subset, origin='lower')
    plt.subplot(224)
    plt.imshow(cii_mass_pixel, origin='lower')
    plt.show()


"""
CII column density
"""

def calculate_cii_column_density(filling_factor=1.0, velocity_limits=None, cutout_reg_stub=None, mask_cutoff=6*u.K):
    """
    October 5, 2023
    Copying things from m16_deepdive.calculate_cii_column_density
    That function says "Following Okada 2015 Sec 3.3, pg 10
    Several equations and some rules on how to assume Tex"
    I will also include velocity and cutout region arguments
    :param filling_factor: float (0-1] assumed beam filling factor of cii emission.
        1.0 (default) is generally okay, implies fully-filled beam.
    :param velocity_limits: tuple(Quantity, Quantity) low, high velocity limits.
        Quantities must be velocities, like km/s.
    :param cutout_reg_stub: (optional) str label for a defined cutout box.
        If given, a spatial cutout will be made of the integrated intensity
        prior to calculation of the column density.
        !!!cutout_reg_stub not yet implemented!!!
    :param mask_cutoff: float or Quantity (Kelvins), value below which CII
        data should be masked out (set to 0 K).
        A float value will be interpreted as a multiplier of the channel noise,
        NOT as a value in K. A Quantity in K will be interpreted directly,
        unrelated to the noise.
        To avoid masking, set mask_cutoff=0
        Default is 6 K
    """
    # Load in file
    # Use the highest resolution full CII map by default. Probably no reason to go into the convolved maps for column density
    cii_cube_fn = get_map_filename('cii')
    # Grab the directory and basename from the CubeData object before trimming down to just SpectralCube
    cii_cube = cube_utils.CubeData(cii_cube_fn)
    savedir = cii_cube.directory
    cii_cube_fn_basename = cii_cube.basename
    cii_cube = cii_cube.data
    # Apply velocity limits
    if velocity_limits is not None:
        cii_cube = cii_cube.spectral_slab(*velocity_limits)
    # Apply region cutout; reuse the slices on the entire cube for efficiency
    if cutout_reg_stub is not None:
        cutout = misc_utils.cutout2d_from_region(cii_cube[0, :, :].to_value(), cii_cube[0, :, :].wcs, get_cutout_box_filename("N19-small"))
        cii_cube = cii_cube[(slice(None), *cutout.slices_original)]
        del cutout # finished with this now

    # Get channel noise
    channel_noise = cube_utils.onesigmas['cii'] * u.K
    # Check mask_cutoff argument
    if not hasattr(mask_cutoff, 'unit'):
        # If no unit, must be a float or other numerical type. Assume multiplier of 1 sigma noise
        mask_cutoff = mask_cutoff * channel_noise
    # Apply mask using mask_cutoff
    print(f"Masking below {mask_cutoff:.2f}")
    cii_cube = cii_cube.with_mask(cii_cube > mask_cutoff).with_fill_value(0*u.K)
    # Get spectral axis in THz units, since equations use frequency
    rest_freq = cii_cube.header['RESTFREQ'] * u.Hz
    freq_axis = cii_cube.spectral_axis.to(u.THz, equivalencies=cii_cube.velocity_convention(rest_freq))
    # Set up some constants
    hnu_kB = const.h * rest_freq / const.k_B
    print(f"T_0 = E_u / k_B = {hnu_kB.decompose():.2f}")
    g0, g1 = 2, 4 # lower, upper
    A10 = 10**(-5.63437) / u.s # Einstein A

    # peak T
    peak_T_map = cii_cube.max(axis=0).quantity
    """
    See 2023-10-05 notes. tau <= 1.3 for Pillars, and turns out that the 13CII
    detection towards the IRAS source in the Bright Northern Ridge (fka RCW 165)
    also gives tau ~ 1.3, with 80 K 12CII and 2 K 13CII.

    For updated optical depth, see 2024-01-17 notes. The 13CII spectrum plots
    show a higher peak optical depth. However, I should stick with 1.3 for ease
    and because the 2.2 higher one is probably only valid towards the MYSO.
    """
    assumed_optical_depth = 1.3

    # Now find excitation temperature
    Tex_map = (hnu_kB / np.log((1 - np.exp(-assumed_optical_depth))*(filling_factor * hnu_kB / peak_T_map) + 1)).decompose()
    original_Tex_map = Tex_map.copy()
    # fixed_Tex_val = np.nanmax(Tex_map) # ~150 for entire map
    # Tex_map[:] = fixed_Tex_val
    """
    The max Tex is like 150, towards fka RCW 165 (IRAS source) where brightest CII emission is.
    I think this is a little too high for the general area, and I think it's likely that that
    ring has special excitation conditions, namely a higher density given that CO column
    densities are high there too.
    I think I might pick a more moderate 120 K, which is a little higher than I
    used in the last paper, but I'll have to have some caveat in the paper
    that it's all guesswork and this is the best we can do without a widespread
    13CII detection.

    What I'll do exactly is make Tex[Tex < 120] = 120 and leave all other bits alone
    so that they can vary
    """
    Tex_cutoff = 120*u.K
    map_max_Tex = np.nanmax(Tex_map)
    if map_max_Tex > Tex_cutoff:
        print(f"Max Tex in map {map_max_Tex:.2f}, higher than {Tex_cutoff}. Using {Tex_cutoff}")
        fixed_Tex_val = 120*u.K
    else:
        print(f"Max Tex in map {map_max_Tex:.2f}, using that.")
        fixed_Tex_val = map_max_Tex
    Tex_map[Tex_map < fixed_Tex_val] = fixed_Tex_val

    # plt.subplot(121)
    # plt.imshow(peak_T_map.to_value(), origin='lower')
    # plt.subplot(122)
    # plt.imshow(Tex_map.to_value(), origin='lower', vmin=120, vmax=160)
    # print(fixed_Tex_val)
    # plt.show()

    """
    Now solve! Copied directly from m16_deepdive.calculate_cii_column_density
    """
    # Error on Tex
    # d/dx (a / log(b/x + 1)) = ab / (x(b+x)log*2((b+x)/x))
    helper_a = hnu_kB
    helper_b = (1 - np.exp(-assumed_optical_depth))*filling_factor*hnu_kB
    err_Tex_map = (channel_noise * (helper_a * helper_b) / (Tex_map * (helper_b + Tex_map) * np.log((helper_b/Tex_map) + 1)**2)).decompose()

    #########################
    # dont need to change much below this
    ####################

    # This is how Tex will often be used, and it needs the extra spectral dimension at axis=0
    hnukBTex = hnu_kB/Tex_map[np.newaxis, :]
    err_hnukBTex = (err_Tex_map * hnu_kB / Tex_map**2)[np.newaxis, :]

    exp_hnukBTex = np.exp(hnukBTex)
    err_exp_hnukBTex = err_hnukBTex * exp_hnukBTex # d(e^(a/x)) = (a dx / x^2) e^(a/x)

    # partition function?
    Z = g0 + g1*np.exp(-hnukBTex) # hnu_kB = Eu/kB since ground is 0 energy (might also be ok if not, but it's definitely ok in this case)
    err_Z = g1 * err_hnukBTex * np.exp(-hnukBTex)

    # optical depth in a given channel
    channel_tau = -1*np.log(1 - ((cii_cube.filled_data[:] / (filling_factor * hnu_kB)) * (exp_hnukBTex - 1))) # 3d cube
    print(channel_tau.unit)

    # Uncertainty on optical depth in channel
    helper_a = (exp_hnukBTex - 1) / (filling_factor * hnu_kB)
    err_channel_tau_from_Tb = channel_noise * helper_a / (1 - helper_a*cii_cube.filled_data[:])
    del helper_a
    # reset the definition of "a"! not the same!
    helper_a = (cii_cube.filled_data[:] / (filling_factor * hnu_kB))
    helper_numerator = helper_a * err_exp_hnukBTex
    helper_denominator = 1. - helper_a*(exp_hnukBTex - 1)
    err_channel_tau_from_Tex = helper_numerator / helper_denominator
    # quick analysis shows approximately equal contributions from each source of uncertainty
    err_channel_tau = np.sqrt(err_channel_tau_from_Tex**2 + err_channel_tau_from_Tb**2).decompose()
    # relatively small percentage of channel_tau values


    # Column density in a given channel
    column_constants = (8*np.pi * (rest_freq / const.c)**2) / (g1*A10)
    channel_column = (
        column_constants * channel_tau * Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex)))
    ).decompose()

    # Uncertainty on column density in channel
    helper_1 = (err_channel_tau * Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex))))**2
    helper_2 = (channel_tau * err_Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex))))**2
    helper_3 = (channel_tau * Z * (err_exp_hnukBTex * exp_hnukBTex * (exp_hnukBTex - 2) / (exp_hnukBTex - 1)**2.))**2
    # Quick analysis shows channel_tau error dominates: factor of 40 over Z err, but only factor of 4 over Tex err
    err_channel_column = (np.sqrt(helper_1 + helper_2 + helper_3) * column_constants).decompose()


    integrated_column_map = np.trapz(channel_column[::-1, :, :], x=freq_axis[::-1], axis=0).to(u.cm**-2)
    # Let's just do quadrature sum * dnu for the integral uncertainty propagation
    dnu = np.median(np.diff(freq_axis[::-1]))
    err_integrated_column_map = (np.sqrt(np.sum(err_channel_column**2, axis=0))*dnu).to(u.cm**-2)
    # looking like a 10% error


    integrated_H_column_map = integrated_column_map / Cp_H_ratio
    err_integrated_H_column_map = err_integrated_column_map / Cp_H_ratio

    particle_mass = Hmass * mean_molecular_weight_neutral
    integrated_mass_column_map = integrated_H_column_map * particle_mass
    err_integrated_mass_column_map = err_integrated_H_column_map * particle_mass

    pixel_scale = misc_utils.get_pixel_scale(cii_cube[0, :, :].wcs)
    pixel_area = (pixel_scale * (los_distance_M16/u.radian))**2
    err_pixel_area = 2 * (pixel_scale/u.radian)**2 * los_distance_M16 * err_los_distance_M16

    integrated_mass_pixel_column_map = (integrated_mass_column_map * pixel_area).to(u.solMass)
    # Include error from column density and from LOS distance
    err_integrated_mass_pixel_column_map_raw = np.sqrt((err_integrated_mass_column_map * pixel_area)**2 + (integrated_mass_column_map * err_pixel_area)**2).to(u.solMass)
    pixels_per_beam = (cii_cube.beam.sr / pixel_scale**2).decompose()
    # sqrt(oversample_factor) to correct for correlated pixels
    err_integrated_mass_pixel_column_map = np.sqrt(pixels_per_beam) * err_integrated_mass_pixel_column_map_raw


    def make_and_fill_header():
        # fill header with stuff, make it from WCS
        hdr = wcs_flat.to_header()
        hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
        hdr['CREATOR'] = f"Ramsey, {__file__}"
        hdr['HISTORY'] = "Using calculate_cii_column_density.py rewritten for large M16"
        hdr['HISTORY'] = f"Original CII map {cii_cube_fn_basename}"
        hdr['HISTORY'] = f"Fixed Tex {fixed_Tex_val:.2f} max Tex calculated using tau={assumed_optical_depth}"
        hdr['HISTORY'] = "if Tex=120, left Tex>120 (up to 150) alone, only raised Tex>120 to 120"
        hdr['HISTORY'] = f"C+/H = {Cp_H_ratio:.2E}"
        hdr['HISTORY'] = f"Hmass = {Hmass:.3E}"
        hdr['HISTORY'] = f"mean molecular weight = {mean_molecular_weight_neutral:.2f}"
        hdr['HISTORY'] = f"adopted particle mass = {particle_mass:.2E}"
        hdr['HISTORY'] = f"pixel scale = {pixel_scale.to(u.arcsec):.3E}"
        hdr['HISTORY'] = f"pixel area = {pixel_area.to(u.pc**2):.3E}"
        hdr['HISTORY'] = f"sqrt(pixels/beam) oversample = {np.sqrt(pixels_per_beam):.2f}"
        hdr['HISTORY'] = f"filling factor = {filling_factor:.2f}"
        hdr['HISTORY'] = f"Masked below {mask_cutoff:.2f}"
        if velocity_limits is not None:
            hdr['HISTORY'] = f"Integrated within {make_simple_vel_stub(velocity_limits)}"

        return hdr

    phdu = fits.PrimaryHDU()
    wcs_flat = cii_cube[0, :, :].wcs

    header1 = make_and_fill_header()
    header1['EXTNAME'] = "C+coldens"
    header1['BUNIT'] = str(integrated_column_map.unit)
    hdu_NCp = fits.ImageHDU(data=integrated_column_map.to_value(), header=header1)

    header2 = make_and_fill_header()
    header2['EXTNAME'] = "mass"
    header2['BUNIT'] = str(integrated_mass_pixel_column_map.unit)
    header2['COMMENT'] = "mass is per pixel on this image"
    hdu_mass = fits.ImageHDU(data=integrated_mass_pixel_column_map.to_value(), header=header2)

    header3 = make_and_fill_header()
    header3['EXTNAME'] = "varyingTex"
    header3['BUNIT'] = str(original_Tex_map.unit)
    header3['COMMENT'] = "This is !!NOT!! the Tex used to calculate column density"
    header3['COMMENT'] = "The fixed Tex (see above) is the max of this image"
    hdu_Tex = fits.ImageHDU(data=original_Tex_map.to(u.K).to_value(), header=header3)

    header4 = make_and_fill_header()
    header4['EXTNAME'] = "Hcoldens"
    header4['BUNIT'] = str(integrated_H_column_map.unit)
    header4['COMMENT'] = "mass is per pixel on this image"
    hdu_NH = fits.ImageHDU(data=integrated_H_column_map.to_value(), header=header4)

    pdrt_density = 2e4 * u.cm**-3
    los_distance_image = (integrated_H_column_map / pdrt_density).to(u.pc)

    header5 = make_and_fill_header()
    header5['EXTNAME'] = "scale_distance"
    header5['BUNIT'] = str(los_distance_image.unit)
    header5['COMMENT'] = f"calculated using PDRT density {pdrt_density:.1E}"
    hdu_distance = fits.ImageHDU(data=los_distance_image.to_value(), header=header5)


    # error maps
    header6 = make_and_fill_header()
    header6['EXTNAME'] = "err_C+coldens"
    header6['BUNIT'] = str(err_integrated_column_map.unit)
    header6['COMMENT'] = "uncertainty propagated"
    hdu_eNCp = fits.ImageHDU(data=err_integrated_column_map.to_value(), header=header6)

    header7 = make_and_fill_header()
    header7['EXTNAME'] = "err_mass"
    header7['BUNIT'] = str(err_integrated_mass_pixel_column_map.unit)
    header7['COMMENT'] = "uncertainty propagated"
    hdu_emass = fits.ImageHDU(data=err_integrated_mass_pixel_column_map.to_value(), header=header7)

    header8 = make_and_fill_header()
    header8['EXTNAME'] = "err_Hcoldens"
    header8['BUNIT'] = str(err_integrated_H_column_map.unit)
    header8['COMMENT'] = "uncertainty propagated"
    hdu_eNH = fits.ImageHDU(data=err_integrated_H_column_map.to_value(), header=header8)


    hdul = fits.HDUList([phdu, hdu_NCp, hdu_NH, hdu_mass, hdu_distance, hdu_Tex,
        hdu_eNCp, hdu_emass, hdu_eNH])

    velocity_stub = "_"+make_simple_vel_stub(velocity_limits) if velocity_limits is not None else ""
    cutout_stub = "_"+cutout_reg_stub if cutout_reg_stub is not None else ""
    savename = cube_utils.os.path.join(savedir, f"Cp_largeM16_coldens_ff{filling_factor:.1f}{velocity_stub}{cutout_stub}.fits")
    print(savename)
    hdul.writeto(savename, overwrite=True)


def calculate_cii_column_density_detection_threshold():
    """
    November 7, 2023
    Make very simple assumptions and find the column density of a 3 km/s wide line with 1 K peak temperature.
    """
    # Velocity array
    v_arr = np.arange(-10, 10.1, 0.5) * kms
    # linewidth
    fwhm = 3 * kms
    # Gaussian model
    g = cps2.models.Gaussian1D(amplitude=1, mean=0, stddev=(fwhm.to_value() / 2.355))
    t_arr = g(v_arr.to_value()) * u.K

    if False:
        plt.plot(v_arr.to_value(), t_arr.to_value())
        plt.show()

    rest_freq = 0.1900536900000e13 * u.Hz # right out of the header
    freq_axis = v_arr.to(u.Hz, equivalencies=u.doppler_radio(rest_freq))
    hnu_kB = const.h * rest_freq / const.k_B
    g0, g1 = 2, 4 # lower, upper
    A10 = 10**(-5.63437) / u.s # Einstein A
    filling_factor = 1.0
    # Let Tex be 100 (but change it and check)
    Tex = 60 * u.K

    hnukBTex = hnu_kB/Tex
    Z = g0 + g1*np.exp(-hnukBTex) # hnu_kB = Eu/kB since ground is 0 energy (might also be ok if not, but it's definitely ok in this case)
    exp_hnukBTex = np.exp(hnukBTex)
    channel_tau = -1*np.log(1 - ((t_arr / (filling_factor * hnu_kB)) * (exp_hnukBTex - 1))) # 3d cube
    column_constants = (8*np.pi * (rest_freq / const.c)**2) / (g1*A10)
    channel_column = (
        column_constants * channel_tau * Z * (exp_hnukBTex / (1 - np.exp(-hnukBTex)))
    ).decompose()
    integrated_column = np.trapz(channel_column[::-1], x=freq_axis[::-1], axis=0).to(u.cm**-2)
    integrated_H_column = integrated_column / Cp_H_ratio

    print(integrated_H_column)



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
        # 2023-04-26, 06-07
        savename = f"/home/ramsey/Pictures/2023-04-26/m16_pv_{orientation}_{slice_idx:03d}.png"
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title=f'{line_stub}, using stub/file {filename}', file=__file__, func='manual_pv_slice_series'))

def pv_slice_series_overlay():
    """
    June 7, 2023
    Use the pvdiagrams.py framework to make PV diagram overlays into a movie.
    It would be cool to do this with the manual_pv_slice_series() code but
    an overlay will involve different pixel grids, so that would be too difficult.
    """
    # Load PV info
    # path_filename_short = "catalogs/m16_pv_vectors_2.reg"; n_steps = 60
    path_filename_short = "catalogs/m16_pv_vectors_3.reg"; n_steps = 85
    path_info = pvdiagrams.linear_series_from_ds9(catalog.utils.search_for_file(path_filename_short), n_steps=n_steps)
    path_stub = os.path.split(path_filename_short)[-1].replace('.reg', '')
    pv_vel_lims = (8*kms, 35*kms)
    pv_vel_intervals = np.arange(16, 33, 2)

    # Load cubes
    img_stub = 'ciiAPEX'
    img_cube_obj = cube_utils.CubeData(get_map_filename(img_stub)).convert_to_kms()

    contour_stub = '12co32'
    contour_cube_obj = cube_utils.CubeData(get_map_filename(contour_stub)).convert_to_kms()

    # Reference image
    ref_vel_lims = (10*kms, 35*kms)
    ref_mom0 = img_cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
    ref_img = ref_mom0.to_value()
    ref_contour_mom0 = contour_cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
    ref_contour = reproject_interp((ref_contour_mom0.to_value(), ref_contour_mom0.wcs), ref_mom0.wcs, ref_mom0.shape, return_footprint=False)

    # Colors
    ref_img_cmap = 'Greys_r'
    ref_contour_cmap = 'magma_r'
    pv_img_cmap = 'plasma'
    pv_img_contours_color = 'k'
    pv_contour_cmap = 'cool'
    reg_color = 'LimeGreen'

    """
    go thru and look at run_plot_and_save_series and plot_path in pvdiagrams.py
    will need to iterate somewhat manually using cues from these two functions
    """

    # Colorscale limits
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


    img_cube = img_cube_obj.data.spectral_slab(*pv_vel_lims)
    contour_cube = contour_cube_obj.data.spectral_slab(*pv_vel_lims)

    # path_info is: center_coord, length_scale, path_generator
    path_generator = path_info[2]
    for i, p in enumerate(path_generator):

        # if i%3 != 0 and i < 44:
        # if i != 14:

        if os.path.isfile(f"/home/ramsey/Pictures/2023-06-13/m16_pv_{path_stub}_{i:03d}.png"):
            continue

        sl_img = pvextractor.extract_pv_slice(img_cube, p)
        sl_contour_raw = pvextractor.extract_pv_slice(contour_cube, p)
        sl_contour_raw.header['RESTFRQ'] = sl_img.header['RESTFRQ']
        sl_wcs = WCS(sl_img.header)
        sl_contour = reproject_interp((sl_contour_raw.data, sl_contour_raw.header), sl_wcs, shape_out=sl_img.data.shape, return_footprint=False)

        fig = plt.figure(figsize=(10, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # Reference image
        ax_ref = fig.add_subplot(gs[0,0], projection=ref_mom0.wcs)
        cbar_ax = ax_ref.inset_axes([1, 0, 0.05, 1])
        cbar_ax2 = ax_ref.inset_axes([0, 1, 1, 0.05])

        im = ax_ref.imshow(ref_img, origin='lower', cmap=ref_img_cmap, vmin=0)
        cbar = fig.colorbar(im, cax=cbar_ax, label=f"{get_data_name(img_stub)} ({ref_mom0.unit.to_string('latex_inline')})")
        ax_ref.text(0.05, 0.93, make_vel_stub(ref_vel_lims), color='k', ha='left', va='bottom', transform=ax_ref.transAxes)

        cs = ax_ref.contour(ref_contour, cmap=ref_contour_cmap, linewidths=0.5, alpha=0.6)
        cbar = fig.colorbar(cs, cax=cbar_ax2, location='top', spacing='proportional', label=f"{get_data_name(contour_stub)} ({ref_contour_mom0.unit.to_string('latex_inline')})")

        ax_ref.plot([c.ra.deg for c in p._coords], [c.dec.deg for c in p._coords], color=reg_color, linestyle='-', lw=1, transform=ax_ref.get_transform('world'))
        ax_ref.text(p._coords[0].ra.deg, p._coords[0].dec.deg + 4*u.arcsec.to(u.deg), 'Offset = 0\"', color=reg_color, fontsize=10, va='center', ha='right', transform=ax_ref.get_transform('world'))

        # Plot the footprint of the overlay if it would be visible at all
        overlay_nan_map = np.isnan(ref_contour)
        if np.any(overlay_nan_map):
            ax_ref.contour(overlay_nan_map.astype(float), levels=[0.5], colors='SlateGray', linestyles=':', linewidths=1)
        del overlay_nan_map

        # Beams
        beam_patch_kwargs = dict(alpha=0.9, hatch='////')
        beam_x, beam_y = 0.93, 0.1
        beam_ecs = [['white', 'grey'], [cs.cmap(cs.norm(cs.levels[j])) for j in [0, 2]]]
        for j, cube in enumerate((img_cube, contour_cube)):
            # Beam is known, plot it
            patch = cube.beam.ellipse_to_plot(*(ax_ref.transAxes + ax_ref.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(ref_mom0.wcs))
            patch.set(**beam_patch_kwargs, facecolor=beam_ecs[j][0], edgecolor=beam_ecs[j][1])
            ax_ref.add_artist(patch)
            beam_x -= 0.03


        # PV diagram
        ax_pv = fig.add_subplot(gs[1,0], projection=sl_wcs)
        cbar_ax = ax_pv.inset_axes([1, 0, 0.05, 1])
        # Image
        im = ax_pv.imshow(sl_img.data, origin='lower', cmap=pv_img_cmap, vmin=0, vmax=pv_vmaxes.get(img_stub, None), aspect=(sl_img.data.shape[1]/(2.5*sl_img.data.shape[0])))
        cbar = fig.colorbar(im, cax=cbar_ax, label=img_cube.unit.to_string('latex_inline'))
        # Contours
        cs = ax_pv.contour(sl_img.data, colors=pv_img_contours_color, linewidths=1, linestyles=':', levels=_get_levels(img_stub))
        for l in cs.levels:
            cbar.ax.axhline(l, color=pv_img_contours_color)
        cs = ax_pv.contour(sl_contour, cmap=pv_contour_cmap, linewidths=1.5, levels=_get_levels(contour_stub), vmax=pv_vmaxes.get(contour_stub, None))
        for l in cs.levels:
            cbar.ax.axhline(l, color=cs.cmap(cs.norm(l)))

        # Plot horizontal gridlines
        xlim = ax_pv.get_xlim() # save existing xlim to reintroduce them later
        x_length = p._coords[0].separation(p._coords[1]).deg
        for v in pv_vel_intervals: # these mess up the xlim
            ax_pv.plot([0, x_length], [v*1e3]*2, color='grey', alpha=0.7, linestyle='--', transform=ax_pv.get_transform('world'))
        # Label observation names
        ax_pv.text(0.05, 0.95, "Image: " + cube_utils.cubenames[img_stub], fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)
        ax_pv.text(0.05, 0.90, "Contour: " + cube_utils.cubenames[contour_stub], fontsize=13, color='w', va='top', ha='left', transform=ax_pv.transAxes)
        # Put xlim back in
        ax_pv.set_xlim(xlim)


        ax_pv.coords[1].set_format_unit(u.km/u.s)
        ax_pv.coords[1].set_major_formatter('x.xx')
        ax_pv.coords[0].set_format_unit(u.arcsec)
        ax_pv.coords[0].set_major_formatter('x.xx')

        plt.tight_layout()

        # 2023-06-12,13
        savename = f"/home/ramsey/Pictures/2023-06-13/m16_pv_{path_stub}_{i:03d}.png"
        fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='pv movie',
            file=__file__, func='pv_slice_series_overlay'))

        plt.close(fig)




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

def convolve_32_to_10_pmo():
    """
    October 19, 2023
    Quick convolve CO(3-2) to PMO CO(1-0) (Like 55 arcsec or something)
    12 and 13 CO 1-0 by PMO reportedly have the exact same beam (unlike 12 and 13 CO 3-2)
    """
    reference_cube_fn = get_map_filename('12co10-pmo')
    reference_cube = cube_utils.CubeData(reference_cube_fn).convert_to_K().data
    reference_beam = reference_cube.beam
    del reference_cube # save memory
    print(f"Convolving to PMO CO 1-0 beam {reference_beam.major.to(u.arcsec):.2f} X {reference_beam.minor.to(u.arcsec):.2f}")
    print("Confirmed that 13co10-pmo beam is the same.")

    target_cube_fn = get_map_filename('13co32') # 12 and 13 co 32
    target_cube = cube_utils.CubeData(target_cube_fn).convert_to_K()
    savename = target_cube.full_path.replace('.fits', '.PMObeam.fits')
    target_cube = target_cube.data
    old_beam = target_cube.beam
    print(f"original beam {old_beam.major.to(u.arcsec):.2f} X {old_beam.minor.to(u.arcsec):.2f}")
    print("writing to ", savename)

    print("convolving...")
    convolved_cube = target_cube.convolve_to(reference_beam)
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

def overlay_moment(background='8um', overlay='cii', velocity_limits=None, velocity_limits2=None, data_memo=None, data_memo_rules='background', cutout_reg_stub='N19', reg_filename_or_idx=None, plot_stars=False):
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
        #### TODO: this does not work in Galactic, need to load them in as SkyCoords (or convert) and then plot based on image frame
        del star_df

    # Load in background
    img, img_info = get_2d_map(background_stub, velocity_limits=velocity_limits, average_Tmb=True, data_memo=data_memo)
    # Cutout background (and update WCS)
    # Set align_with_frame="icrs" so that all data is aligned with RA-Dec, including originally galactic data
    if cutout_reg_stub is not None:
        img_info['cutout'] = misc_utils.cutout2d_from_region(img, img_info['wcs'], get_cutout_box_filename(cutout_reg_stub), align_with_frame='icrs')
        img = img_info['cutout'].data
        img_info['wcs'] = img_info['cutout'].wcs
    # Memoize it if it's not already there
    if dmr%2 == 0 and img_info['vlim_hash'] not in data_memo:
        data_memo[img_info['vlim_hash']] = (img, img_info)

    # Load in overlay
    if velocity_limits2 is None:
        velocity_limits2 = velocity_limits
    overlay, overlay_info = get_2d_map(overlay_stub, velocity_limits=velocity_limits2, average_Tmb=True, data_memo=data_memo)
    # Regrid overlay to background
    try:
        overlay_regrid = reproject_interp((overlay, overlay_info['wcs']), img_info['wcs'], shape_out=img.shape, return_footprint=False)
    except Exception as e:
        print(e)
        print(overlay.shape)
        print(overlay_info['wcs'])
        print(img.shape)
        print(img_info['wcs'])
        raise RuntimeError
    # From here on out, don't use the overlay wcs, use the img wcs
    # Memoize it if it's not already there
    if dmr > 0 and overlay_info['vlim_hash'] not in data_memo:
        data_memo[overlay_info['vlim_hash']] = (overlay_regrid, overlay_info)

    # print("img hash", img_info['vlim_hash'])
    # print("overlay hash", overlay_info['vlim_hash'])

    # Figure
    figsizes = {'n19': (10, 10), 'med-large': (13, 9), 'med': (10, 9)}
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
        # Generic vlim is also used for contour colors
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
    velocity_stub = " "+make_vel_stub(velocity_limits2) if (overlay_info['obs_type'] == 'cube' and velocity_limits2) else ""
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
        # print("Plotting stars?")
        # print(star_ra)
        ax.plot(star_ra, star_dec, mec=marcs_colors[3], mfc='k', marker='o', linestyle='none', transform=ax.get_transform('world'))
    ax.set_xlim(ref_xlim)
    ax.set_ylim(ref_ylim)



    velocity_stub = "" if not velocity_limits else "_"+make_simple_vel_stub(velocity_limits)
    if velocity_limits2 != velocity_limits:
        velocity_stub += "_"+make_simple_vel_stub(velocity_limits2)
    if overlay_info['obs_type'] != 'cube' and img_info['obs_type'] != 'cube':
        # Override velocity stub with empty string if neither image is a cube
        velocity_stub = ""
    cutout_stub = "" if cutout_reg_stub is None else f"cutout {cutout_reg_stub} from {os.path.basename(get_cutout_box_filename(cutout_reg_stub))}"
    if reg_fn_stub:
        reg_fn_stub = "_"+reg_fn_stub
    if plot_stars:
        reg_fn_stub += '_with-stars'
    # 2023-05-03,04,05,09,10,11,26,30, 06-02,07,14
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"overlay_{overlay_stub}_on_{background_stub}{velocity_stub}{reg_fn_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title=cutout_stub, file=__file__, func="overlay_moment"))
    # Some cleanup since things seem to pile up
    plt.close(fig)

def overlay_two_moments(background='8um', overlays=('cii', '12co32'), velocity_limits=None, cutout_reg_stub='N19', reg_filename_or_idx=None, plot_stars=False, **kwargs):
    """
    October 24, 2023
    Mostly same as overlay_moment() but with two overlays.
    The overlays will be in single colors, not in colormaps. Like my CS/CII/3um image from the pillar paper
    I'll keep it simple at first so I can churn out an image, but can dress it up later.

    :param background: data stub. If it is not 2D photometry, it must be one of
        the overlays
    :param overlays: 2-tuple of data stubs, cubes
    :param velocity_limits: either a single tuple of (lo, hi) limits (velocity Quantity)
        or two nested tuples as above; if two nested tuples, then the first applies
        to the first overlay and so on. Nested tuples like ((lo, hi), (lo, hi))
    :param cutout_reg_stub: same as usual, will cutout the background and reproject
        the moment overlays to that.
    :param kwargs: other kwargs:
        'levels': a dictionary whose keys are the stubs in overlays and values are
            arrays of contour levels. OK if not all keys are in this dict or if it contains other keys.
        'vlims': a dictionary whose keys are background image stubs and values are
            tuples (vmin, vmax). Either or both vmin, vmax can be None.
    disabling the other inputs until later if I need them. I'll just copy them from overlay_moment
    """
    # Redefine the conveniently-named input keywords
    background_stub = background
    overlay_stubs = list(overlays) # must be mutable; edited during saving to add velocity stubs
    del background, overlays # they will be reused, but I want to be clear here
    # Parse velocity limits
    if velocity_limits is None:
        vel_lims_1 = vel_lims_2 = None
    else:
        assert isinstance(velocity_limits, tuple)
        if isinstance(velocity_limits[0], tuple):
            # two different limits
            vel_lims_1, vel_lims_2 = velocity_limits
        else:
            # same limits
            # make sure it's really limits
            assert velocity_limits[0].to(kms) # will throw error if problem
            vel_lims_1 = vel_lims_2 = velocity_limits
    # Reuse argument name, now it will definitely be tuple of two limits
    velocity_limits = (vel_lims_1, vel_lims_2)

    # Some helper functions
    def _load_overlay(i):
        if overlay_stubs[i] is None:
            # Short circuit if stub is None
            return None, None
        # Load
        overlay, overlay_info = get_2d_map(overlay_stubs[i], velocity_limits=velocity_limits[i], average_Tmb=True)
        # Reproject
        try:
            overlay_regrid = reproject_interp((overlay, overlay_info['wcs']), img_info['wcs'], shape_out=img.shape, return_footprint=False)
        except Exception as e:
            print(e)
            raise RuntimeError("exiting manually, go add in the other debug lines")
        del overlay
        return overlay_regrid, overlay_info

    def _get_levels(i):
        """
        Get levels from the kwargs, or something else if not specified
        """
        line_stub = overlay_stubs[i]
        kwarg_levels = kwargs.get("levels", {}).get(line_stub, None)
        if kwarg_levels is not None:
            return kwarg_levels
        else:
            return None #get_levels(line_stub)

    def _get_vlims():
        """
        Get image vlims from kwargs, or something else if not specified
        """
        kwarg_vlims = kwargs.get("vlims", {}).get(background_stub, None)
        if kwarg_vlims is not None:
            vlims = kwarg_vlims
        else:
            vlims = (None, None)

        vlims_dict = {}
        vlims_keys = ['vmin', 'vmax']
        for vi, v in enumerate(vlims):
            if v is not None:
                vlims_dict[vlims_keys[vi]] = v
        return vlims_dict


    # Load in background
    if background_stub in overlay_stubs:
        # Background is one of the overlays; find out which
        background_idx = overlay_stubs.index(background_stub)
        # Load that one
        print(velocity_limits[background_idx])
        img, img_info = get_2d_map(overlay_stubs[background_idx], velocity_limits=velocity_limits[background_idx], average_Tmb=True)
    else:
        # 2D photometry background
        img, img_info = get_2d_map(background_stub, average_Tmb=True)
        background_idx = None
    # Cutout background (and update WCS)
    # Set align_with_frame="icrs" so that all data is aligned with RA-Dec, including originally galactic data
    if cutout_reg_stub is not None:
        img_info['cutout'] = misc_utils.cutout2d_from_region(img, img_info['wcs'], get_cutout_box_filename(cutout_reg_stub), align_with_frame='icrs')
        img = img_info['cutout'].data
        img_info['wcs'] = img_info['cutout'].wcs

    # Figure
    figsizes = {'n19': (10, 10), 'med-large': (13, 9), 'med': (10, 9)}
    fig = plt.figure(figsize=figsizes.get(cutout_reg_stub, (10, 10)))
    ax = fig.add_subplot(111, projection=img_info['wcs'])
    # Plot image
    # Use generic vlims as a backup for specific vlims
    img_vlim = _get_vlims()
    if not _get_vlims():
        img_vlim.update(get_generic_vlim(background_stub))
        img_vlim.update(get_vlim(img_info['vlim_hash']))
    im = ax.imshow(img, origin='lower', **img_vlim, cmap='cividis')
    # Plot contours
    contour_colors = ('k', 'w')
    contour_lws = (2, 1)
    overlay_info_list = []
    for i in range(2):
        if background_idx is not None and background_idx==i:
            # background is also overlay
            overlay, overlay_info = img, img_info
            # Don't add overlay_info to list
        else:
            overlay, overlay_info = _load_overlay(i)
            if overlay is None:
                continue
            overlay_info_list.append(overlay_info)
        cs = ax.contour(overlay, colors=contour_colors[i], linewidths=contour_lws[i], linestyles='-', levels=_get_levels(i))
        print(overlay_stubs[i])
        print(np.nanmin(overlay), np.nanmax(overlay))
        print(cs.levels)

    # Image colorbar (right)
    cbar_ax = ax.inset_axes([1, 0, 0.05, 1])
    cbar = fig.colorbar(im, cax=cbar_ax, label=f"{get_data_name(background_stub)} ({img_info['unit'].to_string('latex_inline')})")
    # Beams
    beam_patch_kwargs = dict(alpha=0.9, hatch='////')
    beam_x, beam_y = 0.93, 0.1
    beam_ecs = [['white', 'grey'], ['grey', 'k'], ['grey', 'white']]
    for i, data_info_dict in enumerate([img_info] + overlay_info_list):
        if 'beam' not in data_info_dict:
            continue
        # Beam is known, plot it
        patch = data_info_dict['beam'].ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([beam_x, beam_y]), misc_utils.get_pixel_scale(img_info['wcs']))
        patch.set(**beam_patch_kwargs, facecolor=beam_ecs[i][0], edgecolor=beam_ecs[i][1])
        ax.add_artist(patch)
        beam_x -= 0.03

    # Save
    overlay_stub_final = ""
    for i in range(2):
        if overlay_stubs[i] is not None:
            if velocity_limits[i] is not None:
                overlay_stubs[i] += "-" + make_simple_vel_stub(velocity_limits[i])
            overlay_stub_final += overlay_stubs[i]
    cutout_stub = "" if cutout_reg_stub is None else f"cutout {cutout_reg_stub} from {os.path.basename(get_cutout_box_filename(cutout_reg_stub))}"
    savename = f"overlay2_{background_stub}_{overlay_stub_final}.png"
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), savename),
        metadata=catalog.utils.create_png_metadata(title=cutout_stub, file=__file__, func="overlay_two_moments"))
    plt.close(fig)


"""
Save moment 0
"""
def save_moment0(line_stub='cii', velocity_limits=None, cutout_reg_stub='N19'):
    """
    October 26, 2023
    Create a moment 0 within the velocity limits, optionally cutout, and save
    to FITS
    """
    # Let it be real moment 0 units
    img, img_info = get_2d_map(line_stub, velocity_limits=velocity_limits, average_Tmb=False)
    if cutout_reg_stub is not None:
        img_info['cutout'] = misc_utils.cutout2d_from_region(img, img_info['wcs'], get_cutout_box_filename(cutout_reg_stub), align_with_frame='icrs')
        img = img_info['cutout'].data
        img_info['wcs'] = img_info['cutout'].wcs

    hdr = img_info['wcs'].to_header()
    hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
    hdr['CREATED'] = f"Ramsey, {__file__}"
    hdr['BUNIT'] = str(img_info['unit'])
    hdr.update(img_info['beam'].to_header_keywords())
    hdr['HISTORY'] = f"From line {line_stub}"
    hdr['HISTORY'] = img_info['original_cube_basename']
    if cutout_reg_stub is not None:
        hdr['HISTORY'] = f"Cutout {get_cutout_box_filename(cutout_reg_stub)}"
    if velocity_limits is not None:
        hdr['HISTORY'] = f"Integrated within {make_simple_vel_stub(velocity_limits)}"
    phdu = fits.PrimaryHDU(data=img, header=hdr)
    vel_stub = "" if velocity_limits is None else "_"+make_simple_vel_stub(velocity_limits)
    cutout_stub = "" if cutout_reg_stub is None else "_"+cutout_reg_stub
    savename = f"{line_stub}_mom0{vel_stub}{cutout_stub}.fits"
    savename = os.path.join(img_info['original_cube_directory'], savename)
    print("Writing to ", savename)
    phdu.writeto(savename)


def fast_pv(reg_filename_or_idx=0, line_stub_list=['12co32', 'ciiAPEX'], velocity_limits=None, **kwargs):
    """
    May 5, 2023
    Re-create the PV diagrams from real_medium_spectra but without the other
    stuff on the plots.
    Formerly real_medium_pv, but planning to use it live so gave it a better name.

    :param velocity_limits: tuple (Quantity, Quantity) in velocity units.
        Limits of PV diagram. Defaults to (8 km/s, 35 km/s) if None.
    :param vmax: (optional) dictionary with keys matching elements of line_stub_list
        Dictionary values should be vmax value for the PV diagram of that line.
        If that line is contoured in color, that will be vmax for contour colorscale.
    :param levels: (optional) dictionary with keys matching elements of line_stub_list
        Dictionary values should be a list-like of float values giving levels for
        PV diagram of that line.
    :param velocity_intervals: (optional) iterable of float values (km/s implied)
        where horizontal lines should be drawn across the PV diagram for reference.
        If not given or None (default), lines will be drawn every 2 km/s between 20-34 km/s.
        If set to False or an empty iterable, no lines will be drawn.
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


    # pv_vmaxes = {'ciiAPEX': 20, '12co32': 30, '13co32': 15}
    # pv_levels = {'ciiAPEX': (3, 37, 4), '12co32': (5, 41, 5), '13co32': (1, 27, 2.5)}
    def _get_levels(line_stub):
        """
        OLD <Get levels from the above dictionary. Return None if not present.>
        NEW Get levels from the vlim memo functions
            future: get levels from vlim memo only if not specified in function call
        """
        kwarg_levels = kwargs.get("levels", {}).get(line_stub, None)
        if kwarg_levels is not None:
            return kwarg_levels
        else:
            return get_levels(line_stub)
        # if line_stub in pv_levels:
        #     return np.arange(*pv_levels[line_stub])
        # else:
        #     return None
    def _get_pv_vmax(line_stub):
        """
        Get vmax from vlim memo functions
            future: get from vlim memo only if not specified in function call
        """
        kwarg_vmax = kwargs.get("vmax", {}).get(line_stub, None)
        if kwarg_vmax is not None:
            return kwarg_vmax
        else:
            return get_generic_vlim(line_stub).get("vmax", None)

    # PV limits
    if velocity_limits is None:
        velocity_limits = (8*kms, 35*kms)

    # Horizontal lines on PV diagram
    velocity_intervals = kwargs.get("velocity_intervals", None)
    if velocity_intervals is None:
        velocity_intervals = np.arange(20, 35, 2)
    elif velocity_intervals is False:
        velocity_intervals = []

    # Most of these files have 3 points and a vector
    # In theory, it can have any number of points which will be handled correctly
    # If it doesn't have points, ignore the points and use the vector.
    pv_path, point_reg_list, reg_fn_stub = get_pv_and_regions(reg_filename_or_idx)

    fig = plt.figure(figsize=kwargs.get('figsize', (16, 9)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 5], hspace=0, wspace=0)
    ax_ref_img = fig.add_subplot(gs[0,0], projection=ref_wcs)

    # Plot reference image
    ax_ref_img.imshow(ref_img, origin='lower', vmin=45, vmax=290, cmap='Greys_r')

    pv_slices = []
    for line_stub in line_stub_list:
        if False and 'cii' in line_stub:
            print("INTERCEPTING CII TO USE OLD SMALLER MAP FOR TESTING")
            cube = cube_utils.CubeData(line_stub).convert_to_K()
        else:
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
    aspect_ratio = (sl0.data.shape[1]/(1.*sl0.data.shape[0]))
    print("default aspect", aspect_ratio)
    if kwargs.get('aspect', None) is not None:
        aspect_ratio = kwargs.get('aspect')
    im = ax_pv.imshow(sl0.data, origin='lower', cmap='plasma', vmin=0, vmax=_get_pv_vmax(ls0), aspect=aspect_ratio)
    cs = ax_pv.contour(sl0.data, colors='k', linewidths=1, linestyles=':', levels=_get_levels(ls0))
    pv_cbar = fig.colorbar(im, ax=ax_pv, location='right', label=cube.data.unit.to_string('latex_inline'))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='k')
    ax_pv.text(0.05, 0.95, "Image: "+get_data_name(ls0), fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)
    # Overlay [1]
    cs = ax_pv.contour(sl1_reproj, cmap='cool', linewidths=1.5, levels=_get_levels(ls1), vmax=_get_pv_vmax(ls1))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='w', linewidth=2)
    ax_pv.text(0.05, 0.90, "Contour: "+get_data_name(ls1), fontsize=13, color='w', va='top', ha='left', transform=ax_pv.transAxes)

    # Plot velocity intervals
    x_length = pv_path._coords[0].separation(pv_path._coords[1]).deg
    # Save xlims to reapply later, since the horizontal lines will stretch the limits
    pv_xlim = ax_pv.get_xlim()
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
    ax_pv.set_xlim(pv_xlim)
    # Fix PV ylim so that low velocity is always on the bottom
    """
    TODO: debug and finish this! I couldn't figure it out in time, need to meet with Marc soonish. Try combining with transData or transAxes to see if anything useful happens... or figure coords?
    """
    if False and False and False:
        pv_ylim = ax_pv.get_ylim() # could theoretically do all this in one step safely, but for clarity's sake, splitting it into two lines
        print(pv_ylim)
        d2a = u.arcsec.to(u.deg)
        print("d2a", d2a)
        print(ax_pv.get_transform('world').transform([-1*d2a, -1]))
        print(ax_pv.get_transform('world').transform([0*d2a, 0]))
        print(ax_pv.get_transform('world').transform([1*d2a, 1]))
        print(sl0.data.shape)
        # print([ax_pv.get_transform('world').inverted().transform([0, x]) for x in pv_ylim])
        print(sorted(pv_ylim))
        return
        ax_pv.set_ylim(sorted(pv_ylim)) # sorted() returns a new list, sorted
    elif kwargs.get("invert", False):
        print("INVERTING!"+"!"*20)
        ax_pv.invert_yaxis()

    plt.tight_layout()
    # 2023-05-06,11, 06-30
    savename = os.path.join(catalog.utils.todays_image_folder(), f"pv_{ls1}_on_{ls0}_along_{reg_fn_stub}.png")
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='pv',
        file=__file__, func="fast_pv"))

def channel_movie(data_stub, vel_lims=(0, 45)):
    """
    May 31, 2023
    Quick! Channel movie
    """
    fn = get_map_filename(data_stub)
    cube_obj = cube_utils.CubeData(fn)

    savename_f = lambda i : f"/home/ramsey/Pictures/2023-05-31/movie/{data_stub}_{i:03d}.png"

    ilo, ihi = [cube_obj.data.closest_spectral_channel(v*kms) for v in vel_lims]
    for i in range(ilo, ihi+1):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection=cube_obj.wcs_flat)
        ax.imshow(cube_obj.data[i].to_value(), origin='lower', **get_generic_vlim(data_stub), cmap='plasma')
        ax.set_title(f"{cube_obj.data.spectral_axis[i].to(kms):.2f}")
        fig.savefig(savename_f(i), metadata=catalog.utils.create_png_metadata(title='for movie', file=__file__, func="channel_movie"))
        print(f"{ilo}->{i}->{ihi}")
        plt.close(fig)

def plot_spectra(reg_set_number=1, line_set_number=1, velocities_to_mark=None):
    """
    June 6, 2023
    Compare some spectra
    At first, don't worry about spatial resolution
    Follow m16_pictures.paper_spectra
    """
    lines = ['cii', '12co10-pmo']

    if reg_set_number == 1:
        reg_filename_short = "catalogs/N19_points.reg"
    elif reg_set_number == 2:
        reg_filename_short = "catalogs/N19_points_2.reg"
    elif reg_set_number == 3:
        reg_filename_short = "catalogs/m16_northridge_points.reg"
    elif reg_set_number == 4:
        reg_filename_short = "catalogs/N19_shell_edge.reg"
    elif reg_set_number == 5:
        reg_filename_short = "catalogs/m16_points_blueshifted_clump.reg"
    elif reg_set_number == 6:
        reg_filename_short = "catalogs/N19_points_all_across.reg"
    elif reg_set_number == 7:
        reg_filename_short = "catalogs/N19_points_all_across_2.reg"
    elif reg_set_number == 8:
        reg_filename_short = "catalogs/m16_spectrum_samples.reg"
    else:
        raise NotImplementedError(f"reg_set_number =/= {reg_set_number}")

    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    multiplier = {'cii': 1,
        '12co10': 0.5, '13co10': 1, 'c18o10': 15,
        '12co32': 1, '13co32': 1,
    }

    def get_multiplier(stub):
        if '-' in stub:
            stub = stub.split('-')[0]
        return multiplier[stub]

    """ Make 2 sets of spectra """
    if line_set_number == 1:
        short_names = ['cii', '12co10-pmo', '12co32',]; set_stub = "12"
    elif line_set_number == 2:
        short_names = ['cii', '13co10-pmo', '13co32',]; set_stub = "13"
    elif line_set_number == 3:
        # specifically for blueshifted clump, reg_set_number 5
        short_names = ['cii']; set_stub = ""
    else:
        raise NotImplementedError(f"line_set_number =/= {line_set_number}")

    # Create Figure and Axes

    if reg_set_number == 1:
        figsize = (15, 5)
        grid_shape = (1, 3)
    elif reg_set_number in [2, 3]:
        figsize = (15, 10)
        grid_shape = (3, 3)
    elif reg_set_number in [4]:
        figsize = (16, 9)
        grid_shape = (2, 3)
    elif reg_set_number == 5:
        figsize = (10, 10)
        grid_shape = (2, 2)
    elif reg_set_number in (6, 7):
        figsize = (10, 20)
        grid_shape = (len(reg_list), 1)
    elif reg_set_number == 8:
        figsize = (13, 8)
        grid_shape = (4, 2)
    else:
        raise NotImplementedError

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(*grid_shape, hspace=0.05, wspace=0.05, left=0.05, right=0.95, bottom=0.05, top=0.95)
    axes = [] # hold the axes in same order as reg_list
    # Iterate thru reg_list and make an Axes for each
    for reg_idx in range(len(reg_list)):
        ax = fig.add_subplot(gs[np.unravel_index(reg_idx, grid_shape)])
        axes.append(ax)

    # Iterate thru cubes, and then thru regs inside of those (load 1 cube at a time, regs are much quicker already loaded)
    for line_idx, line_stub in enumerate(short_names):
        fn = get_map_filename(line_stub)
        cube = cube_utils.CubeData(fn)
        # cube.convert_to_K()
        cube.data = cube.data.with_spectral_unit(kms)
        for reg_idx, reg in enumerate(reg_list):
            pixreg = reg.to_pixel(cube.wcs_flat)
            j, i = [int(round(c)) for c in pixreg.center.xy]
            try:
                spectrum = cube.data[:, i, j]
            except IndexError:
                spectrum = np.full(cube.data.shape[0], np.nan) * cube.data.unit
            multiplier_stub = '' if get_multiplier(line_stub) == 1 else f' $\\times${get_multiplier(line_stub)}'
            line_name = get_data_name(line_stub)
            p = axes[reg_idx].plot(cube.data.spectral_axis.to_value(), spectrum.to_value()*get_multiplier(line_stub), label=f"{line_name}{multiplier_stub}", alpha=0.75)

    # Iterate thru regs/Axes again and dress up the plots
    for reg_idx in range(len(reg_list)):
        ax = axes[reg_idx]
        ss = ax.get_subplotspec()
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(direction='in', which='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        if not ss.is_last_row():
            # hide ticklabels except bottom row
            ax.tick_params(axis='x', labelbottom=False)
        if not ss.is_first_col():
            # hide ticklabels except left column
            ax.tick_params(axis='y', labelleft=False)

        if reg_idx == len(reg_list)-1:
            ax.legend(loc='upper right', fontsize=13)
        if reg_set_number in range(5):
            ax.set_xlim((5, 41))
        elif reg_set_number == 5:
            ax.set_xlim((-5, 46))
        elif reg_set_number in (6, 7):
            ax.set_xlim((0, 40))

        if line_set_number == 1:
            ax.set_ylim((-5, 27)) # or 46
        elif line_set_number == 2:
            ax.set_ylim((-3, 13)) # or 23
        elif reg_set_number == 5:
            ax.set_ylim((-3, 17))

        try:
            txt = reg_list[reg_idx].meta['text']
        except:
            txt = f"REG {reg_idx}"
        ax.text(0.06, 0.94, txt, transform=ax.transAxes, fontsize=15, color='k', ha='left', va='center')
        for v in range(10, 31, 2):
            # Some light velocity gridlines around the important velocities
            ax.axvline(v, color='k', alpha=0.07)
        if velocities_to_mark is not None:
            for v in velocities_to_mark:
                ax.axvline(v, color='k', alpha=0.4)
        ax.axhline(0, color='k', alpha=0.1)
        # Use this line to verify background subtractions done correctly
        # print(f"{reg_list[reg_idx].meta['text']} {check_if_region_is_southern(reg_list[reg_idx].meta['text'])}")

    fig.supxlabel(f"LSR Velocity ({kms.to_string('latex_inline')})")
    fig.supylabel(f"Line intensity ({u.K.to_string('latex_inline')})")

    reg_fn_stub = str(reg_set_number) #os.path.basename(reg_filename_short).replace('.reg', '')
    # 2023-06-06,07
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"sample_spectra_{reg_fn_stub}_{set_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title=f"points: {reg_filename_short}",
        file=__file__, func="plot_spectra"))

"""
Various stuff about comparing 8 micron to CII.
The compare_8micron_and_cii_intensities is pretty old (June 2023) and a little
misdirected I think. The correlation_plot_8um_cii is a test from January 2024
to make the Cornelia-style correlation plots, since Maitraiyee and Lars are
making these too.
"""

def compare_8micron_and_cii_intensities(velocity_limits, reg_index=0):
    """
    June 27, 2023
    At a certain point (set by point region), compare the 8 micron intensity
    to the CII integrated intensity using Cornelia's conversion.
    Show a spectrum of CII along with the comparison to help with integration
    limits, and a moment image.
    """
    reg_set_number = 4
    if reg_set_number == 1:
        reg_filename_short = "catalogs/N19_points.reg"
    elif reg_set_number == 2:
        reg_filename_short = "catalogs/N19_points_2.reg"
    elif reg_set_number == 3:
        reg_filename_short = "catalogs/m16_northridge_points.reg"
    elif reg_set_number == 4:
        reg_filename_short = "catalogs/N19_shell_edge.reg"
    else:
        raise NotImplementedError(f"reg_set_number =/= {reg_set_number}")

    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    reg = reg_list[reg_index]

    # CII
    line_stub = 'cii'
    cube_fn = get_map_filename(line_stub)
    cube_obj = cube_utils.CubeData(cube_fn).convert_to_kms()
    wcs_flat = cube_obj.wcs_flat

    # 8 micron
    fn_8um = catalog.utils.search_for_file("spitzer/SPITZER_I4_mosaic_ALIGNED.fits")
    img_8, hdr_8 = fits.getdata(fn_8um, header=True)
    wcs_8 = WCS(hdr_8)

    # Convolve 8um to cii
    convolve_8 = False
    bmaj, bmin, pa = photometry_beams['8um']
    beam_8 = cube_utils.Beam(bmaj*u.arcsec, bmin*u.arcsec, pa*u.deg)
    beam_cii = cube_obj.data.beam
    beam_conv = beam_cii.deconvolve(beam_8)
    pixel_scale = misc_utils.get_pixel_scale(wcs_8)
    if convolve_8:
        kernel = beam_conv.as_kernel(pixel_scale)
        img_8 = convolve_fft(img_8, kernel)
        img_8 = reproject_interp((img_8, wcs_8), wcs_flat, shape_out=cube_obj.data.shape[1:], return_footprint=False)
        wcs_8 = wcs_flat
        # kernel_fwhm = np.sqrt(beam_cii.major**2 - beam_8.major**2).to(u.arcsec)
        # kernel_width = (2 * kernel_fwhm / (pixel_scale * 2.355)).decompose() # -1 to +1 sigma in pixels
        # sample_region_width = math.ceil(kernel_width*3)
        # print(kernel_width, sample_region_width)
        # img_8 = reproject_adaptive((img_8, wcs_8), wcs_flat, shape_out=cube_obj.data.shape[1:], return_footprint=False, kernel_width=kernel_width, sample_region_width=sample_region_width)
        # wcs_8 = wcs_flat

    figsize = (13, 10)
    fig = plt.figure(figsize=figsize)
    grid_shape = (2, 3)
    gs = fig.add_gridspec(*grid_shape)

    ax_spec = fig.add_subplot(gs[0, :])
    ax_img1 = fig.add_subplot(gs[1, 0], projection=wcs_flat)
    ax_img2 = fig.add_subplot(gs[1, 1], projection=wcs_flat)
    ax_img3 = fig.add_subplot(gs[1, 2], projection=wcs_8)
    axes = [ax_img1, ax_img2, ax_img3]

    full_velocity_limits = (0*kms, 45*kms)
    all_limits = (full_velocity_limits, velocity_limits)

    imgs = [cube_obj.data.spectral_slab(*vl).moment0() for vl in all_limits]

    # Grab these for later
    cii_line_center = cube_obj.data.header['RESTFREQ'] * u.Hz
    velocity_axis_freq = cube_obj.data.spectral_axis[::-1].to(u.Hz, equivalencies=u.doppler_radio(cii_line_center))
    channel_width_freq = np.mean(np.diff(velocity_axis_freq))
    channel_width_vel = np.mean(np.diff(cube_obj.data.spectral_axis))

    # work with the 8 micron
    irac4_effective_width = 3.94e12 * u.Hz # 3.94 THz from Table 4.3 of IRAC Instrument Handbook (per irac8um_to_cii_figure)
    pixreg = reg.to_pixel(wcs_8)
    pj, pi = [int(round(c)) for c in pixreg.center.xy]
    img_unit = u.Unit(hdr_8['BUNIT'])
    intensity_unit_cgs = u.Unit('erg s-1 cm-2 sr-1')
    intensity_8 = (img_8[pi, pj] * img_unit * irac4_effective_width).to(intensity_unit_cgs)
    img_8 = (img_8 * img_unit * irac4_effective_width).to(intensity_unit_cgs)
    """
    cii = 10^b * (8um)^a
    for a, b = 0.70, -1.79
    """
    def convert_8um_to_cii(value):
        # use Cornelia's numbers from 2021 paper
        value = value.to(intensity_unit_cgs) # must be in these units
        a, b = 0.70, -1.79
        value_cii = ((10**b) * (value.to_value()**a)) * value.unit
        # Now convert that to K km/s
        # inverse of this
        # cii_integrated_intensity_cgs = (cii_integrated_intensity/channel_width_vel).to(u.Jy/u.sr, equivalencies=u.brightness_temperature(line_center)) * channel_width_freq
        value_cii_kkms = (value_cii/channel_width_freq).to(u.K, equivalencies=u.brightness_temperature(cii_line_center)) * channel_width_vel
        return value_cii_kkms.to(u.K * kms)

    intensity_8_cii = convert_8um_to_cii(intensity_8)
    img_8 = convert_8um_to_cii(img_8)
    imgs.append(img_8)

    save_img = False
    if save_img:
        keys_to_add = [
            ('DATE', f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"),
            ('CREATOR', f"Ramsey, {__file__}.compare_8micron_and_cii_intensities"),
            ('BUNIT', f"{img_8.unit.to_string('latex_inline')}"),
            ('BMAJ', (f"{beam_cii.major.to(u.deg).to_value()}", 'deg')),
            ('BMIN', (f"{beam_cii.minor.to(u.deg).to_value()}", 'deg')),
            ('BPA', ("0", 'deg')),
            ('TELESCOP', 'Spitzer'),
            ('CHNLNUM', '4'),
            ('HISTORY', "CII resolution"),
            ('HISTORY', "8 um converted to CII"),
        ]
        new_hdr = wcs_flat.to_header()
        for k, v in keys_to_add:
            new_hdr[k] = v
        hdu = fits.PrimaryHDU(data=img_8.to_value(), header=new_hdr)
        hdu.writeto(os.path.join(os.path.dirname(fn_8um), "irac4_to_cii_kkms.fits"))

    save_cii = False


    if save_cii:
        keys_to_add = [
            ('DATE', f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"),
            ('CREATOR', f"Ramsey, {__file__}.compare_8micron_and_cii_intensities"),
            ('BUNIT', f"{imgs[1].unit.to_string('latex_inline')}"),
            ('BMAJ', (f"{beam_cii.major.to(u.deg).to_value()}", 'deg')),
            ('BMIN', (f"{beam_cii.minor.to(u.deg).to_value()}", 'deg')),
            ('BPA', ("0", 'deg')),
            ('TELESCOP', 'SOFIA'),
            ('COMMENT', "Moment 0"),
            ('COMMENT', f"velocity interval {make_vel_stub(velocity_limits)}"),
        ]
        new_hdr = wcs_flat.to_header()
        for k, v in keys_to_add:
            new_hdr[k] = v
        hdu = fits.PrimaryHDU(data=imgs[1].to_value(), header=new_hdr)
        hdu.writeto(os.path.join(cube_obj.directory, f"m16_cii_moment0_{make_simple_vel_stub(velocity_limits)}.fits"))



    pixreg = reg.to_pixel(wcs_flat)
    pj, pi = [int(round(c)) for c in pixreg.center.xy]
    cii_integrated_intensity = imgs[1][pi, pj]

    for i in range(3):
        im = axes[i].imshow(imgs[i].to_value(), origin='lower', vmin=0, vmax=(None if i<2 else 150))
        fig.colorbar(im, ax=axes[i], label=f"{imgs[i].unit.to_string('latex_inline')}")
        if i < 2:
            axes[i].set_title(f"{line_stub} {make_vel_stub(all_limits[i])}")
            reg_patch = pixreg.as_artist()
        else:
            axes[i].set_title(f"{line_stub} from {cube_utils.cubenames['8um']}")
            reg_patch = reg.to_pixel(wcs_8).as_artist()
        reg_patch.set(mec='k')
        axes[i].add_artist(reg_patch)

    spectrum = cube_obj.data[:, pi, pj]

    ax_spec.plot(cube_obj.data.spectral_axis.to_value(), spectrum.to_value(), label='cii')
    ax_spec.set_xlim((-5, 50))
    ax_spec.set_ylim((-1, 10))

    vlcolors = ['grey', 'k']
    for i, vl in enumerate(all_limits):
        for j, v in enumerate(vl):
            ax_spec.axvline(v.to_value(), color=vlcolors[i], alpha=0.7, label=(make_vel_stub(vl) if j == 0 else None))
    ax_spec.axhline(0, color='grey', alpha=0.2)

    ax_spec.legend()

    reg_stub = os.path.basename(reg_filename_short).replace('.fits', '') + f"_{reg_index}"
    # 2023-05-03,04,05,09,10,11,26,30, 06-02,07,14
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"cii_irac4_{make_simple_vel_stub(velocity_limits)}_{reg_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title='cii and irac4-cii', file=__file__, func="compare_8micron_and_cii_intensities"))

def correlation_plot_8um_cii():
    """
    January 10, 2024
    Drafting a correlation plot between CII and 8 micron. I'll be using a lot of
    code/inspiration from m16_pictures.irac8um_to_cii_figure_2p5beam
    For whatever reason, I didn't use the reproject_adaptive there,
    I just used interp. I think that's fine.
    """
    intensity_unit_cgs = u.Unit('erg s-1 cm-2 sr-1')
    """ Get CII and convert to cgs """
    cii_cube_obj = cube_utils.CubeData(get_map_filename('cii')).convert_to_kms()
    cii_slab = cii_cube_obj.data.spectral_slab(0*kms, 40*kms)
    cii_mom0 = cii_slab.moment0()
    # Calculate 1sigma uncertainty on moment0
    n_channels = cii_slab.shape[0]
    channel_width_vel = np.mean(np.diff(cii_cube_obj.data.spectral_axis))
    channel_noise = get_onesigma('cii') * u.K
    mom0_noise = channel_noise * channel_width_vel * np.sqrt(n_channels)
    print("CII noise", mom0_noise)
    # Mask CII by 3sigma like Cornelia does in the 2021 paper, Sec 3.2
    cii_mom0[cii_mom0 < mom0_noise] = np.nan*u.K*kms
    # Convert to cgs
    line_center = cii_cube_obj.data.header['RESTFREQ'] * u.Hz
    # Next parts copied from m16_pictures.irac8um_to_cii_figure_2p5beam
    # Calculate velocity axis in frequency, do the same interval thing for 1 km/s -> Hz conversion
    velocity_axis_freq = cii_cube_obj.data.spectral_axis[::-1].to(u.Hz, equivalencies=u.doppler_radio(line_center))
    # Ratio of channel widths in freq and velocity converts km/s to Hz correctly (I think)
    channel_width_freq = np.mean(np.diff(velocity_axis_freq)) # last element, so it matches the first below
    # The conversion from T_B to S_nu is linear, so we can divide out velocity and multiply back in frequency later with no consequence.
    cii_mom0_cgs = (cii_mom0/channel_width_vel).to(u.Jy/u.sr, equivalencies=u.brightness_temperature(line_center)) * channel_width_freq
    cii_mom0_cgs = cii_mom0_cgs.to(intensity_unit_cgs) # finish the conversion
    """ Get IRAC 4 and convolve/reproject to CII """
    irac4_conv_fn = "irac4_ciigrid_cgs.fits" # The savename I'm using
    if False:
        img_8, img_info_8 = get_2d_map("8um")
        # Convolve to CII
        bmaj, bmin, pa = photometry_beams['8um']
        beam_8 = cube_utils.Beam(bmaj*u.arcsec, bmin*u.arcsec, pa*u.deg)
        beam_cii = cii_cube_obj.data.beam
        beam_conv = beam_cii.deconvolve(beam_8)
        pixel_scale = misc_utils.get_pixel_scale(img_info_8['wcs'])
        kernel = beam_conv.as_kernel(pixel_scale)
        img_8 = convolve_fft(img_8, kernel, preserve_nan=True) # keep the nans from becoming 0s
        img_8 = reproject_interp((img_8, img_info_8['wcs']), cii_cube_obj.wcs_flat, shape_out=cii_mom0.shape, return_footprint=False)
        # Convert 8 micron units to cgs
        irac4_effective_width = 3.94e12 * u.Hz # 3.94 THz from Table 4.3 of IRAC Instrument Handbook (per irac8um_to_cii_figure)
        img_unit = img_info_8['unit']
        intensity_unit_cgs = u.Unit('erg s-1 cm-2 sr-1')
        intensity_8um_obs = (img_8 * img_unit * irac4_effective_width).to(intensity_unit_cgs)
        # Save this because the convolution takes a minute
        keys_to_add = [
            ('DATE', f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"),
            ('CREATOR', f"Ramsey, {__file__}.correlation_plot_8um_cii"),
            ('BUNIT', f"{intensity_unit_cgs.to_string('latex_inline')}"),
            ('BMAJ', (f"{beam_cii.major.to(u.deg).to_value()}", 'deg')),
            ('BMIN', (f"{beam_cii.minor.to(u.deg).to_value()}", 'deg')),
            ('BPA', ("0", 'deg')),
            ('TELESCOP', 'Spitzer'),
            ('CHNLNUM', '4'),
            ('HISTORY', "CII resolution"),
            ('HISTORY', "8 um converted to cgs and on CII grid and beam"),
        ]
        new_hdr = cii_cube_obj.wcs_flat.to_header()
        for k, v in keys_to_add:
            new_hdr[k] = v
        hdu = fits.PrimaryHDU(data=intensity_8um_obs.to_value(), header=new_hdr)
        hdu.writeto(os.path.join(os.path.dirname(img_info_8['full_path']), irac4_conv_fn), overwrite=False)
    else:
        img_8, hdr_8 = fits.getdata(catalog.utils.search_for_file(f"spitzer/{irac4_conv_fn}"), header=True)

    # All convolved and converted now
    img_cii = cii_mom0_cgs.to_value()
    nanmask = (np.isfinite(img_8) & np.isfinite(img_cii)).flatten()
    # Use mask to trim down the number of points plotted
    skip_mask = nanmask.copy()
    skip_mask[:] = False
    skip_mask[::10] = True
    nanmask = nanmask & skip_mask

    # Adjust the IRAC map by 2e-3; see my 2024-01-10 notes and Cornelia's 2021 sec 3.2
    img_8 = img_8 - 2e-3

    # Using i index as proxy for declination just to see something
    ii, jj = np.meshgrid(np.arange(img_cii.shape[0]), np.arange(img_cii.shape[1]), indexing='ij')

    values_cii = img_cii.flatten()[nanmask]
    values_8 = img_8.flatten()[nanmask]
    values_dec = ii.flatten()[nanmask]

    ax = plt.subplot(211)
    plt.scatter(values_8, values_cii, marker='.', alpha=0.2, c=values_dec, cmap='jet')
    # Plot Cornelia's fit to the Orion data
    xlim = (1e-4, 5e-1)
    ylim = (1e-5, 1e-2)
    x = np.array(xlim)
    y = 10**(np.log10(x)*0.7 - 1.79) # point correlation 2021
    y2 = 10**(np.log10(x)*0.65 - 1.83) # point density correlation 2021
    y3 = 2.2e-2 * x**0.79 # Horsehead 2017 paper
    plt.plot(x, y, color=marcs_colors[1])
    plt.plot(x, y2, color=marcs_colors[2])
    plt.plot(x, y3, color=marcs_colors[3])

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("8 micron (cgs)")
    plt.ylabel("CII (cgs)")

    plt.subplot(223)
    plt.imshow(img_cii, origin='lower')
    plt.subplot(224)
    plt.imshow(ii, cmap='jet', origin='lower')

    plt.show()




def n19_shell_dust_mass():
    """
    June 28, 2023
    Plan: use T-tau_colorsolution.fits tau map to get column density.
    Trim them using cube_utils.cutout2d_from_region.
    Use CII 17 km/s channel to get shell mask and limit with box too.
    Reproject it to the cutout2d tau map. Mask, sum, find mass.
    Identify background to tau map and subtract background * pixels under mask.
    Can background in DS9, average/median in circle.
    """
    ...

"""
CHANNEL MAPS
"""

def cii_channel_maps():
    """
    September 27, 2023
    Use the channel_maps.py functions to make channel map figures that look nice
    """
    fn = get_map_filename("cii-30")
    cube = cube_utils.CubeData(fn)
    channel_maps_utils.channel_maps_figure(
        cube.data, name="cii-30", grid_shape=(5, 4), figsize=(12, 13),
        velocity_limits=(5, 41), panel_offset=0, vmin=0, vmax=25,
        text_y=0.92, text_x=0.05, ha='left', tick_labelrotation=40, tick_labelpad=20,
        left=0.08, right=0.89, bottom=0.08, cmap="plasma",
        figure_save_path=catalog.utils.todays_image_folder(),
        metadata=catalog.utils.create_png_metadata(title="channel maps cii using channel_maps.py functions",
            file=__file__, func="cii_channel_maps")
    )

def co_channel_maps():
    """
    Do this!! need to rebin the CO 3-2, so that'll take some time...

    November 17, 2023
    Only now getting around to this, this function name has been a placeholder since the cii_channel_maps()
    one was written, end of September...

    Rebin the 12co32 spectra to 2 km/s. Use the same cii-30 limits, so 5 to 41 (inclusive) in steps of 2 (odd numbers).
    """
    # Rebin
    if False:
        fn = get_map_filename("12co32")
        cube = cube_utils.CubeData(fn)
        full_path = cube.full_path
        cube = cube.data.spectral_slab(3*kms, 43*kms)
        channel_maps_utils.rebin_spectra(cube, velocity_limits=(5, 41, 2), name="12co32", data_filename=full_path)
        print("done")
    fn = get_map_filename("12co32-2kms")
    cube = cube_utils.CubeData(fn)
    channel_maps_utils.channel_maps_figure(
        cube.data, name="12co32-2kms", grid_shape=(5, 4), figsize=(12, 13),
        velocity_limits=(5, 41), panel_offset=0, vmin=0, vmax=25,
        text_y=0.92, text_x=0.05, ha='left', tick_labelrotation=40, tick_labelpad=20,
        left=0.08, right=0.89, bottom=0.08, cmap='plasma',
        figure_save_path=catalog.utils.todays_image_folder(),
        metadata=catalog.utils.create_png_metadata(title="channel maps 12co32-2kms rebin; channel_maps.py functions",
            file=__file__, func="co_channel_maps")
    )
    print("done")


def channel_maps_scratchpad():
    """
    February 17, 2025
    I'm like past 6 months working at SDGR and here I am making new function

    The idea here is to identify indices for cropping the Gal-oriented CII/CO

    Following m16_pictures_2.py::big_average_spectrum_figure for the regridding
    """
    fn = get_map_filename("cii-30")
    cube = cube_utils.CubeData(fn)
    i_27kms = bisect.bisect_right(cube.data.spectral_axis.to(kms).to_value(), 27) - 1

    fn_co = get_map_filename("12co32-2kms")
    cube_co = cube_utils.CubeData(fn_co)
    i_27kms_co = bisect.bisect_right(cube_co.data.spectral_axis.to(kms).to_value(), 27) - 1

    cutout = misc_utils.cutout2d_from_region(cube.data[i_27kms].to_value(), cube.wcs_flat, get_cutout_box_filename('med'), align_with_frame='galactic')
    ref_wcs = cutout.wcs
    ref_img = cutout.data
    ref_shape = ref_img.shape

    co_reproj = reproject_interp((cube_co.data[i_27kms_co].to_value(), cube_co.wcs_flat), ref_wcs, shape_out=ref_shape, return_footprint=False)

    crop_slices = (slice(4, 146), slice(21, 138))
    # crop_slices = (slice(None), slice(None))

    ref_wcs = ref_wcs[crop_slices]

    fig = plt.figure()
    ref_ax = fig.add_subplot(121, projection=ref_wcs)
    im = ref_ax.imshow(ref_img[crop_slices], origin='lower', cmap=cmocean.cm.matter)

    co_ax = fig.add_subplot(122, projection=ref_wcs)
    im_co = co_ax.imshow(co_reproj[crop_slices], origin='lower', cmap=cmocean.cm.algae)

    plt.show()


def cii_co_combined_channel_maps():
    """
    February 17, 2025
    Ambitious attempt to put CII and CO side-by-side channel-by-channel.
    This will have to be split into two images. I can fit 10 channels on each
    image, and I have 19.
    """
    # Config
    line_names = ['cii-30', '12co32-2kms']
    velocity_limits = [7, 41] * kms
    channel_grid_shape = (6, 3)
    line_vlims = {'cii-30': [0, 25], '12co32-2kms': [0, 25]}
    dpi = 100

    # Load data
    cube_fns = [get_map_filename(line_name) for line_name in line_names]
    cubes = [cube_utils.CubeData(fn).convert_to_K().convert_to_kms() for fn in cube_fns]
    # Get the wcs_flat and slices
    def calc_wcs_flat():
        # Start with an arbitrary CII slice to get the ref wcs
        cutout = misc_utils.cutout2d_from_region(cubes[0].data[0].to_value(), cubes[0].wcs_flat, get_cutout_box_filename('med'), align_with_frame='galactic')
        uncropped_wcs = cutout.wcs
        crop_slices = (slice(4, 146), slice(21, 138))
        ref_wcs = uncropped_wcs[crop_slices]
        ref_img = cutout.data[crop_slices]
        """ :return: WCS, shape """
        return ref_wcs, ref_img.shape
    ref_wcs, ref_shape = calc_wcs_flat()

    fig = plt.figure(figsize=(13.5, 14.5))
    mega_gridspec = fig.add_gridspec(right=0.89, left=0.07, top=0.98, bottom=0.06)
    # Create an axis so we can anchor the colorbar(s)
    mega_axis = mega_gridspec.subplots()
    mega_axis.set_axis_off()
    # Make a "channel" gridspec; this will be 5x2. Each frame is one channel.
    channel_gridspec = mega_gridspec[0,0].subgridspec(*channel_grid_shape, hspace=0.02, wspace=0.05)

    # Make the frame gridspecs. Should be 5x2 of them, and they should each contain 1x2 grids
    frame_gridspecs = []
    # the subplotspecs have more information about the frame position in the channel grid
    frame_subplotspecs = []
    for frame_idx in range(channel_grid_shape[0] * channel_grid_shape[1]):
        frame_subplotspec = channel_gridspec[np.unravel_index(frame_idx, channel_grid_shape)]
        frame_subplotspecs.append(frame_subplotspec)
        frame_gridspec = frame_subplotspec.subgridspec(1, len(line_names), wspace=0, hspace=0)
        frame_gridspecs.append(frame_gridspec)
    # Create and organize the axes
    def get_ax_stub(line_name, frame_idx):
        return f"{line_name}_{frame_idx:02d}"
    axes_lookup = {}
    def get_axis(line_name, frame_idx):
        stub = get_ax_stub(line_name, frame_idx)
        if stub not in axes_lookup:
            frame_gridspec = frame_gridspecs[frame_idx]
            ax = fig.add_subplot(frame_gridspec[0, line_names.index(line_name)], projection=ref_wcs)
            axes_lookup[stub] = ax
        return axes_lookup[stub]

    # Map velocities to channels
    channel_indices = [[cube.data.closest_spectral_channel(v) for v in velocity_limits] for cube in cubes]
    """ Below here, heavily copied from the regular channel maps script """

    # Text defaults
    text_x = 0.05
    text_y = 0.92
    # ha/va are horizontal and vertical alignment
    ha = 'left'
    # the color I use there is from Marc's collection of colorblind-friendly colors and works well against "plasma"
    default_text_kwargs = dict(fontsize=14, color='k', ha=ha, va='center')
    tick_labelsize = 14
    tick_labelrotation = 40
    tick_labelpad = 20

    # Colors
    cmaps = [cmocean.cm.matter, cmocean.cm.dense] # Image colormap
    # cmap = cmaps[0] # stick with one for now
    beam_patch_ec = "grey" # edge color
    beam_patch_fc = "white" # face color
    pixel_scale = ref_wcs.proj_plane_pixel_scales()[0] # gives RA, Dec scales; pick one. They're almost certainly equal, so doesn't matter

    # Save the last image from each line for colorbar making
    im_cache = [None]*len(line_names)

    # Loop thru cubes
    for cube_idx, line_name in enumerate(line_names):
        cube = cubes[cube_idx]
        vlims = dict(zip(['vmin', 'vmax'], line_vlims[line_name]))
        first_channel_idx, last_channel_idx = channel_indices[cube_idx]
        cmap = cmaps[cube_idx]
        # Loop through channels and plot
        for frame_idx, channel_idx in enumerate(range(first_channel_idx, last_channel_idx+1)):
            if frame_idx >= channel_grid_shape[0]*channel_grid_shape[1]:
                break
            velocity = cube.data.spectral_axis[channel_idx]
            unaligned_channel_data = cube.data[channel_idx].to_value()
            channel_data = reproject_interp((unaligned_channel_data, cube.wcs_flat), ref_wcs, shape_out=ref_shape, return_footprint=False)

            print(f"Frame {frame_idx:2d}, Channel {channel_idx:3d}")
            ### print the [min, mean, median, max] for each panel so that we can find the best vlims (min, max) for all of them
            # print([f(channel_data) for f in (np.nanmin, np.nanmean, np.nanmedian, np.nanmax)])


            # Setup Axes
            ax = get_axis(line_name, frame_idx)
            # Remove x and y labels on individual panels (use the "super" titles)
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")
            ss = frame_subplotspecs[frame_idx]
            # Coordinate labels
            if ss.is_last_row() and ss.is_first_col():
                # Coordinates only on bottom left corner panel
                # AND only Dec on first line, then RA on both lines
                # Mess around with the rotation, position, and size of coordinate labels
                ax.coords[0].set_ticklabel(rotation=tick_labelrotation, rotation_mode='anchor', pad=tick_labelpad, fontsize=tick_labelsize, ha='right', va='top')
                ax.coords[0].set_major_formatter('d.dd')
                if cube_idx == 0:
                    ax.coords[1].set_major_formatter('d.dd')
                    ax.coords[1].set_ticklabel(fontsize=tick_labelsize)
                else:
                    ax.tick_params(axis='y', labelleft=False)
            else:
                # If not the bottom left panel, no coordinates (panels have no space in between)
                # Hide coordinates
                ax.tick_params(axis='x', labelbottom=False)
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(axis='both', direction='in')
            # Plot
            im_cache[cube_idx] = ax.imshow(channel_data, origin='lower', cmap=cmap, **vlims)
            # Label velocity on each panel
            if cube_idx == 0:
                ax.text(text_x, text_y, f"{velocity.to_value():.0f} {velocity.unit.to_string('latex_inline')}", transform=ax.transAxes, **default_text_kwargs)
            if frame_idx == 0:
                ax.text(text_x, 1-text_y, get_data_name(line_name.split('-')[0]), transform=ax.transAxes, **default_text_kwargs)
            # Beam on every panel
            beam_patch = cube.data.beam.ellipse_to_plot(*(ax.transAxes + ax.transData.inverted()).transform([0.9, 0.1]), pixel_scale)
            beam_patch.set(alpha=0.9, facecolor=beam_patch_fc, edgecolor=beam_patch_ec)
            ax.add_artist(beam_patch)

    # Colorbar
    # Create a space to the right of the panels using the height/location of the mega_axis as an anchor
    cbar_pad = 0.025
    for cbar_i in range(len(line_names)):
        cbar_ax = mega_axis.inset_axes([1. + 0.015 + cbar_pad*(cbar_i), 0, cbar_pad, 1])
        cbar_kwargs = {}
        if cbar_i == len(line_names)-1:
            cbar_kwargs['label'] = label='T$_{\\rm MB}$ (K)'
        cbar = fig.colorbar(im_cache[cbar_i], cax=cbar_ax, **cbar_kwargs)
        if cbar_i != len(line_names)-1:
            cbar_ax.tick_params(axis='y', labelright=False, right=False)

    # Titles
    fig.supxlabel("Galactic Longitude")
    fig.supylabel("Galactic Latitude")

    dpi_stub = f"_dpi{dpi}"

    fig_save_name = f"channel_maps_{'-'.join(line_names)}{dpi_stub}.png"
    savefig_kwargs = {"dpi": dpi}
    savefig_kwargs['metadata'] = catalog.utils.create_png_metadata(title=f"channel maps {', '.join(line_names)} rebin",
        file=__file__, func="cii_co_combined_channel_maps")

    figure_save_path = catalog.utils.todays_image_folder()

    full_save_path = os.path.join(figure_save_path, fig_save_name)
    # fig.savefig(full_save_path, **savefig_kwargs)
    fig.savefig(full_save_path.replace('.png', '.pdf'))
    print(f"Figure saved to {os.path.join(figure_save_path, fig_save_name)}")


"""
Various other interesting things
"""

def peak_T_velocity_map():
    """
    October 5, 2023
    While making the CII column densities, I kinda liked the way that the peak T
    map showed some of the features. I want to do an argmax of the peak T map
    and see what the velocities look like. Maybe compare to moment 0 and 1, for
    kicks.

    This is a neat map!
    Lots of interesting shapes appear. I don't have time in my thesis to go
    deep on this stuff, but there's lots of little rings that someone can
    find someday.

    What this does show is that moment 1 and 2 maps make the northern cloud pop
    really well, as it causes wide (composite) line profiles and lies at a much
    lower velocity, near 16 km/s
    """
    mask_cutoff = 6*u.K
    velocity_limits = None

    cii_cube_fn = get_map_filename('cii')
    cii_cube = cube_utils.CubeData(cii_cube_fn).convert_to_kms().data
    # Apply velocity limits
    if velocity_limits is not None:
        cii_cube = cii_cube.spectral_slab(*velocity_limits)
    print(f"Masking below {mask_cutoff:.2f}")
    cii_cube = cii_cube.with_mask(cii_cube > mask_cutoff).with_fill_value(0*u.K)

    peak_T_map = cii_cube.max(axis=0).quantity
    plt.subplot(221)
    plt.imshow(peak_T_map.to_value(), origin='lower')
    amax = cii_cube.argmax(axis=0)
    peak_velocity_map = cii_cube.spectral_axis[amax]
    peak_velocity_map[amax==0] = np.nan
    plt.subplot(222)
    plt.imshow(peak_velocity_map.to_value(), origin='lower', vmin=20, vmax=30)

    mom0 = cii_cube.moment0()
    plt.subplot(223)
    plt.imshow(mom0.to_value(), origin='lower')
    # plt.imshow((mom0/peak_T_map).to_value(), origin='lower', vmin=0, vmax=15)
    mom1 = cii_cube.moment1()
    # mom2 = cii_cube.linewidth_fwhm()
    plt.subplot(224)
    plt.imshow(mom1.to_value(), origin='lower', vmin=20, vmax=30)
    # plt.imshow(mom2.to_value(), origin='lower', vmin=0, vmax=15)

    plt.show()

def peak_T_and_moment_maps_CO(isotope='12', transition='32', velocity_limits=None):
    """
    In a similar vein to the peak_T_velocity_map for CII, I'm doing a check of
    peak T vs moment 0 for 12 and 13 CO 3-2 and 1-0
    For a given isotope and transition, produce a 3-extension FITS file, peak T,
    moment 0, and peak T velocity (just for reference)
    :param isotope: str, '12' or '13'
    :param transition: str, '32' or '10' CO line transition
    :param velocity_limits: (optional) tuple(Quantity low, Quantity high)
        If given, should be a 2-tuple of velocity Quantities (low, high) limits.
        If given, the peak temperature and moment 0 will both be calculated from
        within these limits. If not given or None (default), the full velocity
        range is used.
    """
    line_stub = isotope + "co" + transition
    if transition == '10':
        # Use Purple Mountain
        line_stub += '-pmo'
    cube_obj = cube_utils.CubeData(get_map_filename(line_stub))
    cube = cube_obj.data.with_spectral_unit(kms)
    if velocity_limits is not None:
        cube = cube.spectral_slab(*velocity_limits)
    mom0 = cube.moment0().to(u.K * kms)
    peak_T_argmax = cube.argmax(axis=0)
    peak_T_velocity = cube.spectral_axis[peak_T_argmax]
    # peak_T_velocity[peak_T_argmax==0] = np.nan # get rid of junk
    peak_T = cube.max(axis=0).quantity
    header_template = mom0.wcs.to_header()
    header_template.update(cube.beam.to_header_keywords())
    header_template['COMMENT'] = f"{line_stub} CO data from file"
    header_template['COMMENT'] = f"{cube_obj.basename}"
    if velocity_limits is not None:
        header_template['COMMENT'] = f"Between velocities {make_vel_stub(velocity_limits)}"
        velocity_stub = "_" + make_simple_vel_stub(velocity_limits)
    else:
        header_template['COMMENT'] = "Full velocity range of data"
        velocity_stub = ""
    def make_and_fill_hdu(extname, data, bunit):
        header = header_template.copy()
        header['EXTNAME'] = extname
        header['BUNIT'] = str(bunit)
        hdu = fits.ImageHDU(data=data, header=header)
        return hdu
    hdul = fits.HDUList([fits.PrimaryHDU(),
        make_and_fill_hdu("moment_0", mom0.to_value(), mom0.unit),
        make_and_fill_hdu("peak_T", peak_T.to_value(), peak_T.unit),
        make_and_fill_hdu("peak_velocity", peak_T_velocity.to_value(), peak_T_velocity.unit),
    ])
    savename = os.path.join(cube_obj.directory, f"{line_stub}_mom0_peakT_peakvel{velocity_stub}.fits")
    print("Done, writing to "+savename)
    hdul.writeto(savename)


def convert_pacs_tau_to_coldens():
    """
    December 7, 2023
    Quick one-time use code to convert the tau160 from 70-160 to a column density N_H (total)
    Use the Cext/H from the paper, 1.9e-25 cm2/H
    N_H = tau / Cext/H
    """
    fn = "herschel/T-tau_colorsolution_70zeroedat160.fits"
    full_fn = catalog.utils.search_for_file(fn)
    cexth = 1.9e-25 * u.cm**2
    with fits.open(full_fn) as hdul:
        tau160 = hdul['tau']
        nhtot = (tau160.data / cexth).to(u.cm**-2)
        hdr = tau160.header.copy()
    del hdr['EXTNAME']
    hdr['BUNIT'] = str(nhtot.unit)
    hdr['COMMENT'] = "value is total H column N_H"
    hdr['HISTORY'] = f"tau160 converted to NHtot using Cext(160)/H 1.9e-25 cm2/H"
    hdr['HISTORY'] = "written by m16_bubble.convert_pacs_tau_to_coldens"
    hdr['HISTORY'] = f"using {fn}"
    hdr['DATE'] = "December 9, 2023"
    # background = 1e22 * u.cm**-2
    background = 0 * u.cm**-2
    if background.to_value() != 0:
        hdr['HISTORY'] = f"subtracted {background:.2E}"
    new_hdu = fits.PrimaryHDU(data=(nhtot - background).to_value(), header=hdr)
    savename = os.path.join(os.path.dirname(full_fn), "coldens_70-160_colorsolution_70zeroedat160.fits")
    new_hdu.writeto(savename, overwrite=True)


def spitzer_expansion_plot():
    """
    January 23, 2024
    Spitzer expansion plot from Xander's data.
    Data in misc_data/spitzer_expansion.txt
    """
    def _calc_normalized_mass(shell_mass, q0):
        """ See 2024-01-23 notes for details; M_norm is in cgs with units cm3 """
        particle_mass = Hmass * 1.33
        beta = 2.6e-13 * u.cm**3 / u.s
        Mnorm = (beta * shell_mass / particle_mass / q0).decompose()
        return Mnorm.to(u.cm**3).to_value()

    m16_params = {
        # shell mass, Q0
        'm16': (1e4*u.solMass, 8.43e49 / u.s),
        'n19': (650*u.solMass, 7.43e47 / u.s),
    }
    m16_velocities = {'m16': 10, 'n19': 4}

    m16_vel_errors = {'m16': 2, 'n19': 1} # absolute
    m16_mass_errors = {'m16': 0.5, 'n19': 0.5} # fractional

    data_fn = os.path.join(catalog.utils.misc_data_path, "spitzer_expansion.txt")
    data = np.loadtxt(data_fn, skiprows=3)
    with open(data_fn, 'r') as f:
        densities = [float(x) for x in f.readline().split()[1:]]
    colors = {1e5: 'orange', 1e4: 'red', 1e3: 'green', 1e2: 'blue'}
    colors = {n: marcs_colors[i] for n, i in zip((1e5, 1e4, 1e3, 1e2), (2, 3, 4, 5))}
    v_array = data[:, 0]
    fig = plt.figure(figsize=(8, 8))
    for i in list(range(1, data.shape[1]))[::-1]:
        exp_str = f"{int(np.log10(densities[i-1]))}"
        label = "$10^{" + exp_str + "}$ " + (u.cm**-3).to_string('latex_inline')
        plt.plot(v_array, 1./data[:, i], color=colors[densities[i-1]], zorder=3)
        plt.text(v_array[v_array.size*(30+i)//48]/1.02, 1./data[:, i][v_array.size*(30+i)//48]*1.005, label, color=colors[densities[i-1]], rotation=-54, rotation_mode='anchor')
        # plt.plot([v_array[v_array.size*2//3]], [1./data[:, i][v_array.size*2//3]], marker='D')

    data_table = """| region | v | M | Nlyc | M_nor |
| ---- | ---- | ---- | ---- | ---- |
| NGC 1977 | 1.5 | 700 | 1 | 9.89E-01 |
| M43 | 6 | 7 | 1.5 | 6.59E-03 |
| Orion Veil | 13 | 1500 | 70 | 3.03E-02 |
| RCW 36 | - | 1000 | 6 | 2.36E-01 |
| RCW 120 | 15 | 500 | 38 | 1.86E-02 |
| RCW 49 | 13 | 24000 | 3900 | 8.70E-03 |"""
# | 30 Dor | 25 | 4.50E+05 | 1.20E+05 | 5.30E-03 | # 30 Dor goes way off the page
# | N19_X | 5 | 700 | 18 | 5.50E-02 |
    # colnames = [x.strip() for x in [0].split('|') if x.strip()]
    rows = [[x.strip() for x in line.split('|') if x.strip()] for line in data_table.split('\n')]
    colnames = rows[0]
    rows = rows[2:]
    df = pd.DataFrame(rows, columns=colnames)
    df = df.replace(to_replace='-', value=np.nan)
    for col in list(df.columns):
        if col == 'region':
            continue
        df[col] = df[col].astype('float')

    source_vel_errors = {
        "NGC 1977": 0.3, "M43": 0.3, "Orion Veil": 0.3, "RCW 36": 0.1, "RCW 120": 0.1, "RCW 49": 0.3,
    } # fractional
    source_mass_errors = {
        "NGC 1977": 0.5, "M43": 0.5, "Orion Veil": 0.5, "RCW 36": 0.3, "RCW 120": ((500-40.)/500, (520.-500)/500), "RCW 49": 0.3,
    } # fractional

    def _get_vel_errors(name, vel):
        if name in m16_vel_errors:
            verr = m16_vel_errors[name]
            # These are absolute
        else:
            verr = source_vel_errors[name]
            verr = vel * verr # these are fractional
        return verr

    def _get_mass_errors(name, mass):
        if name in m16_mass_errors:
            merr = m16_mass_errors[name]
        else:
            merr = source_mass_errors[name]
        # All fractional. One is asymmetric
        if isinstance(merr, tuple):
            # asymmetric
            merr = mass * np.array(merr)[::-1][:, np.newaxis]
        else:
            merr = mass * merr
        return merr

    # adjust approximately by my N19 Q0 value
    # df.loc[df['region']=='N19', 'M_nor'] = 2 * df[df['region']=='N19']['M_nor']
    # essentially doesn't matter, only pushes the point up a tiny bit, stays within the same two density contours

    x_text_adjust = 1.06
    y_text_adjust = 1.19
    text_kwargs = dict(ha='center', fontsize=15)

    xy_text_adjust = {
        "NGC 1977": (1/1.05, 1.7),
        "M43": (1, 1.6),
        "Orion Veil": (1.1, 1/2.9),
        "RCW 36": (1, 1),
        "RCW 120": (1.02, 1.14),
        "RCW 49": (1, 1.4),
        "m16": (1/1.5, 1/1.1),
        "n19": (1.2, 1.2),
    }


    # plt.plot(df['v'], 1./df['M_nor'], linestyle='none', marker='o', markersize=10, color=marcs_colors[0])
    # plt.plot(df['v'], 1./df['M_nor'], linestyle='none', marker='o', markersize=10, color=marcs_colors[0])
    for i in df.index:
        row = df.loc[i]
        name = row['region']
        v = row['v']
        mnorm = row['M_nor']
        if np.isnan(v):
            continue
        plt.errorbar(v, 1./mnorm, xerr=_get_vel_errors(name, v), yerr=_get_mass_errors(name, 1./mnorm), linestyle='none', marker='s', markersize=10, color=marcs_colors[0], capsize=5)
        x_text_adjust, y_text_adjust = xy_text_adjust[name]
        plt.text(v*x_text_adjust, 1./mnorm*y_text_adjust, name, **text_kwargs, zorder=15)

    for name in m16_velocities:
        v = m16_velocities[name]
        mnorm = _calc_normalized_mass(*m16_params[name])
        # plt.plot(v, 1./mnorm, 'D', markersize=10, color=marcs_colors[1], zorder=10)
        plt.errorbar(v, 1./mnorm, xerr=_get_vel_errors(name, v), yerr=_get_mass_errors(name, 1./mnorm), linestyle='none', marker='s', markersize=10, color=marcs_colors[1], capsize=5)
        x_text_adjust, y_text_adjust = xy_text_adjust[name]
        plt.text(v*x_text_adjust, 1./mnorm*y_text_adjust, name.capitalize(), **text_kwargs, zorder=15)

    if False:
        """ Test out an error bar or limit thing for N19 considering H2 mass """
        n19_mass, n19_q0 = m16_params['n19']
        n19_mass_2 = 4200*u.solMass + n19_mass
        n19_mnorm_both = np.array([_calc_normalized_mass(x, n19_q0) for x in [n19_mass, n19_mass_2]])
        n19_v_both = [m16_velocities['n19']]*2
        plt.plot(n19_v_both, 1./(n19_mnorm_both), marker='_', linestyle='-', markersize=10, color=marcs_colors[1], zorder=5)

        n19_mass, n19_q0 = m16_params['m16']
        n19_mass_2 = 100*u.solMass
        n19_mnorm_both = np.array([_calc_normalized_mass(x, n19_q0) for x in [n19_mass, n19_mass_2]])
        n19_v_both = ([m16_velocities['m16']]*2)
        plt.plot(n19_v_both, 1./(n19_mnorm_both), marker='_', linestyle='-', markersize=10, color=marcs_colors[1], zorder=5)


    plt.legend(handles=[
        mpatches.Patch(color=marcs_colors[i], label=["Literature", "This work"][i]) for i in range(2)
    ], loc='lower left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1, 20))
    plt.gca().invert_xaxis()
    # plt.ylim((1e-3, 10))
    plt.ylim((0.1, 1e3))
    plt.xlabel(f"Expansion Velocity [{kms.to_string('latex_inline')}]", fontsize=15)
    # plt.ylabel("M$_{\\rm norm}$" + f" ({(u.cm**3).to_string('latex_inline')})", fontsize=15)
    plt.ylabel("\"Equivalent\" density " + f" [{(u.cm**-3).to_string('latex_inline')}]", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "spitzer_expansion.png"),
        metadata=catalog.utils.create_png_metadata(title="both n19 Q0 values",
            file=__file__, func="spitzer_expansion_plot"))

def integrate_cii_and_FIR_luminosities():
    """
    January 24, 2024
    Sum the FIR and CII fluxes to get energy for Xander's tables.

    The next step would be to mask this to get N19; this is for M16 + N19

    Result:
    CII 1024.7933376112644 solLum
    FIR (40-500 from 70,160,250) 437548.4931927458 solLum
    """
    cii_fn = "sofia/cii_integrated_intensity_cgs.fits"
    fir_fn = "herschel/m16-I_FIR.fits"
    def _load(short_fn):
        img, hdr = fits.getdata(catalog.utils.search_for_file(short_fn), header=True)
        wcs = WCS(hdr)
        unit = u.Unit(hdr['BUNIT'])
        return img*unit, wcs
    cii_img, cii_wcs = _load(cii_fn)
    fir_img, fir_wcs = _load(fir_fn)
    cii_reproj = reproject_interp((cii_img.to_value(), cii_wcs), fir_wcs, shape_out=fir_img.shape, return_footprint=False)
    fir_img[np.isnan(cii_reproj)] = np.nan
    del cii_reproj
    # Works great!
    # Now sum over pixels. Do it at native resolutions.
    def _calc_lum(img, wcs):
        pixel_scale = misc_utils.get_pixel_scale(wcs)
        pixel_area = (pixel_scale * los_distance_M16 / u.radian)**2
        flux_sum = np.nansum(img)
        return (flux_sum * pixel_area * 4 * np.pi * u.sr).to(u.solLum)
    cii_lum = _calc_lum(cii_img, cii_wcs)
    fir_lum = _calc_lum(fir_img, fir_wcs)
    print(f"CII {cii_lum}")
    print(f"FIR (40-500 from 70,160,250) {fir_lum}")


def ekin_ew_vs_age_plot(setting=0):
    """
    January 29, 2024
    Recreate the figure from Xander's 2024-01-23 email
    Ekin/Ew vs Age
    """
    existing_data = {
        'RCW 36\n(bipolar)': (0.45, 7.38e-3), # old age was 0.7 which was the "constant velocity" age
        'RCW 120': (0.15, 8.21e-1),
        'M42': (0.2, 4.89e-1),
        'RCW 49': (2*0.97, 6.7e-3),
    }
    data_errs = {
        'RCW 36\n(bipolar)': 0.36, # 36% KE error per 2024-03-29 calculation
        'RCW 120': (1./12, 1), # see 2024-03-29 and 2024-04-01
        'M42': 0.78,
        'RCW 49': 0.67,
        'N19': 0.71,
        'M16': (0.001, 1.4),
    }

    age_errors = {
        'RCW 36\n(bipolar)': (0.3, 0.6), # 0.3-0.6
        'RCW 120': (0.15, 0.4),
        'M42': (0.15, 0.25), # 0.2 +/- 0.05
        'RCW 49': (0.5, 2.0),
        'N19': (0.2, 0.7),
        'M16': (1, 3),
    }

    age_type = {
        'RCW 36\n(bipolar)': 'k',
        'RCW 120': 'k',
        'M42': 'k',
        'RCW 49': 's',
        'N19': 'k',
        'M16': 's',
    }

    new_data = {
        'N19': {
            'age': 0.5, 'v': 4, 'mshell': 650, 'Ew': 1.1e47,
        },
        'M16': {
            'age': 2*1.03, 'v': 10, 'mshell': 1e4, 'Ew': 9.8e50,
        },
    }
    units = {'age': u.Myr, 'v': kms, 'mshell': u.solMass, 'Ew': u.erg}
    def _get_age(name):
        return new_data[name]['age'] #* units['age']

    def _get_ekin_ew(name):
        m = new_data[name]['mshell'] * units['mshell']
        v = new_data[name]['v'] * units['v']
        wind_energy = new_data[name]['Ew'] * units['Ew']
        kinetic_energy = 0.5 * m * v**2
        kinetic_to_wind = (kinetic_energy / wind_energy).decompose()
        return kinetic_to_wind.to_value()

    def _calc_energy_error_existing(name, ratio):
        error_pct = data_errs[name]
        # Let the Ew error be 0
        if isinstance(error_pct, tuple):
            ratio_err = np.abs(ratio * (1 - np.array(error_pct)[:, np.newaxis])) # these are expressed as limit values, fractions of the actual value
        else:
            ratio_err = ratio * error_pct
        return ratio_err

    def _get_age_errors(name, age):
        # Simpler version, can be used for horizontal errors
        age_lims = age_errors[name]
        age_err = np.abs(age - np.array(age_lims)[:, np.newaxis])
        return age_err

    def _make_age_error_line(name, age, ratio):
        # Diagonal error bars
        age_lims = age_errors[name]
        x_vals = np.array(age_lims)
        y_vals = ratio * age / x_vals
        return x_vals, y_vals

    marker_key = {'k': 's', 's': 'D'}
    def _get_marker(name):
        return marker_key[age_type[name]]

    xy_text_adjust = {
        # "NGC 1977": (1/1.05, 1.7),
        # "M43": (1, 1.6),
        "M42": (1.2, 1.2),
        "RCW 36\n(bipolar)": (1/1.5, 1.2),
        "RCW 120": (1, 1.3),
        "RCW 49": (1/1.4, 1/1.4),
        "M16": (1/1.3, 1.1),
        "N19": (1.3, 1.3),
    }

    if setting == 0:
        fontsize = 15
    elif setting == 1:
        fontsize = 25
    text_kwargs = dict(ha='center', fontsize=fontsize)
    if setting == 0:
        # x_text_adjust, y_text_adjust = 1.1, 1.18
        markersize = 10
    elif setting == 1:
        # x_text_adjust, y_text_adjust = 1.34, 1.18
        markersize = 15
    horizontal_errors = True
    plt.figure(figsize=(8, 8))
    for name in existing_data:
        t, ratio = existing_data[name]
        ratio_err = _calc_energy_error_existing(name, ratio)
        if horizontal_errors:
            age_err = _get_age_errors(name, t)
            plt.errorbar(t, ratio, yerr=ratio_err, xerr=age_err, marker=_get_marker(name), markersize=markersize, color=marcs_colors[0], capsize=5)
        else:
            plt.errorbar(t, ratio, yerr=ratio_err, marker=_get_marker(name), markersize=markersize, color=marcs_colors[0], capsize=5)
            age_x, age_y = _make_age_error_line(name, t, ratio)
            plt.plot(age_x, age_y, marker='|', markersize=markersize, linestyle='-', color=marcs_colors[0])
        x_text_adjust, y_text_adjust = xy_text_adjust[name]
        plt.text(t*x_text_adjust, ratio*y_text_adjust, name, **text_kwargs)
    for name in new_data:
        t = _get_age(name)
        ratio = _get_ekin_ew(name)
        ratio_err = _calc_energy_error_existing(name, ratio)
        if horizontal_errors:
            age_err = _get_age_errors(name, t)
            plt.errorbar(t, ratio, yerr=ratio_err, xerr=age_err, marker=_get_marker(name), markersize=markersize, color=marcs_colors[1], capsize=5)
        else:
            plt.errorbar(t, ratio, yerr=ratio_err, marker=_get_marker(name), markersize=markersize, color=marcs_colors[1], capsize=5)
            age_x, age_y = _make_age_error_line(name, t, ratio)
            plt.plot(age_x, age_y, marker='|', markersize=markersize, linestyle='-', color=marcs_colors[1])
        x_text_adjust, y_text_adjust = xy_text_adjust[name]
        plt.text(t*x_text_adjust, ratio*y_text_adjust, name, **text_kwargs)

    if False:
        """ Add limit thing for N19 """
        name = 'N19'
        mass = new_data[name]['mshell']
        mass_2 = 4200 + mass
        mass_both = [mass, mass_2] * units['mshell']
        def _ke_w_f(n, mass):
            v = new_data[n]['v'] * units['v']
            ke = 0.5 * mass * v**2
            wind = new_data[n]['Ew'] * units['Ew']
            return (ke/wind).decompose().to_value()
        kew_both = [_ke_w_f(name, m) for m in mass_both]
        age_both = [_get_age(name)]*2
        plt.plot(age_both, kew_both, marker='_', linestyle='-', color=marcs_colors[1])

        """ Same for M16 """
        if False:
            name = 'M16'
            mass = new_data[name]['mshell']
            mass_2 = 100
            mass_both = [mass, mass_2] * units['mshell']
            def _ke_w_f(n, mass):
                v = new_data[n]['v'] * units['v']
                ke = 0.5 * mass * v**2
                wind = new_data[n]['Ew'] * units['Ew']
                return (ke/wind).decompose().to_value()
            kew_both = [_ke_w_f(name, m) for m in mass_both]
            age_both = [_get_age(name)]*2
            plt.plot(age_both, kew_both, marker='_', linestyle='-', color=marcs_colors[1])

    """ Draw a line which shows a theoretical bubble perfectly coupled for 100,000 yrs and then bursting and having 0 coupling after """
    linestyles = ['-', '--', '-.', ':']
    burst_times = [0.03, 0.1, 0.3, 1]
    ratio_0s = [1]
    if True:
        t_arr = np.arange(0.1, 10, 0.1)
        colors = marcs_colors
        for i, t_burst in enumerate(burst_times):
            for j, ratio_0 in enumerate(ratio_0s):
                ratio_arr = np.ones(t_arr.size) * ratio_0
                ratio_arr[t_arr > t_burst] = ratio_0 * t_burst / t_arr[t_arr > t_burst]
                # labeltext = (f"{t_burst} Myr" if j==0 else None)
                plt.plot(t_arr, ratio_arr, color='grey', linestyle=linestyles[i])

    plt.xlim((0.1, 10))
    # plt.ylim((3e-3, 3))
    plt.ylim((3e-3, 9))
    plt.xscale('log')
    plt.yscale('log')
    if setting == 0:
        fontsizes = [16, 17]
    elif setting == 1:
        fontsizes = [22, 22]
        plt.gca().tick_params(axis='both', labelsize=20)
    plt.xlabel("age [Myr]", fontsize=fontsizes[0])
    plt.ylabel("E$_{\\rm kin}$ / E$_{\\rm w}$", fontsize=fontsizes[1])
    plt.legend(handles=[
        Line2D([], [], color='grey', linestyle=linestyles[i], label=f"{burst_times[i]} Myr") for i in range(len(burst_times))
    ] + [
        Line2D([], [], color='grey', marker=marker_key[x], markersize=markersize, linestyle='none', label={'s': "Stellar age", 'k': "Kinematic age"}[x])  for x in 'ks'
    ] + [
        mpatches.Patch(color=marcs_colors[i], label=["Literature", "This work"][i]) for i in range(2)
    ], ncols=2)
    plt.tight_layout()

    plt.savefig(os.path.join(catalog.utils.todays_image_folder(), "ekin_ew_vs_age.png"),
        metadata=catalog.utils.create_png_metadata(title="from Xanders 2024-01-23 email",
            file=__file__, func="ekin_ew_vs_age_plot"))


def n19_self_absorption():
    """
    Feb 25, 2024
    Quick try a figure about self absorption along N19 ring. Zoom in real close.
    use zoom box "N19-small" key
    use regions in catalogs/N19_shell_edge_selfabs.reg
    """
    use_CO = False
    # Load cube
    line_stub = 'cii'
    fn = get_map_filename(line_stub)
    cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()
    if use_CO:
        co_cube_obj = cube_utils.CubeData(get_map_filename("13co32")).convert_to_K().convert_to_kms()
    # Ref image, rotated
    velocity_limits = (15*kms, 21*kms)
    mom0 = cube_obj.data.spectral_slab(*velocity_limits).moment0()
    unit = mom0.unit
    cutout_name = 'N19-med'
    cutout = misc_utils.cutout2d_from_region(mom0.to_value(), mom0.wcs, get_cutout_box_filename(cutout_name), align_with_frame='galactic')
    ref_img = cutout.data
    ref_wcs = cutout.wcs

    # Plot setup
    fig_ref = plt.figure(figsize=(7, 6))
    fig_spec = plt.figure(figsize=(7, 6))
    # gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.3)

    # Plot
    # ref_ax = fig.add_subplot(gs[0, 0], projection=ref_wcs)
    ref_ax = fig_ref.add_subplot(111, projection=ref_wcs)
    im = ref_ax.imshow(ref_img, origin='lower', cmap="Greys_r", vmin=10, vmax=60)
    cax = ref_ax.inset_axes([1, 0, 0.05, 1])
    fig_ref.colorbar(im, cax=cax, label=f"{get_data_name(line_stub)} {make_vel_stub(velocity_limits)} ({unit.to_string('latex_inline')})")
    lat, lon = (ref_ax.coords[i] for i in range(2))
    for l in (lat, lon):
        l.set_major_formatter("d.dd")
    lat.set_axislabel("Galactic Latitude", fontsize=12)
    lon.set_axislabel("Galactic Longitude", fontsize=12)
    ref_ax.tick_params(labelsize=12)

    # spec_ax = fig.add_subplot(gs[0, 1])
    spec_ax = fig_spec.add_subplot(111)
    spec_ax.set_xlabel(f"V ({kms.to_string('latex_inline')})")
    spec_ax.set_ylabel(f"{get_data_name(line_stub)} " + "T$_{\\rm MB}$" + f" ({u.K.to_string('latex_inline')})")

    # Load regions
    reg_filename_short = "catalogs/N19_shell_edge_selfabs_2.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # Grab spectra from (0-indexed) 0, [3 or 4], and 5(circle). The rest are points.
    xaxis = cube_obj.data.spectral_axis.to_value()

    fixed_diameter = 3
    diameter_stub = "_default" # f"_{fixed_diameter}beamsacross"

    # Function to change the radius of a Circle or convert a Point to a Circle
    def _set_circle_radius(reg, beams_across):
        return reg
        radius_arcsec = beams_across * 15.5*u.arcsec / 2
        if isinstance(reg, regions.CircleSkyRegion):
            reg.radius = radius_arcsec
            # print("HAS RADIUS", reg)
        else:
            # print("HAS NO RADIUS", reg)
            reg = regions.CircleSkyRegion(center=reg.center, radius=radius_arcsec)
            # print("IS NOW", reg)
        return reg

    # First, points
    selected_points = [0, 1]
    colors = ['k', marcs_colors[2]]
    names = ["N19 shell edge", "Off position toward\nNorthern Cloud"]
    for idx, reg_idx in enumerate(selected_points):
        # Old ways
        # pixreg = reg_list[reg_idx].to_pixel(cube_obj.wcs_flat)
        # j, i = [int(round(c)) for c in pixreg.center.xy]
        # spectrum = cube_obj.data[:, i, j].to_value()
        # New ways, only circles
        # reg = _set_circle_radius(reg_list[reg_idx], fixed_diameter)
        reg = reg_list[reg_idx]
        subcube = cube_obj.data.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2))
        # print("JI", j, i)
        p = spec_ax.step(xaxis, spectrum, color=colors[idx], label=names[idx], linewidth=3, linestyle='-', where='mid', zorder=5 - idx)
        # Remake pixreg because different WCS for image
        reg.to_pixel(ref_wcs).plot(ax=ref_ax, color=p[0].get_c())
        # reg_patch = reg_list[reg_idx].to_pixel(ref_wcs).as_artist()
        # reg_patch.set(color=p[0].get_c(), mec=p[0].get_c()) # Line2D because points!!!
        # ref_ax.add_artist(reg_patch)

        #### CO
        if use_CO:
            subcube = co_cube_obj.data.subcube_from_regions([reg])
            spectrum = subcube.mean(axis=(1, 2))
            spec_ax.step(co_cube_obj.data.spectral_axis.to_value(), spectrum.to_value(), color=colors[idx], linewidth=1, where='mid')

    # Next, circle
    # circ_idx = 5
    # subcube = cube_obj.data.subcube_from_regions([reg_list[circ_idx]])
    # spectrum = subcube.mean(axis=(1, 2))
    # p = spec_ax.plot(xaxis, spectrum, label="Circle", color='k')
    # reg_list[circ_idx].to_pixel(ref_wcs).plot(ax=ref_ax, color=p[0].get_c())

    spec_ax.legend(loc='upper left')
    spec_ax.set_xlim((7, 23))

    fig_spec.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.99)


    co_stub = "_withCO" if use_CO else ""
    savename = f"{line_stub}_n19_self_absorption_spectrum{diameter_stub}{co_stub}"
    info_txt = f"{reg_filename_short} points: {selected_points} zoom: {cutout_name}"
    for fig, stub in zip([fig_ref, fig_spec], ["REF", "SPEC"]):
        fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"{savename}_{stub}.png"),
            metadata=catalog.utils.create_png_metadata(title=info_txt, file=__file__,
                func="n19_self_absorption"))


def energetics_calculations():
    """
    March 4, 2024
    Run thru all the energetics calculations in one place.
    These have been spread throughout my notes and I want to consolidate,
    particularly in case I have to redo them with different numbers (again)

    This will be a long function
    """
    txt = "Calculations"
    txt2 = '-'*len(txt) + " " + txt + " " + '-'*len(txt)
    dashes = '-' * len(txt2)
    print(dashes)
    print(txt2)
    print(dashes)
    print()

    """ Part 1: M16 shell dimensions, surface area, energy """
    def _estimate_ellipsoid_dimensions():
        """
        Estimate the dimensions of the M16 cavity
        From 2024-01-14 notes
        """
        reg_label = ['right', 'left']
        size_label = ['tall', 'wide']
        x_sky = [[26, 36], [22, 45]] * u.arcmin
        x_phys = (x_sky * 1740*u.pc / u.radian).to(u.pc)
        for i, reg in enumerate(reg_label):
            for j, dimension in enumerate(size_label):
                print(reg, dimension)
                print(x_sky[i, j])
                print(x_phys[i, j])
                print()
        print(x_sky)
        print(x_phys)
        print(f"Averages are {np.mean(x_phys, axis=0)}")
        print("Calling it 12 pc tall (total) and 20 pc wide (each)")
        semimajor, semiminor = [20, 6] * u.pc
        print(f"Which means {semimajor} semimajor and {semiminor} semiminor.")
        return semimajor, semiminor

    def _ellipsoidal_surface_area(a, b):
        """
        From 2024-01-14 notes
        Both a, b are "semi" i.e. half width
        :param a: the repeated axis (major in our case)
        :param b: the different axis (minor for ours)
        """
        # a = repeated axis
        p = 1.6075
        first = a**p * b**p
        second = a**(2*p)
        inner = (2*first + second) / 3
        return 4 * np.pi * inner**(1./p)

    m16_shell_v = 10 * kms
    nh_detection_limit = 1e21 * u.cm**-2
    m16_shell_thickness = 0.5 * u.pc
    def _m16_shell_mass_energy(a):
        """
        From 2024-01-14 notes (but that's not the final version)
        :param a: surface area
        """
        surface_mass_density = nh_detection_limit * Hmass * mean_molecular_weight_neutral
        mass = a * surface_mass_density
        mass = (mass).to(u.solMass)
        print(f"M16 shell mass {mass:.1E} ~ 10^{np.round(np.log10(mass/u.solMass))}")
        energy = 0.5 * mass * m16_shell_v**2.
        energy = energy.to(u.erg)
        print(f"M16 shell energy {energy:.1E} ~ 10^{np.round(np.log10(energy/u.erg))}")
        density = nh_detection_limit / m16_shell_thickness
        density = density.to(u.cm**-3)
        print(f"M16 shell density < {density:.1f} ~ {np.round(density, -2):.0f}")
        return mass, density

    amaj, bmin = _estimate_ellipsoid_dimensions()
    surface_area = _ellipsoidal_surface_area(amaj, bmin)
    m16_shell_mass, density = _m16_shell_mass_energy(surface_area)
    pdr_temp = 100*u.K
    m16_shell_therm_e = ((m16_shell_mass / (Hmass * mean_molecular_weight_neutral)) * (3./2) * (const.k_B * pdr_temp)).to(u.erg)
    print(f"M16 PDR thermal energy {m16_shell_therm_e:.1E} -> {m16_shell_therm_e:.0E}")


    print(dashes)
    print()

    """ Part 2: M16 shell pressures """
    p_unit = u.K * u.cm**-3
    p_conv = lambda p : (p/const.k_B).to(p_unit)
    def _m16_shell_therm_p(dens):
        therm_p = dens * pdr_temp
        print(f"M16 shell thermal pressure {therm_p:.2E} -> {therm_p:.0E}")
        return therm_p
    def _m16_shell_tot_p(therm_p):
        tot_p = 3*therm_p
        print(f"M16 shell total pressure {tot_p:.2E} -> {tot_p:.0E}")
        return tot_p
    cgs_to_gauss = (u.Gauss / (u.cm**(-1/2) * u.g**(1/2) * u.s**-1))
    def _reverse_engineer_B_field(p):
    	print(f"For pressure P = {p:.1E}, ", end='')
    	b = ((p*8*np.pi*const.k_B)**(1/2) * cgs_to_gauss).to(u.microGauss)
    	print(f"B = {b:.2f}")

    therm_p = _m16_shell_therm_p(density)
    _m16_shell_tot_p(therm_p)
    _reverse_engineer_B_field(therm_p)
    _reverse_engineer_B_field(np.round(therm_p, -int(np.log10(therm_p/p_unit))))
    print(dashes)
    print()

    """ Part 3: M16 HII pressures """
    higgs_1d_turb = 7*kms # Down from 12, which was the 3D turb vel
    hester_n = 58 * u.cm**-3
    hii_temp = 8000 * u.K
    def _m16_hii_therm_p():
        therm_p = hester_n * hii_temp
        print(f"M16 HII thermal pressure {therm_p:.2E} -> {therm_p:.1E}")
        return therm_p
    def _m16_hii_turb_p():
        particle_mass = Hmass * mean_molecular_weight_neutral / 2
        rho = hester_n * particle_mass
        turb_p = p_conv(rho * higgs_1d_turb**2)
        print(f"M16 HII turbulent pressure {turb_p:.2E} -> {turb_p:.1E}")
    _m16_hii_therm_p()
    _m16_hii_turb_p()
    print(dashes)
    print()


    """ Part 4: N19 PDR """
    print('-'*17 + " N19 " + '-'*16)
    def _generic_shell_volume(r1, r2):
        """ r1 < r2, both in distance units """
        return (4./3)*np.pi * (r2**3 - r1**3)
    def _limb_brightening_path(r1, r2):
        """ r1 < r2, both in distance units """
        return 2 * np.sqrt(r2**2 - r1**2)
    n19_shell_v = 4 * kms
    n19_cii_col = 1e22 * u.cm**-2
    n19_size = [1.8, 2.3] * u.pc
    l = _limb_brightening_path(*n19_size)
    print(f"PDR limb brightening l: {l:.1f}, l/2: {l/2:.1f}")
    l = l/2 # Half shell limb brightened path
    n19_cii_n_1 = (n19_cii_col / l).to(u.cm**-3) # atomic dens via coldens / distance
    print(f"N19 PDR shell density via N(H)/limb bright {n19_cii_n_1:.0f} -> {np.round(n19_cii_n_1, -2):.0f}")
    # 0.5 pc shell thickness, 5e21 upper limit coldens thru FG shell
    n19_cii_col_fg = 5e21 * u.cm**-2
    n19_cii_n_fg = (n19_cii_col_fg / (n19_size[1] - n19_size[0])).to(u.cm**-3)
    print(f"N19 PDR shell density via N(H) upper limit / foreground shell thickness {n19_cii_n_fg:.0f} -> {np.round(n19_cii_n_fg, -2):.0f}")
    n19_cii_n = 1500 * u.cm**-3
    n19_fg_col_estimate = (n19_cii_n * (n19_size[1] - n19_size[0])).to(u.cm**-2)
    print(f"N19 PDR shell N(H) 1500cm-3 x foreground shell thickness {n19_fg_col_estimate:2E} -> {n19_fg_col_estimate:.1E}")
    vol = _generic_shell_volume(*n19_size) / 2
    print(f"vol: {vol:.1f} -> {vol:.0f}")
    n19_cii_mass = 650 * u.solMass
    n19_cii_n_2 = (n19_cii_mass / (Hmass * mean_molecular_weight_neutral * vol)).to(u.cm**-3)
    print(f"N19 PDR shell density via mass/vol {n19_cii_n_2:.0f} -> {np.round(n19_cii_n_2, -2):.0f}")
    n19_cii_n_2 = (700*u.solMass / (Hmass * mean_molecular_weight_neutral * vol)).to(u.cm**-3)
    print(f"(if I use 700 solMass) N19 PDR shell density via mass/vol {n19_cii_n_2:.0f} -> {np.round(n19_cii_n_2, -2):.0f}")
    print(dashes)
    print()

    """ Part 4: N19 Molecular """
    n19_h2_n = 5.6e3 * u.cm**-3
    n19_h2_col = 1.1e22 * u.cm**-2
    n19_h2_mass = 4200 * u.solMass
    n19_size_mol = [2, 3] * u.pc
    l_mol = _limb_brightening_path(*n19_size_mol)
    print(f"Molecular limb brightening l: {l_mol:.1f}, l/2: {l_mol/2:.1f}")
    s_mol = (n19_h2_col / n19_h2_n).to(u.pc)
    print(f"Molecular col/n = s: {s_mol:.1f} -> {s_mol:.0f}")
    print(dashes)
    print()

    """ Part 4: N19 PDR Energetics """
    n19_cii_n = 1500 * u.cm**-3
    n19_pdr_t = 100 * u.K
    def _generic_therm_p(t, n):
        therm_p = t * n
        return therm_p, f"{therm_p:.2E} -> {therm_p:.1E}"
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    print(f"fwhm conv {fwhm_conv:.3f}")
    def _generic_turb_p(fwhm, n, mmw_mod):
        particle_mass = Hmass * mean_molecular_weight_neutral * mmw_mod
        rho = n * particle_mass
        turb_p = p_conv(rho * (fwhm/fwhm_conv)**2)
        return turb_p, f"{turb_p:.2E} -> {turb_p:.1E}"
    pdr_therm_p, txt = _generic_therm_p(n19_pdr_t, n19_cii_n)
    print(f"N19 PDR shell thermal pressure {txt}")
    pdr_turb_p, txt = _generic_turb_p(3.5*kms, n19_cii_n, 1)
    print(f"N19 PDR shell turbulent pressure {txt}")
    print("N19 PDR magnetic:")
    _reverse_engineer_B_field(pdr_turb_p)
    _reverse_engineer_B_field(np.round(pdr_turb_p, -int(np.log10(pdr_turb_p/p_unit))))
    pdr_tot_p = pdr_therm_p + 2*pdr_turb_p
    print(f"total pressure {pdr_tot_p:.2E} -> {pdr_tot_p:.1E}")
    ke_f = lambda m : (0.5 * m * n19_shell_v**2).to(u.erg)
    # n19_pdr_ke = (0.5 * n19_cii_mass * n19_shell_v**2).to(u.erg)
    n19_pdr_ke = ke_f(n19_cii_mass)
    print(f"N19 PDR kinetic energy {n19_pdr_ke:.2E} -> {n19_pdr_ke:.0E}")
    n19_pdr_therm_e = ((n19_cii_mass/(Hmass * mean_molecular_weight_neutral)) * (3./2) * (const.k_B * n19_pdr_t)).to(u.erg)
    print(f"N19 PDR thermal energy {n19_pdr_therm_e:.1E} -> {n19_pdr_therm_e:.0E}")
    print(dashes)
    print()

    """ Part 4: N19 PDR Energetics """
    h2_t = 30 * u.K
    h2_therm_p, txt = _generic_therm_p(h2_t, n19_h2_n)
    print(f"N19 H2 shell thermal pressure {txt}")
    h2_turb_p, txt = _generic_turb_p(1*kms * fwhm_conv, n19_h2_n, 2)
    print(f"N19 H2 shell turbulent pressure {txt}")
    print("N19 H2 magnetic:")
    _reverse_engineer_B_field(h2_turb_p)
    _reverse_engineer_B_field(np.round(h2_turb_p, -int(np.log10(h2_turb_p/p_unit))))
    h2_tot_p = h2_therm_p + 2*h2_turb_p
    print(f"total pressure {h2_tot_p:.2E} -> {h2_tot_p:.1E}")
    # n19_h2_ke = (0.5 * n19_h2_mass * n19_shell_v**2).to(u.erg)
    n19_h2_ke = ke_f(n19_h2_mass)
    print(f"N19 H2 kinetic energy {n19_h2_ke:.2E} -> {n19_h2_ke:.0E}")
    n19_h2_therm_e = ((n19_h2_mass/(Hmass * 2 * mean_molecular_weight_neutral)) * (const.k_B * h2_t)).to(u.erg)
    print(f"N19 H2 thermal energy {n19_h2_therm_e:.1E} -> {n19_h2_therm_e:.0E}")
    print(dashes)
    print()

    print(f"Mass ratio {n19_h2_mass}/{n19_cii_mass} = {(n19_h2_mass/n19_cii_mass).decompose()} and ke ratio {(n19_h2_ke/n19_pdr_ke).decompose()}")
    print()

    """ Part 5: N19 HII density and energy """
    n19_tot_p = 3e6 * p_unit
    def _hii_density_from_pressure(p):
        ntot = (n19_tot_p / (8000*u.K)).to(u.cm**-3)
        return ntot/2
    n19_hii_n = _hii_density_from_pressure(n19_tot_p)
    print(f"N19 HII density {n19_hii_n:.0f} -> {np.round(n19_hii_n, -int(np.log10(n19_hii_n*u.cm**3))):.0f}")
    n19_hii_vol = (2*u.pc)**3 * (4*np.pi/3)
    n19_hii_eth = (n19_tot_p*const.k_B*n19_hii_vol).to(u.erg)
    print(f"N19 HII E_therm {n19_hii_eth:.1E}")
    print(dashes)
    print()


    """ Part 6: Xray plasma """
    src_area = 886837.460929 * u.arcsec**2
    Y = 1.54 * 10**58 * u.m**-3
    em = Y * src_area.to_value()
    print(f"Y = {em:.4E} = {em.to(u.cm**-3):.1E}")

    src_area_phys = (src_area * (1740*u.pc/u.radian)**2).decompose()
    eff_circle_radius = np.sqrt(src_area_phys/np.pi)
    eff_circle_vol = (4./3) * np.pi * eff_circle_radius**3

    n = np.sqrt(em/eff_circle_vol).to(u.cm**-3)
    tkev = 0.150 * u.keV
    t = (tkev/const.k_B).to(u.K)
    p = (t*n).to(u.K * u.cm**-3)
    # First, observed volume approximation
    eth = (n*tkev*eff_circle_vol).to(u.erg)

    # Now big ellipsoid!
    a, b = [20, 6]*u.pc
    ell_vol = (4.*np.pi/3) * a**2 * b
    ell_eth = (n*tkev*ell_vol).to(u.erg)

    print(f"SRC AREA {src_area.to(u.arcmin**2):.2f}")
    print(f"SRC AREA {src_area_phys.to(u.pc**2):.2f}")
    print(f"EFF RADIUS {eff_circle_radius.to(u.pc):.2f}")
    print(f"EFF VOL {eff_circle_vol.to(u.pc**3):.0f}")
    print(f"plasma n   {n:.2f}")
    print(f"plasma T   {t:.2E}")
    print(f"plasma Pth {p:.2E} -> {p:.1E}")
    print(f"E_therm obs {eth:.2E} -> {eth:.1E}")
    print(f"BIG VOL {ell_vol.to(u.pc**3):.0f}")
    print(f"E_therm est {ell_eth:.2E} -> {ell_eth:.1E}")



def n19_time_calculation():
    """
    March 7, 2024
    Day my thesis is due!! and here i am doing analysis
    """
    beta = 2.6e-13 * u.cm**3 / u.s
    q0 = 7.43e47 / u.s
    n = [200, 300, 400, 1000] * u.cm**-3
    def _stromgren(density):
        return (((3./(4*np.pi)) * q0 / (density**2 * beta))**(1./3)).to(u.pc)
    r0 = _stromgren(n)
    if True:
        print("Stromgren")
        print([f"{x:.0f}" for x in n])
        print([f"{x:.3f}" for x in r0])

    r_shell = 2*u.pc
    cs = 10*kms
    v = 4*kms
    def _age_from_r(density):
        # recalculating R0 inside here, no need to do it outside
        r = _stromgren(density)
        coeff = (4*r)/(7*cs)
        brackets = (r_shell/r)**(7./4) - 1
        return (coeff*brackets).to(u.Myr)
    def _age_from_v(density):
        # recalculating R0 inside here, no need to do it outside
        coeff = (4*_stromgren(density))/(7*cs)
        brackets = (v/cs)**(-7./3) - 1
        return (coeff*brackets).to(u.Myr)
    def _vel_from_r(density):
        t = _age_from_r(density)
        return cs * (1 + (7*t*cs)/(4*_stromgren(density)))**(-3./7)
    if True:
        print([f"{x:.3f}" for x in _age_from_r(n)])
        print([f"{x:.3f}" for x in _age_from_v(n)])
        print([f"{x:.3f}" for x in _vel_from_r(n)])

    print("Weaver")
    lmech = 2.23e47 * u.erg/u.Myr
    l36 = (lmech/(1e36*u.erg/u.s)).decompose()
    def _w77_t_from_r(density):
        dens = density.to(u.cm**-3).to_value()
        first = (1./27)**(5./3)
        mid = (dens/l36)**(1./3)
        last = r_shell.to(u.pc).to_value()**(5./3)
        return (first*mid*last).decompose() * u.Myr
    def _w77_t_from_v(density):
        dens = density.to(u.cm**-3).to_value()
        first = 16**(5./2)
        mid = (l36/dens)**(1./2)
        last = v.to(kms).to_value()**(-5./2)
        return (first*mid*last).decompose() * u.Myr
    def _w77_v_from_r(density):
        dens = density.to(u.cm**-3).to_value()
        t = _w77_t_from_r(density).to(u.Myr).to_value()
        return 16 * (l36/(dens * t**2))**(1./5) * kms
    print([f"{x:.3f}" for x in _w77_t_from_r(n)])
    print([f"{x:.3f}" for x in _w77_t_from_v(n)])
    print([f"{x:.3f}" for x in _w77_v_from_r(n)])


def cii_channel_maps_photoseries_slides_animation():
    """
    March 26, 2024
    Defense day! going to see if I can pull this quick visualization off
    make a movie (no coordinates) of CII every 1 km/s (mom0s) between 10 and 35 or so
    """
    cube = cube_utils.CubeData(get_map_filename('cii')).convert_to_K().convert_to_kms()
    v_start, v_stop, v_step = 10, 37, 2
    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(111)
    for v in np.arange(v_start, v_stop+v_step, v_step):
        v0 = v*kms
        v1 = (v+v_step)*kms
        mom0 = cube.data.spectral_slab(v0, v1).moment0()
        img_raw = mom0.to_value()
        cutout = misc_utils.cutout2d_from_region(img_raw, mom0.wcs, get_cutout_box_filename('med'), align_with_frame='galactic')
        img = cutout.data
        ax.imshow(img, origin='lower', vmin=0, vmax=40, cmap='Greys_r')
        ax.text(0.1, 0.9, f"{v+1:2d} km / s", color='k', fontsize=25, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.1, 0.95, "[C II]", color='k', fontsize=25, ha='left', va='top', transform=ax.transAxes)
        ax.set_axis_off()
        fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.1)
        fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"anim/cii_mom0_{v:02d}.png"))
        ax.clear()







"""
Bubble diagram stuff
"""

def bubble_geometry():
    """
    January 31, 2024
    try visualizing a biconcave disc with matplotlib 3d
    """
    import mpl_toolkits.mplot3d.art3d as art3d

    def test_z(r):
        # r is radius from 0
        return np.cos(np.pi*r/2)

    diameter = 2
    a = (.05, 2, 0) # a0, a1, a2
    def biconcave_z(r):
        # r is radius from 0
        rd2 = (r/diameter)**2 # appears multiple times
        first = diameter * np.sqrt(1 - 4*rd2)
        second = a[0] + a[1]*rd2 + a[2]*(rd2**2)
        return first*second

    # axis_array = np.linspace(-1.2, 1.2, 200)
    # xx, yy = np.meshgrid(axis_array, axis_array, indexing='xy')
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xx = np.outer(np.cos(u), np.sin(v))
    yy = np.outer(np.sin(u), np.sin(v))
    rr = np.sqrt(xx**2 + yy**2)
    # rr[rr>1] = np.nan
    zz = biconcave_z(rr) * np.sign(np.cos(v))

    # zz[xx < -0.8] = np.nan

    # fig = plt.figure(figsize=(13, 6)) # 12, 10 for 3 panels
    fig1 = plt.figure(figsize=(8, 6)) # 12, 10 for 3 panels
    fig2 = plt.figure(figsize=(8, 5)) # 12, 10 for 3 panels
    figs = [fig1, fig2]

    def _plot3d(ax_description, wire=False, alpha=1):
        # ax = fig.add_subplot(ax_description, projection='3d', computed_zorder=False)
        ax = figs[ax_description%10 - 1].add_subplot(111, projection='3d', computed_zorder=False)
        if wire:
            f = ax.plot_wireframe
        else:
            f = ax.plot_surface
        f(xx, yy, zz, alpha=alpha, color="SeaGreen", zorder=50)
        ax.set_aspect("equal")
        return ax

    zz[(yy < -0.8) & (zz < 0.15)] = np.nan
    ax1 = _plot3d(121)
    ax2 = _plot3d(122, wire=True, alpha=0.7)
    # ax3 = _plot3d(224, alpha=0.7)
    axes = [ax1, ax2]

    """
    publication_ready = 2 is good for the paper
    publication_ready = 3 is good for the presentation (took out some of the pink boxes)
    """
    publication_ready = 3
    """ """
    if publication_ready == 0:
        fig.subplots_adjust(hspace=-0.5, wspace=-0.05, top=1.2, bottom=-0.2, left=-0.1, right=1.1)
    elif publication_ready == 1:
        fig.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
    elif publication_ready >= 2:
        fig1.subplots_adjust(top=1.05, bottom=-0.05, left=-0.25, right=1.2) # solid surface
        fig2.subplots_adjust(top=1.25, bottom=-0.45, left=-0.65, right=1.6) # wireframe with NC and N19
    # ax2 = fig.add_subplot(122)
    # ax2.imshow(rr, origin='lower')

    for j, ax in enumerate(axes):
        for i_axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            # i_axis.set_ticklabels([])
            i_axis.set_tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
            for s in ['inward_factor', 'outward_factor']:
                i_axis._axinfo['tick'][s] = 0.0
        ax.patch.set_alpha(0)
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        ax.set_zlim((-0.45, 0.45))
        """ Plot the filament """
        l = 1.2
        lw = 3
        if False: # turn it off, zorder gets messed up with the diagram frames
            if j < 1:
                ax.plot([0, 0], [0, 0], [-l, -biconcave_z(0)], color='k', linewidth=lw, zorder=0)
                ax.plot([0, 0], [0, 0], [biconcave_z(0), l], color='k', linewidth=lw, zorder=100)
            else:
                ax.plot([0, 0], [0, 0], [-l, -biconcave_z(0)], color='k', linewidth=lw, zorder=0)
                ax.plot([0, 0], [0, 0], [biconcave_z(0), l], color='k', linewidth=lw, zorder=0)


    """ Plot the diagram crosscut locations """
    def _plot_square(ax, line_kwargs, direction="pos", dz=0.5, dx=1, loc=0, zorders=[100, 100, 0, 0]):
        dz = 0.6
        dx = 1
        if direction == 'pos':
            flip_arg = False
        elif direction == 'los':
            flip_arg = True
        loc_arr = [loc, loc]
        _flip_arg_f = lambda x : (loc_arr, x) if flip_arg else (x, loc_arr)
        ax.plot(*_flip_arg_f([-dx, dx]), [dz, dz], zorder=zorders[0], **line_kwargs) # top
        ax.plot(*_flip_arg_f([dx, dx]), [dz, -dz], zorder=zorders[1], **line_kwargs) # right
        ax.plot(*_flip_arg_f([-dx, -dx]), [-dz, dz], zorder=zorders[2], **line_kwargs) # left
        ax.plot(*_flip_arg_f([dx, -dx]), [-dz, -dz], zorder=zorders[3], **line_kwargs) # bottom

    def _plot_square_split_y(ax, split_y_loc, line_kwargs, dz=0.5, dx=1, loc=0):
        # Only for LOS squares. Fixes Zorder when something crosses at a y loc
        loc_arr = [loc, loc]
        _flip_arg_f = lambda x : (loc_arr, x)
        assert -dx < split_y_loc < dx # here, x = y. Confusing, yes. Correct, yes. I'm copying code from above
        zorders = [100, 0]
        ax.plot(*_flip_arg_f([-dx, split_y_loc]), [dz, dz], zorder=zorders[0], **line_kwargs) # top
        ax.plot(*_flip_arg_f([split_y_loc, dx]), [dz, dz], zorder=zorders[1], **line_kwargs) # top
        ax.plot(*_flip_arg_f([dx, dx]), [dz, -dz], zorder=zorders[1], **line_kwargs) # right
        ax.plot(*_flip_arg_f([-dx, -dx]), [-dz, dz], zorder=zorders[0], **line_kwargs) # left
        ax.plot(*_flip_arg_f([dx, split_y_loc]), [-dz, -dz], zorder=zorders[1], **line_kwargs) # bottom
        ax.plot(*_flip_arg_f([split_y_loc, -dx]), [-dz, -dz], zorder=zorders[0], **line_kwargs) # bottom

    # N19 ypos
    n19_y = -0.6

    # On-sky diagram
    line_kwargs_dict = dict(color='Orchid', linewidth=3)
    if publication_ready < 3:
        _plot_square(ax1, line_kwargs_dict, direction='pos')
    # _plot_square(ax2, line_kwargs_dict, direction='pos')
    # line_kwargs_dict = dict(color='LimeGreen', linewidth=3)
    if publication_ready < 3:
        _plot_square(ax1, line_kwargs_dict, direction='los', dx=1.1, zorders=[100, 0, 100, 0], loc=-0.1)
    _plot_square(ax1, line_kwargs_dict, direction='los', dx=1.1, zorders=[100, 0, 100, 0], loc=0.3)
    _plot_square_split_y(ax2, n19_y, line_kwargs_dict, dx=1.1, loc=-0.1)
    if publication_ready < 3:
        _plot_square_split_y(ax2, n19_y, line_kwargs_dict, dx=1.1, loc=0.35)

    if publication_ready < 3:
        ax1.text(0.8, 0, 0.6+0.02, "1", color=line_kwargs_dict['color'], zorder=101)
        ax1.text(-0.1-0.04, -0.7, 0.6+0.03, "2", color=line_kwargs_dict['color'], zorder=101)
        ax1.text(0.3-0.04, -0.7, 0.6+0.03, "3", color=line_kwargs_dict['color'], zorder=101)
        ax2.text(-0.1+0.04, -0.7, 0.6+0.03, "2", color=line_kwargs_dict['color'], zorder=101)
        ax2.text(0.3+0.04, -0.7, 0.6+0.03, "3", color=line_kwargs_dict['color'], zorder=101)


    """ Try plotting Northern Cloud and N19 """
    # Northern Cloud
    u, v = np.mgrid[0:10, 0:10]
    u = u*2.7
    v = v*1.6
    uv = np.array([u, v]).reshape(2, v.size)
    northerncloud_rot = 47
    theta = np.deg2rad(northerncloud_rot)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x, y = rot_matrix.dot(uv).reshape(2, *v.shape)*0.6/10
    x = x-0.66
    y = y-0.55
    z = np.ones_like(y)*n19_y
    northerncloud_color = "SaddleBrown"
    ax2.plot_surface(x, z, y, zorder=90, alpha=0.4, color=northerncloud_color)
    # N19 circle
    circ_n19 = mpatches.Circle((-0.1, 0.3), 0.15, color='Moccasin', alpha=0.4, zorder=90)
    ax2.add_patch(circ_n19)
    art3d.pathpatch_2d_to_3d(circ_n19, z=n19_y, zdir='y')
    ax2.text(-0.7, n19_y+0.01, 0.6, "Northern Cloud", (np.cos(np.deg2rad(northerncloud_rot)), 0, np.sin(np.deg2rad(northerncloud_rot))), color=northerncloud_color, fontsize=13, ha='left', va='bottom', zorder=101)
    ax2.text(-0.04, n19_y+0.01, 0.3, "N19", color='b', fontsize=13, ha='center', va='center', zorder=101)


    # ax2.view_init(elev=7, azim=270)
    ax2.view_init(elev=4, azim=273)
    # ax3.view_init(elev=2, azim=270)

    ax1.yaxis.set_label_text("Observer's Line of Sight")
    ax1.zaxis.set_label_text("Parallel to filament (Gal. $b$)")
    ax1.xaxis.set_label_text("Perpendicular to filament (Gal. $l$)")

    ax2.yaxis.set_label_text("LOS")
    # ax2.zaxis.set_label_text("Approx. Gal $b$")
    # ax2.xaxis.set_label_text("Approx. Gal $l$")
    ax2.zaxis.set_label_text("Parallel to filament (Gal. $b$)")
    ax2.xaxis.set_label_text("Perpendicular to filament (Gal. $l$)")


    for i in range(2):
        if publication_ready < 3:
            figs[i].savefig(os.path.join(catalog.utils.todays_image_folder(), f"biconcave_disc_panel_{chr(i+65)}.pdf"),)
                # metadata=catalog.utils.create_png_metadata(title=f"biconcave D={diameter} a = {a}",
                #     file=__file__, func="bubble_geometry"))
            print("SAVED AS PDF")
        else:
            figs[i].savefig(os.path.join(catalog.utils.todays_image_folder(), f"biconcave_disc_panel_{chr(i+65)}_{publication_ready}.png"),
                metadata=catalog.utils.create_png_metadata(title=f"biconcave D={diameter} a = {a}",
                    file=__file__, func="bubble_geometry"))
            print("SAVED AS PNG")


def bubble_cross_cut(**kwargs):
    """
    February 1, 2024
    Try to make the shell cross cut diagram in matplotlib with fill-between
    """

    # Editing hatch thickness
    matplotlib.rcParams['hatch.linewidth'] = 6

    def biconcave_z(r, diameter, a):
        """
        The 3d viz uses:
        diameter = 2
        a = (.05, 2, 0) # a0, a1, a2

        r is radius from 0
        """
        rd2 = (r/diameter)**2 # appears multiple times
        first = diameter * np.sqrt(1 - 4*rd2)
        second = a[0] + a[1]*rd2 + a[2]*(rd2**2)
        return first*second

    def make_concentric_biconcave(diameter, a):
        x0 = np.linspace(-diameter/2, diameter/2, 100)
        x = np.concatenate([x0, x0[::-1]])
        z0 = biconcave_z(x0, diameter, a)
        z = np.concatenate([z0, -z0[::-1]])
        return x, z

    # molecular
    x1, z1 = make_concentric_biconcave(2, (0.1, 2, 0))
    # atomic
    x2, z2 = make_concentric_biconcave(1.95, (0.09, 2, 1))
    z2 *= 0.9
    # HII
    x3, z3 = make_concentric_biconcave(1.9, (0.08, 2, 2))
    z3 *= 0.8
    # plasma
    x4, z4 = make_concentric_biconcave(1.8, (0.07, 2, 2))
    z4 *= 0.7

    def _filter(x, z, n):
        z[(x > n) | (x < -n)] = np.nan

    def _filter2(x, z, n, n2):
        z[((x > n) | (x < -n)) & (z < n2)] = np.nan

    def _arrow_dxdy(length, angle_deg):
        """ Return dx, dy from length, angle. Angle in float degrees """
        dx = length * np.cos(np.deg2rad(angle_deg))
        dy = length * np.sin(np.deg2rad(angle_deg))
        return dx, dy

    general_arrow_kwargs = dict(zorder=10, width=0.008, head_width=0.05)

    switch_color = True
    color_h2 = marcs_colors[0]
    color_h2_preex = "SlateGray"
    color_h2_preex_alt = "#282828"
    color_h2_preex_n19 = "SaddleBrown"
    color_pdr = marcs_colors[2]
    color_hii = "Moccasin" if switch_color else "PapayaWhip"
    color_hii_alt = "Goldenrod"
    color_plasma = "AliceBlue" if switch_color else "Lavender"
    color_plasma_alt = "MediumPurple"
    color_star = "Turquoise"

    """ """
    select = kwargs.get("select", 2)
    # Select == 3 now means "broken open with labels" because I need it for the presentation
    """ """

    if select == 2:
        fig = plt.figure(figsize=(14, 8))
    else:
        fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot()

    ax.set_xlim((-1.15, 1.15))
    ax.set_ylim((-0.53, 0.53))
    ax.axvspan(-0.5, 0.5, color=color_h2_preex, alpha=0.8, zorder=0)
    ax.axvspan(-0.1, 0.1, color=color_h2_preex, alpha=1, zorder=0)

    if select == 0 or select >= 3:
        """ Gas Phase Labels """
        phases = [
            ("Molecular Gas Shell (T $\sim 30$ K)", color_h2, (-0.85, 0.5), ('left', 0.02, 'normal')),
            ("Atomic PDR Shell (T $\sim 100$ K)", color_pdr, (-0.55, 0.36), ('left', 0.32, 'normal')),
            ("Photoionized H II (T $\sim 10^4$ K)", color_hii_alt, (0.79, 0.378), ('right', 0.98, 'normal')),
            ("Shocked Wind Plasma\n(T $\sim 10^6$ K)", color_plasma_alt, (0.8, 0.0), ('right', None, 'normal')),
            ("Pre-existing Molecular\nFilament (T $\lesssim 30$ K)", color_h2_preex_alt, (0.0, 0.4), ('center', None, 'normal'))]
        def annotate_phase_arrow(index):
            ptext, pcolor, xy, (ha, tx, fw) = phases[index]
            ax.annotate(ptext, xy=xy, xycoords='data', xytext=(tx, 0.96), textcoords='figure fraction', arrowprops=dict(facecolor=pcolor), color=pcolor, fontsize=16, fontweight=fw, ha=ha, va='bottom', zorder=11)
        def annote_phase_inplace(index):
            ptext, pcolor, xy, (ha, tx, fw) = phases[index]
            ax.text(*xy, ptext, color=pcolor, fontsize=16, fontweight=fw, ha=ha, va='center', zorder=11)

        for i in range(3):
            annotate_phase_arrow(i)
        for i in range(3, 5):
            annote_phase_inplace(i)
        ax.text(0.007, 0.06, "NGC 6611", ha='center', va='center', color='k', fontsize=13, fontweight='bold', zorder=13)
        # for i, p in enumerate(phases):
        #     ax.text(0.12, 0.625 - 0.05*i, p, color=phases[p], fontsize=16, transform=ax.transAxes, zorder=11, ha='left', va='center')


    if select == 0:
        # Compass
        if False:
            # Compass in data coords
            ax.arrow(1.05, -0.48, *_arrow_dxdy(0.1, 90), width=0.005, color='k', zorder=11)
            ax.arrow(1.05, -0.48, *_arrow_dxdy(0.1, 180), width=0.005, color='k', zorder=11)
            # ax.text(1.05, -0.48+0.15, 'g lat', color='k', fontsize=12, zorder=11, ha='center', va='center')
            # ax.text(1.05-0.15, -0.48, 'g lon', color='k', fontsize=12, zorder=11, ha='center', va='center')
            ax.text(1.05+0.04, -0.48+0.12, 'g lat', color='k', fontsize=12, zorder=11, ha='center', va='center')
            ax.text(1.05-0.1, -0.48+0.02, 'g lon', color='k', fontsize=12, zorder=11, ha='center', va='center')
        else:
            # Compass in figure inches coords
            fx, fy = fig.get_size_inches()
            compass_pad = 0.3 # inches
            compass_len = 0.7
            compass_text_pad = 0.12
            up_arrow = mpatches.Arrow(fx - compass_pad, compass_pad, *_arrow_dxdy(compass_len, 90), width=0.1, color='k', zorder=11, transform=fig.dpi_scale_trans)
            left_arrow = mpatches.Arrow(fx - compass_pad, compass_pad, *_arrow_dxdy(compass_len, 180), width=0.1, color='k', zorder=11, transform=fig.dpi_scale_trans)
            fig.add_artist(up_arrow)
            fig.add_artist(left_arrow)
            fig.text(fx - compass_pad, compass_pad + compass_len + compass_text_pad, 'g lat', color='k', fontsize=12, zorder=11, ha='center', va='center', transform=fig.dpi_scale_trans)
            fig.text(fx - compass_pad - compass_len + compass_text_pad, compass_pad + compass_text_pad, 'g lon', color='k', fontsize=12, zorder=11, ha='center', va='center', transform=fig.dpi_scale_trans)




    elif select == 1 or select >= 3:
        _filter(x1, z1, 1.6/2)
        # _filter(x2, z2, 1.95/2)
        # _filter(x3, z3, 1.95/2)

        _filter2(x1, z1, 1.05/2, 0)
        _filter2(x2, z2, 1.1/2, 0)
        _filter2(x3, z3, 1.2/2, 0)

        # _filter(x4, z4, -1.9)

        arrow_kwargs = dict(color=color_plasma_alt, **general_arrow_kwargs)
        for xsign in (-1, 1):
            ax.arrow(-0.87*xsign, -0.1, *_arrow_dxdy(0.1, 270-(xsign*75)), **arrow_kwargs)
            ax.arrow(-0.84*xsign, -0.24, *_arrow_dxdy(0.1, 270-(xsign*50)), **arrow_kwargs)
            ax.arrow(-0.69*xsign, -0.31, *_arrow_dxdy(0.1, 270-(xsign*20)), **arrow_kwargs)

        if select == 1 or select > 3:
            los_arrow_kwargs = dict(**general_arrow_kwargs)
            los_arrow_kwargs["width"] = los_arrow_kwargs["width"]*0.7
            los_arrow_kwargs["head_width"] = los_arrow_kwargs["width"]*4
            ax.arrow(-1.08, 0.3, 2.16, 0, **los_arrow_kwargs, color='k')

            if select == 1 or select > 3.1:
                los_crossing_positions = [-0.92, -0.47, 0.47, 0.92]
                for i, lcp in enumerate(los_crossing_positions):
                    if select == 1: # for the paper
                        ax.add_patch(mpatches.Circle((lcp, 0.3), 0.07, edgecolor='k', linestyle='--', linewidth=2, facecolor='none', zorder=11))
                    else:
                        ax.add_patch(mpatches.Circle((lcp, 0.3), 0.07, edgecolor='k', linestyle='--', linewidth=2, facecolor=['b', 'g', 'g', 'r'][i], alpha=0.5, zorder=11))
                    ax.text(lcp-0.02, 0.32, f"{i+1}", color='k', fontsize=16, fontweight='bold', ha='center', va='center', zorder=11)

    elif select == 2:
        # _filter(x1, z1, 1/2)
        # _filter(x2, z2, 1.05/2)
        # _filter(x3, z3, 1.1/2)
        # _filter(x4, z4, 1.1/2)
        # ax.set_xlim((-1.5, 0.1))
        # ax.set_xlim((-0.4, 0.05))
        ax.set_xlim((-0.533, 0.183))
        ax.set_ylim((-0.205, 0.205))
        cr = 0.05 # "Center radius" except  none of these numbers actually use it unmodified.
        # cpos = (-1.1, 0.3)
        cpos = (-0.28, 0.08)

        r_white = mpatches.Rectangle((cpos[0], cpos[1]-cr*1.5), -cr*20, cr*20, zorder=5, color='w')
        # c1 = mpatches.Circle(cpos, cr*1.7, color=color_h2, zorder=6)
        # Use wedges to show partial rings
        # Bottom wedge first. Use the old rectangle displacement (x = -cr, +cr) to find wedge theta limits
        # Get relative theta to the horizontal (need to modify it more). use degrees, it's what wedge uses
        theta_relative = np.rad2deg(np.arccos(1/1.7))*0.9 # cr / cr*1.7, where cr*1.7 is the outer radius of the wedge. Decrement it by a little so that it wraps around the rectangle properly
        # Bottom molecular gas wedge
        w1_lo = mpatches.Wedge(cpos, cr*1.7, theta1=(180+theta_relative), theta2=(360-theta_relative), color=color_h2, zorder=7)
        # Top molecular gas wedge
        w1_hi = mpatches.Wedge(cpos, cr*1.7, theta1=theta_relative, theta2=(180-theta_relative), color=color_h2, zorder=7)
        # Northern Cloud
        r1 = mpatches.Rectangle((cpos[0]-cr, cpos[1]+cr*1.3), cr*2, cr*3, color=color_h2_preex_n19, alpha=1, zorder=6)
        ax.add_patch(r_white)
        ax.add_patch(r1)
        ax.add_patch(w1_lo)
        ax.add_patch(w1_hi)

        # r_h2_compressed = mpatches.Rectangle((cpos[0]+cr*0.27, cpos[1]-cr*1.6), cr*2, -cr, angle=45, color=color_h2, zorder=8)
        # ax.add_patch(r_h2_compressed)

        # r2_w = mpatches.Rectangle((cpos[0] - cr*2, cpos[1]-2*cr), cr, cr*5, color='w', zorder=8)
        # c2 = mpatches.Circle(cpos, cr*1.5, color=color_pdr, zorder=9)
        # Partial
        if False:
            w2 = mpatches.Wedge(cpos, cr*1.5, color=color_pdr, theta1=0, theta2=360, zorder=9)
        else:
            w2 = mpatches.Wedge(cpos, cr*1.5, color=color_pdr, theta1=theta_relative, theta2=(360-theta_relative), zorder=9)
        w2_1 = mpatches.Wedge(cpos, cr*1.9, color=color_pdr, theta1=(180+(theta_relative*1)), theta2=(360-(theta_relative*0.9)), zorder=6)
        ax.add_patch(w2)
        ax.add_patch(w2_1)
        # c3 = mpatches.Circle(cpos, cr*1.3, color=color_hii, zorder=10)
        if False:
            w3 = mpatches.Wedge(cpos, cr*1.3, color=color_hii, theta1=theta_relative, theta2=(360-theta_relative), zorder=10)
            ax.add_patch(w3)
        elif False:
            w3 = mpatches.Wedge(cpos, cr*1.3, color=color_hii, theta1=0, theta2=360, zorder=10)
            ax.add_patch(w3)
        else:
            # Partial circle, opening to NGC 6611
            w3 = mpatches.Wedge(cpos, cr*1.3, color=color_hii, theta1=theta_relative, theta2=(360-(theta_relative*0.8)), zorder=10)
            # Wedge at the bottom
            w4 = mpatches.Wedge(cpos, cr*2.1, color=color_hii, theta1=(180+(theta_relative*1)), theta2=(360-(theta_relative*0.8)), zorder=5)
            # Hatched side towards 6611
            w5 = mpatches.Wedge(cpos, cr*1.3, ec=color_hii, fill=False, linewidth=0, theta2=theta_relative, theta1=(360-(theta_relative*0.8)), hatch="-", zorder=10)
            w6 = mpatches.Wedge(cpos, cr*1.5, ec=color_pdr, fill=False, linewidth=0, theta2=theta_relative, theta1=(360-(theta_relative*0.9)), hatch="-", zorder=9)
            ax.add_patch(w3)
            ax.add_patch(w4)
            ax.add_patch(w5)
            ax.add_patch(w6)
        c4 = mpatches.Circle(cpos, cr*0.9, color=color_plasma, zorder=12)
        ax.add_patch(c4)



        """ Text labels """
        # W584 and N19
        ax.text(cpos[0] + cr*0.1, cpos[1] + cr*0.15, "W584 (O9 V)", ha='center', va='center', color='k', fontsize=13, zorder=13)
        ax.text(cpos[0] + cr*0.0, cpos[1] - cr*0.5, "N19", ha='center', va='center', color='k', fontsize=17, fontweight='bold', zorder=13)
        # NGC 6611
        ax.text(0.007, 0.04, "NGC 6611", ha='center', va='center', color='k', fontsize=17, fontweight='bold', zorder=13)
        # Northern Cloud
        ax.text(cpos[0] + cr*0.0, cpos[1] + cr*2, "Northern Cloud", ha='center', va='center', color='w', fontsize=13, zorder=13)
        # Filament
        # ax.text(0, 0.17, "Filament", ha='center', va='center', color='k', fontsize=13, zorder=13)

        # One star
        ax.scatter([cpos[0]], [cpos[1]], s=90, marker='*', facecolor=color_star, edgecolor='k', zorder=13)

        """ Scale bar with "broken" thing """
        # px implies "point x", idk, i just need to use something different than x1 because I have arrays named that
        px0 = cpos[0]
        px1 = 0
        xlen = px1-px0
        xlo, xhi = px0 + xlen*0.4, px0 + xlen*0.6
        y = [-cpos[1]]*2
        scalebar_kwargs = dict(color='k', linewidth=2, zorder=13)
        ax.plot([px0, xlo], y, **scalebar_kwargs)
        ax.plot([xhi, px1], y, **scalebar_kwargs)
        dy = 0.01
        for x in [px0, px1]:
            ax.plot([x, x], [y[0]-dy, y[0]+dy], **scalebar_kwargs)
        dx1 = 0.004 # put slashes at an angle
        dx2 = 0.0005 # correction for a visual bug
        dy = 0.006
        for x in [xlo, xhi]:
            ax.plot([x-dx1+dx2, x+dx1+dx2], [y[0]-dy, y[0]+dy], **scalebar_kwargs)
        ax.text(px0 + 0.5*xlen, y[0], "d = ?", color='k', fontsize=14, fontweight='bold', ha='center', va='center', zorder=13)


    if select > 0:
        # fx, fy = fig.get_size_inches()
        los_line_pad = 0.3 # inches
        los_line_len = 1.5
        los_line_text_pad = 0.12
        los_line_arrow = mpatches.Arrow(los_line_pad, los_line_pad, los_line_len, 0, width=0.15, color='k', zorder=11, transform=fig.dpi_scale_trans)
        fig.add_artist(los_line_arrow)
        fig.text(los_line_pad, los_line_pad + los_line_text_pad, 'Observer LOS', color='k', fontsize=12, zorder=11, ha='left', va='center', transform=fig.dpi_scale_trans)


    ax.fill_between(x1, z1, color=color_h2, zorder=1)
    ax.fill_between(x2, z2, color=color_pdr, zorder=2)
    ax.fill_between(x3, z3, color=color_hii, zorder=3)
    ax.fill_between(x4, z4, color=color_plasma, zorder=4)

    # stars
    rng = np.random.default_rng(seed=77544)
    star_x = rng.uniform(low=-0.04, high=0.04, size=9)
    star_y = rng.uniform(low=-0.04, high=0.04, size=9)
    star_s = (1 - rng.power(2, size=9))*300 + 75
    ax.scatter(star_x, star_y, s=star_s, marker='*', facecolor=color_star, edgecolor='k', zorder=13)

    if select <= 2:
        # Add "Frame label" from bubble_geometry(), the pink frames
        frame_number = [1, 3, 2][select]
        fig.text(0.02, 0.9, str(frame_number), color='Orchid', fontsize=21, ha='left', va='center')

    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    if select >= 3:
        plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"biconcave_disc_crosscut_{select}.png"),
            metadata=catalog.utils.create_png_metadata(title="largest shell D=2 a=(0.1, 2, 0)",
                file=__file__, func="bubble_cross_cut"))
        print("SAVED AS PNG")
    else:
        plt.savefig(os.path.join(catalog.utils.todays_image_folder(), f"biconcave_disc_crosscut_{select}.pdf"),)
            # metadata=catalog.utils.create_png_metadata(title="largest shell D=2 a=(0.1, 2, 0)",
            #     file=__file__, func="bubble_cross_cut"))
        print("SAVED AS PDF")


def fake_bubble_spectra(setting=1):
    """
    Feb 4, 2024
    Demonstrate the spectra through the bubble_cross_cut() diagram
    Put next to the real CII spectra from the western cavity, like
    m16_pictures_2.m16_expanding_shell_spectra.
    Those regions are in catalogs/m16_west_cavity_spec_regions.reg
    setting: 1 for paper, 2 for presentation
    """
    reg_filename_short = "catalogs/m16_west_cavity_spec_regions.reg"
    savename_stub = "west_cavity_3am_circles"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))

    # Fig
    fig = plt.figure(figsize=(16, 9))
    # Axes
    gs = fig.add_gridspec(1, 2)
    img_axes = []
    # Load cube
    line_stub = "cii"
    fn = get_map_filename(line_stub)
    cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()
    # Reference contour moment image
    if True:
        # Model spectra
        gmid_1 = cps2.models.Gaussian1D()
        gmid_2 = cps2.models.Gaussian1D()
        x = np.linspace(-8, 8, 100)
        ymid_1 = gmid_1(x)
        ymid_2 = gmid_2(x)
        gr = cps2.models.Gaussian1D(mean=4)
        gb = cps2.models.Gaussian1D(mean=-4)
        yr = gr(x)
        yb = gb(x)
        ytot = ymid_1 + ymid_2 + yr + yb
        dy = 1.5
        dy2 = 0.05
        model_ax = fig.add_subplot(gs[0, 0])
        model_ax.plot(x, yb + dy*-2, color='b', linestyle=':')
        model_ax.plot(x, ymid_1 + dy*-1, color='green', linestyle=':')
        model_ax.plot(x, ymid_2 + dy*1, color='green', linestyle=':')
        model_ax.plot(x, yr + dy*2, color='r', linestyle=':')
        model_ax.plot(x, ytot-dy*0.3, color='k', linestyle='-', linewidth=2)
        model_ax.spines[['top', 'right', 'left']].set_visible(False)
        model_ax.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False)
        model_ax.set_xticks([-7, 7], labels=["Low velocity", "High velocity"])

        names = ["Blueshifted shell", "Filament", "Filament", "Redshifted shell"]
        x_positions = [7, 7, -7, -7]
        name_colors = ['b', 'g', 'g', 'r']
        for i, j in enumerate([-2, -1, 1, 2]):
            model_ax.text(-8.3, dy*j, f"{i+1}", fontsize=15, color='k', fontweight='bold', ha='right', va='center')
            model_ax.text(x_positions[i], dy*j+dy2, names[i], fontsize=15, color=name_colors[i], ha=('left' if i > 1 else 'right'), va='bottom')
        model_ax.text(-8.3, -dy*0.3, "Sum", fontsize=15, color='k', fontweight='bold', ha='right', va='center')


    def _norm_spectrum(spec):
        return spec/np.nanmax(spec)

    if True:
        # Observed spectra
        spec_ax = fig.add_subplot(gs[0, 1])

        # Add the averaged spectrum from both circles
        if False:
            # Decided not to use this, it's not as good as North circle
            subcube = cube_obj.data.subcube_from_regions(reg_list)
            spectrum = subcube.mean(axis=(1, 2))
            spec_ax.plot(subcube.spectral_axis.to_value(), _norm_spectrum(spectrum.to_value()), color='k', linewidth=3, label='Both circles')

        reg_labels = ["South circle", "Western cavity circle"]
        if False:
            # Skip doing both, only do the north circle
            for j, reg in enumerate(reg_list):
                subcube = cube_obj.data.subcube_from_regions([reg])
                spectrum = subcube.mean(axis=(1, 2))
                spec_ax.plot(subcube.spectral_axis.to_value(), _norm_spectrum(spectrum.to_value()), linewidth=1, linestyle=':', color=marcs_colors[1-j], label=reg_labels[j])

        # North Circle spectrum only
        north_circle_idx = 1
        subcube = cube_obj.data.subcube_from_regions([reg_list[north_circle_idx]])
        spectrum = subcube.mean(axis=(1, 2))
        spec_ax.plot(subcube.spectral_axis.to_value(), _norm_spectrum(spectrum.to_value()), linewidth=3, linestyle='-', color='k', label=reg_labels[north_circle_idx])

        # Get full avg spec above 6 K
        if setting == 0:
            # Marc says this doesn't add anything, they already know the emission is at 26 km/s
            spectrum = cube_obj.data.mean(axis=(1, 2))
            coeff = 3
            spec_ax.plot(subcube.spectral_axis.to_value(), _norm_spectrum(spectrum.to_value()), color="grey", linewidth=3, linestyle='-.', label=f"Entire field average")

            # Grab the blueshifted clump spectra
            reg_filename_short_bc = "catalogs/m16_points_blueshifted_clump.reg"
            reg_list_bc = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short_bc))

            def _plot_blue_clump_spec(cube_object):
                pixreg = reg_list_bc[1].to_pixel(cube_object.wcs_flat)
                j, i = [int(round(c)) for c in pixreg.center.xy]
                spectrum = cube_object.data[:, i, j]
                return cube_object.data.spectral_axis.to_value(), spectrum.to_value()

            coeff = 4
            if False:
                co_cube_obj = cube_utils.CubeData(get_map_filename("12co32")).convert_to_K().convert_to_kms()
                x, y = _plot_blue_clump_spec(co_cube_obj)
                green_mask = (x > 16) & (x < 23)
                y[green_mask] = np.nan
                spec_ax.plot(x, _norm_spectrum(y), color=marcs_colors[0], linestyle='-', linewidth=1.5, label=f'CO, blue clump')

            x, y = _plot_blue_clump_spec(cube_obj)
            spec_ax.plot(x, _norm_spectrum(y), color=marcs_colors[0], linestyle='-', linewidth=1.5, label=f'Blueshifted clump')

        spec_ax.spines[['top', 'right']].set_visible(False)
        spec_ax.axhline(0, color='grey', linestyle="--", alpha=0.2)
        spec_ax.set_xlabel("V$_{\\rm LSR}$ " + f"({kms.to_string('latex_inline')})")
        spec_ax.set_ylabel(f"Normalized {get_data_name(line_stub)} line intensity")
        spec_ax.set_xlim((-4, 49))
        spec_ax.set_ylim((-0.4, 1.1))
        if setting == 0:
            spec_ax.legend(loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.1, left=0.05, top=0.95, right=0.95)
    if setting == 0:
        fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"expanding_shell_diagram_spectrum.pdf"),)
            # metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short} {reg_filename_short_bc}", file=__file__, func="fake_bubble_spectra"))
        print("SAVED AS PDF")
    elif setting == 1:
        fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"expanding_shell_diagram_spectrum.png"),
            metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short}", file=__file__, func="fake_bubble_spectra"))
        print("SAVED AS PNG")

def bubble_simulated_projection():
    """
    Feb 24, 2024
    Not a good use of time but I want to know
    Quickly
    """

    diameter_1 = 2
    diameter_2 = 1.75
    a_1 = (.1, 2, 0) # a0, a1, a2
    a_2 = (.09, 1.6, 1) # a0, a1, a2
    height_mod = 0.98
    def biconcave_z(r, diameter, a):
        # r is radius from 0
        rd2 = (r/diameter)**2 # appears multiple times
        first = diameter * np.sqrt(1 - 4*rd2)
        second = a[0] + a[1]*rd2 + a[2]*(rd2**2)
        return first*second

    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xx = np.outer(np.cos(u), np.sin(v))
    yy = np.outer(np.sin(u), np.sin(v))

    # Use the simple grid approach
    axis_array = np.linspace(-1.1, 1.1, 60) # 30
    # 3D grids
    xx, yy = np.meshgrid(axis_array, axis_array, indexing='xy')
    # Get cylindrical r coord on xy plane
    rr = np.sqrt(xx**2 + yy**2)
    # Find the z heights of 2 biconcave discs as function of r plane
    # Reshape everything to 3 dimensions but with empty z (0) axis
    z_1 = biconcave_z(rr, diameter_1, a_1)[np.newaxis, :, :]
    z_2 = biconcave_z(rr, diameter_2, a_2)[np.newaxis, :, :] * height_mod # make it even smaller
    # Add an r plane
    r = rr[np.newaxis, :, :]
    # Reshape the axis_array
    z = axis_array[:, np.newaxis, np.newaxis]
    # Add an x axis array
    x = axis_array[np.newaxis, :, np.newaxis]

    # Make 3D z array (basically a meshgrid component)
    # zzz = np.zeros((axis_array.size,)*3)
    vol = np.zeros((axis_array.size,)*3) # zzz.copy()
    # zzz[:] = axis_array.reshape(axis_array.size, 1, 1)
    rmin = 0 # carve out the core
    xmin = 0.6 # remove gas that would be travelling perpendicular to LOS (so would be in green channel maps, not red/blue)
    xmax = 0.85 # remove gas too extreme (set to 2 or something to remove)
    vol[(((z < z_1) & (z > 0) & (r > rmin)) | ((z > -z_1) & (z < 0) & (r > rmin))) & (((x < -xmin) & (x > -xmax)) | ((x > xmin) & (x < xmax)))] = 1.0
    vol[((z < z_2) & (z > 0)) | ((z > -z_2) & (z < 0))] = 0.0 #  & (r > 0.5)

    # Project along some axis
    plt.subplot(221)
    plt.imshow(np.sum(vol, axis=0), origin='lower')
    plt.subplot(222)
    plt.imshow(np.sum(vol, axis=1), origin='lower')
    plt.subplot(212)
    plt.plot(axis_array, biconcave_z(axis_array, diameter_1, a_1))
    plt.plot(axis_array, biconcave_z(axis_array, diameter_2, a_2)*height_mod)
    plt.show()




"""
13 CII
"""

def set_circle_radius(reg_list, fixed_diameter_beams):
    """
    Feb 28, 2024
    Set the radius of circles in reg_list to be fixed_diameter_beams across
    15.5 arcsec is beamwidth
    Meant for use with the 13CII circles in the two functions below
    :returns: string stub with info about circle diameter
    """
    if fixed_diameter_beams is None:
        diameter_stub = ""
    else:
        diameter_stub = f"_{fixed_diameter_beams}beamsacross"

    for reg in reg_list:
        print(reg.radius/15.5 * 2, " beams across")
        if fixed_diameter_beams is not None:
            fixed_radius_arcsec = 15.5*u.arcsec * fixed_diameter_beams / 2
            reg.radius = fixed_radius_arcsec
    return diameter_stub



def spectra_13cii(fixed_diameter=None):
    """
    January 15, 2024
    Compare 13CII spectra to 12CII
    Use the regions in sofia/13cii_spots.reg; see 2024-01-15 notes
    That file contains [Circle, Circle, Point, Point]
    Continued Jan 16, modeling the figure after Guevara 2020 figs
    """
    coeff_13 = ratio_12co_to_13co / 0.625 # co ratio == c+ ratio
    def cii_ratio_as_function_of_tau(tau):
        """
        Calculate the ratio of 12CII / 13CII brightness as a function of tau_12,
        the optical depth of 12CII.
        """
        return (1. - np.exp(-tau)) / (1. - np.exp(-tau / coeff_13))

    def make_tau_spline():
        """
        Fit a spline interpolating tau_12 values from 12/13 cii ratio values
        """
        tau_array = np.arange(0.01, 4., 0.01)
        cii_ratio_array = cii_ratio_as_function_of_tau(tau_array)
        tau_spline = UnivariateSpline(cii_ratio_array[::-1], tau_array[::-1], s=0)
        return tau_spline

    # Check and plot the 12cii/13cii relationship with tau
    if False:
        tau_array = np.arange(0.01, 4., 0.01)
        cii_ratio_array = cii_ratio_as_function_of_tau(tau_array)
        tau_spline = UnivariateSpline(cii_ratio_array[::-1], tau_array[::-1], s=0)
        plt.plot(cii_ratio_array[::-1], tau_array[::-1])
        x = 40
        y = tau_spline(x)
        print(x, y)
        x = np.arange(20, 70, 1)
        plt.plot(x, tau_spline(x), 'x')
        plt.show()
        return


    from astropy.convolution import CustomKernel
    reg_filename_short = "sofia/13cii_spots.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    reg_list = reg_list[:2] # just the two Circles
    # Set circle radius
    diameter_stub = set_circle_radius(reg_list, fixed_diameter)

    reg_dict = {reg.meta['text']: reg for reg in reg_list}
    print(reg_dict)
    # Load CII
    cii_stub = 'cii'
    cii_cube_obj = cube_utils.CubeData(get_map_filename(cii_stub)).convert_to_kms().convert_to_K()
    # Figure
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2)
    # gs = fig.add_gridspec(2, 1)

    assumed_13cii_dv = 11.2

    line_limits_lookup = [(25.5, 31.5), (24.5, 30.5)]

    for i, vel_str in enumerate(reg_dict):
        ax = fig.add_subplot(gs[0, i])
        reg = reg_dict[vel_str]

        # Get spectrum from region somehow
        if True:
            subcube = cii_cube_obj.data.subcube_from_regions([reg])
            # Get the number of pixels averaged under this mask
            n_pixels_in_avg = np.sum(~subcube.mask.view()[0])
            spectrum = subcube.mean(axis=(1, 2))
        else:
            # Centers of circles
            pj, pi = [int(round(p)) for p in reg.to_pixel(cii_cube_obj.wcs_flat).center.xy]
            spectrum = cii_cube_obj.data[:, pi, pj]
            n_pixels_in_avg = 1

        vel_axis = cii_cube_obj.data.spectral_axis.to_value()
        new_vel_axis = np.arange(-20, 76, 1) # 1 km/s interpolation

        spectrum_12 = spectrum.spectral_smooth(CustomKernel(signal.triang(3)))
        spectrum_12 = spectrum_12.spectral_interpolate(new_vel_axis*kms).to_value()
        # ax.plot(vel_axis, spectrum.to_value(), color='grey')
        ax.step(new_vel_axis, spectrum_12, label="[CII]", where='mid', color='k')
        # Convolve CII subcube
        spectrum_13 = spectrum.spectral_smooth(CustomKernel(signal.triang(3)))
        # Add 0.2 to the interpolation grid so that when we correct with -11.2, 13CII ends up sampled at whole number velocities
        # spectrum_13 = spectrum_13.spectral_interpolate((new_vel_axis+0.2)*kms).to_value()
        # vel_axis_13 = (new_vel_axis+0.2) - 11.2
        # spectrum_13 = spectrum_13.spectral_interpolate(new_vel_axis*kms).to_value()
        # vel_axis_13 = new_vel_axis - assumed_13cii_dv
        spectrum_13 = spectrum_13.spectral_interpolate((new_vel_axis+assumed_13cii_dv)*kms).to_value()
        vel_axis_13 = new_vel_axis

        # spectrum_13[vel_axis < 36] = np.nan
        # print(new_vel_axis)
        # print(vel_axis_13)
        ax.step(vel_axis_13, spectrum_13 * coeff_13, label="[$^{13}$CII] x 71.44", where='mid', color='r')

        err_12cii = get_onesigma(cii_stub)/np.sqrt(n_pixels_in_avg)
        err_13cii = err_12cii*coeff_13
        ax.axhline(err_13cii, color='r', alpha=0.5)

        # New axis for ratio, sharex with spectrum plot
        ratio_ax = fig.add_subplot(gs[1, i], sharex=ax)
        # Plot ratio
        # Use 24-30 km/s
        lo, hi = line_limits_lookup[i]
        vel_subset_12 = new_vel_axis[(new_vel_axis > lo) & (new_vel_axis < hi)]
        # vel_subset_13 = vel_axis_13[(vel_axis_13 > 23.5) & (vel_axis_13 < 30.5)] ## equivalent to vel_subset_12, but in practice off by float precision (so that 23.00...0x > 23) for vel_subset_13 but not 12
        spec_subset_12 = spectrum_12[(new_vel_axis > lo) & (new_vel_axis < hi)]
        spec_subset_13 = spectrum_13[(vel_axis_13 > lo) & (vel_axis_13 < hi)]
        # print(spec_subset_12.size)
        # print(spec_subset_13.size)

        ratio_subset = spec_subset_12/spec_subset_13
        # Clean the ratios; nan out values where 13cii is below 1sigma and where 13cii is below 12cii.
        ratio_subset[spec_subset_13*coeff_13 < err_13cii] = np.nan
        ratio_subset[spec_subset_13*coeff_13 < spec_subset_12] = np.nan
        # ratio_ax.plot(vel_subset_12, ratio_subset, marker='x', color='grey', linestyle='none')
        ratio_bar_plot = ratio_ax.bar(vel_subset_12, ratio_subset, color='grey', alpha=0.6, width=(vel_subset_12[1]-vel_subset_12[0]))
        ratio_ax.set_ylim((20, 80))

        # Figure out ratio error. Geometric, so fractional error easy
        frac_err_12 = err_12cii / spec_subset_12
        frac_err_13 = err_13cii / (spec_subset_13*coeff_13)
        ratio_err = np.sqrt(frac_err_12**2 + frac_err_13**2)*ratio_subset

        # New twin Axes for optical depth
        tau_ax = ratio_ax.twinx()
        tau_spline = make_tau_spline()
        tau_subset = tau_spline(ratio_subset)
        # tau_ax.plot(vel_subset_12, tau_subset, color='b', where='mid', linestyle='none')
        tau_ax.set_ylim((0, 3))
        # Errors on tau; do hi and lo, +/- onesigma. hi ratio => lo tau, so they're reversed
        tau_lo = tau_spline(ratio_subset + ratio_err)
        tau_hi = tau_spline(ratio_subset - ratio_err)
        tau_hi_errorbar = tau_hi - tau_subset
        tau_lo_errorbar = tau_subset - tau_lo



        tau_plot = tau_ax.errorbar(vel_subset_12, tau_subset, marker='o', yerr=(tau_lo_errorbar, tau_hi_errorbar), color='b', capsize=2, linestyle='none')

        print(max(tau_subset))


        if False:
            """
            Tried fitting and subtraction. The Gaussian fit to the 12CII line is too inaccurate that deep in the line wings to help.
            The fitted Gaussian doesn't affect the 13CII line at all, and makes a strange looking beat pattern in the main line.
            That beat pattern usually means you're fitting with too thin a line. There's probably multiple components in the 12CII line,
            and we have no hope of separating them with our data.
            """
            # Fit the 12CII to correct emission
            fitter = cps2.fitting.LevMarLSQFitter()
            g0 = cps2.models.Gaussian1D(amplitude=55, mean=28, stddev=4/2.355,
                bounds={'amplitude': (40, 60), 'mean': (26, 30), 'stddev': (3/2.355, 5.5/2.355)})
            velocity_mask = (vel_axis > 20) & (vel_axis < 35)
            g_fit = fitter(g0, vel_axis[velocity_mask], spectrum.to_value()[velocity_mask])
            print(g_fit)
            # Plot fit
            ax.step(new_vel_axis, g_fit(new_vel_axis), color='DarkGreen', where='mid')
            ax.step(vel_axis_13, g_fit(new_vel_axis)*coeff_13, color='LimeGreen', where='mid')
            # Try subtraction
            ax.step(vel_axis_13, spectrum_13 - g_fit(new_vel_axis)*coeff_13, color='RoyalBlue', where='mid')

        ax.set_ylim((-30, 150))
        # plt.ylim((-330, 630))
        ax.set_xlim((19.5, 37.5))
        # ax.set_xlim((0, 55))
        if i == 0:
            ax.legend()
            handles = [ratio_bar_plot, tau_plot]
            labels = ["T$_{\\rm MB\\,[CII]}$ / T$_{\\rm MB\\,[^{13}CII]}$", "$\\tau_{\\rm [CII]}$"]
            tau_ax.legend(handles=handles, labels=labels)

        ratio_ax.set_xlabel(f"Velocity ({kms.to_string('latex_inline')})")
        ratio_ax.set_ylabel("Line ratio [$^{12}$CII]/[$^{13}$CII]")
        tau_ax.set_ylabel("[$^{12}$CII] optical depth")
        ax.set_ylabel("T$_{\\rm MB}$ (K)")

    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.93, top=0.96)
    # plt.show()
    # Formerly: cii_13cii_tau_{assumed_13cii_dv:.1f}_circles.png (added "spectra_" 2024-02-28)
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"spectra_cii_13cii_tau_{assumed_13cii_dv:.1f}_circles{diameter_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title="Guevara 2020 plot inspiration",
            file=__file__, func="spectra_13cii"))


def contours_13cii(fixed_diameter=None):
    """
    January 17, 2024
    Going to see what the zoomed-in contours of 13CII look like
    """
    reg_filename_short = "sofia/13cii_spots.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    reg_list = reg_list[:2] # just the two Circles

    diameter_stub = set_circle_radius(reg_list, fixed_diameter)

    # Load CII
    cii_stub = "cii"
    cii_cube_obj = cube_utils.CubeData(get_map_filename(cii_stub)).convert_to_kms().convert_to_K()
    # subcube = cii_cube_obj.data[:, 205:265, 200:305] # small box
    # subcube = cii_cube_obj.data[:, 155:325, 150:365] # too wide
    # subcube = cii_cube_obj.data[:, 185:285, 180:325] # a bit too wide
    subcube = cii_cube_obj.data[:, 210:255, 215:285] # narrower
    # Figure
    fig = plt.figure(figsize=(10, 6))
    # Moment 0
    velocity_limits = (26*kms, 31*kms)
    velocity_limits_13 = tuple(x + 11.2*kms for x in velocity_limits)
    mom0_12 = subcube.spectral_slab(*velocity_limits).moment0()
    slab_13 = subcube.spectral_slab(*velocity_limits_13)
    mom0_13 = slab_13.moment0()
    # Get uncertainty for 13CII contours
    n_channels = slab_13.shape[0]
    dv = np.mean(np.diff(subcube.spectral_axis)).to_value()
    channel_noise = get_onesigma(cii_stub)
    onesigma_mom0 = channel_noise * dv * np.sqrt(n_channels)
    print(onesigma_mom0)

    ax = plt.subplot(111, projection=subcube[0].wcs)
    im = ax.imshow(mom0_12.to_value(), origin='lower')
    fig.colorbar(im, ax=ax, label="[$^{12}$CII] integrated intensity " + f"({mom0_12.unit.to_string('latex_inline')})")
    # ax = plt.subplot(122, sharex=ax, sharey=ax)
    # im = ax.imshow(mom0_13.to_value(), origin='lower', vmin=-5, vmax=5)
    # fig.colorbar(im, ax=ax)
    ax.contour(mom0_13.to_value(), levels=[x*onesigma_mom0 for x in (2, 3, 4)], colors='k')

    for reg in reg_list:
        pixreg = reg.to_pixel(subcube[0].wcs)
        pixreg.visual['facecolor'] = pixreg.visual['edgecolor'] = 'white'
        pixreg.visual['linewidth'] = 2
        pixreg.plot(ax=ax)

    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    plt.tight_layout()

    # plt.show()
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), f"contours_13cii_{cii_stub}_{make_simple_vel_stub(velocity_limits)}{diameter_stub}.png"),
        metadata=catalog.utils.create_png_metadata(title=f"13cii contours 2,3,4 sigma {onesigma_mom0:.2f}",
            file=__file__, func="contours_13cii"))


"""
Dark / IR Clouds
"""
def plot_dark_clouds():
    """
    Feb 18, 2024
    SCOPE 850 micron Eden et al. 2019 ([2019MNRAS.485.2895E](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.2895E))
    """
    full_filename = catalog.utils.search_for_file("catalogs/G016.96p00.27_SCOPE850um_clouds_2.tsv") # there is ...clouds.tsv and ...clouds_2.tsv
    df = pd.read_csv(full_filename, skiprows=46, sep=";", header=None)
    print(df)
    s = SkyCoord(ra=df[2].values, dec=df[3].values, unit=(u.deg, u.deg), frame='fk5')
    pointskyregs = regions.Regions([regions.PointSkyRegion(center=x) for x in s])
    save_filename = os.path.join(os.path.dirname(full_filename), os.path.basename(full_filename).replace(".tsv", ".reg"))
    pointskyregs.write(save_filename)

def vizier_query_dark_clouds():
    """
    Feb 18, 2024
    Try vizier querying the SCOPE SCUBA-2 850 micron catalog
    Based on the vizier_queries_m16_g0.py code
    """
    from astroquery.vizier import Vizier
    # Vizier.ROW_LIMIT = -1 # Always get all the rows
    scope_cat_name = "J/MNRAS/485/2895"
    # catalog_dict = Vizier.find_catalogs(scope_cat)
    cat = Vizier(row_limit=-1).query_constraints(catalog=scope_cat_name,
        RAJ2000="274.0 .. 275.4", DEJ2000="-13.11 .. -14.3")
    print(cat[0].to_pandas())

    # catalogs = Vizier.get_catalogs(catalog_dict[scope_cat])
    # print(catalogs)
    # sptype_catalog = catalogs[1] # returns 2 catalogs
    # del catalogs, catalog_dict # save memory
    # catalog_df = sptype_catalog.to_pandas(index='ID')



"""
RRLs from Loren
"""

def plot_rrls():
    """
    Just basic plotting, try a spectrum, see if the data axes are messed up
    """
    # RRL filename
    fn_short = "greenbank/M16_halpha_2pol_average.fits"
    cube_obj = cube_utils.CubeData(fn_short).convert_to_K().convert_to_kms()
    spec = cube_obj.data[:, 40, 40].to_value()
    spectral_axis = cube_obj.data.spectral_axis.to_value()
    plt.plot(spectral_axis, spec)
    plt.show()



def shell_expansion_rrls_and_cii():
    """
    greenbank/M16_halpha_2pol_average.fits
    Following m16_pictures_2.m16_expanding_shell_spectra
    """
    # RRL filename
    rrl_fn_short = "greenbank/M16_halpha_2pol_average.fits"
    # Copy-paste-edited from m16_expanding_shell_spectra
    reg_filename_short = "catalogs/m16_west_cavity_spec_regions.reg"
    spec_labels = ("1", "2")
    colors = marcs_colors[:2][::-1]
    savename_stub = "west_cavity_3am_circles"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # Fig
    fig = plt.figure(figsize=(13, 6))
    # Axes
    gs = fig.add_gridspec(2, 3)
    img_axes = []
    # Load cube
    line_stub = "cii"
    fn = get_map_filename(line_stub)
    cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()
    # Reference contour moment image
    ref_vel_lims = (12*kms, 30*kms)
    ref_mom0 = cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
    # Moment images
    # velocity_intervals = [(2, 12), (35, 45)]
    velocity_intervals = [(5, 15), (30, 40)]
    for i, vel_lims in enumerate(velocity_intervals):
        vel_lims = tuple(v*kms for v in vel_lims)
        mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0()
        ax = fig.add_subplot(gs[i, 0], projection=cube_obj.wcs_flat)
        im = ax.imshow(mom0.to_value(), origin='lower', vmin=-10, vmax=45, cmap='plasma')
        fig.colorbar(im, ax=ax, label=f"{get_data_name(line_stub)} {make_vel_stub(vel_lims)} ({mom0.unit.to_string('latex_inline')})")
        ax.contour(ref_mom0.to_value(), levels=np.arange(75, 400, 75), colors='k', linewidths=0.7)
        # Plot circles
        for j, reg in enumerate(reg_list):
            reg.to_pixel(cube_obj.wcs_flat).plot(ax=ax, color=colors[j])
        img_axes.append(ax)
    # Load RRL data and use all same settings
    rrl_cube_obj = cube_utils.CubeData(rrl_fn_short).convert_to_K().convert_to_kms()
    # spectrum multiplier to highlight RRL data
    rrl_mult = 10.
    # Spectra, both on same figure
    spec_ax = fig.add_subplot(gs[:, 1:])
    for j, reg in enumerate(reg_list):
        # CII
        subcube = cube_obj.data.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2))
        spec_ax.plot(subcube.spectral_axis.to_value(), spectrum.to_value(), color=colors[j], linestyle='-', linewidth=1, label=spec_labels[j])
        # RRL spectra on the figure too
        subcube = rrl_cube_obj.data.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2)) * rrl_mult
        spec_ax.plot(subcube.spectral_axis.to_value(), spectrum.to_value(), color=colors[j], linestyle='-', linewidth=2)



    # Mark moment velocities on spectrum plot
    for vel_lims in velocity_intervals:
        plt.axvspan(*vel_lims, color='grey', alpha=0.3)
    # Extra plot dressing
    spec_ax.axhline(0, color='grey', linestyle="--", alpha=0.2)
    spec_ax.set_xlabel("V$_{\\rm LSR}$ " + f"({kms.to_string('latex_inline')})")
    spec_ax.set_ylabel(f"{get_data_name(line_stub)} line intensity ({spectrum.unit.to_string('latex_inline')})")
    spec_ax.set_xlim([-10, 60])
    plt.subplots_adjust(left=0.09, right=0.97, top=0.95, wspace=0.45, bottom=0.09)
    vel_stub = "-and-".join([make_simple_vel_stub(tuple(v*kms for v in vel_lims)) for vel_lims in velocity_intervals])

    savename = f"expanding_shell_spectra_{line_stub}_{vel_stub}.png"
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), savename),
        metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short}",
            file=__file__, func="m16_expanding_shell_spectra"))



if __name__ == "__main__":
    pass # in case nothing else is commented in, just so this is syntactically correct

    """
    RRLs
    """
    # shell_expansion_rrls_and_cii()

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
    # overlay_moment(background='250um', overlay='ciiAPEX', velocity_limits=(21*kms, 22*kms), cutout_reg_stub='med', reg_filename_or_idx="catalogs/N19_pv_2.reg", plot_stars=False)

    # for i in range(10, 34, 2):
    # i = 22
    # vel_lims = (i*kms, (i+2)*kms)
    # overlay_moment(background='ciiAPEX', overlay='12co10-pmo', velocity_limits=(15*kms, 18*kms), velocity_limits2=(18*kms, 20*kms), cutout_reg_stub='med', plot_stars=False, reg_filename_or_idx="catalogs/N19_points_2.reg")
    # overlay_moment(background='13co10-nob', overlay='90cm', cutout_reg_stub='med-large', velocity_limits=(0*kms, 40*kms), plot_stars=False, reg_filename_or_idx=None)
    # vel_lims_list = [
    #     (5*kms, 15*kms),
    #     (5*kms, 40*kms),
    #     (30*kms, 40*kms)
    # ]
    # for i in range(len(vel_lims_list)):
    #     vel_lims = vel_lims_list[i]
    #     overlay_moment(background='8um', overlay='cii-30', velocity_limits2=vel_lims, cutout_reg_stub='med', plot_stars=True, reg_filename_or_idx=("catalogs/m16_west_cavity_pvs.reg", 6))

    # overlay_moment(background='8um', overlay='cii', velocity_limits2=(7*kms, 10*kms), cutout_reg_stub='blueclump', plot_stars=False, reg_filename_or_idx=("catalogs/m16_points_blueshifted_clump.reg"))
    # v = 24
    # dv = 1
    # while v < 29:
    #     vel_lims = (v*kms, (v+dv)*kms)
    #     overlay_moment(background='cii', overlay='cii', velocity_limits2=(17*kms, 20*kms),
    #         velocity_limits=vel_lims, cutout_reg_stub='N19', plot_stars=False, reg_filename_or_idx=("catalogs/N19_shell_edge.reg"))
    #     v += dv

    # for c in ['cii']:
    #     overlay_moment(background=c, overlay=c, velocity_limits=(13*kms, 17*kms),
    #         velocity_limits2=(17*kms, 20*kms), cutout_reg_stub='N19', plot_stars=False, reg_filename_or_idx=("catalogs/N19_points_all_across_2.reg"))

    # cii = 'ciiAPEX'
    # co = '12co32'
    # bgs = ['8um'] #[str(x)+"um" for x in (70, 160, 250, 350)]
    # for bg in bgs:
    #     overlay_two_moments(background=bg, overlays=(cii, co), velocity_limits=((17*kms, 21*kms), (18*kms, 21*kms)),
    #         vlims={'ciiAPEX': (-1.5, 14.5), '70um': (0, 1), '160um': (0, 0.8), '250um': (500, 2500), '350um': (240, 1000), '500um': (100, 350),
    #                 '8um': (45, 200)},
    #         levels={'ciiAPEX': np.arange(-2.5, 16, 2.5)})

    # overlay_moment(background='12co10-nob', overlay='ciiAPEX', velocity_limits=(18*kms, 24*kms), velocity_limits2=(14*kms, 21*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/SE_bubbles_cii_pvs_2.reg", 1))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(15*kms, 16*kms), velocity_limits=(18*kms, 19*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(16*kms, 17*kms), velocity_limits=(19*kms, 20*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(17*kms, 18*kms), velocity_limits=(20*kms, 21*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(18*kms, 19*kms), velocity_limits=(21*kms, 22*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(19*kms, 20*kms), velocity_limits=(22*kms, 23*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(20*kms, 21*kms), velocity_limits=(23*kms, 24*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))
    # overlay_moment(background='13co32', overlay='ciiAPEX', velocity_limits2=(21*kms, 22*kms), velocity_limits=(24*kms, 25*kms), cutout_reg_stub='med-large', plot_stars=False, reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs.reg", 5))

    # v_start, v_increment = 14, 2
    # while v_start < 20:
    #     vel_lims = (v_start*kms, (v_start + v_increment)*kms)
    #     overlay_moment(background='ciiAPEX', overlay='160um', velocity_limits=vel_lims, cutout_reg_stub='med', plot_stars=True, reg_filename_or_idx=("catalogs/SE_bubbles_cii_pvs_2.reg", 1))
    # #     overlay_moment(background='ciiAPEX', overlay='160um', velocity_limits=vel_lims, velocity_limits2=(15*kms, 30*kms), cutout_reg_stub='med', plot_stars=True, reg_filename_or_idx=("catalogs/m16_west_cavity_pvs.reg", 0))
    #     v_start += v_increment

    # vcii = (28*kms, 29*kms)
    # vco = (25*kms, 26*kms)
    # vcii = (28*kms, 29*kms)
    # vco = (25*kms, 26*kms)
    # l1 = '12co32'
    # l2 = 'ciiAPEX'
    # # for l in ('12co32',):# '12co10-pmo'):
    # for dv1 in np.arange(-2, 6, 1):
    #     for dv2 in np.arange(-2, 6, 1):
    #         overlay_moment(background=l1, overlay=l2, velocity_limits=[np.around(v + dv1*kms, 2) for v in vco], velocity_limits2=[np.around(v + dv2*kms, 2) for v in vco], cutout_reg_stub='med')

    ####
    ## Save moment 0 to FITS
    ####
    # save_moment0(line_stub='12co32', velocity_limits=(23*kms, 27*kms), cutout_reg_stub=None)
    # line = '12co10-nob'
    # vel_stubs = [(10*kms, 21*kms), (17*kms, 21*kms)]
    # for vs in vel_stubs:
    #     for line in ['12co10-nob', '13co32', '13co10-pmo', '13co10-nob']:
    #         try:
    #             save_moment0(line_stub=line, velocity_limits=vs, cutout_reg_stub=None)
    #         except Exception as e:
    #             print(f"Exception from {line}", vs)
    #             print(e)
    #             print("ignoring and continuing")
    # save_moment0(line_stub='cii-30', velocity_limits=(6*kms, 11*kms), cutout_reg_stub=None)
    # save_moment0(line_stub='cii-30', velocity_limits=(15*kms, 30*kms), cutout_reg_stub=None)
    # save_moment0(line_stub='cii-30', velocity_limits=(21*kms, 27*kms), cutout_reg_stub=None)
    # save_moment0(line_stub='12co10-pmo', velocity_limits=(10*kms, 27*kms), cutout_reg_stub=None)
    # save_moment0(line_stub='12co10-nob', velocity_limits=(10*kms, 27*kms), cutout_reg_stub=None)


    """ Comparing 8micron and CII """
    # for i in range(5):
    #     compare_8micron_and_cii_intensities((20*kms, 21*kms), i)
    # correlation_plot_8um_cii()



    """
    PV examples
    """
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['13co32', '13co10-pmo'])
    # fast_pv(reg_filename_or_idx=0)
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['ciiAPEX', '12co32'])
    # fast_pv(reg_filename_or_idx=("catalogs/m16_west_cavity_pvs.reg", 0), line_stub_list=['cii-30', '12co32'])

    # error = False
    # i = 1
    # while not error:
    #     try:
    #         fast_pv(reg_filename_or_idx=("catalogs/N19_pv_paths_4.reg", i), line_stub_list=['ciiAPEX', '12co32'])
    #         i += 1
    #     except:
    #         error = True

    # blue clump large PV
    # fast_pv(reg_filename_or_idx=("catalogs/m16_pvs_blueshifted_clump.reg", 0), line_stub_list=['ciiAPEX', '12co32'], velocity_limits=(-5*kms, 50*kms))

    # blueshifted large-scale CO (in the Nobeyama data)
    # error = False
    # i = 2
    # while not error and i < 3:
    #     try:
    #         fast_pv(reg_filename_or_idx=("catalogs/m16_across_large_galactic_co10-nob.reg", i), line_stub_list=['12co10-pmo', '13co10-pmo'], velocity_limits=(10*kms, 40*kms),
    #             levels={'13co10-pmo': np.arange(2, 30, 3)}, figsize=(20, 9), aspect=0.5) # aspect = 7 for Nob, aspect = 0.5 for pmo
    #         print(i, "done")
    #     except:
    #         error = True
    #         print("exiting")
    #     i += 1

    ##### make all PVs from a given .reg file
    # error = False
    # i = 0
    # while not error and i<6:
    #     try:
    #         # fast_pv(reg_filename_or_idx=("catalogs/m16_west_cavity_pvs.reg", i), line_stub_list=['cii-30', '12co10-nob'],
    #
    #         # fast_pv(reg_filename_or_idx=("catalogs/N19_and_E-bubble_GMF_pvs.reg", i), line_stub_list=['13co10-pmo', '13co32'],
    #             # vmax={'13co10-pmo': 8, '12co10-pmo': 35},
    #         # fast_pv(reg_filename_or_idx=("catalogs/N19_and_E-bubble_GMF_pvs.reg", i), line_stub_list=['12co10-pmo', '13co10-pmo'],
    #             # vmax={'13co10-pmo': 8, '12co10-pmo': 20},
    #         # fast_pv(reg_filename_or_idx=("catalogs/N19_and_E-bubble_full_pvs_2.reg", i), line_stub_list=['13co10-nob', 'ciiAPEX'],
    #         #     vmax={'13co10-pmo': 8, '12co10-pmo': 35, '13co10-nob': 10},
    #         fast_pv(reg_filename_or_idx=("catalogs/m16_pv_vectors_4.reg", i), line_stub_list=['ciiAPEX', '13co32'],
    #             vmax={'13co10-pmo': 8, '12co10-pmo': 35, '13co10-nob': 10, '13co32': 15},
    #             velocity_limits=(5*kms, 45*kms),
    #             velocity_intervals=np.arange(10, 23, 2), # horizontal lines on PV
    #             levels={
    #                 # 'ciiAPEX': np.concatenate([np.arange(2, 61, 3), np.arange(65, 126, 20)]),
    #                 # 'ciiAPEX': np.concatenate([np.arange(2, 15, 5), np.arange(17, 126, 15)]),
    #                 # '12co10-pmo': np.concatenate([np.arange(0.75, 5, 1), np.arange(5.75, 26, 4)]),
    #                 # '12co10-pmo': np.arange(5, 55, 5),
    #                 # '12co10-nob': np.concatenate([np.arange(2, 5, 2), np.arange(6, 26, 6)]),
    #                 # '12co10-nob': np.arange(5, 55, 5),
    #                 '12co32': np.concatenate([np.arange(0.5, 5, 1.5)*2, np.arange(5, 51, 5)*2]),
    #             }
    #         )
    #         plt.close(plt.gcf())
    #         i += 1
    #     except:
    #         error = True

    # pv_slice_series_overlay()

    """
    CO/CII column
    """
    # find_co10_noise()
    velocity_limits = {
        ## the experimental stuff
        'redshifted_1': (29*kms, 45*kms), 'blueshifted_1': (0*kms, 13*kms),
        'north_cloud_1': (13*kms, 20*kms), '25kms_1': (20*kms, 29*kms),
        'co32_red': (23.3*kms, 28*kms),
        'big_molecular_cloud': (22*kms, 27*kms), # the greenish-red molecular cloud that crosses over M16 east of the pillars/spire
        'super-red-stuff': (27*kms, 30*kms), # probably not useful but CO 3-2 has the MYSO and then a small bright rim close to the cluster.
        'north_cloud_2': (11*kms, 21*kms),
        ### the good stuff
        'redshifted_2': (21*kms, 27*kms), # the originals
        'green-cloud': (21*kms, 23*kms), 'red-cloud': (23*kms, 27*kms), # the main green/red stuff, split more finely
        'north_cloud_3': (10*kms, 21*kms),
        ### new wave experimental 2024-01-15
        'blue_clump': (6*kms, 11*kms), 'high_velocity': (35*kms, 40*kms),
    }
    # for s in ['green-cloud', 'red-cloud', 'redshifted_2', 'north_cloud_2']:
    # for s in ['redshifted_2', 'north_cloud_2']:
    # for s in ['north_cloud_3']:
    #     co_column_manage_inputs(line='10', isotope='13', velocity_limits=velocity_limits[s], cutout_reg_stub=None)
        # get_co32_to_10_ratio_for_density(velocity_limits=velocity_limits[s], isotope10='13', noise_cutoff=0)
        # get_13co10_to_c18o10_ratio_for_opticaldepth(velocity_limits=velocity_limits[s])

    # sample_multiple_maps_regions(velocity_limits=velocity_limits['north_cloud_2'])
    # sample_multiple_maps_regions(velocity_limits=velocity_limits['redshifted_2'])
    # sample_masked_map("BNR")
    # sample_masked_map("N19")

    # calculate_cii_column_density(mask_cutoff=3*u.K, velocity_limits=velocity_limits['north_cloud_2'])
    # calculate_cii_column_density(mask_cutoff=3*u.K, velocity_limits=velocity_limits['redshifted_2'])
    # calculate_cii_column_density(mask_cutoff=3*u.K, velocity_limits=velocity_limits['green-cloud'])

    # calculate_cii_column_density(mask_cutoff=6*u.K, velocity_limits=velocity_limits['north_cloud_3'])
    # calculate_cii_column_density(mask_cutoff=6*u.K, velocity_limits=velocity_limits['redshifted_2'])
    # calculate_cii_column_density(mask_cutoff=6*u.K, velocity_limits=velocity_limits['red-cloud'])
    # get_co_spectra_for_radex()

    # make_more_radex_grids()
    # for i in [28]:
    #     k = f"t{i}"
    # compare_data_with_radex_grid("t28")
    # scatter_radex_grid()
    # calc_mass_from_masked_data()
    # sum_chisq_to_get_masked_area_errorbars()
    # individual_pixel_ensemble_chisq()
    # for l in ["N19", "BNR"]:
    #     unified_chisq_plotting_system(region_label=l, abscal_pct=10)
    #     for a in [5, 10, 15]:
    #         unified_chisq_plotting_system(region_label=l, abscal_pct=a)
    # print_out_particle_mass_for_rho()
    # mask_footprint_reference_plot()

    # calculate_cii_column_density_detection_threshold()
    # calculate_co_column_density_detection_threshold()

    # convert_pacs_tau_to_coldens()

    """
    Channel maps/movies
    """
    # cii_channel_maps()
    # channel_movie('ciiAPEX', vel_lims=(0, 10))
    # co_channel_maps()
    cii_co_combined_channel_maps()
    # channel_maps_scratchpad()

    """
    Spectra
    """
    # for i in range(1, 3):
    #     for j in range(1, 3):
    #         if i == j == 1:
    #             continue
    #         print("Doing", i, j)
    # plot_spectra(reg_set_number=4, line_set_number=1, velocities_to_mark=(17.5, 19.5, 21.5))
    # plot_spectra(reg_set_number=8, line_set_number=1, velocities_to_mark=(17.5, 19.5, 21.5))

    # blueshifted clump
    # plot_spectra(reg_set_number=5, line_set_number=3, velocities_to_mark=(8,))

    # N19 shell stuff; N19_points_all_across and _2; reg sets 6 and 7
    # for i in (6,7):
    #     plot_spectra(reg_set_number=i, line_set_number=2, velocities_to_mark=(17, 18, 19, 20, 21))

    """
    Misc
    """
    # Peak temperature map and velocity map
    # peak_T_velocity_map()
    # peak_T_and_moment_maps_CO(isotope='13', transition='10', velocity_limits=velocity_limits['north_cloud_2'])
    # peak_T_and_moment_maps_CO(isotope='12', transition='10', velocity_limits=velocity_limits['north_cloud_2'])
    # contours_13cii(x)
    # spectra_13cii(x)
    # spitzer_expansion_plot()
    # ekin_ew_vs_age_plot()
    # integrate_cii_and_FIR_luminosities()
    # trim_CO_mass_to_CII_grid(velocity_limits=(11*kms, 21*kms))
    # bubble_geometry()
    # bubble_cross_cut(select=3.2)
    # fake_bubble_spectra()
    # bubble_simulated_projection()
    # vizier_query_dark_clouds()
    # n19_self_absorption()
    # energetics_calculations()
    # n19_time_calculation()
    # cii_channel_maps_photoseries_slides_animation()
