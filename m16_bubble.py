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
import math

# from math import ceil
# from scipy import signal
# from scipy.interpolate import UnivariateSpline

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


ratio_12co_to_H2 = 8.5e-5
Cp_H_ratio = 1.6e-4 # Sofia et al 2004, what Tiwari et al 2021 used in the N(C+)->N(H) section

ratio_12co_to_13co = 44.65 # call it 45 in paper, the difference will be miniscule

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
        '12co10-pmo': "purplemountain/G17co12.fits", '13co10-pmo': "purplemountain/G17co13.fits",
        'c18o10-pmo': "purplemountain/G17c18o.fits",
        '12co10-nob': "nobeyama/FGN_01700+0000_2x2_12CO_v1.00_cube.fits", '13co10-nob': "nobeyama/FGN_01700+0000_2x2_13CO_v1.00_cube.fits",
        'c18o10-nob': "nobeyama/FGN_01700+0000_2x2_C18O_v1.00_cube.fits",
        'rrl': "thor_vla/RRL_stacked_L17.50_deg.m16cutout.fits",
}
observatory_names = {'pmo': "PMO", 'nob': "NRO"}
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
    'med-large': "m16_cutout_box_medium-large.reg", # sort of like the Hill 2012 footprint, includes some of the filament. Goes well outside CII
    'med': "m16_cutout_box_medium.reg", # CII and CO (3-2) footprint, aligned in equatorial (so there will be NaN gaps)
}
vlim_memo = { # hash things somehow and put them here
    '8um': (40, 320), 'irac4': (40, 320),
    'cii.levels': np.concatenate([np.arange(2.5, 61, 5), np.arange(65, 126, 15)]), 'cii.generic': (0, 25), #65 is better for most, 14 good for some
    '12co32.levels': np.arange(2.5, 51, 5), '12co32.generic': (0, 25), # or 40
    '13co32.levels': np.arange(1, 27, 2.5), '13co32.generic': (0, 18),
    'rrl.levels': np.arange(3, 30, 6), # RRLs should be "sliced" into a 1 km/s channel
    '90cm.levels': np.arange(0, 0.61, 0.06), '850um': (0, 0.5), '850um.levels': np.concatenate([np.arange(.15, 2.1, 0.3), np.arange(3, 6, 1)]), '850um.generic': (0.15, 2),

    # '13co10.levels': np.arange(0.25, 5, 0.5), '13co10.generic': (0, 4), # these are good for large velocity intervals
    '13co10.levels': np.arange(1, 22, 2), '13co10.generic': (0, 20), # small velocity intervals (like 1 km/s)

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
onesigmas = {'12co10-pmo': 0.415, '13co10-pmo': 0.200, 'c18o10-pmo': 0.202}

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
    if '-' in key:
        # Get rid of the observatory tag
        key = key.split('-')[0]
    vlims = {}
    vlims_keys = ['vmin', 'vmax']
    if key in vlim_memo:
        val = vlim_memo[key]
        for k, v in zip(vlims_keys, val):
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
    if '-' in data_stub:
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
        return cube_utils.cubenames[data_stub.split('-')[0]] + " " + observatory_names[data_stub.split('-')[1]]
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
        img, hdr = fits.getdata(catalog.utils.search_for_file(data_filename), header=True)
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


def co_column_manage_inputs():
    """
    May 16, 2023
    Wrapper function for the more general function below.
    This function is specific to the M16 CO 1-0 and 3-2 maps we have.
    It does the work of m16_deepdive.integrate_13_and_18_co_for_column_density
    and m16_deepdive.get_excitation_temperature_12co.

    The only user input is whether to use 1-0 or 3-2.
    Can consider adding C18O 1-0 later.
    And write in the capability (to be refined later) to use specific velocity limits.
    """
    line = '10'
    if line == '32':
        thick_cube_stub = '12co32'
        thin_cube_stub = '13co32'
        thick_channel_uncertainty = cube_utils.onesigmas[thick_cube_stub] * u.K
        thin_channel_uncertainty = cube_utils.onesigmas[thin_cube_stub] * u.K
    elif line == '10':
        thick_cube_stub = '12co10'
        thin_cube_stub = '13co10'
        thick_channel_uncertainty = onesigmas[thick_cube_stub] * u.K
        thin_channel_uncertainty = onesigmas[thin_cube_stub] * u.K

    # Placeholder for velocity limits argument
    velocity_limits = None
    # Expose the if logic as the top layer, since we only need to check once!
    if velocity_limits is None:
        def _apply_velocity_limits(cube):
            return cube
    else:
        def _apply_velocity_limits(cube):
            return cube.spectral_slab(*velocity_limits)

    # Load 12 (optically thick) and 13 or 18 (optically thin) maps.
    # Get excitation temperature from optically thick cube
    thick_cube_fn = get_map_filename(thick_cube_stub)
    thick_cube_obj = cube_utils.CubeData(thick_cube_fn)
    thick_cube = _apply_velocity_limits(thick_cube_obj.data)
    # Save this in final cube
    peak_temperature = thick_cube.max(axis=0).to(u.K)
    del thick_cube_obj, thick_cube

    # Get integrated optically thin emission
    thin_cube_fn = get_map_filename(thin_cube_stub)
    thin_cube_obj = cube_utils.CubeData(thin_cube_fn).convert_to_kms()
    thin_cube = _apply_velocity_limits(thin_cube_obj.data)
    # Grab a couple other things from the CubeData object
    beam = thin_cube_obj.data.beam
    save_path = os.path.dirname(thin_cube_obj.full_path)
    # Todo: make a cleaner moment image by trimming out the emission-free parts
    # Save this in final cube
    mom0 = thin_cube.moment0()
    # Noise
    cube_dv = np.abs(np.diff(thin_cube.spectral_axis[:2]))[0].to(kms)
    n_channels = thin_cube.shape[0]
    moment0_noise = thin_channel_uncertainty * cube_dv * np.sqrt(n_channels)
    del thin_cube_obj, thin_cube

    # plt.imshow(mom0.to(u.K*kms).to_value(), origin='lower')
    # plt.show()

    col_dens_calculator = COColumnDensity(thin_cube_stub)
    col_dens_calculator.set_data(peak_temperature, mom0.to(u.K*kms))
    col_dens_calculator.set_uncertainty(thick_channel_uncertainty, moment0_noise)
    col_dens_calculator.set_abundance_ratios(ratio_12co_to_13co, ratio_12co_to_H2)
    col_dens_calculator.calculate_column_density()
    col_dens_calculator.calculate_mass_per_pixel(mom0.wcs, los_distance=los_distance_M16, e_los_distance=err_los_distance_M16, thin_line_beam=beam)

    # plt.subplot(121)
    # plt.imshow(col_dens_calculator.H2_column_density.to_value(), origin='lower')
    # plt.subplot(122)
    # plt.imshow(col_dens_calculator.mass_per_pixel.to_value(), origin='lower')
    # plt.show()

    """ Write results to FITS """
    savename = os.path.join(save_path, f"column_density_{thin_cube_stub}.fits")

    header_pairs = col_dens_calculator.create_header_comments()

    def make_and_fill_header():
        # fill header with stuff, make it from WCS
        hdr = mom0.wcs.to_header()
        for k, v in header_pairs:
            hdr[k] = v
        return hdr

    # common messages
    h2_msg = "This is MOLECULAR hydrogen (H2)"
    mass_msg = "mass is per pixel on this image"

    extensions = [
        ('H2coldens', col_dens_calculator.H2_column_density, h2_msg),
        ('err_H2coldens', col_dens_calculator.e_H2_column_density, h2_msg),
        ('12COcoldens', col_dens_calculator.thick_line_column_density),
        ('err_12COcoldens', col_dens_calculator.e_thick_line_column_density),
        ('13COcoldens', col_dens_calculator.thin_line_column_density),
        ('err_13COcoldens', col_dens_calculator.e_thin_line_column_density),
        ('mass', col_dens_calculator.mass_per_pixel, mass_msg),
        ('err_mass', col_dens_calculator.e_mass_per_pixel, mass_msg),
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
    def __init__(self, optically_thin_line_stub):
        """
        Initialize instance using a line stub like '13co10', whatever the
        optically thin species is.
        The function will then obtain molecular constants from internal
        dictionaries.
        :param optically_thin_line_stub: str, line stub describing optically
            thin CO line
        """
        # Constants, in order:
        # B_0 (MHz), E_upper (K), mu (Debye), nu (GHz), J_upper
        # From the above, we construct g and S
        # Looks like B_0 is constant to the molecule, and does not change with transtion
        _constants = {
            '13co10': (55101.01, 5.28880, 0.11046, 110.20135400, 1),
            '13co32': (55101.01, ), # see 2023-09-11 notes, still re-researching this
            # 'c18o10': (),
        }
        if optically_thin_line_stub not in '13co10':
            raise RuntimeError(f"Line {optically_thin_line_stub} not supported.")
        B0, Eu, mu, nu, self.Ju = _constants[optically_thin_line_stub]
        self.B0 = B0 * u.MHz
        self.Eu = Eu * u.K
        self.mu = mu * u.Debye
        # relying on the units to correct bugs from mixing up "mu" and "nu"
        self.nu = nu * u.GHz

    def set_data(self, peak_temperature, integrated_intensity):
        """
        Provide data to the COColumnDensity instance.
        Both inputs should be Quantity arrays of matching dimension and share
        WCS, so that a given pixel references the same sky position
        in both images. WCS info does not need to be specified at this stage,
        just needs to be the same for both.
        :param peak_temperature: Quantity, peak temperature of optically thick
            line to be used as excitation temperature Tex
        :param integrated_intensity: Quantity, integrated intensity of optically
            thin line to be used for column density calculation
        """
        self.peak_temperature = peak_temperature
        self.integrated_intensity = integrated_intensity

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

    def _calculate_thin_line_column_density(self):
        """
        Calculate the column densities and propagate uncertainty.
        Makes use of astropy.constants (const)
        """
        # Let Tex be equal to peak MB temperature in the opt. thick line.
        # This is an assumption, so I'll keep them as separate variables.
        self.Tex = self.peak_temperature
        self.e_Tex = self.e_pt
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
        self.thin_line_column_density = (prefactor * (Qrot/g) * exp_term * self.integrated_intensity).to(u.cm**-2)
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
    if cutout_reg_stub is not None:
        img_info['cutout'] = misc_utils.cutout2d_from_region(img, img_info['wcs'], get_cutout_box_filename(cutout_reg_stub))
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
    im = ax_pv.imshow(sl0.data, origin='lower', cmap='plasma', vmin=0, vmax=pv_vmaxes.get(ls0, None), aspect=(sl0.data.shape[1]/(2.5*sl0.data.shape[0])))
    cs = ax_pv.contour(sl0.data, colors='k', linewidths=1, linestyles=':', levels=_get_levels(ls0))
    pv_cbar = fig.colorbar(im, ax=ax_pv, location='right', label=cube.data.unit.to_string('latex_inline'))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='k')
    ax_pv.text(0.05, 0.95, "Image: "+get_data_name(ls0), fontsize=13, color=marcs_colors[1], va='top', ha='left', transform=ax_pv.transAxes)
    # Overlay [1]
    cs = ax_pv.contour(sl1_reproj, cmap='cool', linewidths=1.5, levels=_get_levels(ls1), vmax=pv_vmaxes.get(ls1, None))
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='w', linewidth=2)
    ax_pv.text(0.05, 0.90, "Contour: "+get_data_name(ls1), fontsize=13, color='w', va='top', ha='left', transform=ax_pv.transAxes)

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
        ax.set_xlim((5, 41))
        if line_set_number == 1:
            ax.set_ylim((-5, 27)) # or 46
        elif line_set_number == 2:
            ax.set_ylim((-3, 13)) # or 23

        ax.text(0.06, 0.94, reg_list[reg_idx].meta['text'], transform=ax.transAxes, fontsize=15, color='k', ha='left', va='center')
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
    # overlay_moment(background='250um', overlay='ciiAPEX', velocity_limits=(21*kms, 22*kms), cutout_reg_stub='med', reg_filename_or_idx="catalogs/N19_pv_2.reg", plot_stars=False)

    # for i in range(10, 34, 2):
    # i = 22
    # vel_lims = (i*kms, (i+2)*kms)
    # overlay_moment(background='ciiAPEX', overlay='12co10-pmo', velocity_limits=(15*kms, 18*kms), velocity_limits2=(18*kms, 20*kms), cutout_reg_stub='med', plot_stars=False, reg_filename_or_idx="catalogs/N19_points_2.reg")
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

    # for i in range(5):
    #     compare_8micron_and_cii_intensities((20*kms, 21*kms), i)

    """
    PV examples
    """
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['13co32', '13co10-pmo'])
    # fast_pv(reg_filename_or_idx=0)
    # fast_pv(reg_filename_or_idx="catalogs/N19_pv_2.reg", line_stub_list=['ciiAPEX', '12co32'])
    # error = False
    # i = 0
    # while not error:
    #     try:
    #         fast_pv(reg_filename_or_idx=("catalogs/N19_pv_paths_2.reg", i), line_stub_list=['12co32', 'ciiAPEX'])
    #         plt.close(plt.gcf())
    #         i += 1
    #     except:
    #         error = True

    # pv_slice_series_overlay()

    """
    CO column
    """
    # find_co10_noise()

    """
    Channel movies
    """
    # channel_movie('ciiAPEX', vel_lims=(0, 10))

    """
    Spectra
    """
    # for i in range(1, 3):
    #     for j in range(1, 3):
    #         if i == j == 1:
    #             continue
    #         print("Doing", i, j)
    plot_spectra(reg_set_number=4, line_set_number=1, velocities_to_mark=(17.5, 19.5, 21.5))
    plot_spectra(reg_set_number=4, line_set_number=2, velocities_to_mark=(17.5, 19.5, 21.5))
