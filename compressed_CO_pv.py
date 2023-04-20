"""
I want to look at a couple features in RCW 49 that I think are CO clouds with
a shock front(?) being pushed through them.
I'll make a couple PV diagrams here to see what's going on.
Created: October 8, 2020
"""
__author__ = "Ramsey Karim"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import pvextractor
import regions

from . import catalog
from . import cube_utils
# This has some stuff I'll probably want to use
from . import cube_pixel_spectra as cps1
from . import cube_pixel_spectra_2 as cps2
from . import pvdiagrams
from . import misc_utils

western_info_dict = {
    'filename': catalog.utils.search_for_file("catalogs/western-COcloud-pv-cuts.reg"),
    'vlims': (-15, 8),
    'cii_img_lims': (0, 120),
    'cii_pv_lims': (0, 11),
    '12co_img_lims': (0, 130),
    '12co_pv_lims': (0, 20),
}

ridge_info_dict = {
    'filename': catalog.utils.search_for_file("catalogs/ridge-COcloud-pv-cuts.reg"),
    'vlims': (12, 30),
    'cii_img_lims': (0, 210),
    'cii_pv_lims': (0, 35),
    '12co_img_lims': (0, 50),
    '12co_pv_lims': (0, 12),
}

data_filenames = {
    'cii': catalog.utils.search_for_file("sofia/rcw49-cii.fits"),
    '12co': catalog.utils.search_for_file("apex/apexCO/RCW49_12CO.fits")
}

region_info_dict = western_info_dict
dataset = 'cii'

def plot_pv(reg_index):
    """
    First, simplest attempt: plot a PV diagram
    """
    data_filename = data_filenames[dataset]

    path_list = pvextractor.paths_from_regfile(region_info_dict['filename'])
    reg_list = regions.read_ds9(region_info_dict['filename'])
    # print(len(reg_list))
    # print([x.meta['text'] for x in reg_list])
    # return
    subcube = cps2.cutout_subcube(data_filename=data_filename, reg_filename=region_info_dict['filename']).spectral_slab(*(x*u.km/u.s for x in region_info_dict['vlims']))

    reference_image = subcube.moment(order=0)
    ref_img_stretch = misc_utils.check_stretch('linear')
    ref_img_val = reference_image.to_value()
    ref_lo, ref_hi = misc_utils.flquantiles(ref_img_val[np.isfinite(ref_img_val)], 100)
    ref_med = np.nanmedian(ref_img_val)
    ref_lo = max(ref_med, ref_lo)
    print(f"Image: low: {ref_lo}, med: {ref_med}, high: {ref_hi}")

    fig_img = plt.figure("Reference image", figsize=(5, 4))
    ax_img = fig_img.add_axes([0.05, 0.05, 0.8, 0.9], projection=reference_image.wcs)
    ref_lo, ref_hi = region_info_dict[dataset+'_img_lims']
    im = ax_img.imshow(ref_img_stretch(ref_img_val), origin='lower', vmin=ref_img_stretch(ref_lo), vmax=ref_img_stretch(ref_hi))
    ax_img_divider = make_axes_locatable(ax_img)
    ax_img_cb = ax_img_divider.append_axes('top', size='7%', pad='2%', axes_class=mpl_axes.Axes)
    fig_img.colorbar(im, cax=ax_img_cb, orientation='horizontal')
    ax_img_cb.xaxis.set_ticks_position('top')
    linestyles = ['-', '--', 'dotted', '-.']
    used_colors = dict()
    def check_color(color):
        if color in used_colors:
            try:
                return next(used_colors[color])
            except StopIteration as e:
                raise RuntimeError(f"Ran out of linestyles for {color}") from e
        else:
            used_colors[color] = iter(linestyles)
            return next(used_colors[color])

    artists = []
    fig_pv = plt.figure("PV diagrams", figsize=(6, 6))
    for i, (path, reg) in enumerate(zip(path_list, reg_list)):
        sl = pvextractor.extract_pv_slice(subcube, path)
        sl_stretch = misc_utils.check_stretch('linear')
        sl_val = sl_stretch(sl.data)
        sl_lo, sl_hi = misc_utils.flquantiles(sl_val[np.isfinite(sl_val)].flatten(), 300)
        sl_med = np.nanmedian(sl_val)
        sl_lo = min(sl_med, sl_lo)

        ax_pv = plt.subplot2grid(total_len_to_grid(len(reg_list)), index_to_grid(i, len(reg_list)), colspan=8, projection=WCS(sl.header), fig=fig_pv)
        sl_lo, sl_hi = region_info_dict[dataset+'_pv_lims']
        im = ax_pv.imshow(sl.data, origin='lower', aspect=(sl.data.shape[1]/sl.data.shape[0]), vmin=sl_lo, vmax=sl_hi)
        # fig_pv.colorbar(im, ax=ax_pv)
        ax_pv.coords[1].set_format_unit(u.km/u.s)
        ax_pv.set_title(reg.meta['text'])

        reg_pix = reg.to_pixel(reference_image.wcs)
        reg_pix.visual.update(reg.visual)
        reg_pix.meta.update(reg.meta)
        linestyle = check_color(reg_pix.visual['color'])
        artist = reg_pix.as_artist(width=12, linewidth=1.2, fill=False, linestyle=linestyle)
        ax_img.add_artist(artist)
        artists.append(artist)

    ax_img.legend(handles=ax_img.artists, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig_pv.subplots_adjust(left=0.03, right=0.9, wspace=0.1, hspace=0.2, bottom=0.04, top=0.975)
    ax_pv_cb = fig_pv.add_axes([0.92, 0.03, 0.04, 0.95])
    fig_pv.colorbar(im, cax=ax_pv_cb)
    plt.show()


def total_len_to_grid(total_len):
    if total_len == 9:
        return (3, 25)
    elif total_len == 5:
        return (2, 25)

def index_to_grid(index, total_len):
    """
    Convert a flat index (integer) to a grid location.
    """
    if total_len == 9:
        i, j = np.unravel_index(index, (3, 3))
        return i, j*8
    elif total_len == 5:
        i, j = np.unravel_index(index, (2, 3))
        return i, j*8

if __name__ == "__main__":
    plot_pv(0)
