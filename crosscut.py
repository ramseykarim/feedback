import sys
import numpy as np
import matplotlib
font = {'family': 'sans', 'weight': 'normal', 'size': 6}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from scipy.interpolate import interpn
from math import ceil

from . import misc_utils
from . import catalog

"""
Functions and script to make a cross-cut plot using several different data sources
"""

def get_xcut_length(xcut_coords):
    """
    Separation between the two coordinates, returned as Quantity
    :param xcut_coords: tuple pair of SkyCoord objects
    :returns: Quantity angular distance
    """
    return xcut_coords[0].separation(xcut_coords[1])


def find_n_samples(wcs, xcut_length):
    """
    Given the WCS object for a FITS image, find the number of points along
    the xcut_length such that the original pixel scale is Nyquist sampled
    Uses ceiling rounding for integer number of samples
    :param wcs: WCS object
    :param xcut_length: Quantity angular separation
    :returns: integer number of samples
    """
    pixel_scale = misc_utils.get_pixel_scale(wcs)
    return int(ceil(xcut_length.deg / pixel_scale.to(u.deg).to_value())) * 2


def cross_cut(image, wcs, xcut_coords, n_points):
    """
    Return the values along a linear cut across an 2D image
    :param image: should be 2d array
    :param wcs: should be for 2d image
    :param xcut_coords: should be tuple(SkyCoords) start, end
    :param n_points: number of points across cut, including start & end
    :returns: 1D numpy array of length n_points, with values interpolated
        from the image
    """
    p0, p1 = (wcs.world_to_pixel(c) for c in xcut_coords)
    j_range, i_range = (np.linspace(p0[x], p1[x], n_points) for x in range(2))
    p_range = np.stack([i_range, j_range], axis=1)
    return interpn(tuple(np.arange(x) for x in image.shape), image, p_range, fill_value=np.nan, bounds_error=False, method='linear')

def get_angle_axis(xcut_coords, n_points):
    """
    Get angle separation from starting coordinate (xcut_coords[0])
      for each point along cross cut
    Returns arcseconds
    :param xcut_coords: tuple pair of SkyCoord objects
    :param n_points: number of samples between the coordinates, including
        the coordinates themselves (numpy.linspace style)
    :returns: 1D array of length n_points containing the distance from each
        point to the first coordinate, terminating at the second coordinate.
        Distances are in arcseconds, but the array is a vanilla numpy array
    """
    ra_range = np.linspace(*(xcut_coords[x].ra.to_value() for x in range(2)), n_points)
    dec_range = np.linspace(*(xcut_coords[x].dec.to_value() for x in range(2)), n_points)
    point_coords = SkyCoord(ra_range, dec_range, unit=u.deg)
    return point_coords.separation(xcut_coords[0]).to(u.arcsec).to_value()

def get_velocity_axis(wcs, vaxis=0):
    """
    Use WCS to make the velocity axis for a cube
    Always returns units in km/s
    It can be assumed that the velocity axis is sorted in increasing order.
    It would be the fault of the underlying header info if it weren't
    :param wcs: WCS object
    :param vaxis: NAXIS 3 (axis 0) should be velocity, but can use vaxis to specify
    """
    array_coords = np.zeros((wcs.pixel_n_dim, wcs.array_shape[vaxis]))
    array_coords[vaxis, :] = np.arange(wcs.array_shape[vaxis])
    v = wcs.array_index_to_world_values(*array_coords)[wcs.pixel_n_dim - vaxis - 1]
    # Use world_axis_units to convert to km/s
    return v * u.Unit(wcs.world_axis_units[wcs.pixel_n_dim - vaxis - 1]).to("km/s")

def load_image(filename, ext=0):
    """
    Get image and WCS for a regular 2D image
    :param filename: file path to 2D FITS image file
    :param ext: extension number, default 0
    :returns: 2D img array, WCS object
    """
    img, header = fits.getdata(filename, ext=ext, header=True)
    return img, WCS(header)

def load_cube(filename, vmin, vmax):
    """
    Get moment-0 image and WCS for a cube, RA-DEC-Velocity
    Use vmin and vmax to specify limits for moment-0 calculation
    Assumes velocity axis is 0 (NAXIS 3)
    :param filename: file path to 3D FITS cube file
    :param vmin: minimum velocity for cubes, in km/s
    :param vmax: maximum velocity for cubes, in km/s
    :returns: 2D moment 0 array, WCS object for 2D image
    """
    cube, header = fits.getdata(filename, header=True)
    vaxis = get_velocity_axis(WCS(header))
    # If vmin==vmax is in vaxis, then idx_min + 1 == idx_max
    idx_min = np.searchsorted(vaxis, vmin, side='left')
    idx_max = np.searchsorted(vaxis, vmax, side='right')
    img = np.mean(cube[idx_min:idx_max, :, :], axis=0)
    return img, WCS(header, naxis=2)


def load_general(filename, *args, **kwargs):
    # General function for doing all the loading
    # FILENAME should be the complete path of a FITS file
    # Either 2, 3, or 4 additional args should be given, depending
    #   on whether filename points to a 2D image or a 3D cube
    #   or if the image should be read from a different extension.
    # If 2D: 2 args, coords and n_points for cross_cut function
    # If 3D: 4 args, vmin, vmax for load_cube and then
    #   coords and n_points.
    # If different extension: 3 args, first arg is extension number,
    #   then coords and n_points.
    # Returns arguments for cross_cut
    if len(args) == 3:
        load_args = args[:2] # vmin, vmax
        args = args[2:] # coords; THIS IS TOO HACKY
        img, wcs = load_cube(filename, *load_args)
    elif len(args) == 2:
        load_args = args[0]
        args = args[1:]
        img, wcs = load_image(filename, ext=load_args)
    else:
        img, wcs = load_image(filename)
    # if 'f_to_apply' in kwargs:
    #     img = kwargs['f_to_apply'](img)
    img = np.log10(img - np.nanmin(img)) # INTERESTING: subtracting before log has an (unpredictable?) effect on the log curve
    return (img, wcs, *args)

def normalize_crosscut(xcut, rescale=True):
    """
    A few operations to comfortably line up all the cross cuts
    :param xcut: array to work on
    :param rescale: True if there should be a geometric factor applied to the
        array. False is useful if the array is log, in which case
        a geometric factor affects the apparent power law.
    :returns: same array as xcut argument, but offset and maybe normalized
    """
    # Subtract median
    subtracted = xcut - np.nanmedian(xcut)
    # Get rid of stars (mostly for HST)
    subtracted[subtracted > np.nanstd(subtracted)*5] = np.nan ## TODO: make this star-proof for F814W. note that stars are several pixels wide. (fit gaussian?)
    # Add back some value to get it above 0
    if rescale:
        # Normalize, if applicable
        subtracted /= (4 * np.nanstd(subtracted[np.isfinite(subtracted)]))
    return subtracted


data_path = catalog.utils.ancillary_data_path

cross_cuts_coords = {
    0: ("10:24:07.3706 -57:45:04.036", "10:24:39.7421 -57:41:21.431", -4.7, -3.7), # First one I tried
    "WR20b_1": ("10:24:23.3160 -57:48:22.958", "10:24:35.8602 -57:48:51.026", -10.8, -6.8),
    "WR20b_2": ("10:24:19.0298 -57:48:54.625", "10:24:15.1469 -57:50:00.728", 8., 10.4),
    "Wd2_N": ("10:23:55.2786 -57:42:33.456", "10:23:54.8181 -57:41:20.952", -6.6, -5.2),
    "Wd2_N_near": ("10:24:01.4203 -57:43:56.432", "10:24:01.4203 -57:42:43.929", -9.6, -8.2),
    "clear-across-1": ("10:23:29.8019 -57:46:56.589", "10:24:43.2086 -57:41:05.669", -25, 0), # To accompany the cartoon
    "clear-across-2": ("10:24:49.8687 -57:45:21.769", "10:23:20.1454 -57:45:33.656", -12, -4),
    "from-center-1": ("10:23:58.1 -57:45:49", "10:25:05.5470 -57:40:17.746", -25, 0), # Center from WR20a (Wd2 center) thru Wd2 MC to faraway
}


if False:
    data_path = catalog.utils.ancillary_data_path
    chandra_fn = f"{data_path}chandra/full_band.fullfield.diffuse_filled.flux"
    hdul = fits.open(chandra_fn)
    data = hdul[0].data
    print(data.shape)
    plt.imshow(data, origin='lower')
    plt.show()
    hdul.close()

if False:
    selection = "from-center-1"
    coord_start_xcut, coord_end_xcut = (SkyCoord(x, unit=(u.hourangle, u.deg)) for x in cross_cuts_coords[selection][:2])
    approx_midpoint = SkyCoord((coord_start_xcut.ra + coord_end_xcut.ra)/2, (coord_start_xcut.dec + coord_end_xcut.dec)/2)
    vlims = cross_cuts_coords[selection][2:]

    coords_xcut = (coord_start_xcut, coord_end_xcut)
    xcut_len = get_xcut_length(coords_xcut)
    n_points = 50
    xcut_args = (coords_xcut,)

    cuts_to_make = {
        # images just need filenames. cubes need velocity limits too.
        # "500 um": "herschel/helpssproc/processed/1342255009/SPIRE500um-image.fits",
        # "350 um": "herschel/helpssproc/processed/1342255009/SPIRE350um-image.fits",
        # "70 um": "herschel/helpssproc/processed/1342255009/PACS70um-image.fits", # GOOD
        "F814W": "hst/F814W.fits", # GOOD
        "843 MHz": "most/J1024M56.FITS",
        # "12CO": ("apex/apexCO/RCW49_12CO.fits",), # GOOD
        # "13CO": ("apex/apexCO/RCW49_13CO.fits",),
        "8 um": "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits", # GOOD
        "CII": ("sofia/rcw49-cii.fits",), # GOOD
        # "$\\tau_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 2),
        # "870 um": "apex/atlasgal/J102414-574658.fits", # GOOD
        # "$T_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 1),
        "0.5-7 keV": ("chandra/full_band.fits", np.arcsinh),
    }

    cuts_to_plot = {}

    for data_name in cuts_to_make:
        kwargs = {}
        if isinstance(cuts_to_make[data_name], str):
            # this is an image
            load_args = (data_path + cuts_to_make[data_name], *xcut_args)
            cuts_to_plot_key = data_name
        elif len(cuts_to_make[data_name]) == 1: # so messy. need object if using this in future.
            # this is a cube
            label = f"{data_name} [{vlims[0]:.1f}, {vlims[1]:.1f}] km/s"
            load_args = (data_path + cuts_to_make[data_name][0], *vlims, *xcut_args)
            cuts_to_plot_key = label
        elif len(cuts_to_make[data_name]) == 2:
            if isinstance(cuts_to_make[data_name][1], int):
                # This is an image to be read from another extension
                # At this point, these should probably be objects.........
                load_args = (data_path + cuts_to_make[data_name][0], cuts_to_make[data_name][1], *xcut_args)
                cuts_to_plot_key = data_name
            else:
                # this is an image with a funnction to apply
                load_args = (data_path + cuts_to_make[data_name][0], *xcut_args)
                cuts_to_plot_key = data_name
                kwargs['f_to_apply'] = cuts_to_make[data_name][1]
        # LoL this is so hacky
        args = load_general(*load_args, **kwargs)
        # This separation gives me a chance to intercept the arguments if I want
        w = args[1]
        cuts_to_plot[cuts_to_plot_key] = cross_cut(*args, find_n_samples(w, xcut_len))

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    for label in cuts_to_plot:
        normed_cut = normalize_crosscut(cuts_to_plot[label], rescale=False)
        angle_axis = get_angle_axis(*xcut_args, len(normed_cut))
        alpha = 0.2 if len(normed_cut) > 20000 else .9 # So hacky; please rewrite this
        plt.plot(np.log10(angle_axis[1:]), normed_cut[1:], label=label, linestyle='-', marker=None, alpha=alpha, lw=0.7)

    ############ NEED TO SORT THIS OUT, MAKE IT phi^-3 backwards!!!
    phi = angle_axis[1:]
    phi3 = -3 * np.log10(phi)
    phi3 -= np.min(phi3)
    phi3 /= (np.log10(2.) + 2)
    plt.plot(np.log10(phi), phi3, '--', lw=1, alpha=0.5, label='$\\phi^{-3}$')

    plt.ylabel("Normalized log intensity")
    plt.xlabel("Log$_{10}$ Distance along cross-cut [arcseconds]")
    # plt.ylim([-0.5, 1.2])
    plt.ylim([-0.5, 2.2])
    plt.xlim([1, plt.xlim()[1]])
    plt.axvline(x=np.log10(34), linestyle='-.', label='Wd2')
    plt.legend()

    img, w = load_image(data_path + cuts_to_make["8 um"])
    width = 15*u.arcmin # was 10
    img_cutout = Cutout2D(img, approx_midpoint, [width, width], wcs=w)
    plt.subplot(122, projection=img_cutout.wcs)
    plt.imshow(np.arcsinh(img_cutout.data), origin='lower', vmin=np.arcsinh(11), vmax=np.arcsinh(900), cmap='Greys_r')
    arrow = True # Looks a little better without arrow
    if arrow:
        x, y = coord_start_xcut.ra.deg, coord_start_xcut.dec.deg
        dx = (coord_end_xcut.ra - coord_start_xcut.ra).deg
        dy = (coord_end_xcut.dec - coord_start_xcut.dec).deg
        plt.arrow(x, y, dx, dy,
            transform=plt.gca().get_transform('world'), color='r', length_includes_head=True, width=0.003)
    else:
        plt.plot([coord_start_xcut.ra.deg, coord_end_xcut.ra.deg],
            [coord_start_xcut.dec.deg, coord_end_xcut.dec.deg],
            transform=plt.gca().get_transform('world'), color='r')
    plt.show()
    # plt.savefig(f"/home/rkarim/Pictures/4-07-20-work/crosscut_{selection}.png")




class DataLayer:
    """
    This class is designed to hold one source of image or cube data.
    It will implement loading functions that make it very easy to take
        cross cuts or azimuthal averages without knowing the specifics of the
        data source.
    Written: July 6, 2020
    """
    def __init__(self, name, filepath, cube=False, extension=0, f_to_apply=None,
        alpha=0.9):
        self.name = name
        self.filepath = catalog.utils.ancillary_data_path + filepath
        self.is_cube = cube
        self.extension = extension
        self.f_to_apply = f_to_apply
        self.alpha = alpha # For plotting

    def load(self, vmin=None, vmax=None):
        """
        Use the pre-built (by me, in first version) load functions for cubes
            and images.
        Could, in the future, update this to use SpectralCube for cubes, but
            I don't think there's a need for that (other than readability)
        :param vmin: minimum velocity for cubes, in km/s. Not used for images.
        :param vmax: maximum velocity for cubes, in km/s. Not used for images.
        """
        if self.is_cube:
            img, wcs = load_cube(self.filepath, vmin, vmax)
        else:
            img, wcs = load_image(self.filepath, ext=self.extension)
        if self.f_to_apply is not None:
            img = self.f_to_apply(img)
        return img, wcs

    def __hash__(self):
        """
        In case we want to store this in a set
        """
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


    def cross_cut(self, cross_cut_obj):
        """
        Use the load function to make a cross cut; return both the spatial (x)
            and intensity (y) arrays
        :param cross_cut_obj: CrossCut instance
        Remaining kwargs (vmin, vmax) are passed to load
        """
        img, wcs = self.load(**cross_cut_obj.vlim_kwargs())
        n_samples = find_n_samples(wcs, cross_cut_obj.len)
        x_array = get_angle_axis(cross_cut_obj.coords, n_samples)
        y_array = cross_cut(img, wcs, cross_cut_obj.coords, n_samples)
        return x_array, y_array

    def label(self, cross_cut_obj):
        if self.is_cube:
            return f"{self.name} [{cross_cut_obj.vlims[0]:.1f}, {cross_cut_obj.vlims[1]:.1f}] km/s"
        else:
            return self.name



class CrossCut:
    """
    This class will manage an entire cross-cut figure.
    Written: July 6-7, 2020
    """
    def __init__(self, xcut_coords, vlims=None, log=False):
        """
        :param xcut_coords: tuple of 2 SkyCoord objects describing the
            beginning and end of the 1D cross-cut
        :param vlims: tuple of 2 numbers describing the low and high velocity
            limits for cube moment calculation, in km/s. If it is left None,
            a default of [-100, 100] will be used.
        """
        self.coords = xcut_coords
        self.len = get_xcut_length(xcut_coords)
        self.approx_midpoint = SkyCoord((xcut_coords[0].ra + xcut_coords[1].ra)/2, (xcut_coords[0].dec + xcut_coords[1].dec)/2)
        if vlims is None:
            self.vlims = (-100, 100)
        else:
            self.vlims = tuple(vlims)
        self.layers = {}
        self.already_plotted = set()
        self.fig, self.axes = None, None
        self.log = log

    def add_data_layer(self, *data_layers):
        """
        Add an arbitrary number of DataLayer objects to the cross cut diagram
        :param data_layers: any number of DataLayer objects as arguments
        """
        for layer in data_layers:
            self.layers[layer.name] = layer

    def setup_figure(self, fig=None, figsize=(16, 8), xcut_axis=None):
        """
        Create figure and axes
        """
        if fig is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig
        if xcut_axis is None:
            self.axes = {'xcut': plt.subplot(121), 'img': None}
        else:
            self.axes = {'xcut': xcut_axis, 'img': None}

    def set_axis_limits(self, xlim, ylim):
        """
        Set x and y limits on the cross cut figure
        :param xlim: argument to plt.xlim
        :param ylim: argument to plt.ylim
        """
        self.switch_axes()
        plt.xlim(xlim)
        plt.ylim(ylim)

    def switch_axes(self, subplot_name='xcut'):
        """
        Quickly switch axes using the axis tag
        """
        plt.sca(self.axes[subplot_name])

    def update_plot(self):
        """
        Add any layers that haven't been plotted already
        """
        # Switch to xcut axis (default)
        self.switch_axes()
        for layer_name in self.layers:
            if layer_name in self.already_plotted:
                # Already done, skip it
                continue
            layer = self.layers[layer_name]
            # Give this instance as an argument to DataLayer.cross_cut;
            # we already have all the info it needs
            angle_array, cut_array = layer.cross_cut(self)
            # Apply log; get rid of the first element (0 distance) since log
            if self.log:
                # cut_array[cut_array <= 0] = np.nanmin(cut_array[cut_array > 0])/10
                cut_array = np.log10(cut_array[1:])
                angle_array = np.log10(angle_array[1:])
            # Decide whether we should rescale while offsetting
            rescale = not self.log
            # Normalize/offsef the array
            cut_array = normalize_crosscut(cut_array, rescale=rescale)
            plt.plot(angle_array, cut_array, label=layer.label(self),
                linestyle='-', marker=None, alpha=layer.alpha, lw=0.7)
            # Record that we already did this
            self.already_plotted.add(layer.name)
        plt.legend()

    def overplot_power_law(self, exponent=-3, x_intercept=9., exp_label=None,
        end_x=5, linestyle='--', alpha=0.5, **plot_kwargs):
        """
        Overplot a power law onto the cross cut diagram.
        :param exponent: the exponent of the power law
        :param x_intercept: the x-intercept of the log power law;
            alternatively, the log10 x value when linear y = 1
        """
        x_array = np.linspace(-1, end_x, 3)
        y_intercept = -1 * exponent * x_intercept
        y_array = exponent * x_array + y_intercept
        self.switch_axes()
        if exp_label is None:
            exp_label = f'{exponent}'
        label = '$\\phi^{' + exp_label + '}$'
        plt.plot(x_array, y_array, linestyle=linestyle, alpha=alpha, label=label,
            **plot_kwargs)
        plt.legend()


    def plot_image(self, layer_to_plot, vlims=None, stretch='arcsinh',
        subplot_number=122):
        """
        Plot an image with a superimposed arrow illustrating the cross cut`
        :param layer_to_plot: the layer name of the layer to use.
            If it's a cube, it'll be the moment 0 map with limits described here
        :param vlims: visual limits for plotting the image, specified in linear
        """
        layer = self.layers[layer_to_plot]
        img, wcs = layer.load(**self.vlim_kwargs())
        # Make a cutout about 2x the length of the cross cut
        width = 2 * self.len
        img_cutout = Cutout2D(img, self.approx_midpoint, [width, width], wcs=wcs,
            mode='partial', fill_value=np.nan)
        # Use specified stretch
        if isinstance(stretch, str):
            stretch = {'arcsinh': np.arcsinh, 'linear': lambda x: x, 'log': np.log10, 'sqrt': np.sqrt}[stretch]
        stretched_image = stretch(img_cutout.data)
        # Use flquantiles for min, max unless we specified through vlims
        if vlims is None:
            lo, hi = misc_utils.flquantiles(stretched_image[np.isfinite(stretched_image)].flatten(), 10000)
        else:
            lo, hi = stretch(np.array(vlims))
        self.axes['img'] = plt.subplot(subplot_number, projection=img_cutout.wcs)
        plt.imshow(stretched_image, origin='lower', vmin=lo, vmax=hi, cmap='Greys_r')
        # Prepare to plot the line or arrow showing the cross cut
        plot_kwargs = dict(color='r', transform=plt.gca().get_transform('world'))
        coord_start_xcut, coord_end_xcut = self.coords
        arrow = True # can think about this later
        if arrow:
            x, y = coord_start_xcut.ra.deg, coord_start_xcut.dec.deg
            dx = (coord_end_xcut.ra - coord_start_xcut.ra).deg
            dy = (coord_end_xcut.dec - coord_start_xcut.dec).deg
            plt.arrow(x, y, dx, dy, length_includes_head=True, width=0.002,
                **plot_kwargs, alpha=0.3)
        else:
            plt.plot([coord_start_xcut.ra.deg, coord_end_xcut.ra.deg],
                [coord_start_xcut.dec.deg, coord_end_xcut.dec.deg],
                **plot_kwargs, alpha=0.3)



    def vlim_kwargs(self):
        """
        Return vlims as dict for kwargs, with some write-protection on the
            values (returning new dictionary)
        """
        return dict(vmin=self.vlims[0], vmax=self.vlims[1])

# (__name__ == "__main__")
if True:
    selection = "from-center-1"
    coords = tuple(SkyCoord(x, unit=(u.hourangle, u.deg)) for x in cross_cuts_coords[selection][:2])
    vlims = cross_cuts_coords[selection][2:]
    cross_cut_obj = CrossCut(coords, vlims=vlims, log=True)
    cross_cut_obj.setup_figure()
    layers = [
        DataLayer("CII", "sofia/rcw49-cii.fits", cube=True, alpha=0.5),
        DataLayer("843 MHz", "most/J1024M56.FITS"),
        DataLayer("8 um", "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits"),
        DataLayer("F814W", "hst/F814W.fits", alpha=0.2),
        DataLayer("0.5-7 keV", "chandra/full_band.fits"),
    ]
    cross_cut_obj.add_data_layer(*layers)
    cross_cut_obj.update_plot()
    cross_cut_obj.overplot_power_law(x_intercept=2.6, alpha=0.7, linestyle='-.')
    cross_cut_obj.overplot_power_law(exponent=-(4./3), x_intercept=2.45, exp_label="-4/3", end_x=2.45)
    cross_cut_obj.overplot_power_law(exponent=-1, x_intercept=1.5, lw=0.5)
    cross_cut_obj.set_axis_limits((0, 3), (-1.5, 1.5))
    # CII: vlims=(0, 11), 8um: vlims=(11, 900)
    cross_cut_obj.plot_image('8 um', stretch='arcsinh', vlims=(11, 900))
    plt.show()



"""
======================================================================
try the Churchwell-like azimuthal average (maybe in 8 sections or something)

use astropy azimuth thing
"""


if False:
    data_path = catalog.utils.ancillary_data_path
    center_coord = catalog.utils.wd2_center_coord
    full_radius = 7.*u.arcmin
    n_points = 50
    xcut_args = (center_coord, full_radius, n_points)

    cuts_to_make = {
        # images just need filenames. cubes need velocity limits too.
        # "500 um": "herschel/helpssproc/processed/1342255009/SPIRE500um-image.fits",
        # "350 um": "herschel/helpssproc/processed/1342255009/SPIRE350um-image.fits",
        # "70 um": "herschel/helpssproc/processed/1342255009/PACS70um-image.fits", # GOOD
        # "843 MHz": "most/J1024M56.FITS",
        # "12CO": ("apex/apexCO/RCW49_12CO.fits",), # GOOD
        # "13CO": ("apex/apexCO/RCW49_13CO.fits",),
        "CII": ("sofia/rcw49-cii.fits",), # GOOD
        # "8 um": "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits", # GOOD
        # "F814W": "hst/F814W.fits", # GOOD
        # "$\\tau_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 2),
        # "870 um": "apex/atlasgal/J102414-574658.fits", # GOOD
        # "$T_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 1),
        # "0.5-7 keV": ("chandra/full_band.fits", np.arcsinh),
    }
