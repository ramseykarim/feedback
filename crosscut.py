import sys
import numpy as np
import matplotlib
if __name__ == "__main__":
    font = {'family': 'sans', 'weight': 'normal', 'size': 6}
    matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib import patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from scipy.interpolate import interpn
from math import ceil

from . import pvdiagrams
misc_utils = pvdiagrams.misc_utils
catalog = pvdiagrams.catalog
cube_utils = pvdiagrams.cube_utils
mpl_cm = pvdiagrams.mpl_cm
mpl_colors = pvdiagrams.mpl_colors

"""
Functions and script to make a cross-cut plot using several different data sources
Updated July 27, 2020 to work with some M16 data. Hoping to generalize this stuff.
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


def offset_crosscut(xcut):
    """
    A few operations to comfortably line up all the cross cuts
    :param xcut: array to work on
    :returns: same array as xcut argument, but offset
    """
    # Subtract median
    subtracted = xcut - np.nanmedian(xcut)
    # Get rid of stars (mostly for HST)
    subtracted[subtracted > np.nanstd(subtracted)*5] = np.nan ## TODO: make this star-proof for F814W. note that stars are several pixels wide. (fit gaussian?)
    # Add back some value to get it above 0
    return subtracted

def normalize_crosscut(xcut):
    """
    Rescale operation to comfortably line up all the cross cuts
    :param xcut: array to work on
    :returns: same array as xcut argument, but normalized
    """
    # Normalize
    return xcut / (4 * np.nanstd(xcut[np.isfinite(xcut)]))


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
    "thru-clcenter-1": ("10:22:59.3030 -57:49:59.699", "10:25:02.3860 -57:41:11.252", -25, 0), # similar to from-center-1
    "thru-clcenter-2": ("10:23:49.0980 -57:53:50.384", "10:24:14.4021 -57:37:19.339", -25, 0), # more N-S than thru-clcenter-1
    "thru-clcenter-3": ("10:23:12.0398 -57:41:53.733", "10:25:02.2100 -57:49:57.528", -25, 0), # crosses bright ridge
    "M16-marc-pillar2": ("18:18:55.2663 -13:51:17.4481", "18:18:47.7341 -13:49:35.1483", 19, 24), # Marc's pillar 2 PV diagram cut
    "M16-pillar1": ("18:19:00.5191 -13:51:16.046", "18:18:49.8401 -13:48:29.025", 19, 24), # Marc's pillar 2 PV diagram cut

}


def coords_from_selection(selection):
    return tuple(SkyCoord(x, unit=(u.hourangle, u.deg)) for x in cross_cuts_coords[selection][:2])


def vlims_from_selection(selection):
    return cross_cuts_coords[selection][2:]


def coords_from_region(reg_file_name, index=0):
    """
    Use the path_from_ds9 function in pvdiagrams.py to get linear paths
    The index shouldn't count non-line/vector regions in the file.
    Nov 12, 2020: updated to return list of paths if path_from_ds9 returns list
        this would happen if index==None
    """
    # Load the vector or line at the given index (defaults to the first one)
    p = pvdiagrams.path_from_ds9(reg_file_name, index)
    # Pull the coordinates from the line/vector and return SkyCoord tuple
    if isinstance(p, list):
        return [tuple(SkyCoord(x) for x in path._coords) for path in p]
    else:
        return tuple(SkyCoord(x) for x in p._coords)


file_info = {
    ###### images just need filenames. cubes need velocity limits too.
    "500 um": "herschel/helpssproc/processed/1342255009/SPIRE500um-image.fits",
    "350 um": "herschel/helpssproc/processed/1342255009/SPIRE350um-image.fits",
    "70 um": "herschel/helpssproc/processed/1342255009/PACS70um-image.fits", # GOOD
    "F814W": "hst/F814W.fits", # GOOD
    "843 MHz": "most/J1024M56.FITS",
    "12CO": "apex/apexCO/RCW49_12CO.fits", # GOOD
    "13CO": "apex/apexCO/RCW49_13CO.fits",
    "8 um": "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits", # GOOD
    "CII": "sofia/rcw49-cii.fits", # GOOD
    "$\\tau_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 2),
    "870 um": "apex/atlasgal/J102414-574658.fits", # GOOD
    "$T_{d}$": ("herschel/RCW49large_2p_2BAND_500grid_beta1.7.fits", 1),
    "0.5-7 keV": "chandra/full_band.fits",
}


class DataLayer:
    """
    This class is designed to hold one source of image or cube data.
    It will implement loading functions that make it very easy to take
        cross cuts or azimuthal averages without knowing the specifics of the
        data source.
    Written: July 6, 2020
    Updated July 27, 2020: now accepts CubeData instances under the "filename"
        argument, and can "load" and plot them properly
    Updated September 9, 2020: supports a DataLayer-level velocity limit
        that supersedes the CrossCut-level limit, so that multiple velocity
        components from the same cube can be overplotted.
        If vlims is set here, then a CubeData object can be reused.
    """

    def __init__(self, name, filepath, cube=False, extension=0, f_to_apply=None,
        alpha=0.9, offset=False, vlims=None, color=None, linestyle='-', norm=None, linewidth=0.7):
        self.name = name
        if isinstance(filepath, str):
            # Direct load from filepath method
            self.filepath = catalog.utils.search_for_file(filepath)
            self.cube_obj = None
            self.is_cube = cube
        elif isinstance(filepath, cube_utils.CubeData):
            # CubeData instance method
            self.cube_obj = filepath
            self.filepath = self.cube_obj.full_path
            if self.name is None:
                self.name = self.cube_obj.name
            self.is_cube = True  # ignore the cube keyword
        self.extension = extension
        self.f_to_apply = f_to_apply
        self.alpha = alpha # For plotting
        self.color = color # For plotting
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.norm_coeff = 1
        if misc_utils.is_number(norm):
            self.norm = True
            self.norm_coeff = norm
        else:
            self.norm = norm
        if offset:
            # Some kind of vertical offsetting procedure
            if callable(offset):
                self.offset = offset
            elif misc_utils.is_number(offset):
                # Some zero point to subtract
                self.offset = lambda x: x - float(offset)
            else:
                self.offset = offset_crosscut
        else:
            self.offset = lambda x: x
        # Set personal velocity limits; defaults to None
        assert (vlims is None) or (hasattr(vlims, '__len__') and len(vlims) == 2)
        self.vlims = vlims
        # A way to save moment images and stuff that isn't that memory intensive
        # Format will be [img, wcs, vlims], and only saves one at a time
        self.img_to_reuse = None


    def load(self, vmin=None, vmax=None):
        """
        Use the pre-built (by me, in first version) load functions for cubes
            and images.
        Could, in the future, update this to use SpectralCube for cubes, but
            I don't think there's a need for that (other than readability)
        Update July 27, 2020: this now works for CubeData instances, which
            wrap SpectralCube instances. I'll still rely on the regular
            file-reading version for some types of data.
        :param vmin: minimum velocity for cubes, in km/s. Not used for images.
        :param vmax: maximum velocity for cubes, in km/s. Not used for images.
        If self.vlims is not None, the vlim arguments here are ignored
            and self.vlims is used instead
        """
        if self.vlims is not None:
            vmin, vmax = self.vlims
        if self.is_cube:
            if self.img_to_reuse is not None and (vmin, vmax) == self.img_to_reuse[2]:
                # We have a moment image cached
                img, wcs = self.img_to_reuse[:2]
            else:
                # No saved img for these vlims, make a new moment0
                if self.cube_obj is None:
                    img, wcs = load_cube(self.filepath, vmin, vmax)
                else:
                    km_s = u.km / u.s
                    mom0 = self.cube_obj.data.spectral_slab(vmin*km_s, vmax*km_s).moment(order=0)
                    try:
                        img = mom0.to(u.K*km_s).to_value()
                    except:
                        img = ((mom0/km_s).to(u.K, equivalencies=self.cube_obj.equivalency())*km_s).to_value()
                    wcs = self.cube_obj.wcs_flat
                # Cache the image for later
                self.img_to_reuse = [img, wcs, (vmin, vmax)]
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
        if self.vlims is None:
            vmin, vmax = cross_cut_obj.vlims
        else:
            vmin, vmax = self.vlims
        if self.is_cube:
            return f"{self.name} [{vmin:.1f}, {vmax:.1f}] km/s"
        else:
            return self.name


class CrossCut:
    """
    This class will manage an entire cross-cut figure.
    Written: July 6-7, 2020
    Nov 12 2020 Note: i should rewrite this so I can efficiently plot several
        cuts
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
            plt.figure(fig.number)
            self.fig = fig
        if xcut_axis is None:
            self.axes = {'xcut': plt.subplot(121), 'img': None}
        else:
            if isinstance(xcut_axis, int):
                xcut_axis = plt.subplot(xcut_axis)
            self.axes = {'xcut': xcut_axis, 'img': None}
        return self.fig

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
        :returns: the switched-to axis
        """
        plt.sca(self.axes[subplot_name])
        return self.axes[subplot_name]

    def update_plot(self, norm=True, legend=True):
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
            # Normalize/offset the array
            cut_array = layer.offset(cut_array)
            if (not self.log) and (norm if layer.norm is None else layer.norm):
                cut_array = normalize_crosscut(cut_array) * layer.norm_coeff
            plt.plot(angle_array, cut_array, label=layer.label(self),
                marker=None, alpha=layer.alpha, lw=layer.linewidth, color=layer.color, linestyle=layer.linestyle)
            # Record that we already did this
            self.already_plotted.add(layer.name)
        if legend:
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
        subplot_number=122, line_color='r', cmap='Greys_r', cutout=True):
        """
        Plot an image with a superimposed arrow illustrating the cross cut`
        :param layer_to_plot: the layer name of the layer to use.
            If it's a cube, it'll be the moment 0 map with limits described here
        :param vlims: visual limits for plotting the image, specified in linear
        """
        if isinstance(layer_to_plot, str):
            layer = self.layers[layer_to_plot]
            img, wcs = layer.load(**self.vlim_kwargs())
            if cutout:
                # Make a cutout about 2x the length of the cross cut
                width = 2 * self.len
                img_cutout = Cutout2D(img, self.approx_midpoint, [width, width], wcs=wcs,
                    mode='partial', fill_value=np.nan)
                img = img_cutout.data
                wcs = img_cutout.wcs
            # Find the specified stretch, or confirm callable
            stretch = misc_utils.check_stretch(stretch)
            # Use specified stretch
            stretched_image = stretch(img)
            # Use flquantiles for min, max unless we specified through vlims
            if vlims is None:
                lo, hi = misc_utils.flquantiles(stretched_image[np.isfinite(stretched_image)].flatten(), 10000)
            else:
                lo, hi = stretch(np.array(vlims))
            self.axes['img'] = plt.subplot(subplot_number, projection=wcs)
            plt.imshow(stretched_image, origin='lower', vmin=lo, vmax=hi, cmap=cmap)
        elif isinstance(layer_to_plot, CrossCut):
            layer_to_plot.switch_axes('img')
        # Prepare to plot the line or arrow showing the cross cut
        plot_kwargs = dict(color=line_color, transform=catalog.utils.get_transform())
        coord_start_xcut, coord_end_xcut = self.coords
        arrow = False # can think about this later
        line_arrow_alpha = 0.8
        if arrow:
            x, y = coord_start_xcut.ra.deg, coord_start_xcut.dec.deg
            dx = (coord_end_xcut.ra - coord_start_xcut.ra).deg
            dy = (coord_end_xcut.dec - coord_start_xcut.dec).deg
            arrow_width = data_length_from_display_length(0.0011661145290986497, coord_start_xcut)  # Converted from 0.002 for RCW 49
            arrow_head_width = arrow_width * 10  # Converted from 0.02 for RCW 49
            arrow_head_length = arrow_head_width * 2  # Converted from 0.04 for RCW 49
            plt.arrow(x, y, dx, dy, length_includes_head=True, width=arrow_width,
                **plot_kwargs, alpha=line_arrow_alpha,
                head_width=arrow_head_width, head_length=arrow_head_length)
        else:
            plt.plot([coord_start_xcut.ra.deg, coord_end_xcut.ra.deg],
                [coord_start_xcut.dec.deg, coord_end_xcut.dec.deg],
                **plot_kwargs, alpha=line_arrow_alpha, lw=3)

    def mark_radius(self, radius, label=False, **plot_kwargs):
        """
        Overplot onto the image axis a circle of given radius originating
        from the first coordinate.
        Also mark this radius on the cross-cut diagram with a vertical line
        of the same appearance.
        :param radius: must be Quantity, angular unit
        :param plot_kwargs: any kwargs to pass to BOTH Ellipse and axvline.
        """
        # Make the circle
        self.switch_axes('img')
        center_coord = self.coords[0]
        x, y = center_coord.ra.deg, center_coord.dec.deg
        width_dec = radius.to(u.deg).to_value() * 2
        width_ra = width_dec / np.cos(center_coord.dec.rad)
        circle_patch = patches.Ellipse((x, y), width_ra, width_dec,
            transform=catalog.utils.get_transform(), fill=False,
            **plot_kwargs)
        self.axes['img'].add_patch(circle_patch)
        # Now make the vertical line
        self.switch_axes('xcut')
        radius_to_mark = radius.to(u.arcsec).to_value()
        if self.log:
            radius_to_mark = np.log10(radius_to_mark)
        if label:
            if not isinstance(label, str):
                label = f"$r = {radius.to(u.arcsec).to_value():.1f}''$"
        plt.axvline(radius_to_mark, **plot_kwargs, label=label)
        if label:
            plt.legend()

    def mark_distance(self, distance, label=False, **plot_kwargs):
        """
        Very similar to mark_radius, but doesn't imply azimuthal symmetry.
        Instead of a circle on the image plot, overlays a small hatch mark.
        :param distance: must be a Quantity, angular unit
        :param plot_kwargs: any kwargs to pass to BOTH the small mark and the
            axvline
        """
        ax = self.switch_axes('img')
        """
        TODO: use matplotlib Arc patch, could copy a lot of the code from above
        Need to get the position angle of the cross cut endpoints, SkyCoord
        has a method for that. Also arc length, probably just limit it to a
        useful, general, fixed length
        """


    def vlim_kwargs(self):
        """
        Return vlims as dict for kwargs, with some write-protection on the
            values (returning new dictionary)
        """
        return dict(vmin=self.vlims[0], vmax=self.vlims[1])


"""
Some limited-use plotting helper functions
These work with transforms between data and display coordinates
"""

def display_length_from_data_length(data_length, reference_coord, axis=None):
    """
    Calculate a LENGTH in display units using a known length
    in data units.
    This is helpful for the ARROW function in matplotlib.

    This relies on modifying the DEC coordinate to check length, so as long as
    the reference_coord isn't so close to a pole that adding the data_length to
    it pushes it past the pole, then we should be fine regardless of Dec.
    :param data_length: float length in data units (probably degrees for a
        plot using WCS)
    :param reference_coord: SkyCoord reference (data) coordinate
    :param axis: optional, axis for transformation. If None, uses plt.gca()
    :returns: float length in display units
    """
    if axis is None:
        axis = plt.gca()
    # Split up x (RA) and y (DEC) data coordinates
    x0, y0 = reference_coord.ra.deg, reference_coord.dec.deg
    # Displace y (DEC, since dec displacement is always in degrees, RA displacement varies with DEC)
    y1 = y0 + data_length
    # Transform both of these coordinates to display coordinates
    # and then return the separation between display coordinates
    return np.sqrt(np.sum(np.subtract(*axis.transData.transform([(x0, y0), (x0, y1)]))**2.))


def data_length_from_display_length(display_length, reference_coord, axis=None):
    """
    Calculates a LENGTH in data units using a known length in display units.
    Inverse of the display_length_from_data_length function above.

    Do NOT trust these 100% as physical distances. ARROW clearly mishandles
    sky coordinate transformations, so this function hacks around that.
    These are pretty close to physical lengths! But can't be too careful.
    Same as above, independent of DEC unless very close to poles.
    :param display_length: float length in display units
    :param reference_coord: SkyCoord reference (data) coordinate
    :param axis: optional, axis for transformation. If None, uses plt.gca()
    :returns: float length in data units
    """
    if axis is None:
        axis = plt.gca()
    # Generate the inverse transform
    inv = axis.transData.inverted()
    # Split up x (RA) and y (DEC)
    x0, y0 = reference_coord.ra.deg, reference_coord.dec.deg
    # Convert the reference data coord to a display coord
    reference_display_coord = axis.transData.transform((x0, y0))
    # Modify the display y coordinate by the length
    reference_display_coord[1] += display_length
    # Transform the modified display coord back to data coords
    xy1 = inv.transform(reference_display_coord)
    # Find and return the separation between these data coordinates
    return np.sqrt(np.sum((np.array([x0, y0]) - xy1)**2.))


"""
Actual cross-cut stuff again
"""

def prepare_layers(target='rcw49'):
    if target.lower() == 'rcw49':
        layers = [
            DataLayer("CII", "sofia/rcw49-cii.fits", cube=True, alpha=0.7, offset=-0.1),
            DataLayer("843 MHz", "most/J1024M56.FITS", offset=-0.1),
            DataLayer("8 um", "spitzer/irac/30002561.30002561-28687.IRAC.4.median_mosaic.fits", offset=2.2),
            DataLayer("F814W", "hst/F814W.fits", alpha=0.2, offset=-0.8),
            DataLayer("0.5-7 keV", "chandra/full_band.fits", offset=-9),
        ]
    elif target.lower() == 'm16':
        layers = [
            DataLayer("CII", "sofia/M16_CII_U.fits", cube=True, alpha=0.7),
            DataLayer("12CO(1-0)", cube_utils.CubeData("bima/M16_12CO1-0_7x4.fits"), cube=True, alpha=0.7),
            # DataLayer("13CO(1-0)", cube_utils.CubeData("bima/M16_13CO1-0_7x4.fits"), cube=True, alpha=0.7),
            # DataLayer("12CO(3-2)", "")
            DataLayer("12CO(3-2)", "apex/M16_12CO3-2.fits", cube=True, alpha=0.7),
            DataLayer("13CO(3-2)", "apex/M16_13CO3-2.fits", cube=True, alpha=0.7),
            DataLayer("5.6 um", "spitzer/SPITZER_I3_6049792_0000_5_E8698528_maic.fits"),
        ]
    return layers

def single_plot_rcw49():
    selection = "from-center-1"
    coords = coords_from_selection(selection)
    vlims = vlims_from_selection(selection)
    cross_cut_obj = CrossCut(coords, vlims=vlims, log=True)
    cross_cut_obj.setup_figure()
    layers = prepare_layers()
    cross_cut_obj.add_data_layer(*layers)
    cross_cut_obj.update_plot()
    cross_cut_obj.overplot_power_law(x_intercept=2.6, alpha=0.7, linestyle='-.', lw=0.7)
    cross_cut_obj.overplot_power_law(exponent=-(4./3), x_intercept=2.5, exp_label="-4/3", end_x=2.5, lw=0.7)
    cross_cut_obj.overplot_power_law(exponent=-1, x_intercept=2., lw=0.5)
    cross_cut_obj.set_axis_limits((0.5, 3), (-1., 1.5))
    cross_cut_obj.switch_axes('xcut')
    plt.ylabel("Normalized log intensity")
    plt.xlabel("Log$_{10}$ Distance along cross-cut [arcseconds]")
    plt.title("Cross cut")
    # CII: vlims=(0, 11), 8um: vlims=(11, 900)
    cross_cut_obj.plot_image('8 um', stretch='arcsinh', vlims=(11, 900))
    cross_cut_obj.switch_axes('img')
    plt.title("8 um image, cross cut overlaid")
    cross_cut_obj.mark_radius((10**2.25)*u.arcsec, color='k', lw=0.3, linestyle='--')
    shell_radius = np.sin(5/4160.)*u.rad
    cross_cut_obj.mark_radius(shell_radius, color='k', lw=0.5, linestyle='-')
    shell_radius = np.sin(6/4160.)*u.rad
    cross_cut_obj.mark_radius(shell_radius, color='k', lw=0.5, linestyle='-')
    plt.show()


def double_plot_rcw49():
    selection_n = 1
    selection = f'thru-clcenter-{selection_n}'
    terminal_coords = coords_from_selection(selection)
    center_coord = catalog.utils.wd2_cluster_center_coord
    coord_pair_1 = (center_coord, terminal_coords[1]) # forwards
    coord_pair_2 = (center_coord, terminal_coords[0]) # backwards
    vlims = vlims_from_selection(selection)
    layers = prepare_layers()
    # Two nearly identical CrossCut instances
    cross_cut_objects = tuple(CrossCut(coords, vlims=vlims, log=True) for coords in (coord_pair_1, coord_pair_2))
    fig = cross_cut_objects[0].setup_figure(figsize=(14, 14), xcut_axis=221)
    cross_cut_objects[1].setup_figure(fig=fig, xcut_axis=223)
    for subplot_n, cco in zip((222, 224), cross_cut_objects):
        cco.add_data_layer(*layers)
        cco.update_plot()
        cco.set_axis_limits((0, 3), (-1.5, 1.5))
        cco.switch_axes('xcut')
        plt.ylabel("Log$_{10}$ Normalized intensity")
        plt.xlabel("Log$_{10}$ Distance along cross-cut [arcseconds]")
        cco.overplot_power_law(x_intercept=2.6, alpha=0.7, linestyle='-.', lw=0.7)
        # cco.overplot_power_law(exponent=-(4./3), x_intercept=2.45, exp_label="-4/3", end_x=2.45, lw=0.7)
        cco.overplot_power_law(exponent=-2, x_intercept=2.4, lw=0.5)
        cco.overplot_power_law(exponent=-1, x_intercept=1.5, lw=0.5)
        cco.plot_image('8 um', stretch='arcsinh', vlims=(11, 900), subplot_number=subplot_n)
        plt.title("8 um image, cross cut overlaid")
        cco.mark_radius((10**2.05)*u.arcsec, color='k', lw=0.3, linestyle='--')
        cco.mark_radius(np.sin(5/4160.)*u.rad, color='k', lw=0.5, linestyle='-')
        cco.mark_radius(np.sin(6/4160.)*u.rad, color='k', lw=0.5, linestyle='-')
    plt.savefig(f"/home/ramsey/Pictures/7-07-20-work/double_crosscut_{selection_n}.png") # , dpi=fig.dpi*1.2


def plot_m16():
    selection = "M16-marc-pillar2"
    coords = coords_from_selection(selection)
    vlims = vlims_from_selection(selection)
    vlims = (21.5, 22.)
    cross_cut_obj = CrossCut(coords, vlims=vlims, log=False)
    cross_cut_obj.setup_figure()
    layers = prepare_layers(target='m16')
    cross_cut_obj.add_data_layer(*layers)
    cross_cut_obj.update_plot()
    # cross_cut_obj.set_axis_limits()
    cross_cut_obj.switch_axes('xcut')
    plt.ylabel("Normalized intensity")
    plt.xlabel("Distance along cross-cut (arcseconds)")
    plt.title("Cross cut")

    cross_cut_obj.plot_image('5.6 um', stretch='arcsinh', vlims=(15, 190))
    cross_cut_obj.switch_axes('img')
    plt.title("IRAC 3 image, cross cut overlaid")
    plt.show()


def setup_paths(n, select=0):
    """
    Helper function, November 13, 2020
    Feels ok to put it in crosscut.py, everything's already so mixed up it doesn't
    really matter at this point.
    This is a good central location to handle hardcoding different reg files.
    Just handles the hardcoding I have to do for setup for all the regions
    n =
        0: across_each_pillar
        1: p3_shelves
        2: across all pillars (Jan 29 2021 image)
    select: depends on the region, means different things
        n=0: the pillar number (0-indexed)
        n=1: ??
        n=2: which across-all path (0 and 2 are good candidates)
    """
    if n == 0:
        # Colors
        cmap = mpl_cm.get_cmap('autumn')
        colors = [mpl_colors.to_hex(cmap(x)) for x in (0, 0.33, 0.66, 0.99)]
        # Load coord pairs
        reg_filename = catalog.utils.search_for_file("catalogs/across_each_pillar.reg") # 5 regions in this now
        path_list = coords_from_region(reg_filename, index=None)
        # Specific to across_each_pillar; select pillar
        selected_pillar = select
        pillar_names = ['Pillar 1', 'Pillar 2', 'Pillar 3']
        region_name = pillar_names[selected_pillar]
        # Gather 4 (or 3) paths for each possible selected pillar
        path_name = ['North', 'Mid-N', 'Mid', 'South']
        path_list = path_list[selected_pillar*4:(selected_pillar+1)*4]
        if selected_pillar == 2:
            path_name.pop(1)
        else:
            path_name[2] += '-S'
        # Vlims for this cut
        vlims = (23, 27)
        # Number of paths -> grid shape
        grid_shape = (len(path_list), 2)
    elif n == 1:
        # Colors
        cmap = mpl_cm.get_cmap('autumn')
        colors = [mpl_colors.to_hex(cmap(x)) for x in (0, 0.33, 0.66, 0.99)]
        # Load coord pairs
        reg_filename = catalog.utils.search_for_file("catalogs/p3_shelves.reg")
        path_list = coords_from_region(reg_filename, index=None)
        # Select region
        selected_region = select # can be 0 1 2 3
        region_names = ["P3 West Tail", "P3 East Tail", "Southern Shelf 1", "Southern Shelf 2"]
        region_name = region_names[selected_region]
        # If 0 or 1, include the 0th region
        if selected_region == 0:
            path_list = path_list[:4]
        elif selected_region == 1:
            path_list = [path_list[0]] + path_list[4:7]
        elif selected_region == 2:
            path_list = path_list[7:9]
        else:
            path_list = path_list[9:]
        # Vlims for this cut
        vlims = (19, 24)
        grid_shape = (len(path_list), 2)
        path_name = [str(x) for x in range(1, 1+len(path_list))]
    elif n == 2:
        # Colors
        colors = ['g']
        # Load coord pairs and select path
        reg_filename = catalog.utils.search_for_file("catalogs/across_all_pillars.reg")
        path_list = [coords_from_region(reg_filename, index=select)]
        region_name = "M16 pillars"
        path_name = [region_name]
        vlims = (20, 28)
        grid_shape = (1, 2) # subplot2grid
    else:
        raise NotImplementedError("I haven't set that reg file up yet")
    return colors, path_list, path_name, vlims, grid_shape, region_name




if __name__ == '__main__':
    plot_m16()

"""
======================================================================
try the Churchwell-like azimuthal average (maybe in 8 sections or something)

use astropy azimuth thing
"""


def plot_azimuthal():
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
