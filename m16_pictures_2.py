"""
A second  designated place for M16 region / bubble pictures.

This file's emphasis is on publication-centered figures rather than trials, drafts,
and analysis like those in m16_bubble.py.

Created: November 19, 2023
(in Michigan for TG)
M16 pillars paper is published as of this month, and I've sent around a text draft
of the new paper. Needs lots of images, which is where this file comes in.

This file is an extension of m16_bubble.py, so it imports * from there.
Reference m16_bubble.py while using this file.
"""

from .m16_bubble import *


def big_average_spectrum_figure():
    """
    November 19, 2023
    Big average spectrum figure
    CII and CO 3-2
    Rectangles in catalogs/north-south_spectrum_box.reg

    First, just one line. Next, consider iterating thru cubes too
    Done, works well.
    Also adapted it for the small-area regions, since code is identical, just
    swap region filenames (and labels, colors)
    """
    # Load regions
    select = 3
    if select == 0:
        reg_filename_short = "catalogs/north-south_spectrum_box.reg"
        spec_labels = ('north', 'south')
        colors = marcs_colors[:2]
        savename_stub = "big_avg_spectra"
    elif select == 1:
        reg_filename_short = "catalogs/north-south_spectrum_box_with_samples.reg"
        spec_labels = ('north', 'south', 'ridge', 'center', 'N19')
        colors = [None]*len(spec_labels)
        savename_stub = "big_avg_spectra_moreregs"
    elif select == 2:
        reg_filename_short = "catalogs/m16_co_small_samples.reg"
        spec_labels = ('ridge', 'center', 'N19')
        colors = marcs_colors[2:5]
        savename_stub = "small_avg_spectra_moreregs"
    # New batch of regions! Circles, 10x 15.5'' CII beams across
    elif select == 3:
        subselect = "_5" # or "" empty string
        reg_filename_short = f"catalogs/m16_spectrum_samples{subselect}.reg"
        colors = marcs_colors[:8]
        savename_stub = f"circle_samples_10beam{subselect}"

    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))

    if select <= 2:
        """ Old setup """
        # Setup fig
        fig = plt.figure(figsize=(13, 6))
        # Setup axes; one for each line?
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 2])
        def _index_gridspec(flatindex):
            # Index the gridspec. Depends on the shape and number of regions
            return gs[j, 1:]
        # Gridspec loc for ref image
        ref_gridspec_loc = gs[:, 0]

    elif select == 3:
        """ New setup """
        fig = plt.figure(figsize=(8, 11))
        mega_gs = fig.add_gridspec(3, 1, hspace=0, wspace=0, height_ratios=[1, 5, 4])
        gs_shape = (5, 2)
        gs = mega_gs[2, 0].subgridspec(*gs_shape, hspace=0, wspace=0)
        legend_ax_anchor = fig.add_subplot(mega_gs[2, 0])
        # legend_ax_anchor.set_axis_off()
        top_gs = mega_gs[0, 0].subgridspec(1, 3, hspace=0, wspace=0, width_ratios=[1, 4, 1])
        def _index_gridspec(flatindex):
            # Unravel as if 4, 2
            if flatindex == 0:
                return top_gs[0, 1]
            else:
                i_index = (flatindex - 1) // 2
                j_index = (flatindex - 1) % 2
                return gs[i_index, j_index]
        ref_gridspec_loc = mega_gs[1, 0].subgridspec(3, 1, height_ratios=(1, 17, 4), hspace=0, wspace=0)[1, 0]



    axes_dict = {}
    def _get_spec_axis(flatindex):
        # create or grab axis for flat index
        if flatindex not in axes_dict:
            axes_dict[flatindex] = fig.add_subplot(_index_gridspec(flatindex))
        return axes_dict[flatindex]



    reference_line_stub = "cii"
    ref_wcs, ref_shape = None, None
    line_stub_list = ("cii", "12co32", "13co32")
    line_plot_colors = marcs_colors[:3]
    xlims = (8, 37)
    noise_cutoff = 0 # sigma; set to 0 or below to turn off. good if > 5-10
    multipliers = {'13co32': 3}
    def _get_line_label(line_index):
        label = f"{get_data_name(line_stub_list[line_index])}"
        mult = multipliers.get(line_stub_list[line_index], None)
        if mult is not None:
            label = f"{label} $\\times${mult}"
        return label
    def _get_region_label(reg_idx):
        if reg_idx > 0:
            return f"{reg_idx}"
        else:
            return "Large"



    """ Setup reference img panel """
    ref_vel_lims = (10*kms, 27*kms)
    ref_img_misaligned, ref_img_info = get_2d_map(reference_line_stub, velocity_limits=ref_vel_lims)
    ref_unit = ref_img_info['unit']
    cutout = misc_utils.cutout2d_from_region(ref_img_misaligned, ref_img_info['wcs'], get_cutout_box_filename('med'), align_with_frame='galactic')
    ref_wcs = cutout.wcs
    ref_img = cutout.data
    ref_shape = ref_img.shape
    del ref_img_misaligned
    ref_ax = fig.add_subplot(ref_gridspec_loc, projection=ref_wcs)
    im = ref_ax.imshow(ref_img, origin='lower', cmap=cmocean.cm.matter, vmin=0, vmax=150)
    cax = ref_ax.inset_axes([1, 0, 0.05, 1])
    fig.colorbar(im, cax=cax, label=f"{get_data_name(reference_line_stub)} integrated intensity between {make_vel_stub(ref_vel_lims)} ({ref_unit.to_string('latex_inline')})", orientation='vertical')

    """
    Spectra

    Loop over spectral lines first (j)
    that way we can load each file once and use it a bunch

    Then loop over regions (i)
    """
    for j, line_stub in enumerate(line_stub_list):
        # if j > 0:
        #     print("SKIPPING ", end="")
        #     continue
        # Load cube
        fn = get_map_filename(line_stub)
        cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()

        # Plot both sets of spectra
        for i in range(len(reg_list)):
            # if i > 2:
            #     print("SKIPPING ", end="")
            #     continue
            # Setup axes
            # Extract subcube
            subcube = cube_obj.data.subcube_from_regions([reg_list[i]])
            # Mask by noise too?
            if noise_cutoff > 0:
                # Flip the .view() with ~, since view uses the numpy convention that "true" is masked out
                mask = np.any(~(subcube > get_onesigma(line_stub)*noise_cutoff*u.K).view(), axis=0)
                subcube = subcube.with_mask(mask)
            # Axis
            ax = _get_spec_axis(i)
            # Extract spectrum
            spectrum = subcube.mean(axis=(1, 2))
            # spec = spectrum.to_value()/np.nanmax(spectrum.to_value())
            spec = spectrum.to_value()
            if line_stub in multipliers:
                spec = spec * multipliers[line_stub]
            ax.plot(subcube.spectral_axis.to_value(), spec, color=line_plot_colors[j])
            if j == 0:
                # Once per region (so only on line j == 0) Add box regions to reference with appropriate colors
                pixreg = reg_list[i].to_pixel(ref_wcs)
                pixreg.plot(ax=ref_ax, color='w', lw=1)
                if i > 0:
                    ref_ax.text(*pixreg.center.xy, _get_region_label(i), ha="center", va="center", color='w', fontsize=12)
                ax.text(0.02, 0.9, _get_region_label(i), ha="left", va="top", color='k', fontsize=14, transform=ax.transAxes)

    """ Format Axes """
    for i, ax in axes_dict.items():
        ax.axhline(0, color='k', alpha=0.25, linestyle=':')
        # ax.axhline(0.5, color='k', alpha=0.25, linestyle=':')
        ax.set_xlim(xlims)
        for v in range(17, 28, 2):
            ax.axvline(v, color='grey', alpha=0.25, linestyle='--')
        ss = ax.get_subplotspec()
        if not ss.is_last_row() and i < 8:
            ax.xaxis.set_tick_params(labelbottom=False, direction='in', top=True)
        else:
            ax.xaxis.set_tick_params(direction='in', top=True)
        if ss.is_last_col() and not ss.is_first_col():
            # no yaxis label
            ax.yaxis.set_tick_params(labelleft=False, labelright=True, left=True, right=True, direction='in')
        else:
            ax.yaxis.set_tick_params(direction='in', left=True, right=True)
        ylims = ax.get_ylim()
        if i == 0:
            pass
        elif i == 7:
            ax.set_ylim((-0.5, 4.99))
            # ax.set_ylim((-1., 9.5))
        elif i <= 4 or i == 9 or i == 6:
            ax.set_ylim((-1.5, 19.9))
        else:
            ax.set_ylim((-3, 38))

    legend_ax_anchor.set_xlabel(f"V ({kms.to_string('latex_inline')})", labelpad=17, fontsize=12)
    legend_ax_anchor.set_ylabel("T$_{\\rm MB}$" + f" ({u.K.to_string('latex_inline')})", labelpad=22, fontsize=12)
    legend_ax_anchor.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    legend_ax_anchor.tick_params(axis='both', labelleft=False, labelbottom=False, left=False, bottom=False)


    # Legend
    legend_ax_anchor.legend(handles=[
        mpatches.Patch(color=line_plot_colors[i], label=_get_line_label(i))
            for i in range(len(line_stub_list))
    ], bbox_to_anchor=[0, 1, 1, 0.1], loc='center', ncols=len(line_stub_list))

    lat, lon = (ref_ax.coords[i] for i in range(2))
    for l in (lat, lon):
        l.set_major_formatter("d.dd")
    lat.set_axislabel("Galactic Latitude", fontsize=12)
    lon.set_axislabel("Galactic Longitude", fontsize=12)
    ref_ax.tick_params(labelsize=12)

    savename = os.path.join(catalog.utils.todays_image_folder(),
        f"{savename_stub}")
    fig.subplots_adjust(top=0.98, bottom=0.05, left=0.07, right=0.95)
    print()
    fig.savefig(savename+".pdf",)
        # metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short}, noise {noise_cutoff}",
        #     file=__file__, func="big_average_spectrum_figure"))


def full_picture_integrated_intensity():
    """
    November 19, 2023
    Compare integrated intensity maps of different velocity intervals.
    Also use a red-blue or RGB image to compare spatial positions more directly

    I want to do 2 intervals for CII and 3 for CO. I think I can use the same
    code, with a bit of handling for the RGB stuff and number of axes, etc.
    """
    select = 0
    if select == 0:
        line_stub = "cii"
        velocity_intervals = [(15, 21), (21, 23), (23, 27)]
        # velocity_intervals = [(10, 21), (21, 30)]
        # norms = [1, 0.7]
        norms = [1, 0.7, 0.7]
    elif select == 1:
        line_stub = "12co32"
        velocity_intervals = [(15, 21), (21, 23), (23, 27)]
        norms = [0.8, 0.7, 1]
    # Fig setup
    fig = plt.figure(figsize=(15, 7))
    # Load
    fn = get_map_filename(line_stub)
    cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()
    # Save mom0s
    imgs = []
    for i, vel_lims in enumerate(velocity_intervals):
        # Units
        vel_lims = [v*kms for v in vel_lims]
        # Axes
        ax = plt.subplot(1, len(velocity_intervals)+1, 1 + i, projection=cube_obj.wcs_flat)
        # Image
        mom0 = cube_obj.data.spectral_slab(*vel_lims).moment0()
        im = ax.imshow(mom0.to_value(), origin='lower')
        fig.colorbar(im, ax=ax, orientation='horizontal', label=f"{get_data_name(line_stub)} {make_vel_stub(vel_lims)} ({mom0.unit.to_string('latex_inline')})")
        imgs.append(mom0.to_value())
    # Make extra axis
    ax = plt.subplot(1, len(velocity_intervals)+1, len(velocity_intervals)+1, projection=cube_obj.wcs_flat)
    # Compose rgb/rb
    img_arr = []
    for i, img in enumerate(imgs):
        norm_img = img/np.nanmax(img) / norms[i]
        norm_img[np.isnan(img)] = 1
        img_arr.append(norm_img)
    # flip so red is high velocity
    img_arr = img_arr[::-1]
    if len(velocity_intervals) < 3:
        green_fill_val = 0.4
        green_img = (img_arr[0] + img_arr[1]) * green_fill_val
        green_img[np.isnan(imgs[0])] = 1
        img_arr.insert(1, green_img)
    img_arr = np.moveaxis(img_arr, 0, -1)
    ax.imshow(img_arr, origin='lower')
    plt.show()


def m16_expanding_shell_spectra():
    """
    November 28, 2023
    Averaged spectra from the M16 west cavity to show blue (questionable) and red (definitely good) emission
    Circles for regions in catalogs/m16_west_cavity_spec_regions.reg
    Follow big_average_spectrum_figure() for reg extraction code.
    """
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
    # Spectra, both on same figure
    spec_ax = fig.add_subplot(gs[:, 1:])
    for j, reg in enumerate(reg_list):
        subcube = cube_obj.data.subcube_from_regions([reg])
        spectrum = subcube.mean(axis=(1, 2))
        spec_ax.plot(subcube.spectral_axis.to_value(), spectrum.to_value(), color=colors[j], label=spec_labels[j])
    # Mark moment velocities on spectrum plot
    for vel_lims in velocity_intervals:
        plt.axvspan(*vel_lims, color='grey', alpha=0.3)
    # Extra plot dressing
    spec_ax.axhline(0, color='grey', linestyle="--", alpha=0.2)
    spec_ax.set_xlabel("V$_{\\rm LSR}$ " + f"({kms.to_string('latex_inline')})")
    spec_ax.set_ylabel(f"{get_data_name(line_stub)} line intensity ({spectrum.unit.to_string('latex_inline')})")
    plt.subplots_adjust(left=0.09, right=0.97, top=0.95, wspace=0.45, bottom=0.09)
    vel_stub = "-and-".join([make_simple_vel_stub(tuple(v*kms for v in vel_lims)) for vel_lims in velocity_intervals])

    savename = f"expanding_shell_spectra_{line_stub}_{vel_stub}.png"
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), savename),
        metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short}",
            file=__file__, func="m16_expanding_shell_spectra"))


def m16_blue_clump():
    """
    November 29, 2023
    Highlight the blue clump to show that it is blueshifted and a clump
    Use 8, 70 micron, and CII and CO spectra.

    Going to implement some memoization to disk to make this routine faster?

    This function uses dictionaries to save info for reuse.
    """
    # plot defaults
    default_label_text_size = 17

    include_zoomouts = False

    # Select the blue clump field
    cutout_reg_stub = "blueclump-large2"
    def _load_helper(stub, reproject_wcs_shape=None):
        """
        Now with memoization! because this takes a long time to run if we keep loading the big files
        """
        # Check memoization
        memo_name = f"misc_regrids/{stub}.{cutout_reg_stub}_regrid.fits"
        try:
            fn = catalog.utils.search_for_file(memo_name)
            print(f"Found memoized data {stub} {cutout_reg_stub}")
        except FileNotFoundError:
            fn = None
        # Use fn to tell if memo found or not
        if fn is None:
            # Memo not found, load normally
            img, info = get_2d_map(stub)
            if reproject_wcs_shape is None:
                # Cutout using cutout_reg_stub
                info['cutout'] = misc_utils.cutout2d_from_region(img, info['wcs'], get_cutout_box_filename(cutout_reg_stub), align_with_frame='icrs')
                info['img'] = info['cutout'].data
                info['wcs'] = info['cutout'].wcs
            else:
                # Reproject using the (wcs, shape) tuple
                wcs_obj, shape_out = reproject_wcs_shape
                info['img'] = reproject_interp((img, info['wcs']), wcs_obj, shape_out=shape_out, return_footprint=False)
                info['wcs'] = wcs_obj
            assert 'stub' not in info
            info['stub'] = stub

            # Save memo
            memo_full_name = f"{catalog.utils.m16_data_path}{memo_name}"
            hdr = info['wcs'].to_header()
            hdr['BUNIT'] = str(info['unit'])
            hdr['DATE'] = f"Created: {datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()}"
            hdr['AUTHOR'] = "Ramsey Karim"
            hdr['CREATOR'] = f"rkarim, via {__file__}.m16_blue_clump"
            hdr['COMMENT'] = "reprojected to aligned cutout from"
            hdr['COMMENT'] = cutout_reg_stub
            hdu = fits.PrimaryHDU(data=info['img'], header=hdr)
            print(f"Memoizing data for {stub} to {memo_full_name}")
            hdu.writeto(memo_full_name)
        else:
            # Memo found, load memo
            info = {}
            data, hdr = fits.getdata(fn, header=True)
            info['wcs'] = WCS(hdr)
            info['img'] = data
            info['unit'] = u.Unit(hdr['BUNIT'])
        # Convert units from anything / px to anything / sr
        try:
            info['unit'].to(u.MJy/u.sr)
        except:
            try:
                pixel_area = misc_utils.get_pixel_scale(info['wcs'])**2
                mjysr = u.MJy/u.sr
                # print((info['unit'] * u.pixel/pixel_area).to(mjysr))
                info['img'] = (info['img'] * info['unit'] * u.pixel/pixel_area).to(mjysr).to_value()
                info['unit'] = mjysr
            except:
                raise RuntimeError("Unit issue!!! Don't know what's wrong with this one, can't convert to MJy/sr from: "+info['unit'].to_string())
        return info

    # The info dicts will carry lots of info in this function. Keep them somewhere accessible
    img_info_dicts = {}
    # Load primary 8 micron and trim to blue clump field
    ref_img_stub = "irac4-large"
    img_info_dicts[ref_img_stub] = _load_helper(ref_img_stub)
    # Load in secondary 70 micron image and reproject to primary
    img2_stub = "160um"
    img_info_dicts[img2_stub] = _load_helper(img2_stub)
    # Figure, Axes
    if include_zoomouts:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(6, 3)
    else:
        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(3, 12, hspace=0.25, wspace=0.1)
    def _make_fig_and_plot(stub, grid_loc, vlims=None, key_suffix=""):
        """
        Generalized helper for turning arrays into figures in this function.
        :param key_suffix: string suffix to add to "wcs", "img", and "ax" for a zoomed or otherwise modified cutout
        """
        info = img_info_dicts[stub]
        ax = fig.add_subplot(gs[grid_loc], projection=info['wcs'+key_suffix])
        cax = ax.inset_axes([1, 0, 0.05, 1])
        if vlims is None:
            vlims_dict = {}
        else:
            vlims_dict = {k: v for k, v in zip(('vmin', 'vmax'), vlims)}
        im = ax.imshow(info['img'+key_suffix], origin='lower', **vlims_dict, cmap=cmocean.cm.matter)
        cbar = fig.colorbar(im, cax=cax, label=info['unit'].to_string('latex_inline'))
        info['ax'+key_suffix] = ax

    if include_zoomouts:
        _make_fig_and_plot(ref_img_stub, (slice(0, 3), 0), vlims=(25, 200))
        _make_fig_and_plot(img2_stub, (slice(3, 6), 0), vlims=(-0.15*4166, 0.8*4166))  # (0, 2500) # (0, 0.8)

    # Zoom in again with another cutout
    cutout_reg_stub_zoom = "blueclump-zoom"
    def _cutout_helper(stub):
        info = img_info_dicts[stub]
        img = info['img']
        # no need to align again
        extra_kw = {}
        if not include_zoomouts:
            extra_kw['align_with_frame'] = "galactic"
        info['cutout-zoom'] = misc_utils.cutout2d_from_region(img, info['wcs'], get_cutout_box_filename(cutout_reg_stub_zoom), **extra_kw)
        # don't write over the first cutout, use -zoom suffix to differentiate
        info['img-zoom'] = info['cutout-zoom'].data
        info['wcs-zoom'] = info['cutout-zoom'].wcs

    _cutout_helper(ref_img_stub)
    _cutout_helper(img2_stub)

    if include_zoomouts:
        gs_loc1 = (slice(0, 3), 1)
        gs_loc2 = (slice(3, 6), 1)
    else:
        gs_loc1 = (slice(0, 2), slice(0, 5))
        gs_loc2 = (slice(0, 2), slice(7, 12))
    _make_fig_and_plot(ref_img_stub, gs_loc1, vlims=(40, 120), key_suffix="-zoom")
    _make_fig_and_plot(img2_stub, gs_loc2, vlims=(-0.025*4166, 0.3*4166), key_suffix="-zoom")

    # Overlay regions on zoomed figure and also zoom boxes on larger figure
    reg_color = "k"
    reg_filename_short = "catalogs/m16_points_blueshifted_clump.reg"
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    box_reg_color = 'k'
    box_reg = regions.Regions.read(get_cutout_box_filename(cutout_reg_stub_zoom))[0]

    def _plot_reg(stub, reg, key_suffix="", color=None, label=False):
        """
        General region plot helper
        :param label: plot reg.meta['text'] slightly offset from the center of the region
        """
        info = img_info_dicts[stub]
        ax = info['ax'+key_suffix]
        pixreg = reg.to_pixel(info['wcs'+key_suffix])
        reg_patch = pixreg.as_artist()
        # Gotta mess around with the lines vs other patches (from m16_bubble.overlay_moment)
        if isinstance(reg_patch, Line2D):
            # Point!!!
            reg_patch.set(mec=color, marker='o', mew=3)
            ax.add_artist(reg_patch)
        else:
            # Anything besides Point
            reg_patch.set(ec=color)
            ax.add_patch(reg_patch)
        if label:
            center = reg.center
            cra, cdec = center.galactic.l.deg, center.galactic.b.deg
            dx = (-20*u.arcsec).to(u.deg).to_value()
            dy = (20*u.arcsec).to(u.deg).to_value()
            x, y = info['wcs'+key_suffix].world_to_pixel_values(cra + dx, cdec + dy)
            ax.text(x, y, reg.meta['text'], ha='center', va='center', fontsize=default_label_text_size+0, fontweight='bold', color=color)

    for stub in img_info_dicts.keys():
        for reg in reg_list:
            # spectrum samples
            _plot_reg(stub, reg, key_suffix="-zoom", color=reg_color, label=True)
        # zoom box
        if include_zoomouts:
            _plot_reg(stub, box_reg, color=box_reg_color)

    """
    Load lines and plot spectra

    Using a series of functions to do this; with the right call order, I can
    avoid loading 2 full cubes at once.
    I can also plot 3 spectra and only 2 moments if i want
    """


    line_stub_list = ['ciiAPEX', '12co32', '12co10-nob']
    line_plot_colors = marcs_colors[:3]
    line_dict = {} # keys are line_stubs, values are dicts containing stuff for each line cube (including entire cube)
    levels_string = "" # string for contour levels
    def _load_cube(line_stub, plot_color=None):
        """
        :param plot_color: set the spectrum line plot color for this line
        """
        # Store cube stuff in a dict for convenient access later, similar to images
        line_info = {}
        fn = get_map_filename(line_stub)
        cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()
        # save stuff to dict
        line_info['CubeData'] = cube_obj
        line_info["color"] = plot_color
        line_dict[line_stub] = line_info

    def _plot_contours(line_stub, img_stub, key_suffix="", velocity_limits=None, levels=None):
        """
        :param img_stub: select the image to overplot contours on, along with key_suffix
        :param velocity_limits: float tuple, km/s implied
        """
        line_info = line_dict[line_stub]
        cube_obj = line_info['CubeData']
        # Apply velocity limits, assuming float tuple implied km/s
        if velocity_limits is not None:
            vel_lims = tuple(v*kms for v in velocity_limits)
            subcube = cube_obj.data.spectral_slab(*vel_lims)
        else:
            subcube = cube_obj.data
        mom0 = subcube.moment0()
        ax = img_info_dicts[img_stub]['ax'+key_suffix]
        mom0_reproj = reproject_interp((mom0.to_value(), mom0.wcs), img_info_dicts[img_stub]['wcs'+key_suffix], shape_out=img_info_dicts[img_stub]['img'+key_suffix].shape, return_footprint=False)
        cs = ax.contour(mom0_reproj, levels=levels, colors='grey', linewidths=1.3, alpha=1)
        # print(line_stub, cs.levels)
        nonlocal levels_string # scope!!! we finally have to use this!
        levels_string += line_stub + " " + str(cs.levels) + ". "
        img_info_dicts[img_stub]['overlaystub'+key_suffix] = line_stub

    spec_axes = {}
    def _extract_and_plot_spectra(line_stub, reg, grid_loc=None):
        """
        Cube already loaded and line info dict exists
        :param grid_loc: gridspec location for subplot creation
            grid_loc can be None IFF Axes already exists;
            will search spec_axes dict for existing Axes using reg.meta['text'] as key
        """
        line_info = line_dict[line_stub]
        cube_obj = line_info['CubeData']
        pixreg = reg.to_pixel(cube_obj.wcs_flat)
        j, i = [int(round(c)) for c in pixreg.center.xy]
        spectrum = cube_obj.data[:, i, j]
        # Check spec_axes for existing axes, or make one
        key = reg.meta['text']
        if key not in spec_axes:
            spec_axes[key] = fig.add_subplot(gs[grid_loc])
        ax = spec_axes[key]
        ax.plot(cube_obj.data.spectral_axis.to_value(), spectrum.to_value(), color=line_info['color'], label=get_data_name(line_stub.replace("APEX", "")))

    contour_vel_lims = (7, 10)

    for i, color, line_stub in zip(range(3), line_plot_colors, line_stub_list):
        # load
        _load_cube(line_stub, color)
        # plot contours if not 12co10
        if '10' not in line_stub:
            img_stub_for_overlay = [ref_img_stub, img2_stub][i]
            _plot_contours(line_stub, img_stub_for_overlay, key_suffix="-zoom", velocity_limits=contour_vel_lims, levels=5)
        for j, reg in enumerate(reg_list):
            if include_zoomouts:
                gs_loc = (slice(2*j, 2*(j+1)), 2)
            else:
                gs_loc = (2, slice(4*j, 4*(j+1)))
            _extract_and_plot_spectra(line_stub, reg, grid_loc=gs_loc)

    # Mark up the spectrum plots
    for i, key in enumerate(spec_axes):
        ax = spec_axes[key]
        ax.text(0.1, 0.9, key, color='k', transform=ax.transAxes, fontsize=default_label_text_size, ha='center', va='center')
        ax.set_xlabel("V$_{\\rm LSR}$ " + f"({kms.to_string('latex_inline')})")
        if int(key) == 1:
            ax.set_ylabel(f"Line intensity ({u.K.to_string('latex_inline')})")
        else:
            ax.yaxis.set_tick_params(labelleft=False, direction='in')
        ax.axvspan(*contour_vel_lims, color='grey', alpha=0.3)
        ax.axhline(0, color='grey', linestyle="--", alpha=0.2)
        if int(key) == 2:
            ax.legend(bbox_to_anchor=[0, 1, 1, 1.1], ncols=3, loc='lower center')
        ax.set_xlim((-4, 28))
        ax.set_ylim((-3, 15))

    # Mark up the image plots
    for stub in img_info_dicts.keys():
        # ax = img_info_dicts[stub]['ax']
        ax = img_info_dicts[stub]['ax-zoom']
        ax.text(0.1, 0.1, "Image: "+get_data_name(stub), color='k', transform=ax.transAxes, fontsize=default_label_text_size, ha='left', va='center')
        line_stub = img_info_dicts[stub]['overlaystub-zoom']
        ax.text(0.95, 0.9, "Contour: "+get_data_name(line_stub.replace("APEX", "")), color='w', transform=ax.transAxes, fontsize=default_label_text_size, ha='right', va='center')
        # General coord formatting
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')
        ax.set_xlabel("Galactic Longitude")
        ax.set_ylabel("Galactic Latitude")

    fig.subplots_adjust(top=1, bottom=0.1, left=0.08, right=0.9)
    levels_string = levels_string.rstrip()
    save_text = f"vellims{contour_vel_lims}. large:{cutout_reg_stub}. {levels_string}"
    savename = f"spectra_{cutout_reg_stub_zoom}" + "-".join(line_stub_list) + ".png"
    fig.savefig(os.path.join(catalog.utils.todays_image_folder(), savename),
        metadata=catalog.utils.create_png_metadata(title=save_text, file=__file__, func="m16_blue_clump"))

    # plt.show()


def pv_n19():
    """
    March 1, 2024
    copying framework form m16_bubble.fast_pv
    """
    reg_filename_tup = ("catalogs/N19_pv_paths_5.reg", 1)
    pv_path, point_reg_list, reg_fn_stub = get_pv_and_regions(reg_filename_tup)
    velocity_limits = (8*kms, 35*kms)

    def _get_levels(line_stub):
        """
        OLD <Get levels from the above dictionary. Return None if not present.>
        NEW Get levels from the vlim memo functions
            future: get levels from vlim memo only if not specified in function call
        """
        return get_levels(line_stub)

    def _get_pv_vmax(line_stub):
        """
        Get vmax from vlim memo functions
            future: get from vlim memo only if not specified in function call
        """
        return get_generic_vlim(line_stub).get("vmax", None)


    fig = plt.figure(figsize=(10, 8))
    line_stub_list = ['ciiAPEX', '12co32']
    pv_slices = []
    for line_stub in line_stub_list:
        cube_filename_short = get_map_filename(line_stub)
        cube = cube_utils.CubeData(cube_filename_short).convert_to_K()
        pv_slices.append(pvextractor.extract_pv_slice(cube.data.spectral_slab(*velocity_limits), pv_path))
    ls0, ls1 = line_stub_list
    sl0, sl1_raw = pv_slices
    aspect_ratio = sl0.data.shape[1]/(1.2*sl0.data.shape[0])
    sl0.header['CTYPE2'] = sl1_raw.header['CTYPE2'] = "VRAD"
    sl1_raw.header['RESTFRQ'] = sl0.header['RESTFRQ']
    sl1_reproj = reproject_interp((sl1_raw.data, sl1_raw.header), sl0.header, return_footprint=False)
    ax = plt.subplot(111, projection=WCS(sl0.header))
    image_cmap = cmocean.cm.matter #'plasma'
    im = ax.imshow(sl0.data, origin='lower', cmap=image_cmap, vmin=0, vmax=_get_pv_vmax(ls0), aspect=aspect_ratio)
    levels =_get_levels(ls0)
    e0 = get_onesigma(ls0)
    print(ls0, e0)
    levels = np.arange(e0*2, 35, e0*4)
    print(ls0, levels)
    cs = ax.contour(sl0.data, colors='k', linewidths=1, linestyles=':', levels=levels)
    cax = ax.inset_axes([1, 0, 0.05, 1])
    pv_cbar = fig.colorbar(im, cax=cax)
    pv_cbar.set_label("T$_{\\rm MB}$ "+f"({cube.data.unit.to_string('latex_inline')})", fontsize=13)
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color='k', linewidth=1.5, linestyle=':')
    image_color = im.cmap(0.8) #matplotlib.colormaps[image_cmap](0.9)
    # Overlay [1]
    contour_cmap = cmocean.cm.algae_r #'cool_r'
    levels = _get_levels(ls1)
    e1 = get_onesigma(ls1)
    levels = np.arange(e1*4, 28, e1*9)
    pvvmax = _get_pv_vmax(ls1)
    print(ls1, e1)
    print(ls1, levels)
    cs = ax.contour(sl1_reproj, cmap=contour_cmap, linewidths=1.5, levels=levels, vmax=pvvmax)
    for l in cs.levels:
        pv_cbar.ax.axhline(l, color=cs.cmap(cs.norm(l)), linewidth=2)
    contour_color = cs.cmap(0.3) #matplotlib.colormaps[contour_cmap](0.05)
    """ Text """
    text_x, dx = 0.01, 0
    text_y, dy = 0.02 + 0.03*2, -0.03
    ax.text(text_x + 0*dx, text_y + 0*dy, get_data_name(ls1)+": solid contour", fontsize=13, fontweight='bold', color=contour_color, va='center', ha='left', transform=ax.transAxes)
    ax.text(text_x + 1*dx, text_y + 1*dy, get_data_name(ls0.replace("APEX", ""))+": image & dotted contour", fontsize=13, fontweight='bold', color=image_color, va='center', ha='left', transform=ax.transAxes)
    ax.text(text_x + 2*dx, text_y + 2*dy, "Both data on the $\sim$20"+u.arcsec.to_string('latex_inline')+" CO beam", fontsize=13, fontweight='bold', color='k', va='center', ha='left', transform=ax.transAxes)
    text_x, dx = 0.01, 0.55
    text_y, dy = 0.02, 0
    ax.text(text_x + 1*dx, text_y + 0*dy, "N19 expanding shell: half-ellipse", fontsize=13, fontweight='bold', color='k', va='center', ha='left', transform=ax.transAxes)
    ax.coords[1].set_format_unit(kms)
    ax.coords[1].set_major_formatter('x')
    ax.coords[1].set_axislabel(f"Velocity ({kms.to_string('latex_inline')})", fontsize=13)
    ax.coords[0].set_format_unit(u.arcsec)
    ax.coords[0].set_major_formatter('x')
    ax.coords[0].set_axislabel(f"Offset, east to west ({u.arcsec.to_string('latex_inline')})", fontsize=13)

    # Mark the shell
    transform = ax.get_transform('world')
    dtheta = np.pi*0.15 * 0
    t = np.linspace(np.pi - dtheta, 2*np.pi + dtheta, 20)
    st, ct = np.sin(t), np.cos(t)
    shell_center = 400*u.arcsec.to(u.deg) # arcsec offset
    exp_shell_setting = 2
    if exp_shell_setting == 0:
        systematic_velocity = 19*1e3
    elif exp_shell_setting == 1:
        systematic_velocity = 20*1e3
    elif exp_shell_setting == 2:
        systematic_velocity = 21*1e3
    def _make_shell_xy(exp_vel_kms, radius_pc):
        shell_radius = ((radius_pc*u.pc)/(los_distance_M16/u.radian)).to(u.deg).to_value()
        expansion_velocity = exp_vel_kms*1e3
        x = shell_radius * ct + shell_center
        y = expansion_velocity * st + systematic_velocity
        return x, y
    if exp_shell_setting == 0:
        ax.plot(*_make_shell_xy(4, 2.3), color='w', linewidth=3, transform=transform)
    elif exp_shell_setting == 1:
        ax.plot(*_make_shell_xy(5, 2.35), color='w', linewidth=3, transform=transform)
    elif exp_shell_setting == 2:
        ax.plot(*_make_shell_xy(6, 2.4), color='w', linewidth=3, transform=transform)
    # ax.plot(*_make_shell_xy(3, 2.0), color='w', linewidth=2, linestyle="--", transform=transform)
    # ax.plot(*_make_shell_xy(6, 2.7), color='w', linewidth=2, linestyle="--", transform=transform)

    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.08, top=0.97)
    # 2023-05-06,11, 06-30
    savename = os.path.join(catalog.utils.todays_image_folder(), f"n19_pv_{ls1}_on_{ls0}_along_{reg_fn_stub}_expshell{chr(65+exp_shell_setting)}.png")
    fig.savefig(savename, metadata=catalog.utils.create_png_metadata(title='pv',
        file=__file__, func="pv_n19"))








if __name__ == "__main__":
    m16_blue_clump()
