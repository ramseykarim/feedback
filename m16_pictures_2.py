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
    select = 1
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
    reg_list = regions.Regions.read(catalog.utils.search_for_file(reg_filename_short))
    # Setup fig
    fig = plt.figure(figsize=(13, 6))
    # Setup axes; one for each line?
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 2])
    axes = []
    reference_line_stub = "cii"
    ref_wcs, ref_shape = None, None
    line_stub_list = ("cii", "12co32", "13co32")
    xlims = (-15, 55)
    noise_cutoff = 0 # sigma; set to 0 or below to turn off. good if > 5-10

    for j, line_stub in enumerate(line_stub_list):
        # Axis
        ax = fig.add_subplot(gs[j, 1:])
        axes.append(ax)
        # Load cube
        fn = get_map_filename(line_stub)
        cube_obj = cube_utils.CubeData(fn).convert_to_K().convert_to_kms()

        if line_stub == reference_line_stub:
            # Setup reference img panel
            ref_wcs = cube_obj.wcs_flat
            ref_shape = cube_obj.data.shape[1:]
            ref_ax = fig.add_subplot(gs[:, 0], projection=ref_wcs)
            ref_vel_lims = (0*kms, 40*kms)
            ref_img = cube_obj.data.spectral_slab(*ref_vel_lims).moment0()
            im = ref_ax.imshow(ref_img.to_value(), origin='lower')
            fig.colorbar(im, ax=ref_ax, label=f"{get_data_name(line_stub)} {make_vel_stub(ref_vel_lims)} ({ref_img.unit.to_string('latex_inline')})", orientation='horizontal')
        else:
            # Reproject to ref wcs (which then needs to be saved) and plot footprint
            fp = np.isfinite(cube_obj.data[0, :, :].to_value()).astype(float)
            fp = reproject_interp((fp, cube_obj.wcs_flat), ref_wcs, shape_out=ref_shape, return_footprint=False)
            ref_ax.contour(fp, levels=[0.5], linestyles="--", colors='grey')

        # Plot both sets of spectra
        for i, label in enumerate(spec_labels):
            # Setup axes
            # Extract subcube
            subcube = cube_obj.data.subcube_from_regions([reg_list[i]])
            # Mask by noise too?
            if noise_cutoff > 0:
                # Flip the .view() with ~, since view uses the numpy convention that "true" is masked out
                mask = np.any(~(subcube > get_onesigma(line_stub)*noise_cutoff*u.K).view(), axis=0)
                subcube = subcube.with_mask(mask)
                ## useful debugging
                # if i==1 and line_stub == '13co32':
                #     plt.figure()
                #     plt.imshow(subcube.moment0().to_value(), origin='lower')
                #     plt.show()
                #     return
            # Extract spectrum
            spectrum = subcube.mean(axis=(1, 2))
            p = ax.plot(subcube.spectral_axis.to_value(), spectrum.to_value()/np.nanmax(spectrum.to_value()), color=colors[i], label=f"{get_data_name(line_stub)} {spec_labels[i]}")
            if line_stub == reference_line_stub:
                # Add box regions to reference with appropriate colors
                pixreg = reg_list[i].to_pixel(cube_obj.wcs_flat)
                pixreg.plot(ax=ref_ax, color=p[0].get_c(), lw=3)


    for ax in axes:
        ax.legend()
        ax.axhline(0, color='k', alpha=0.25, linestyle=':')
        ax.axhline(0.5, color='k', alpha=0.25, linestyle=':')
        ax.set_xlim(xlims)
        for v in range(10, 35, 2):
            ax.axvline(v, color='grey', alpha=0.25, linestyle='--')
        ss = ax.get_subplotspec()
        if ss.is_last_row():
            ax.set_xlabel(f"V ({kms.to_string('latex_inline')})")
    savename = os.path.join(catalog.utils.todays_image_folder(),
        f"{savename_stub}.png")
    fig.savefig(savename,
        metadata=catalog.utils.create_png_metadata(title=f"{reg_filename_short}, noise {noise_cutoff}",
            file=__file__, func="big_average_spectrum_figure"))


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



if __name__ == "__main__":
    m16_expanding_shell_spectra()
