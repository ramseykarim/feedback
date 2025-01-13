#!/bin/bash

# also try upper limits of
# 342
# 1494
# 0.02
# linear looks best but sinh and squared also look reasonable (making the center super washed out but pulling out low-contrast detail on the edge)

current_dir=$(pwd)
work_dir="/home/ramsey/Documents/Research/Feedback/m16_data"
echo "Moving from $current_dir to $work_dir"
pushd $work_dir > /dev/null

function make_rgb_img() {
  ds9 -rgb \
  -red herschel/PACS70um-image.fits -scale LINEAR -scale limits -0.05 0.05 \
  -green wise/2746m137_ac51-w3-int-3.fits -scale LINEAR -scale limits 1140 1682 \
  -blue optical/dss2-red.18.18.45.1-13.47.31.2.fits -scale SQRT -scale limits 3800 18000 \
\
  -frame first -frame delete \
	-frame first -green \
	-geometry 1000x1000 \
	-wcs galactic -frame lock wcs \
	-pan to +17:03:00 +0:42:00 wcs galactic \
	-zoom to 0.2 0.2 \
\
	-wcs degrees -grid yes -grid grid no -grid border yes -grid numerics color white -grid numerics fontweight bold \
	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 -grid tickmarks width 2 \
	-grid tickmarks color white -grid border width 3 -grid border color black -colorbar no \
\

}

coord_color="green"
coord_boilerplate="-wcs degrees -grid yes -grid grid no -grid border yes -grid numerics color ${coord_color} -grid numerics fontweight bold \
-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 -grid tickmarks width 2 \
-grid tickmarks color ${coord_color} -grid border width 3 -grid border color black -colorbar no"

function make_panel_img() {
  ds9 \
  optical/dss2-red.18.18.45.1-13.47.31.2.fits -scale SQRT -scale limits 3800 18000 \
  -frame new herschel/PACS70um-image.fits -scale LINEAR -scale limits -0.05 0.05 \
  -frame new wise/2746m137_ac51-w4-int-3.fits -scale LINEAR -scale limits 315 362 \
  -tile -tile column \
  -frame first -geometry 1400x800 -wcs galactic -frame lock wcs \
  -pan to +16:33:45.939 +0:58:33.875 wcs galactic \
	-zoom to 0.18 0.18 \
  \
  $coord_boilerplate \
  -frame next $coord_boilerplate \
  -frame next $coord_boilerplate \
  -frame next $coord_boilerplate \

}

make_panel_img &
popd > /dev/null
echo "Returned to $(pwd)"




# -green wise/2746m137_ac51-w3-int-3.fits -scale LINEAR -scale limits 1140 1682 \
# -frame new optical/dss2-red.18.18.45.1-13.47.31.2.fits -scale SQRT -scale limits 3800 18000
# -green wise/2746m137_ac51-w4-int-3.fits -scale LINEAR -scale limits 310 362 \
