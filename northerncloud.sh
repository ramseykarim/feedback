#!/bin/bash


sofia="sofia/cii_mom0_10.0.21.0.fits -scale limits 0 100 -scale LINEAR"
optical="optical/dss2-red.18.18.45.1-13.47.31.2.fits -scale limits 3800 18000 -scale SQRT"
pmo="purplemountain/12co10-pmo_mom0_10.0.21.0.fits -scale limits 0 90 -scale LINEAR"
apex="apex/12co32_mom0_10.0.21.0.fits -scale limits 0 65 -scale LINEAR"

regions="-region load catalogs/m16_lobes_outlined_4.reg \
-region select all -region color white -region width 2 \
-region select none"

gridconfig="-wcs degrees -grid yes -grid grid no -grid border no -grid numerics color white \
-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics fontweight normal \
-grid numerics gap1 2 -grid numerics gap2 6 \
-grid tickmarks width 1 -grid tickmarks color white"


function make_rgb_img() {
  ds9 -rgb \
  	-red $optical \
  	$regions \
  \
  	-rgb \
  	-green $pmo \
  	-blue $pmo \
  	-red $optical \
  \
  	-frame first -frame delete \
  	-wcs galactic -frame lock wcs \
  	-tile -tile row -geometry 700x1400 -tile grid gap 1 \
  \
  	-frame first -pan to +17:03:45.000 +0:52:10.000 wcs galactic \
  	-zoom to 0.14 -colorbar no \
  \
  	$gridconfig \
  \
  	-frame next $gridconfig \
  \

}


current_dir=$(pwd)
work_dir="/home/ramsey/Documents/Research/Feedback/m16_data"
echo "Moving from $current_dir to $work_dir"
pushd $work_dir > /dev/null
make_rgb_img &
popd > /dev/null
echo "Returned to $(pwd)"
