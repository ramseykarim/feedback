#!/bin/bash

# 2025-01-25
# Doing three different images here, see notes from this day

function westcavity_cii() {
  ds9 -rgb \
  	-red sofia/cii-30_mom0_35.0.40.0.fits -scale limits 0 20 \
  	-green sofia/cii-30_mom0_21.0.27.0.fits -scale limits 0 145 \
  	-blue sofia/cii-30_mom0_6.0.11.0.fits -scale limits 0 25 \
  	-frame new herschel/PACS160um-image.fits -scale limits -0.25 1 \
  	-frame first -frame delete -frame first \
  	-wcs galactic -frame lock wcs \
  	-region load catalogs/m16_lobes_outlined_4.reg \
  	-region select all \
  	-region load catalogs/m16_west_cavity_spec_regions_northcircle.reg \
  	-region select invert -region color white -region select none \
  \
  	-frame last \
  	-region load catalogs/m16_lobes_outlined_4.reg \
  	-region load catalogs/m16_west_cavity_spec_regions_northcircle.reg \
  	-region select all -region color white -region select none \
  \
  	-geometry 1700x900 -colorbar no \
  	-tile -tile grid gap 0 -pan to +17:00:20.400 +0:52:20.000 wcs galactic \
  	-frame first -zoom to 4.5 \
  	-wcs degrees -grid yes -grid grid no -grid numerics color black \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics fontweight normal \
  	-grid numerics gap1 2 -grid numerics gap2 6 \
  	-grid tickmarks width 1 -grid tickmarks color black \
    -grid border yes -grid border width 3 -grid border color black \
  \
  	-frame next -wcs degrees -grid yes -grid grid no -grid numerics color white \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics fontweight normal \
  	-grid numerics gap1 2 -grid numerics gap2 6 \
  	-grid tickmarks width 1 -grid tickmarks color white \
    -grid border yes -grid border width 3 -grid border color black \

}

function blue_cyan() {
  if [[ $1 -eq 1 ]]; then
    vellims="10.0.27.0"
    rvlims="0 125 "
    bgvlims="10 170 "
  elif [[ $1 -eq 2 ]]; then
    vellims="10.0.21.0"
    rvlims="0 100 "
    bgvlims="0 110 "
  elif [[ $1 -eq 3 ]]; then
    vellims="21.0.23.0"
    rvlims="0 45 "
    bgvlims="0 60 "
  elif [[ $1 -eq 4 ]]; then
    vellims="23.0.27.0"
    rvlims="0 90 "
    bgvlims="10 120 "
  else
    echo "blue_cyan: what? unrecognized arg \"${1}\""
    return 0
  fi
  ds9 -rgb \
  	-red "apex/12co32_mom0_${vellims}.fits" -scale limits $rvlims -scale LINEAR \
  	-green "sofia/cii_mom0_${vellims}.fits" -scale limits $bgvlims -scale LINEAR \
  	-blue "sofia/cii_mom0_${vellims}.fits" -scale limits $bgvlims -scale LINEAR \
  	-frame first -frame delete \
  	-frame first -zoom to fit -zoom to 3.2 -colorbar no \
  	-wcs galactic -frame lock wcs \
  	-geometry 700x1200 \
  	-frame first -wcs degrees -grid yes -grid grid no -grid numerics color black \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 \
  	-grid tickmarks width 2 -grid grid gap1 0.1 -grid grid gap2 0.1 \
    -grid tickmarks color black -grid numerics fontweight bold \
    -grid border yes -grid border width 3 -grid border color black \

}

function velocity_rgb() {
  zscale="LINEAR"
  ds9 -rgb \
  	-red sofia/cii_mom0_23.0.27.0.fits -scale limits 0 100 -scale $zscale \
  	-green sofia/cii_mom0_21.0.23.0.fits -scale limits 0 68 -scale $zscale \
  	-blue sofia/cii_mom0_10.0.21.0.fits -scale limits 0 85 -scale $zscale \
  	-frame first -frame delete \
  	-rgb \
  	-red apex/12co32_mom0_23.0.27.0.fits -scale limits 0 90 -scale $zscale \
  	-green apex/12co32_mom0_21.0.23.0.fits -scale limits 0 40 -scale $zscale \
  	-blue apex/12co32_mom0_10.0.21.0.fits -scale limits 0 75 -scale $zscale \
  \
  	-frame first -zoom to fit -zoom to 1 -colorbar no \
  	-wcs galactic -frame lock wcs \
  	-geometry 1000x700 \
  	-tile -tile grid gap 0 \
  	-frame first -wcs degrees -grid yes -grid grid no -grid numerics color black \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 \
  	-grid tickmarks width 2 -grid grid gap1 0.1 -grid grid gap2 0.1 -grid tickmarks color black \
    -grid border yes -grid border width 3 -grid border color black \
  \
  	-frame next -wcs degrees -grid yes -grid grid no -grid numerics color black \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 \
  	-grid tickmarks width 2 -grid grid gap1 0.1 -grid grid gap2 0.1 -grid tickmarks color black \
    -grid border yes -grid border width 3 -grid border color black \
  # \
  	# -frame next -wcs degrees -grid yes -grid grid no -grid numerics color white \
  	# -grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 \
  	# -grid tickmarks width 2 -grid grid gap1 0.1 -grid grid gap2 0.1 -grid tickmarks color white \
  	# -frame next -wcs degrees -grid yes -grid grid no -grid border no -grid numerics color white \
  	# -grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 \
  	# -grid tickmarks width 2 -grid grid gap1 0.1 -grid grid gap2 0.1 -grid tickmarks color white \
    # -grid border yes -grid border width 3 -grid border color black \
  # \
  # 	-frame last -frame hide -frame last -frame hide -frame first \

}

current_dir=$(pwd)
work_dir="/home/ramsey/Documents/Research/Feedback/m16_data"
echo "Moving from $current_dir to $work_dir"
pushd $work_dir > /dev/null
if [[ $1 -eq 1 ]]; then
  westcavity_cii &
elif [[ $1 -eq 2 ]]; then
  blue_cyan $2 &
elif [[ $1 -eq 3 ]]; then
  velocity_rgb &
else
  echo "${0}: what? argument \"${1}\" unrecognized"
fi
popd > /dev/null
echo "Returned to $(pwd)"


###### from velocity_rgb
# -frame new spitzer/m16_IRAC4_mosaic.fits.fz -scale limits -0.004 288 \
# -crop 7667.5 5918 2478 2571 \
# -frame new herschel/SPIRE250um-image.fits -scale limits 229 2974 \
