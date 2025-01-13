#!/bin/bash


current_dir=$(pwd)
work_dir="/home/ramsey/Documents/Research/Feedback/m16_data"
echo "Moving from $current_dir to $work_dir"
pushd $work_dir > /dev/null


function make_rgb_img() {
  ds9 -rgb \
    -red spitzer/m16_MIPS24um_mosaic.fits.fz \
    -scale SQRT -scale limits 30 1500 \
    -crop 4560 5034 7615 2037 \
\
    -green spitzer/m16_IRAC4_mosaic.fits.fz \
    -scale LINEAR -scale limits 22 260 \
    -crop 7735 6038.5 5059 2378 \
\
    -blue spitzer/m16_IRAC3_mosaic.fits.fz \
    -scale LINEAR -scale limits 7 105 \
    -crop 7735 6038.5 5059 2378 \
\
    -frame first -frame delete \
  	-frame first -red \
  	-geometry 1400x1000 \
  	-wcs galactic -frame lock wcs \
  	-pan to +17:02:09.825 +0:45:57.156 wcs galactic \
    -zoom to 0.4 0.4 \
\
  	-wcs degrees -grid yes -grid grid no -grid border yes -grid numerics color white -grid numerics fontweight bold \
  	-grid axes type exterior -grid format1 d.1 -grid format2 d.1 -grid numerics gap1 2 -grid numerics gap2 6 -grid tickmarks width 2 \
  	-grid tickmarks color white -grid border width 3 -grid border color black -colorbar no \

}

# \
#   	-blue spitzer/m16_IRAC2_mosaic.fits.fz \
#   	-scale LINEAR -scale limits 1 18 \
#   	-crop 7735 6038.5 5059 2378 \


make_rgb_img &
popd > /dev/null
echo "Returned to $(pwd)"
