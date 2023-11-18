#!/bin/sh

color1='magenta'
color2='black'

datapath_her='../m16_data/herschel/'
datapath_spi='../m16_data/spitzer/'
datapath_opt='../m16_data/optical/'

img_path="/home/ramsey/Pictures/2023-11-07/"

gallwidth="1.95"
galbwidth="1.5"
gallcenter="+17:05:05.055"
galbcenter="+0:42:45.713"
crop_limits="-crop $gallcenter $galbcenter $gallwidth $galbwidth wcs galactic degrees"
# I cannot get crop to work anymore

##############################
################################
###########################3
# Herschel


ds9 \
    -rgb \
      -red ${datapath_her}SPIRE250um-image.fits -scale LINEAR -scale limits 140 2100 \
      -green ${datapath_her}PACS160um-image.fits -scale ASINH -scale limits -0.28 2 \
      -blue ${datapath_her}PACS70um-image.fits -scale SQRT -scale limits -0.05 0.5 \
    -wcs galactic \
    -frame lock wcs \
    -grid yes -grid grid no -grid axes color $color2 \
    -grid axes type interior \
    -grid numerics color $color2 -grid tickmarks color $color2 \
    -grid border color $color2 \
    -grid type analysis \
    -grid numerics gap1 -1 -grid numerics gap2 -1 \
    -grid labels gap1 -1 -grid labels gap2 3.5 \
    -view colorbar no \
    -geometry 1100x1100 \
    -pan to $gallcenter $galbcenter wcs galactic \
    -zoom to 0.83 \
    -saveimage eps ${img_path}herschel_ds9.eps


##############################
################################
###########################3
# Spitzer



# ds9 \
#     -rgb \
#       -red ${datapath_spi}m16_MIPS24um_mosaic.fits.fz -scale LOG -scale limits 20 1325 \
#       -green ${datapath_spi}m16_IRAC4_mosaic.fits.fz -scale ASINH -scale limits 25 291 \
#       -blue ${datapath_spi}m16_IRAC2_mosaic.fits.fz -scale ASINH -scale limits 2.5 10 \
#     -wcs galactic \
#     -frame lock wcs \
#     -grid yes -grid grid no -grid axes color $color2 \
#     -grid axes type interior \
#     -grid numerics color $color2 -grid tickmarks color $color2 \
#     -grid border color $color2 \
#     -grid type analysis \
#     -grid numerics gap1 -1 -grid numerics gap2 -1 \
#     -grid labels gap1 -1 -grid labels gap2 3.5 \
#     -view colorbar no \
#     -geometry 1100x1100 \
#     -pan to $gallcenter $galbcenter wcs galactic \
#     -zoom to 0.2 \
#     -saveimage eps ${img_path}spitzer_ds9.eps



    # -red ${datapath_spi}SPITZER_I4_mosaic_ALIGNED.fits -scale LINEAR -scale zscale \
    # -green ${datapath_spi}SPITZER_I2_mosaic_ALIGNED.fits -scale ASINH -scale zscale \
    # -blue ${datapath_spi}SPITZER_I1_mosaic_ALIGNED.fits -scale SQRT -scale zscale \

##############################
################################
###########################3
# DSS2

# ds9 \
#   ${datapath_opt}dss2-red.18.18.45.1-13.47.31.2.fits -scale ASINH -scale minmax \
#   -wcs galactic \
#   -frame lock wcs \
#   -grid yes -grid grid no -grid axes color $color2 \
#   -grid axes type interior \
#   -grid numerics color $color2 -grid tickmarks color $color2 \
#   -grid border color $color2 \
#   -grid type analysis \
#   -grid numerics gap1 -1 -grid numerics gap2 -1 \
#   -grid labels gap1 -1 -grid labels gap2 3.5 \
#   -view colorbar no \
#   -geometry 1100x1100 \
#   -pan to $gallcenter $galbcenter wcs galactic \
#   -zoom to 0.2 \
#   -saveimage eps ${img_path}dss2_ds9.eps
#



# change interior to exterior for publication
# change analysis to publication for publication


# probably not using but useful to have
# -grid grid gap1 0.04166666666666667 \


# ds9 \
#     -rgb \
#       -red ${datapath_spi}m16_MIPS24um_mosaic.fits.fz -scale LOG -scale limits 20 1325 \
#       -green ${datapath_spi}m16_IRAC4_mosaic.fits.fz -scale ASINH -scale limits 25 291 \
#       -blue ${datapath_spi}m16_IRAC2_mosaic.fits.fz -scale ASINH -scale limits 2.5 10 \
#     -wcs galactic \
#     -frame lock wcs \
#     -geometry 1100x1100 \
#     -pan to $gallcenter $galbcenter wcs galactic \
#     -zoom to 0.2 \
