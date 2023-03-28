#!/bin/sh

color1='magenta'
color2='black'

datapath='../m16_data/sofia/'


rawidth_left="12.91325"
rawidth_right="14.07698"
dec_center="-13:51:55.262"
decwidth="12.89194"

crop_limits_left="-crop +18:18:54.3102 $dec_center $decwidth $rawidth_left wcs fk5 arcmin"
crop_limits_right="-crop +18:18:56.8071 $dec_center $decwidth $rawidth_right wcs fk5 arcmin"

ds9 \
    -rgb \
      -red ${datapath}integrated3_2.fits -scale limits 0 200 \
      -green ${datapath}integrated3_1.fits -scale limits 0 80 \
      -blue ${datapath}integrated3_0.fits -scale limits 0 50 \
    -rgbcube ${datapath}../optical/m16_noao_astrometry.fits -frame match wcs \
    -red -scale limits 0 255 -green -scale limits 0 255 -blue -scale limits 0 255 \
    -frame first -frame delete \
    -grid yes -grid grid no -grid axes color $color2 \
    -grid axes type exterior \
    -grid numerics color $color2 -grid tickmarks color $color1 \
    -grid border color $color2 \
    -grid type publication \
    -grid numerics gap1 -1 -grid numerics gap2 -1 \
    -grid labels gap1 -1 -grid labels gap2 3.5 \
    -grid grid gap1 0.04166666666666667 \
    -frame last \
    -grid yes -grid grid no -grid axes color $color2 \
    -grid axes type exterior \
    -grid numerics color $color2 -grid tickmarks color $color2 \
    -grid border color $color2 \
    -grid type publication \
    -grid numerics gap1 -1 -grid numerics gap2 -1 \
    -grid grid gap1 0.04166666666666667 \
    -grid labels no \
    -view colorbar no \
    -geometry 1348x923 \
    -frame first -frame lock wcs \
    -pan to 170 160 physical \
    -red $crop_limits_left \
    -green $crop_limits_left \
    -blue $crop_limits_left \
    -frame last \
    -red $crop_limits_right \
    -green $crop_limits_right \
    -blue $crop_limits_right \
    -zoom to 0.47 \


    # -grid yes -grid grid no -grid axes color $color2 \
    # -grid axes type exterior \
    # -grid numerics color $color2 -grid tickmarks color $color2 \
    # -grid border color $color2 \

    # -grid yes -grid grid no -grid axes color $color1 \
    # -grid axes type exterior \
    # -grid numerics color $color1 -grid tickmarks color $color1 \
    # -grid border color $color1 \
    # -grid type publication \


    # -frame last -frame move first \
