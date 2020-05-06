"""
This was to join the 3d animations of CII and CO
Created: unsure... could check when I uploaded that GIF to my website
"""
import numpy as np
import imageio
import glob

img_path = "figures/vel3d/"
cii_fngen = lambda d: f"anim_CII_{d:04d}.png"
co_fngen = lambda d: f"anim_CO_{d:04d}.png"
final_fngen = lambda d: f"anim_COCII_{d:04d}.png"

n_imgs = len(glob.glob(img_path+"anim_CO_*"))

for i in range(n_imgs):
    img_cii = imageio.imread(img_path+cii_fngen(i))
    img_co = imageio.imread(img_path+co_fngen(i))
    img_both = np.concatenate([img_cii, img_co], axis=1)
    imageio.imwrite(img_path+final_fngen(i), img_both)
    if i % 36 == 0:
        print(i, end=" done, ")
print("all done!")
