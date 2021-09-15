import os

import mantipython

region = 'm16' # or 'rcw49'

if region == "rcw49":
    data_dir = "/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc160/"
    if not os.path.isdir(data_dir):
        # Laptop path
        data_dir = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/processed/1342255009/"
    p70_correction = 80
    p160_correction = 370
elif region == "m16":
    data_dir = "/home/rkarim/Research/Feedback/m16_data/herschel/processed/1342218995_reproc250/"
    if not os.path.isdir(data_dir):
        # Laptop path
        raise RuntimeError("Haven't moved this data to my laptop yet.")
    p70_correction = 268
    p160_correction = 1055


data_fns = {
    70: f"PACS70um-image-remapped-conv-plus{p70_correction:06d}.fits",
    160: f"PACS160um-image-remapped-conv-plus{p160_correction:06d}.fits",
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv-plus6.0pct.fits",
    160: "PACS160um-error-remapped-conv-plus8.0pct.fits",
    250: "SPIRE250um-error-remapped-conv-plus5.5pct.fits",
    350: "SPIRE350um-error-remapped-conv-plus5.5pct.fits",
    500: "SPIRE500um-error-remapped-conv-plus5.5pct.fits",
}

# organize filenames
data_dictionary = {}
for k in data_fns:
    data_dictionary[k] = (data_fns[k], err_fns[k])

# select small cutout area
if region == "rcw49":
    if "350" in data_dir:
        i0, j0 = 726, 466
        width_i, width_j = 220, 280
    elif "160" in data_dir:
        i0, j0 = 2314, 1035
        width_i, width_j = 15, 15 # just the test region
        width_i, width_j = 1286, 1286
    else:
        # 500 grid
        # i0, j0 = 519, 334 # OLD CUTOUT (retired as of March 9 2020)
        # width_i, width_j = 157, 200
        i0, j0 = 543, 337
        width_i, width_j = 294, 294 # NEW CUTOUT (as of March 9 2020, include extended stuff to the W)
        width_i, width_j = 15, 15 # just the test region
elif region == "m16":
    if "250" in data_dir:
        i0, j0 = 1017, 1767
        width_i, width_j = 620, 717
        # width_i, width_j = 10, 10
# decide whether or not this is parallel
n_processes = 10
if region == "rcw49":
    out_path = "/home/rkarim/Research/Feedback/rcw49_data/herschel/"
    if not os.path.isdir(path):
        out_path = "/home/ramsey/Documents/Research/Feedback/rcw49_data/herschel/"
elif region == "m16":
    out_path = "/home/rkarim/Research/Feedback/m16_data/herschel/results/1342218995_reproc250/"
# write_fn = "RCW49large_2p_2BAND_beta2.0.fits"
write_fn = "M16_2p_3BAND_beta2.0.fits"
# write_fn = "TEST.fits"
mantipython.fit_entire_map(data_dictionary, [70,160,250],
    ('T', 'tau'), initial_param_vals={'beta': 2.0},
    param_bounds={'T': (0, None),},
    data_directory=data_dir, log_name_func=lambda s: os.path.join(out_path, f"log{s}.log"),
    n_procs=n_processes, destination_filename=os.path.join(out_path, write_fn),
    cutout=((i0, j0), (width_i, width_j)), fitting_function='jac',
)
