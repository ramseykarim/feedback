import numpy as np
import matplotlib.pyplot as plt
import pickle


def open_FIR_pickle(filename):
    with open(filename, 'rb') as f:
        fit, chisq, models, diffs = pickle.load(f)
    result_dict = {}
    if len(fit) == 3:
        T, tau, beta = fit
        result_dict['beta'] = beta
    else:
        T, tau = fit
    result_dict['T'] = T
    result_dict['tau'] = tau
    result_dict['chisq'] = chisq
    bands = [70, 160, 250, 350, 500]
    for i, m in enumerate(models):
        result_dict[f"model{bands[i]}um"] = m
        result_dict[f"model-obs{bands[i]}um"] = diffs[i]
    return result_dict


def quicklook(filename):
    with open(filename, 'rb') as f:
        fit, chisq, models, diffs = pickle.load(f)

    T, tau, beta = fit
    p70, p160, s250, s350 = diffs
    # beta = np.full(T.shape, 2.0)

    plt.figure(figsize=(16, 9))
    plt.subplot(231)
    plt.imshow(T, origin='lower', vmin=10, vmax=70)
    plt.colorbar()
    plt.title('T')
    plt.subplot(232)
    plt.imshow(tau, origin='lower', vmin=-3.5, vmax=-1)
    plt.colorbar()
    plt.title('tau')
    plt.subplot(233)
    plt.imshow(beta, origin='lower', vmin=0, vmax=2.5)
    plt.colorbar()
    plt.title('beta')

    plt.subplot(234)
    plt.imshow(p70, origin='lower', vmin=-20, vmax=20)
    plt.colorbar()
    plt.title('model-70')
    plt.subplot(235)
    plt.imshow(p160, origin='lower', vmin=-20, vmax=20)
    plt.colorbar()
    plt.title('model-160')
    plt.subplot(236)
    plt.imshow(s350, origin='lower', vmin=-2, vmax=2)
    plt.colorbar()
    plt.title('model-350')

    plt.figure()
    plt.imshow((chisq), origin='lower', vmax=15)
    plt.colorbar()
    plt.show()


def combine(fn_template):
    with open(fn_template(1), 'rb') as f:
        contents = pickle.load(f)
        tile_shape = contents[1].shape
        every_shape = tuple(x.shape for x in contents)
    full_shape = tuple(2*x for x in tile_shape)
    every_template = [np.full(((shape[0], *full_shape) if len(shape)>2 else full_shape), np.nan) for shape in every_shape]
    c1 = (slice(tile_shape[0], full_shape[0]), slice(0, tile_shape[1]))
    c2 = (slice(tile_shape[0], full_shape[0]), slice(tile_shape[1], full_shape[1]))
    c3 = (slice(0, tile_shape[0]), slice(tile_shape[1], full_shape[1]))
    c4 = (slice(0, tile_shape[0]), slice(0, tile_shape[1]))
    cutouts = (c1, c2, c3, c4)
    for i, cutout in zip(range(1, 5), cutouts):
        with open(fn_template(i), 'rb') as f:
            contents = pickle.load(f)
        for target, source in zip(every_template, contents):
            try:
                target[:, cutout[0], cutout[1]] = source[:]
            except:
                target[cutout] = source[:]
    with open(fn_template("FULL"), 'wb') as f:
        pickle.dump(every_template, f)
    print('wrote full map')


if __name__ == "__main__":
    # Desktop directory
    herschel_path = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
    # Laptop directory
    herschel_path = "/home/ramsey/Documents/Research/Feedback/ancillary_data/herschel/"

    # combine(lambda i: f"{herschel_path}RCW49large_350grid_3p_nocal_TILE{i}.pkl")
    quicklook(herschel_path+"RCW49large_350grid_3p_TILEFULL.pkl")
