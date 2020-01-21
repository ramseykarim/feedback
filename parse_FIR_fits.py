import numpy as np
import matplotlib.pyplot as plt
import pickle

testname = "/home/rkarim/Research/mantipython/TEST.pkl"

herschel_path = "/home/rkarim/Research/Feedback/ancillary_data/herschel/"
# fns = ["2p_5b_freshoffsets", "3p_5b_freshoffsets", "2p_3b"]
# fns = [f"{herschel_path}RCW49large_{fn}.pkl" for fn in fns]
with open(herschel_path+"RCW49large_TEST.pkl", 'rb') as f:
    fit, chisq, models, diffs = pickle.load(f)


T, tau = fit
p70, p160, s250, s350, s500 = diffs
beta = np.full(T.shape, 2.0)

plt.figure(figsize=(16, 9))
plt.subplot(231)
plt.imshow(T, origin='lower', vmin=10, vmax=30)
plt.colorbar()
plt.title('T')
plt.subplot(232)
plt.imshow(tau, origin='lower', vmin=-3, vmax=-1)
plt.colorbar()
plt.title('tau')
plt.subplot(233)
plt.imshow(beta, origin='lower', vmin=1, vmax=2.5)
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
plt.imshow(chisq, origin='lower', vmax=50)

plt.show()
