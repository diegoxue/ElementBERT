import joblib
import glob
from scipy.stats import ttest_ind
import numpy as np

prop_id = 2

traj_buffer = []
for f_n in range(1, 9):
    base_path = f'results\prop_{prop_id}_fn_{f_n}\\*.pkl'

    ep_par_buffer = []
    bt_par_buffer = []
    for path in glob.glob(base_path):
        ep_res, bt_res = joblib.load(path)
        ep_res = np.array(ep_res)
        bt_res = np.array(bt_res)

        ep_par_buffer.append(ep_res.mean())
        bt_par_buffer.append(bt_res.mean())

    traj_buffer.append((f_n, 
                        np.mean(ep_par_buffer), np.std(ep_par_buffer),
                        np.mean(bt_par_buffer), np.std(bt_par_buffer)))

np.savetxt(
    f'results\prop_{prop_id}_traj.txt',
    traj_buffer,
    delimiter='\t',
)
