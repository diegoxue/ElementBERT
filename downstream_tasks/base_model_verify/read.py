import joblib
import glob

import numpy as np

base_path = 'results\\*.pkl'

for path in glob.glob(base_path):
    ep_res, bt_res = joblib.load(path)
    ep_res = np.array(ep_res)
    bt_res = np.array(bt_res)

    print('ep', ep_res.mean())
    print('bt', bt_res.mean())
    print('-' * 100)
