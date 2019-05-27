import numpy as np
import os
from data import kddcup10
from dsebm import DSEBM
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def minmax_normalization(x, base):
    min_val = np.min(base, axis=0)
    max_val = np.max(base, axis=0)
    norm_x = (x - min_val) / (max_val - min_val + 1e-10)
    return norm_x

if __name__ == '__main__':
    data = kddcup10.Kddcup('demo/kddcup99-10.data.pp.csv.gz')
    data.get_clean_training_testing_data(0.5)

    base = np.concatenate([data.train_data, data.test_data], axis=0)
    data.data = minmax_normalization(data.train_data, base)
    data.test_data = minmax_normalization(data.test_data, base)

    opts = {
        'lr': 1e-4,
        'batch_size': 1024,
        'epoch_num': 50,
    }

    model = DSEBM(opts)
    model.train(data)