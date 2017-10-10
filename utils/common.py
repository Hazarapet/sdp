import cv2
import numpy as np
import pandas as pd


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]


def aug(df_train, threshold=7e-4):
    target = df_train[df_train['target'] == 1]
    target_mean = np.mean(target, axis=0)

    not_target = df_train[df_train['target'] == 0]
    not_target_mean = np.mean(not_target, axis=0)

    _difference = np.abs(target_mean - not_target_mean)
    _keys = [k for k, item in _difference[_difference < threshold].iteritems()]
    _max = np.max(df_train[_keys], axis=0)

    augment = df_train
    for ind, tr in df_train.iterrows():
        if tr['target'] == 0:
            continue

        _randoms = [np.random.randint(rn, size=1)[0] for rn in np.array(_max)]

        for k, rn in zip(_keys, _randoms):
            tr[k] = rn

        augment = augment.append(pd.DataFrame([tr], columns=augment.columns))

    return augment


def ensemble(array):
    new_array = []
    for cl in range(array.shape[1]):
        cn = list(array[:, cl]).count(1)
        all_cn = array.shape[0]
        if cn >= all_cn / 2.:
            new_array.append(1)
        else:
            new_array.append(0)

    return new_array