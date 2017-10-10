import sys
import numpy as np
import pandas as pd

threshold = 1e-1

df_train = pd.read_csv('resource/train.csv')

target = df_train[df_train['target'] == 1]
target_mean = np.mean(target, axis=0)

not_target = df_train[df_train['target'] == 0]
not_target_mean = np.mean(not_target, axis=0)

_difference = np.abs(target_mean - not_target_mean)
_keys = [k for k, item in _difference[_difference > threshold].iteritems()]
_max = np.max(df_train[_keys], axis=0)
_min = np.min(df_train[_keys], axis=0)

augment = df_train
count = 0

for ind, tr in df_train.iterrows():
    if tr['target'] == 0:
        continue

    _randoms = [np.random.randint(mx, size=1)[0] for mn, mx in np.array(zip(_min, _max))]
    # _randoms = [7777 for rn in np.array(_max)]

    new_tr = tr.copy()
    for k, rn in zip(_keys, _randoms):
        new_tr[k] = rn

    augment = augment.append(pd.DataFrame([new_tr], columns=augment.columns))
    count += 1

    print 'row: {}, count: {}'.format(ind, count)

print df_train.shape
print augment.shape

augment.to_csv('resource/augmented_train.csv', index=False)


