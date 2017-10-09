import json
import numpy as np
import pandas as pd

df_train = pd.read_csv('resource/train.csv')
df_train = df_train.drop(['target', 'id'], axis=1)

train_mean = np.mean(df_train, axis=0).tolist()

with open('mean.json', 'w') as outfile:
    json.dump(train_mean, outfile)

