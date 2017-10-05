import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv')
values = df_train.values

# we should shuffle all examples
np.random.shuffle(values)

# splitting to train and validation set
index = int(len(values) * 0.8)
train, val = values[:index], values[index:]

df_tr = pd.DataFrame([[f, t] for f, t in train])
df_tr.columns = ['id', 'label']

df_tr.to_csv('train_split.csv', index=False)

df_val = pd.DataFrame([[f, t] for f, t in val])
df_val.columns = ['id', 'label']

df_val.to_csv('val_split.csv', index=False)
