import numpy as np
import pandas as pd

df_train = pd.read_csv('resource/train.csv')
values = df_train.values

#  we should shuffle all examples
np.random.shuffle(values)

# splitting to train and validation set
index = int(len(values) * 0.9)
train, val = values[:index], values[index:]

df_tr = pd.DataFrame([f for f in train])
df_tr.columns = df_train.columns

df_tr.to_csv('resource/train_split.csv', index=False)

df_val = pd.DataFrame([f for f in val])
df_val.columns = df_train.columns

df_val.to_csv('resource/val_split.csv', index=False)
