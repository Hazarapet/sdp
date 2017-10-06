import sys
import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import callbacks

st_time = time.time()
BATCH_SIZE = 1000
N_EPOCH = 100

# Read in our input data
df_train = pd.read_csv('resource/train_split.csv')
df_val = pd.read_csv('resource/val_split.csv')

df_test = pd.read_csv('resource/test.csv')

# This prints out (rows, columns) in each dataframe
print('Train shape:', df_train.shape)
print('Val shape:', df_val.shape)
print('Test shape:', df_test.shape)

y_train = np.array(df_train['target'].values)
y_val = np.array(df_val['target'].values)

x_train = np.array(df_train.drop(['target', 'id'], axis=1))
x_val = np.array(df_val.drop(['target', 'id'], axis=1))

id_test = df_test['id'].values
x_test = np.array(df_test.drop(['id'], axis=1))

model = Sequential()
model.add(Dense(32, input_shape=(57,)))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(256, input_shape=(57,)))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization(axis=1))
model.add(Activation('selu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# adam = Adam(lr=1e-4, decay=1e-5)
sgd = SGD(lr=1e-2, momentum=.9, decay=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

rm_cb = callbacks.RemoteMonitor()
ers_cb = callbacks.EarlyStopping(patience=20)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=N_EPOCH, batch_size=BATCH_SIZE, callbacks=[rm_cb, ers_cb])

print('================= Validation =================')
[v_loss, v_acc] = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE, verbose=1)
print('\nVal Loss: {:.5f}, Val Acc: {:.5f}'.format(v_loss, v_acc))

p_test = model.predict(x_test)

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = p_test
sub.to_csv('fast_result.csv', index=False)

print(sub.head())


print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

