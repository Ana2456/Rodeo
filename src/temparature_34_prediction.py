import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt


base_dir = os.path.abspath('../data')
tmax_tmin_2019_file = base_dir + r'/gt-contest_tmax-tmin-14d-2019.h5'
pres_2019_file = base_dir + r'/gt-contest_pres.sfc.gauss.14d-2019.h5'
pevpr_2019_file = base_dir + r'/gt-contest_pevpr.sfc.gauss.14d-2019.h5'
rhum_2019_file = base_dir + r'/gt-contest_rhum.sig995.14d-2019.h5'
slp_2019_file = base_dir + r'/gt-contest_slp.14d-2019.h5'

tmax_tmin_2019 = pd.DataFrame(pd.read_hdf(tmax_tmin_2019_file))
pres_2019 = pd.DataFrame(pd.read_hdf(pres_2019_file))
pevp_2019 = pd.DataFrame(pd.read_hdf(pevpr_2019_file))
rhum_2019 = pd.DataFrame(pd.read_hdf(rhum_2019_file))
slp_2019 = pd.DataFrame(pd.read_hdf(slp_2019_file))

pres_2019.rename_axis(index={'time': 'start_date'}, inplace=True)
pevp_2019.rename_axis(index={'time': 'start_date'}, inplace=True)

# print(tmax_tmin_2019)
# print(pres_2019)
data_2019 = tmax_tmin_2019
print(data_2019.xs(key=[27, 261, '1-1-2019'], level=['lat', 'lon', 'start_date']))
print(pres_2019.xs(key=[27, 261, '1-1-2019'], level=['lat', 'lon', 'start_date']))
print(pevp_2019.xs(key=[27, 261, '1-1-2019'], level=['lat', 'lon', 'start_date']))
data_2019['pres'] = pres_2019['pres']
print(data_2019.xs(key=[27, 261, '1-1-2019'], level=['lat', 'lon', 'start_date']))
# data_2019 = tmax_tmin_2019.join([pres_2019, pevp_2019]) #pevp_2019, rhum_2019, slp_2019
data_2019['temp'] = (data_2019['tmax'] + data_2019['tmin']) / 2
data_2019.drop(['tmin', 'tmax'], axis=1, inplace=True)
# print(data_2019)


def windowed_data(dataframe, window_size, target_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataframe)
    dataset = dataset.window(window_size + target_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + target_size))
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda window: (window[:-target_size], window[-target_size:][-1]))
    # dataset = dataset.map(lambda window: (window[:-target_size], window[-target_size:]))
    dataset = dataset.batch(batch_size).repeat()
    return dataset


def normalize(data, train_split):
    mean = data[:train_split].mean()
    std = data[:train_split].std()
    data = (data - mean) / std
    return data


TRAIN_SPLIT = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 100
WINDOW_SIZE = 14
TARGET_SIZE = 1

features = data_2019.reorder_levels(['start_date', 'lat', 'lon'])
print(features.xs(key=['1-1-2019'], level=['start_date']))
features = features.interpolate(limit_direction='both')
features = features.to_numpy()
print('shape', features.shape)
# print(features)
# features = features.values
features = features.reshape([-1,40,50,2])
print('shape', features.shape)
print(features)

train_dataset = windowed_data(features[:TRAIN_SPLIT], WINDOW_SIZE, TARGET_SIZE, BATCH_SIZE)
validation_dataset = windowed_data(features[TRAIN_SPLIT:], WINDOW_SIZE, TARGET_SIZE, BATCH_SIZE)

for x, y in train_dataset.take(1):
    print(x, y)
    break

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(7, 7),
                                     input_shape=(None, 40, 50, 2),
                                     return_sequences=True,
                                     go_backwards=True,
                                     padding='same',
                                     activation='tanh', recurrent_activation='hard_sigmoid',
                                     kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                     dropout=0.4, recurrent_dropout=0.2
                                     ))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(7, 7),
                                     return_sequences=True,
                                     go_backwards=True,padding='same',
                                     activation='tanh', recurrent_activation='hard_sigmoid',
                                     kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                     dropout=0.4, recurrent_dropout=0.2
                                     ))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(7, 7),
                                     return_sequences=False,
                                     go_backwards=True,padding='same',
                                     activation='tanh', recurrent_activation='hard_sigmoid',
                                     kernel_initializer='glorot_uniform', unit_forget_bias=True,
                                     dropout=0.3, recurrent_dropout=0.2
                                     ))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                 activation='sigmoid',padding='same',
                                 data_format='channels_last'))
#
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.25))
#
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.4))
#
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.4))
#
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.4))
#
# model.add(tf.keras.layers.Dense(1, activation='linear'))

print(model.summary())

optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)

model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
history = model.fit(train_dataset,
                    steps_per_epoch=200,
                    epochs=30,
                    validation_data=validation_dataset,
                    #                     callbacks=[lr_schedule]
                    validation_steps=200
                    )


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()


plot_train_history(history, 'Multi-Step Training and validation loss')
