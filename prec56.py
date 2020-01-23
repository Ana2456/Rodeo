import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xarray as xr
from scipy.io import netcdf
import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
import wget


base_dir = os.path.abspath('./data')
data_file = base_dir + r'/gt-contest_tmp2m-14d-1979-2018.h5'
tmax_file = base_dir + r'/gt-contest_tmax-14d-1979-2018.h5'
tmin_file = base_dir + r'/gt-contest_tmin-14d-1979-2018.h5'
pres_file = base_dir + r'/gt-contest_pres.sfc.gauss-14d-1948-2018.h5'
pevp_file = base_dir + r'/gt-contest_pevpr.sfc.gauss-14d-1948-2018.h5'
rhum_file = base_dir + r'/gt-contest_rhum.sig995-14d-1948-2018.h5'
slp_file = base_dir + r'/gt-contest_slp-14d-1948-2018.h5'

tmax_tmin_2019_file = base_dir + r'/gt-contest-tmax-tmin-14d-2019.h5'
pres_2019_file = base_dir + r'/gt-contest_pres.sfc.gauss.14d-2019.h5'
pevpr_2019_file = base_dir + r'/gt-contest_pevpr.sfc.gauss.14d-2019.h5'
rhum_2019_file = base_dir + r'/gt-contest_rhum.sig995.14d-2019.h5'
slp_2019_file = base_dir + r'/gt-contest_slp.14d-2019.h5'

tmax_tmin_2018_file = base_dir + r'/gt-contest-tmax-tmin-14d-2018.h5'
pres_2018_file = base_dir + r'/gt-contest_pres.sfc.gauss.14d-2018.h5'
pevpr_2018_file = base_dir + r'/gt-contest_pevpr.sfc.gauss.14d-2018.h5'
rhum_2018_file = base_dir + r'/gt-contest_rhum.sig995.14d-2018.h5'
slp_2018_file = base_dir + r'/gt-contest_slp.14d-2018.h5'


tmax_tmin_2019 = pd.DataFrame(pd.read_hdf(tmax_tmin_2019_file))
pres_2019 = pd.DataFrame(pd.read_hdf(pres_2019_file))
pevp_2019 = pd.DataFrame(pd.read_hdf(pevpr_2019_file))
rhum_2019 = pd.DataFrame(pd.read_hdf(rhum_2019_file))
slp_2019 = pd.DataFrame(pd.read_hdf(slp_2019_file))

tmax_tmin_2018 = pd.DataFrame(pd.read_hdf(tmax_tmin_2018_file))
pres_2018 = pd.DataFrame(pd.read_hdf(pres_2018_file))
pevp_2018 = pd.DataFrame(pd.read_hdf(pevpr_2018_file))
rhum_2018 = pd.DataFrame(pd.read_hdf(rhum_2018_file))
slp_2018 = pd.DataFrame(pd.read_hdf(slp_2018_file))

data_2019= pd.concat([tmax_tmin_2019, pres_2019, pevp_2019, rhum_2019, slp_2019], axis=1)
data_2019['temp']=(data_2019['tmax']+data_2019['tmin'])/2
data_2019.drop(['tmin', 'tmax'], axis=1,  inplace=True)

data_2018= pd.concat([tmax_tmin_2018, pres_2018, pevp_2018, rhum_2018, slp_2018], axis=1)
data_2018['temp']=(data_2018['tmax']+data_2018['tmin'])/2
data_2018.drop(['tmin', 'tmax'], axis=1, inplace=True)

data_2018_2019 = pd.concat([data_2018, data_2019])

tmax = pd.DataFrame(pd.read_hdf(tmax_file))
tmin = pd.DataFrame(pd.read_hdf(tmin_file))
pres = pd.DataFrame(pd.read_hdf(pres_file))
pevp = pd.DataFrame(pd.read_hdf(pevp_file))
rhum = pd.DataFrame(pd.read_hdf(rhum_file))
slp = pd.DataFrame(pd.read_hdf(slp_file))
# data_1948_2017 = pd.concat([tmax,tmin,pres,pevp,rhum,slp], axis=1)
# data_1948_2017['temp']=(data_1948_2017['tmax']+data_1948_2017['tmin'])/2
# data_1948_2017.drop(['tmin', 'tmax'], axis=1, inplace=True)
# data_1948_2017.truncate('2016-01-05', '2016-01-10')
# print(tmax)
# print(type(tmax))
# print(tmin.index)

temp = pd.concat([tmax,tmin], axis=1)
# print(temp)

temp['temp']=(temp['tmax']+temp['tmin'])/2
a = temp.query('lat==27.0')
# print(a.tail(200))

# print(slp.head())
temp
data_1979_2017=pd.DataFrame(temp['temp'])


data_1979_2017['pres'] = pres['pres']
data_1979_2017['pevpr'] = pevp['pevpr']
data_1979_2017['rhum'] = rhum['rhum']
data_1979_2017['slp'] = slp['slp']

idx = pd.IndexSlice
data_1979_2017 = data_1979_2017.loc[idx[:,:, pd.Timestamp('1979-01-01'):pd.Timestamp('2017-12-31')],:]
data_1979_2019 = pd.concat([data_1979_2017, data_2018_2019])
data_1979_2019.to_hdf(base_dir + r'/gt-contest_temp.multivariate.14d-1979-2019.h5', key='df')


slp.drop(columns=['slp'], inplace=True)
rhum.drop(columns=['rhum'], inplace=True)
pevp.drop(columns=['pevpr'], inplace=True)

# features=data_1948_2017.dropna()
# features.plot(subplots=True)

def windowed_data(dataframe, window_size, target_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataframe)
    dataset = dataset.window(window_size+target_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+target_size))
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda window: (window[:-target_size],window[-target_size:]))
    # dataset = dataset.map(lambda window: (window[:-target_size], window[-target_size:]))
    dataset = dataset.batch(batch_size).repeat()
    return dataset

def normalize(data, train_split):
    mean = data[:train_split].mean()
    std = data[:train_split].std()
    data = (data-mean)/std
    return data


TRAIN_SPLIT = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 100
WINDOW_SIZE = 14
TARGET_SIZE = 1
data_1979_2019 = pd.read_hdf(base_dir + r'/gt-contest_temp.multivariate.14d-1979-2019.h5')

print(data_1979_2019)
# print(data_1979_2019.values)
features=data_1979_2019.xs(key=[27, 261], level=['lat', 'lon'])
# features=normalize(features.values, TRAIN_SPLIT)
print(features)
features = features.values
train_dataset = windowed_data(features[:TRAIN_SPLIT], WINDOW_SIZE, TARGET_SIZE, BATCH_SIZE)
validation_dataset = windowed_data(features[TRAIN_SPLIT:], WINDOW_SIZE, TARGET_SIZE,BATCH_SIZE)

for x, y in train_dataset.take(1):
    print(x,y)
    break

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(64,
#                      dropout=0.1,
#                      recurrent_dropout=0.1,
                     return_sequences=True,
                     input_shape=(None, 5)))
model.add(tf.keras.layers.GRU(64, activation='relu',
                       # return_sequences=True,
#                      dropout=0.1
#                      recurrent_dropout=0.1))
                             ))
model.add(tf.keras.layers.Dense(5, activation='relu'))

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0)

model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
history = model.fit(train_dataset,
                              steps_per_epoch=200,
                              epochs=30,
                              validation_data=validation_dataset,
#                     callbacks=[lr_schedule]
                              validation_steps=200
                   )

# model = tf.keras.models.Sequential([
#     tf.keras.layers.SimpleRNN(40, return_sequences=True, input_shape=[None, 5]),
#     tf.keras.layers.SimpleRNN(40),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# EVALUATION_INTERVAL = 200
# EPOCHS = 25
#
# history = model.fit(train_dataset, epochs=EPOCHS,
#                       steps_per_epoch=EVALUATION_INTERVAL,
#                       validation_data=validation_dataset, validation_steps=200)

for x, y in train_dataset.take(1):
    print(x,y)
    break


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


def forecast(model, data, window_size, target_size):
    last_date = data.index[-1]
    p_day = datetime(2020, 1, 21)
    delta = p_day - last_date
    x = data.values[-window_size:]
    print(x)
    print(x[:,-1])
    for time in range(0, 5 + delta.days, target_size):
        pred = model.predict(x[np.newaxis, time:time + window_size].astype(np.float32))
        print('pred', pred)
        pred = pred.reshape((target_size, 5))
        # pred = np.rint(pred)
        x = np.concatenate([x, pred], axis=0)

    index = pd.date_range(data.index[-window_size], periods=x.shape[0])
    x = pd.DataFrame(x[:,-1], index=index, columns=['temp34'])
    x.index.rename('Timestep', inplace=True)
    # x = x[p_day:p_day + timedelta(5)]

    x = x.reset_index()
    # x = x[['Scenario', 'Site', 'Timestep', 'Value']]
    return x

temp_data =data_1979_2019.xs(key=[27, 261], level=['lat', 'lon'])

prediction = forecast(model, temp_data, WINDOW_SIZE, TARGET_SIZE )

print(prediction)


def plot_forecast_scene_1(model, data, site, window_size, target_size):
    forecast = []
    values = data.values
    for time in range(0, len(values) - window_size, target_size):
        pred = model.predict(values[time:time + window_size][np.newaxis].astype(np.float32))
        pred = pred.reshape(target_size)
        forecast.append(pred)
    forecast = np.array(forecast).reshape(-1)

    res = pd.concat([data[window_size:].reset_index()[site], pd.DataFrame(forecast)], axis=1)
    res.plot()

plot_forecast_scene_1(model, temp_data, WINDOW_SIZE, TARGET_SIZE)