import os
import struct
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import math

base_dir = os.path.abspath('../data/')



def create_temp_dataframe(file, start_date, end_date):
    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    d = (end - start).days
    date = start
    df = pd.DataFrame(columns=['tmax', 'tmin'])

    with open(file, "rb") as f:
        for i in range(0, d + 1):
            tmax = struct.unpack('<259200f', f.read(4 * 259200))
            nmax = struct.unpack('<259200f', f.read(4 * 259200))
            tmin = struct.unpack('<259200f', f.read(4 * 259200))
            nmin = struct.unpack('<259200f', f.read(4 * 259200))
            # print('tmax:',tmax[0:1000])
            # print('tmin:',tmin[0:1000])
            # df = pd.concat([df,pd.DataFrame(zip(tmax,tmin),index=index, columns=['tmax', 'tmin'])])
            df = df.append(pd.DataFrame(zip(tmax, tmin), columns=['tmax', 'tmin']), ignore_index=True)
            date = date + timedelta(days=1)

        date = pd.date_range(start, end, freq='D')
        lat = np.arange(-89.75, 90.25, 0.5)
        lon = np.arange(0.25, 360.25, 0.5)
        index = pd.MultiIndex.from_product([date, lat, lon], names=['date', 'lat', 'lon'])
        df.set_index(index,inplace=True)
        df = df.reorder_levels(['lat', 'lon', 'date'])

        return df


def interpolate_day(source_df, row):
    lat = row['lat']
    lon = row['lon']
    date = row['date']
    temp = [source_df.xs(key=[lat - 0.25, lon - 0.25, date], level=['lat', 'lon', 'date']),
            source_df.xs(key=[lat - 0.25, lon + 0.25, date], level=['lat', 'lon', 'date']),
            source_df.xs(key=[lat + 0.25, lon - 0.25, date], level=['lat', 'lon', 'date']),
            source_df.xs(key=[lat + 0.25, lon + 0.25, date], level=['lat', 'lon', 'date'])]


    in_tmax = -999.0
    in_tmin = -999.0
    countNA_tmax = 0
    sumval_tmax = 0
    sumw_tmax = 0
    countNA_tmin = 0
    sumval_tmin = 0
    sumw_tmin = 0
    source_lat = [lat - 0.25, lat - 0.25, lat + 0.25, lat + 0.25]
    for i in range(0, 4):
        tmax = temp[i].values[0][0]
        tmin = temp[i].values[0][1]

        if tmax == -999.0:
            countNA_tmax += 1
        else:
            weight = math.cos(source_lat[i] * 3.1416 / 180.0)
            sumval_tmax += weight * tmax
            sumw_tmax += weight

        if tmin == -999.0:
            countNA_tmin += 1
        else:
            weight = math.cos(source_lat[i] * 3.1416 / 180.0)
            sumval_tmin += weight * tmin
            sumw_tmin += weight

    if countNA_tmax > 2:
        in_tmax = -999.0;
    else:
        in_tmax = sumval_tmax / sumw_tmax

    if countNA_tmin > 2:
        in_tmin = -999.0
    else:
        in_tmin = sumval_tmin / sumw_tmin

    series = pd.Series([lat, lon, date, in_tmax, in_tmin], index=['lat','lon','date','tmax', 'tmin'])
    return series


def interpolate(source_df, start_date, end_date):
    target_points = pd.read_csv(base_dir + r'/target_points.csv')
    print(target_points)
    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    date = pd.date_range(start, end, freq='D')
    lat = target_points['lat']
    lon = target_points['lon']

    index = pd.MultiIndex.from_product([date, zip(lat, lon)], names=['date', 'lat-lon'])
    target = pd.DataFrame(index=index)

    target.reset_index(inplace=True)
    target['lat'], target['lon'] = target['lat-lon'].str
    # target.drop(labels=['lat-lon'], inplace=True)
    # print('target', target)

    target = target.apply(lambda x: interpolate_day(source_df, x), axis=1)
    target = target.set_index(['lat', 'lon', 'date'])

    print(target)

    return target


def calc_mean_14days(df):
    def calc_mean(x):
        ret = x[::-1].rolling(window=14, min_periods=1).mean()
        return ret[::-1]

    ret = df.groupby(['lat', 'lon']).apply(lambda x: calc_mean(x))
    ret.rename_axis(index={'date': 'start-date'}, inplace=True)
    return ret


start = '1-1-2019'
end = '3-12-2019'
df = create_temp_dataframe(base_dir + r'/rodeo/' + r'CPC_GLOBAL_T_V0.x_0.5deg.lnx.2019', start, end)
df = interpolate(df, start, end)
df = calc_mean_14days(df)
print(df)

df.to_hdf(base_dir+r'/gt-contest-tmax-tmin-14d-2019.h5')


# ret.to_hdf(base_dir+r'/gt-contest-tmax-tmin-')
# a = df.query('lat==-89.75 and lon==0.25')


# print(df.head(1440))
# df.xs(key=[23.75, 90.25], level=['lat', 'lon'])
