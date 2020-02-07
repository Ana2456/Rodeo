import os
from os import path
import wget
import struct
from dateutil import rrule
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import math
import timeit
import xarray as xr
import gzip

#
# import urllib3.request as request
# from urllib.request import URLopener
# from contextlib import closing

base_dir = os.path.abspath('./data')


def download_precip_data(start_date, end_date, out_dir):
    url = "https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/RT/2019/"
    for dt in rrule.rrule(rrule.DAILY,
                          dtstart=datetime.strptime(start_date, '%Y%m%d'),
                          until=datetime.strptime(end_date, '%Y%m%d')):

        date = dt.strftime('%Y%m%d')
        fn = "PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx." + date + ".RT"
        if not path.isdir(out_dir):
            os.makedirs(out_dir)
        if not path.exists(out_dir + fn):
            wget.download(url + fn, out_dir)


filepath = base_dir + r'/rodeo/noaa-precip/'
#
# start = '20190101'
# end = '20191201'

# download_precip_data(start,end,filepath)

date = datetime.strptime('20190101', '%Y%m%d')


def download__file(url, out_dir, file_name):
    # url = "ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pres.sfc.gauss.2020.nc"
    if not path.isdir(out_dir):
        os.makedirs(out_dir)
    if not path.exists(out_dir + "/" + file_name):
        wget.download(url + file_name, out_dir)


def download(out_dir):
    years = range(1979, 2021, 1)
    url1 = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/'
    url2 = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/'
    url3 = 'ftp://ftp.cpc.ncep.noaa.gov/precip/PEOPLE/wd52ws/global_temp/'
    for year in years:
        download__file(url1, out_dir, "pres.sfc.gauss." + str(year) + ".nc")
        download__file(url1, out_dir, "pevpr.sfc.gauss." + str(year) + ".nc")
        download__file(url2, out_dir, "rhum.sig995." + str(year) + ".nc")
        download__file(url2, out_dir, "slp." + str(year) + ".nc")
        if year != 2020:
            download__file(url3, out_dir, 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.' + str(year) + '.gz')

    download__file(url3, out_dir, 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.' + '2020')


# download(base_dir + '/downloads')


def save_temp_data():
    years = range(1979, 2020, 1)
    data_1979_2020 = pd.DataFrame()
    for year in years:
        start_date = '1-1-' + str(year)
        end_date = '31-12-' + str(year)
        file_name = 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.' + str(year) + '.gz'
        df = save_temp_data_for_specific_time(base_dir + '/downloads', file_name, start_date, end_date)
        print(df)
        data_1979_2020 = pd.concat([data_1979_2020, df])

    data_1979_2020.to_hdf(base_dir + r'/tmax-tmin-14d-1979-2019' + '.h5', key='df')

def concat_df():
    years = range(1979, 2020, 1)
    data_1979_2020 = pd.DataFrame()
    for year in years:
        file_name = base_dir + '/downloads/' + 'gt-contest_tmax-tmin-14d-' + str(year) + '.h5'
        df = pd.read_hdf(file_name)
        print(df)
        data_1979_2020 = pd.concat([data_1979_2020, df])
    file_name = 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.2020'
    start_date = '1-1-2020'
    end_date = '31-1-2020'
    df = save_temp_data_for_specific_time(base_dir + '/downloads', file_name, start_date, end_date)
    print(df)
    data_1979_2020 = pd.concat([data_1979_2020, df])
    data_1979_2020.to_hdf(base_dir + r'/tmax-tmin-14d-1979-2020' + '.h5', key='df')


concat_df()
start_date = '1-1-2020'
end_date = '31-1-2020'
file_name = 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.2020'
df = save_temp_data_for_specific_time(base_dir + '/downloads', file_name, start_date, end_date)
print(df)

def create_temp_dataframe(file, start_date, end_date):
    """

    :param file: str,
            gzip temperature  file path
    :param start_date: str,
            start date, format:-"%d-%m-%Y"
    :param end_date: str,
            end date, format:-"%d-%m-%Y"
    :return: DataFrame,
            pandas dataframe of tmax, tmin data, index ['lat', 'lon', 'date']
    """
    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    d = (end - start).days
    date = start
    df = pd.DataFrame(columns=['tmax', 'tmin'])

    with open(file, "rb") as f:
    # with gzip.open(file, 'rb') as f:
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
        df.set_index(index, inplace=True)
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

    series = pd.Series([lat, lon, date, in_tmax, in_tmin], index=['lat', 'lon', 'date', 'tmax', 'tmin'])
    return series


def interpolate(df, start_date, end_date):
    # target_points = pd.read_csv(base_dir + r'/target_points.csv')
    # print(target_points)
    df.reset_index(inplace=True)
    df['ilat'] = np.rint(df['lat'])
    df['ilon'] = np.rint(df['lon'])
    # print(df)
    df.set_index(keys=['ilat', 'ilon', 'date'], inplace=True)
    # print(df)
    # print(df.xs(key=[-89., 1, start_date], level=['ilat', 'ilon', 'date']))

    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    date = pd.date_range(start, end, freq='D')
    # lat = target_points['lat']
    # lon = target_points['lon']
    #
    # index = pd.MultiIndex.from_product([date, zip(lat, lon)], names=['date', 'lat-lon'])
    # target = pd.DataFrame(index=index)
    #
    # target.reset_index(inplace=True)
    # target['lat'], target['lon'] = target['lat-lon'].str
    # target.drop(columns=['lat-lon'], inplace=True)

    ###
    # lat = np.arange(-90.0, 90, 1)
    # lon = np.arange(0, 359, 1)
    lat = np.arange(20, 60, 1) # if changed, change line 250,251 accordingly
    lon = np.arange(230, 280, 1)
    index = pd.MultiIndex.from_product([date, lat, lon], names=['date', 'lat', 'lon'])
    target = pd.DataFrame(index=index)
    target.reset_index(inplace=True)
    ###

    t = target.join(df, on=['lat', 'lon', 'date'], lsuffix='l', rsuffix='r')

    # print('t', t.head(10))

    # t.set_index(keys=['lat','lon','date'],inplace=True)
    # print(t.xs(key=[27, 261, start_date], level=['lat','lon','date']))
    def func(x):
        # print('x', x)
        a = x[x['tmax'] != -999.0].copy(deep=True)
        a['weight'] = np.cos(a['lat'] * 3.1416 / 180.0)
        a['tmax-val'] = (a['tmax'] * a['weight'])
        a['tmin-val'] = (a['tmin'] * a['weight'])

        b = a.groupby(by='date')[['tmax-val', 'tmin-val', 'weight']].sum()
        b['tmax-val'] = b['tmax-val'] / b['weight']
        b['tmin-val'] = b['tmin-val'] / b['weight']
        b.drop(columns=['weight'], inplace=True)
        b.rename(columns={"tmax-val": "tmax", "tmin-val": "tmin"}, inplace=True)

        return b

    start_time = timeit.default_timer()
    t1 = t.groupby(by=['lat', 'lon']).apply(lambda x: func(x))
    # t1.set_index(['lat', 'lon', 'date'],inplace=True)
    print(timeit.default_timer() - start_time)
    # print('t1', t1)

    # start_time = timeit.default_timer()
    # target = target.apply(lambda x: interpolate_day(source_df, x), axis=1)
    # print(timeit.default_timer() - start_time)
    # target = target.set_index(['lat', 'lon', 'date'])
    # print(target)

    return t1


def extract_data_for_target_points(source_df, start_date, end_date):
    target_points = pd.read_csv(base_dir + r'/target_points.csv')
    # print(target_points)
    # print(df.xs(key=[-89.,1,'2019-01-01'], level=['ilat','ilon','date']))

    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    date = pd.date_range(start, end, freq='D')
    # lat = target_points['lat']
    # lon = target_points['lon']
    # index = pd.MultiIndex.from_product([date, zip(lat, lon)], names=['date', 'lat-lon'])

    lat = np.arange(20, 60, 1)
    lon = np.arange(230, 280, 1)

    index = pd.MultiIndex.from_product([date, lat, lon], names=['date', 'lat', 'lon'])

    target = pd.DataFrame(index=index)

    target.reset_index(inplace=True)
    # target['lat'], target['lon'] = target['lat-lon'].str
    # target.drop(columns=['lat-lon'], inplace=True)
    # print(target)
    # print(source_df)
    t = target.join(source_df, on=['lat', 'lon', 'date'], lsuffix='l', rsuffix='r')
    t.set_index(['lat', 'lon', 'date'], inplace=True)
    return t


def calc_mean_14days(df):
    """

    :param df: pandas.DataFrame
    :return:
    """

    def calc_mean(x):
        ret = x[::-1].rolling(window=14, min_periods=1).mean()
        return ret[::-1]

    ret = df.groupby(['lat', 'lon']).apply(lambda x: calc_mean(x))
    ret.rename_axis(index={'date': 'start_date'}, inplace=True)
    return ret


def resample_mean_by_day(df):
    return df.unstack(level=[0, 1]).resample('D').mean().stack(level=[2, 1]).reorder_levels([2, 1, 0])


def save_data_for_target_points(filename, start_date, end_date):
    """

    :param filename: str
        netcdf filename
    :param start_date: str,
        start date ,format-"%d-%m-%Y"
    :param end_date: str
        end date format-"%d-%m-%Y"
    """
    ds = xr.open_dataset(base_dir + r'/' + filename + r'.nc')
    df = ds.to_dataframe()
    df = resample_mean_by_day(df)
    df.rename_axis(index={'time': 'date'}, inplace=True)
    df = extract_data_for_target_points(df, start_date, end_date)
    df = calc_mean_14days(df)
    pref = ''.join(elem + '.' for elem in filename.split('.')[:-2])
    year = start.split('-')[-1]
    saved_file_name = base_dir + r'/gt-contest_' + pref + r'14d-' + year + r'.h5'
    print(saved_file_name)
    df.to_hdf(saved_file_name, key='df')


def save_temp_data_for_specific_time(out_dir, filename, start_date, end_date):
    """

    :param filename: str
        filename of temparature data
        like 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.2019'
    :param start_date: str
        start date, format:-"%d-%m-%Y"
    :param end_date: str
        end date, format:-"%d-%m-%Y"
    """

    df = create_temp_dataframe(out_dir + r'/' + filename, start_date, end_date)
    df = interpolate(df, start_date, end_date)
    print('interpolated: ', df)
    df = extract_data_for_target_points(df, start_date, end_date)
    print('extracted: ', df)
    df = calc_mean_14days(df)
    # print(df)
    year = start_date.split('-')[-1]
    saved_file_name = out_dir + r'/gt-contest_tmax-tmin-14d-' + year + '.h5'
    df.to_hdf(saved_file_name, key='df')
    print(saved_file_name)
    return df



start = '1-1-2019'
end = '31-12-2019'
df = save_temp_data_for_specific_time(base_dir , 'CPC_GLOBAL_T_V0.x_0.5deg.lnx.2019.gz', start, end)
# df.to_hdf(base_dir + r'/gt-contest-tmax-tmin-14d-' + '2018' + '.h5', key='df')
# ds = df.to_xarray()
df.to_csv(base_dir + r'/gt-contest_tmax-tmin-14d-' + '2019' + '.nc')

save_temp_data()

ds = df.to_xarray()
ds.to_netcdf(base_dir + r'/gt-contest-tmax-tmin-14d-' + '2019' + '.nc')

start_time = timeit.default_timer()
df = create_temp_dataframe(base_dir + r'/CPC_GLOBAL_T_V0.x_0.5deg.lnx.2019', start, end)
print('create_temp_dataframe', timeit.default_timer() - start_time)
print(df)
ds = df.to_xarray()
ds.to_netcdf(base_dir + r'/gt-contest-tmax-tmin-14d-' + '2019' + '.nc')

# df = pd.read_csv(base_dir +  r'/temp-0.5deg-2019.csv')
print(df)
start_time = timeit.default_timer()
df = interpolate(df, start, end)
print('interpolate', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
df = calc_mean_14days(df)
print('calc_mean_14days', timeit.default_timer() - start_time)

print(df)
df.to_hdf(base_dir + r'/gt-contest-tmax-tmin-14d-2019.h5', key='df')

base_dir = os.path.abspath('./')

start = '1-1-2019'
end = '31-12-2019'
save_data_for_target_points('pres.sfc.gauss.1deg.2019', start, end)
save_data_for_target_points('pevpr.sfc.gauss.1deg.2019', start, end)
save_data_for_target_points('rhum.sig995.1deg.2019', start, end)
save_data_for_target_points('slp.1deg.2019', start, end)

start = '1-1-2018'
end = '31-12-2018'
save_data_for_target_points('pres.sfc.gauss.1deg.2018', start, end)
save_data_for_target_points('pevpr.sfc.gauss.1deg.2018', start, end)
save_data_for_target_points('rhum.sig995.1deg.2018', start, end)
save_data_for_target_points('slp.1deg.2018', start, end)

ds = xr.open_dataset(base_dir + r'/' + 'pres.sfc.gauss.1deg.2019' + r'.nc')
df = ds.to_dataframe()
print(df)

df = pd.read_hdf(base_dir + '/gt-contest_pres.sfc.gauss.14d-2019.h5')
print(df)
# a = df.query('lat==-89.75 and lon==0.25')

# df.xs(key=[23.75, 90.25], level=['lat', 'lon'])


start_date = '1-1-2019'
end_date = '31-12-2019'
start = datetime.strptime(start_date, "%d-%m-%Y")
end = datetime.strptime(end_date, "%d-%m-%Y")
date = pd.date_range(start, end, freq='D')
lat = np.arange(10, 70, 1)
lon = np.arange(200, 300, 1)
index = pd.MultiIndex.from_product([lat, lon, date], names=['lat', 'lon', 'start_date'])
