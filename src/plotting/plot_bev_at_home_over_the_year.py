import logging
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure

"""
Calendar heatmaps from Pandas time series data.
Plot Pandas time series data sampled by day in a heatmap per calendar year,
similar to GitHub's contributions calendar.
Based on Martijn Vermaat's calmap:  https://github.com/martijnvermaat/calmap
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_dates(data_raw):
    data = data_raw.copy()
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])

    return data


logging.basicConfig(
    filename='roof_consumption_coverage.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def prepare_df(s1):
    s1['end'] = s1['end'] - pd.Timedelta(seconds=1)
    assert np.all(s1['start'] < s1['end'])

    s1['value'] = 0
    s1.loc[s1.is_home, 'value'] = 1

    s1_start = s1[['start', 'value', 'vin']].copy()
    s1_start['tstamp'] = s1_start['start']

    s1_end = s1[['end', 'value', 'vin']].copy()
    s1_end['tstamp'] = s1_end['end']

    s11 = pd.concat((s1_start[['tstamp', 'value', 'vin']], s1_end[['tstamp', 'value', 'vin']]))
    s11 = s11.set_index('tstamp')

    return s11


output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')
fig_out = os.path.join('plots', 'bev_at_home_over_year', 'bev_at_home_over_year.png')

# baseline = pd.read_csv(os.path.join(output_folder, 'results_baseline.csv'))
# baseline = parse_dates(baseline)

s1 = pd.read_csv(os.path.join(output_folder, 'results_scenario1.csv'))
s1 = parse_dates(s1)

s11 = prepare_df(s1)
all_users = s11.vin.unique()
user_df_resampled_list = []
nb_users = all_users.shape[0]
# user_this = all_users[0]
at_home_time_list = []
for user_this in all_users:
    df = s11[s11['vin'] == user_this]
    df = pd.DataFrame(df['value'])

    # resample dataframe of 1 user
    # https://stackoverflow.com/questions/49191998/pandas-dataframe-resample-from-irregular-timeseries-index/55654486
    resample_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=['value'])
    user_df_resampled = df.combine_first(dummy_frame).interpolate('time')

    # aggregate to 1 hour. Every hour has now the fraction of the time that 1 user was at home
    user_df_resampled = user_df_resampled.resample('1H').sum() / 60
    user_df_resampled_list.append(user_df_resampled)

all_users_df = pd.concat(user_df_resampled_list, axis=1).mean(axis=1)
all_users_df = pd.DataFrame(all_users_df, columns=['value'])
all_users_df['hour'] = all_users_df.index.hour
all_users_df['day'] = all_users_df.index.floor('D')

matrix = all_users_df.pivot(index=['day'], columns=['hour'], values='value')
matrix = matrix.transpose()

# plot
journal = Journal.POWERPOINT_A3
fig, ax = init_figure(nrows=1,
                      ncols=1,
                      fig_height_mm=150,
                      columnes=Columnes.ONE,
                      journal=journal,
                      disabled_spines=[],
                      )

plt.sca(ax)
im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='None')

# plt.colorbar()
plt.grid(b=None)
plt.xlabel("Days in study")
plt.ylabel("Hours of the day")

plt.yticks([0, 6, 12, 18, 24])

im.set_clim([0, 1])
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size="5%", pad=1.0, pack_start=True)
fig.add_axes(cax)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", fraction=0.5)
cb.outline.set_visible(False)
cb.set_label("Percentage of cars at home")
save_figure(fig_out, dpi=10)
