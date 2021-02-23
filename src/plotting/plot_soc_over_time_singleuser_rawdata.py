# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:10:11 2018

@author: martinhe
"""

import datetime
import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from src.db_login import DSN
from src.methods.PV_interface import get_PV_generated
from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure

file_out = os.path.join(".", "data", "bmw_soc_overtime_data.csv")
figure_out = os.path.join(".", "plots", "soc_over_time_rawdata", "bmw_soc_overtime.png")

vin = '' # anonymized
t_start = '2017-06-19T00:00:00'
t_end = '2017-06-26T00:00:00'

dt_start = datetime.datetime.strptime(t_start, '%Y-%m-%dT%H:%M:%S')
dt_end = datetime.datetime.strptime(t_end, '%Y-%m-%dT%H:%M:%S')

# %% plot chargeing state
# download chargeing state from database/file

try:
    all_bmwdata_df = pd.read_csv(file_out)

except FileNotFoundError:
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))
    # download data from server and save to file

    query = """SELECT * from ev_homepv.ecar_data where vin = '{vin}'""".format(vin=vin)  # , t_start, t_end

    all_bmwdata_df = pd.read_sql_query(query, con=engine)
    all_bmwdata_df.to_csv(file_out, index=True)

# parse timestamps...
all_bmwdata_df['timestamp_start_utc'] = all_bmwdata_df['timestamp_start_utc'].astype('datetime64[ns]')
all_bmwdata_df['timestamp_end_utc'] = all_bmwdata_df['timestamp_end_utc'].astype('datetime64[ns]')

all_bmwdata_df = all_bmwdata_df[
    (all_bmwdata_df['vin'] == vin) & ((all_bmwdata_df['timestamp_start_utc'] + pd.Timedelta(hours=24)) >= t_start)
    & ((all_bmwdata_df['timestamp_end_utc'] - pd.Timedelta(hours=24)) < t_end)]

# plug start and end together
charging_state_timeseries1 = pd.DataFrame(data=all_bmwdata_df[['timestamp_start_utc', 'soc_customer_start']].values,
                                          columns=['timestamp', 'soc'])
charging_state_timeseries2 = pd.DataFrame(data=all_bmwdata_df[['timestamp_end_utc', 'soc_customer_end']].values,
                                          columns=['timestamp', 'soc'])

charging_state_timeseries = pd.concat([charging_state_timeseries1, charging_state_timeseries2],
                                      axis=0,
                                      sort=False,
                                      ignore_index=True)

charging_state_timeseries['timestamp'] = charging_state_timeseries['timestamp'].astype(
    'datetime64[ns]')  # transform it to timestamps again!

# set index and plot
charging_state_timeseries.set_index('timestamp', inplace=True)

f, ax = init_figure(nrows=1,
                    ncols=1,
                    columnes=Columnes.ONE,
                    journal=Journal.POWERPOINT_A3,
                    disabled_spines=['top'])

charging_state_timeseries.plot(ax=ax,
                               color="k",
                               linewidth=3,
                               zorder=2,
                               legend=False)

lines, labels = ax.get_legend_handles_labels()
l1 = lines[0]

# %% plot ishome areas
with open(os.path.join(".", "data", "Car_is_at_home_table_UTC.csv"), 'r') as data_file:
    all_data_df = pd.read_csv(data_file)

# transform to date type
all_data_df['start'] = all_data_df['start'].astype('datetime64[ns]')
all_data_df['end'] = all_data_df['end'].astype('datetime64[ns]')

# filter
all_data_df = all_data_df[(all_data_df['vin'] == vin) &
                          ((all_data_df['start'] + pd.Timedelta(hours=24)) >= t_start)
                          & ((all_data_df['end'] - pd.Timedelta(hours=24)) < t_end)]
# sort (just to be sure... the right order actually affects plotting the areas...)
all_data_df.sort_values('start', inplace=True)

# create an extra point for start and end points so that the polygones are cleaner
# for this we append the "end" timestamps to the other dataframe
df2_1 = pd.DataFrame(data=all_data_df[['start', 'is_home']].values, columns=['timestamp', 'is_home'])
df2_1['org_ix'] = df2_1.index
df2_2 = pd.DataFrame(data=all_data_df[['end', 'is_home']].values, columns=['timestamp', 'is_home'])
df2_2['org_ix'] = df2_2.index
df2 = pd.concat([df2_1, df2_2])

# set index for plotting
df2.set_index('timestamp', inplace=True, drop=False)
df2.index.rename('t_ix', inplace=True)  # index should not have the same name as a column...
df2.sort_values(['timestamp', 'org_ix'], ascending=[1, 1],
                inplace=True)  # sort because the order affects the drawing of the polygones

print(dt_start, dt_end)

ax.set_xlim(dt_start, dt_end)

# draw polygones
art = ax.fill_between(df2.index,
                      0,
                      100,
                      where=df2.is_home,
                      alpha=1.0,
                      color="#8da97c",
                      edgecolor="#8da97c",
                      zorder=1,
                      facecolor="#8da97c")

art = ax.fill_between(df2.index,
                      0,
                      100,
                      alpha=0.3,
                      color="#8da97c",
                      edgecolor="#8da97c",
                      zorder=1,
                      facecolor="#8da97c")

l2 = mpatches.Patch(color="#8da97c", alpha=1.0, label="Car is at home location")
l3 = mpatches.Patch(color="#8da97c", alpha=0.3, label="Car is away")

# art.set_edgecolor(None) #edgecolor is ugly with transparency...

# plot pv generation

# print(get_PV_generated(dt_start, dt_end, vin))
ax2 = ax.twinx()
datelist = pd.date_range(dt_start, dt_end, freq='30T', name='timesteps').to_list()
pv_list = [get_PV_generated(start=datelist[i], end=datelist[i + 1], house_ID=vin, pv_model="PVMODEL_SPV170")
           for i in range(len(datelist) - 1)]
pvgen_handle = ax2.plot(datelist[:-1], pv_list, label='PV generation', linewidth=3)

# ax2 = ax


# Put a legend below current axis

lgd = ax.legend(handles=[l1, pvgen_handle[0], l2, l3],
                labels=["SoC", "PV generation", "Car is at home", "Car is away"],
                loc='upper left',
                #                bbox_to_anchor=(0.5, -0.4),
                ncol=4,
                frameon=False,
                fancybox=False,
                shadow=False,
                framealpha=1.0,
                bbox_to_anchor=(-0.12, -0.15))

# make the plot nice

ax.set_xlabel("")
ax.set_ylabel(r"State of Charge (SoC) [\%]")
ax2.set_ylabel(r"PV Generation [kWh]")

plt.sca(ax)
#
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=11))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a'))

plt.sca(ax)
# x axis date formating
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

ax.tick_params(axis='x', which='minor', pad=10)
ax.tick_params(axis='x', which='major', pad=40)
ax.tick_params(axis='x', which='major', bottom=True)
ax.tick_params(axis='x', which='minor', bottom=True)
plt.xticks(rotation=0, horizontalalignment="center")

ax.tick_params(axis='x', which='major', length=0)

for tick in ax.xaxis.get_minor_ticks():
    tick.label.set_fontsize(14)

ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_ylim((0, 101))
ax2.set_ylim((0, 10.1))
frame = lgd.get_frame()
frame.set_facecolor('#ebefe8')

ax.spines['top'].set_visible(False)

save_figure(figure_out, bbox_extra_artists=[lgd], dpi=5)
