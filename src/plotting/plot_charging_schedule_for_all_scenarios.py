import datetime
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from src.methods.PV_interface import get_PV_generated
from src.methods.helpers import soc2remainingCharge


def parse_dates(data_raw):
    data = data_raw.copy()
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])

    return data


def get_charging_schedule(scenario_data):
    rename_dict = {'start': 'tstamp', 'end': 'tstamp', 'soc_start': 'soc',
                   'soc_end': 'soc'}
    start_soc = scenario_data[['vin', 'start', 'soc_start']].copy()
    start_soc.rename(columns=rename_dict, inplace=True)

    end_soc = scenario_data[['vin', 'end', 'soc_end']].copy()
    end_soc.rename(columns=rename_dict, inplace=True)

    return pd.concat((start_soc, end_soc)).sort_values(by=['vin', 'tstamp'])


def filter_by_vin_time(data, vin, t_start, t_end):
    filter_boolean = (data['vin'] == vin) & \
                     ((data['start']) >= t_start) \
                     & ((data['end']) < t_end)
    return data[filter_boolean]


def add_point_for_is_home_polygon(data):
    # create an extra point for start and end points so that the polygones are cleaner
    # for this we append the "end" timestamps to the other dataframe
    df1 = pd.DataFrame(data=data[['start', 'is_home']].values,
                       columns=['timestamp', 'is_home'],
                       index=data.index)
    df2 = pd.DataFrame(data=data[['end', 'is_home']].values,
                       columns=['timestamp', 'is_home'],
                       index=data.index)

    return pd.concat([df1, df2]).sort_index()  # sort_values(by=['timestamp', 'is_home'], )


def draw_is_home_areas(data_raw, ax):
    data = data_raw.copy()
    data = add_point_for_is_home_polygon(data)

    art = ax.fill_between(data['timestamp'],
                          0,
                          100,
                          where=data.is_home,
                          alpha=1.0,
                          color="#8da97c",
                          edgecolor="#8da97c",
                          zorder=1,
                          facecolor="#8da97c")

    art = ax.fill_between(data['timestamp'],
                          0,
                          100,
                          alpha=0.3,
                          color="#8da97c",
                          edgecolor="#8da97c",
                          zorder=1,
                          facecolor="#8da97c")

    l1 = mpatches.Patch(color="#8da97c", alpha=1.0, label="Car is at home location")
    l2 = mpatches.Patch(color="#8da97c", alpha=0.3, label="Car is away")

    return l1, l2


def get_soc_timeseries(data, column_soc_start, column_soc_end):
    ts1 = pd.DataFrame(data=data[['start', column_soc_start]].values,
                       columns=['timestamp', 'soc'])
    ts2 = pd.DataFrame(data=data[['end', column_soc_end]].values,
                       columns=['timestamp', 'soc'])
    ts = pd.concat([ts1, ts2],
                   axis=0,
                   sort=False,
                   ignore_index=True)
    return ts.set_index('timestamp')


def plot_soc_timeseries_for_scenario(data_scenario, vin, t_start, t_end, ax,
                                     color='k', is_home_areas=False,
                                     column_soc_start='soc_start',
                                     column_soc_end='soc_end',
                                     norm=None):
    data = data_scenario.copy()
    data = filter_by_vin_time(data, vin=vin, t_start=t_start, t_end=t_end)
    if is_home_areas:
        draw_is_home_areas(data, ax)

    ts_soc = get_soc_timeseries(data, column_soc_start, column_soc_end)
    if norm is not None:
        ts_soc['soc'] = ts_soc['soc'] * norm

    ts_soc.plot(ax=ax, color=color, linewidth=3, zorder=2, legend=False)

    lines, labels = ax.get_legend_handles_labels()
    line_handle_for_legend = lines[0]

    return line_handle_for_legend


def add_pvgen_on_second_axis(vin, t_start, t_end, ax,
                             label_yax2=r"PV Generation [kWh]"):
    ax2 = ax.twinx()
    datelist = pd.date_range(t_start, t_end, freq='30T', name='timesteps').to_list()
    pv_list = [get_PV_generated(start=datelist[i], end=datelist[i + 1], house_ID=vin, pv_model="PVMODEL_SPV170")
               for i in range(len(datelist) - 1)]
    pvgen_line_handle = ax2.plot(datelist[:-1], pv_list, label='PV generation', linewidth=3)

    ax2.set_ylabel(label_yax2)

    return ax2, pvgen_line_handle


output_folder = os.path.join('.', 'data', 'output')

baseline = pd.read_csv(os.path.join(output_folder, 'results_baseline.csv'))
baseline = parse_dates(baseline)

scenario1 = pd.read_csv(os.path.join(output_folder, 'results_scenario1.csv'))
scenario1 = parse_dates(scenario1)

scenario2 = pd.read_csv(os.path.join(output_folder, 'results_scenario2.csv'))
scenario2 = parse_dates(scenario2)

scenario3 = pd.read_csv(os.path.join(output_folder, 'results_scenario3.csv'))
scenario3 = parse_dates(scenario3)

vin = '' # anonymized

t_start = '2017-03-16T00:00:00'
t_end = '2017-03-24T00:00:00'

t_start = datetime.datetime.strptime(t_start, '%Y-%m-%dT%H:%M:%S')
t_end = datetime.datetime.strptime(t_end, '%Y-%m-%dT%H:%M:%S')

# filter
f, ax = plt.subplots()
l_bs = plot_soc_timeseries_for_scenario(baseline, vin, t_start, t_end, ax,
                                        is_home_areas=True)
l_sc2 = plot_soc_timeseries_for_scenario(scenario2, vin, t_start, t_end, ax,
                                         color='b',
                                         column_soc_start='kWh_start',
                                         column_soc_end='kWh_end',
                                         is_home_areas=True,
                                         norm=100 / soc2remainingCharge(0))

l_sc3 = plot_soc_timeseries_for_scenario(scenario3, vin, t_start, t_end, ax,
                                         color='r',
                                         column_soc_start='kWh_start',
                                         column_soc_end='kWh_end',
                                         norm=100 / soc2remainingCharge(0))

ax2, l_pv = add_pvgen_on_second_axis(vin, t_start, t_end, ax)

ylim = ax.get_ylim()
ax.set_ylim((ylim[0], 101))
ylim = ax.get_ylim()

pv_upper_ylim = 10.1
pv_lower_ylim = pv_upper_ylim * (ylim[0] - ylim[1]) / 101 + pv_upper_ylim
ax2.set_ylim((pv_lower_ylim, 10.1))

plt.savefig(os.path.join('plots', 'charging_schedule_all_scenarios.png'))
