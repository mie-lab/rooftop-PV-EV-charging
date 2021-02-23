import datetime
import os
import sys

sys.path.append('.')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from src.methods.helpers import deltasoc2tocharge
from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure
import numpy as np
from src.methods.loading_and_preprocessing import load_and_prepare_baseline_data
from src.constants import project_constants


def soc_to_kwh(soc_diff):
    return deltasoc2tocharge(-soc_diff)


if __name__ == "__main__":

    data_folder = os.path.join('.', 'data')
    car_is_at_home_file = os.path.join(data_folder, 'Car_is_at_home_table_UTC.csv')
    fig_out = os.path.join('plots', 'energy_demand_ev', 'energy_demand_ev.png')

    data = pd.read_csv(car_is_at_home_file, parse_dates=['start'])

    # load baseline data
    filepath_baseline = os.path.join(data_folder, 'data_baseline.csv')
    data_baseline = load_and_prepare_baseline_data(filepath_baseline)

    # calculate consumption
    delta_soc = data_baseline['soc_start'] - data_baseline['soc_end']
    delta_soc = np.maximum(np.zeros(delta_soc.shape), delta_soc.values)
    consumption_car = list(map(deltasoc2tocharge, delta_soc))
    data_baseline['energy_required_kwh'] = consumption_car

    data_baseline['day'] = data_baseline['start'].dt.floor('d')
    daily_user_consumption = data_baseline.groupby(by=['vin', 'day'])['energy_required_kwh'].sum()
    daily_user_consumption = daily_user_consumption.reset_index()
    daily_user_consumption['dow'] = daily_user_consumption['day'].dt.weekday
    data_to_plot = daily_user_consumption.copy()

    data_to_plot = pd.DataFrame(data_to_plot, columns=['day', 'energy_required_kwh', 'dow'])

    # mask missing data
    dates_to_exclude = [datetime.datetime(2017, 9, 30, 0, 0) + datetime.timedelta(days=i) for i in range(7)]
    data_exclude_ix = data_to_plot['day'].isin(dates_to_exclude)
    data_to_plot.loc[data_exclude_ix, 'energy_required_kwh'] = np.nan

    journal = Journal.POWERPOINT_A3
    fig, axes = init_figure(nrows=1, ncols=2,
                            columnes=Columnes.ONE, fig_height_mm=120,
                            journal=journal, sharex=False, sharey=False,
                            gridspec_kw={'width_ratios': [2, 1]})

    plt.sca(axes[0])
    sns.lineplot(data=data_to_plot, x="day", y="energy_required_kwh",
                 linewidth=2, ax=axes[0], ci=95, err_kws={'alpha': 0.4})
    axes[0].set_xlabel('Days in 2017 [d]')
    axes[0].set_ylabel('Energy Demand of EV per day [kWh]')
    axes[0].set_ylim([-2, 30])

    t_start = project_constants['t_start']
    t_end = project_constants['t_end']
    x_dates = pd.date_range(start=datetime.datetime(2017, 1, 1, 0, 0),
                            end=datetime.datetime(2017, 12, 31, 0, 0),
                            freq='M', ) + datetime.timedelta(days=1)

    axes[0].set_xticks(x_dates[::2])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.sca(axes[1])
    sns.boxplot(x="dow", y="energy_required_kwh", data=data_to_plot, ax=axes[1],
                flierprops={'alpha': .1, 'marker': '+'},
                color='b'
                )
    axes[1].set_xlabel('Weekdays')
    axes[1].set_ylabel('')
    # axes[1].set_ylabel('Energy Demand of EV [kWh]')
    axes[1].set_ylim([-2, 30])
    axes[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
    axes[1].set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'])

    for patch in axes[1].artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .4))

    plt.subplots_adjust(wspace=0.18)

    save_figure(fig_out)

    print(data_to_plot.groupby('dow').median())
