import datetime
import os

import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from src.constants import project_constants
from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure


def parse_dates(data_raw):
    data = data_raw.copy()
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])

    return data


def reshape_for_quantile_plot(df):
    df = df.pivot(index='vin', values='coverage', columns='start')
    return df.columns, df.values  # (x, y)


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # this was just used for the examples

# data
CO2_pv = 53.6 / 1000  # kg/kWh
# for switzerland: Verbraucher-Strommix: 
# https://www.bafu.admin.ch/bafu/de/home/themen/klima/klimawandel--fragen-und-antworten.html#-1202736887
# CO2_swissmix = 181.1/1000  # kg/kWh https://www.bafu.admin.ch/bafu/de/home/themen/klima/klimawandel--fragen-und-antworten.html#-1202736887
# CO2_swissmix = 401/1000
# 181.1 = schweiz
# 401 = Germany
# 5000 charge cycles / 1 cycle per day = 13.6986301369863 years of lifetime
# https://iea-pvps.org/key-topics/environmental-life-cycle-assessment-of-residential-pv-and-battery-storage-systems/
# Emissions are 76.1 gco2/kwh
# Prospective LCA of the production and EoL recycling of a novel type of Li-ion battery for electric vehicles

bat_co2_kwh = 76.1
bat_cap_kwh = 13.5
bat_lifetime_weeks = 5000 / 365 * 52
gCO2_storage_week = bat_cap_kwh * bat_co2_kwh / bat_lifetime_weeks


def get_aggr_df(df):
    df = df.groupby(['vin', pd.Grouper(key='start',
                                       freq='W-MON')])

    df = df[['needed_by_car', 'charged_from_pv', 'charged_from_outside']].sum()
    df = df.reset_index().sort_values('start')

    df.fillna(0, inplace=True)

    df['co2'] = df['charged_from_pv'] * CO2_pv + \
                df['charged_from_outside'] * CO2_swissmix

    end_of_project = project_constants['t_end']
    df = df[df['start'] < end_of_project]

    return df


if __name__ == '__main__':
    output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')

    plot_names = ['switzerland', 'germany']
    for ix, CO2_swissmix in enumerate([181.1 / 1000, 401 / 1000]):
        plot_name = plot_names[ix]
        fig_out = os.path.join('plots', 'co2_plot', 'plot_average_co2_over_year_' + plot_name + '.png')

        baseline = pd.read_csv(os.path.join(output_folder, 'results_baseline.csv'))
        baseline = parse_dates(baseline)
        df_b = get_aggr_df(baseline)

        scenario1 = pd.read_csv(os.path.join(output_folder, 'results_scenario1.csv'))
        scenario1 = parse_dates(scenario1)
        df_s1 = get_aggr_df(scenario1)

        scenario2 = pd.read_csv(os.path.join(output_folder, 'results_scenario2.csv'))
        scenario2 = parse_dates(scenario2)
        df_s2 = get_aggr_df(scenario2)

        scenario3 = pd.read_csv(os.path.join(output_folder, 'results_scenario3.csv'))
        scenario3 = parse_dates(scenario3)
        df_s3 = get_aggr_df(scenario3)

        df_s3['co2'] = df_s3['co2'] + gCO2_storage_week

        df_b['scenario'] = 'Baseline'
        df_s1['scenario'] = 'Scenario 1'
        df_s2['scenario'] = 'Scenario 2'
        df_s3['scenario'] = 'Scenario 3'

        df_all = pd.concat((df_b, df_s1, df_s2, df_s3))

        # mask missing data
        dates_to_exclude = [datetime.datetime(2017, 10, 2, 0, 0) + datetime.timedelta(days=i) for i in range(14)]
        data_exclude_ix = df_all['start'].isin(dates_to_exclude)
        df_all.loc[data_exclude_ix, 'co2'] = np.nan

        journal = Journal.POWERPOINT_A3

        fig, ax = init_figure(nrows=1,
                              ncols=1,
                              columnes=Columnes.ONE,
                              journal=journal, sharex=True, sharey=True)

        sns.lineplot(data=df_all, x="start", y="co2", hue="scenario",
                     linewidth=3, ax=ax)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        handles, labels = ax.get_legend_handles_labels()

        for handle in handles:
            handle.set_linewidth(3)

        leg = ax.legend(handles=handles, labels=labels,
                        loc='upper left', frameon=False, bbox_to_anchor=(-0.14, -0.15),
                        ncol=4)

        plt.xlabel("Time of tracking aggregated by week", labelpad=15)
        plt.ylabel(r"Emissions per Person in $\frac{\text{kg CO2 equivalent}}{\text{week}}$", labelpad=20)

        bbox_extra_artists = [fig, leg]
        save_figure(fig_out, bbox_extra_artists=bbox_extra_artists)
        plt.close(fig)

        # calculate average Co2 savings per user
        df_avco2 = df_all.groupby('scenario')['co2'].mean()
        df_avco2 = df_avco2.Baseline - df_avco2

        print(plot_name)
        print("Average savings per scenario relative to the baseline:")
        print(df_avco2)
        print(10 * "%")
