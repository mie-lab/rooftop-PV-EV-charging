import os

import matplotlib.dates as mdates
import pandas as pd

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

def get_aggr_df(df):
    df = df.groupby(['vin', pd.Grouper(key='start',
                                       freq='W-MON')])

    df = df[['needed_by_car', 'charged_from_pv', 'charged_from_outside']].sum()
    df = df.reset_index().sort_values('start')
    df['coverage'] = df['charged_from_pv'] / df['needed_by_car'] * 100

    end_of_project = project_constants['t_end']
    df = df[df['start'] < end_of_project]

    return df


def tsplot(x, y, n=20, percentile_min=25, percentile_max=75, color='r', plot_mean=False,
           plot_lines=False, plot_median=True, plot_scatter=False, ax=None,
           line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    "https://github.com/arviz-devs/arviz/issues/2"

    if ax is None:
        fig, ax = plt.subplots()

    taus_left = np.linspace(percentile_min, 50, num=n, endpoint=False)
    taus_right = np.linspace(50, percentile_max, num=n + 1)

    perc1 = np.nanpercentile(y, taus_left, axis=0)
    perc2 = np.nanpercentile(y, taus_right[1:], axis=0)

    if plot_scatter:
        for i in range(y.shape[0]):
            scatter_handle = ax.scatter(x, y[i, :], edgecolors='dimgrey',
                                        alpha=0.3, facecolors='none')
        scatter_handle.set_label('data points')

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1 / n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None,
                        label=f'{taus_left[0]:.0f}' + r'${}^{th}$ to ' + f'{taus_right[-1]:.0f}' + r'${}^{th}$ percentile')

    if plot_mean:
        ax.plot(x, np.nanmean(y, axis=0), color=line_color, label='Mean')

    if plot_median:
        ax.plot(x, np.nanmedian(y, axis=0), color=line_color, label='Median')

    return ax


if __name__ == '__main__':
    output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')

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

    data_list = [df_b, df_s1, df_s2, df_s3]
    journal = Journal.POWERPOINT_A3

    fig, axs = init_figure(nrows=2,
                           ncols=2,
                           columnes=Columnes.ONE,
                           journal=journal, sharex=True, sharey=True)
    # fig, axs = plt.subplots(2,2)

    plt_scatter = False

    title_list = ['Baseline', 'Scenario 1', 'Scenario 2', 'Scenario 3']
    for ix, data in enumerate(data_list):
        ax = axs[ix // 2, ix % 2]
        t, y = reshape_for_quantile_plot(data)
        ax = tsplot(t, y, n=1, percentile_min=5, percentile_max=95, plot_median=True, ax=ax, plot_scatter=plt_scatter,
                    color='g', line_color='navy', alpha=0.3)
        ax = tsplot(t, y, n=1, percentile_min=25, percentile_max=75, ax=ax, plot_median=False, color='g',
                    line_color='navy', alpha=0.5)

        ax.set_xticks(ax.get_xticks()[::2])  # skip every second tick
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_title(title_list[ix])
        ax.grid(b=True, which='both')

    # set minor ticks for grid lines
    ax.set_yticks([0, 25, 50, 75, 100], minor=True)

    # labels 
    # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time of tracking aggregated by week", labelpad=15)
    plt.ylabel("Percentage of energy demand covered by PV", labelpad=20)
    plt.grid(b=False)
    # plt.ylabel('Energy demand covered by PV')
    # legend    
    handles, labels = ax.get_legend_handles_labels()

    if plt_scatter:
        leg = plt.figlegend(handles=handles, labels=labels, loc='upper left',
                            ncol=2, frameon=False,
                            bbox_to_anchor=(0.15, 0), labelspacing=0.1,
                            columnspacing=0.1)
        leg.legendHandles[1].set_alpha(1)  # no alpha for scatter circle
        leg.legendHandles[1].set_sizes([200])
    else:

        leg = plt.figlegend(handles=handles, labels=labels, loc='upper left',
                            ncol=3, frameon=False,
                            bbox_to_anchor=(0.03, 0), labelspacing=0.1,
                            columnspacing=1)

    bbox_extra_artists = [fig, leg]

    # save
    save_figure(os.path.join('plots', 'quantile_plots', 'plot_quantile_coverage_year.png'),
                bbox_extra_artists=bbox_extra_artists)
    plt.close(fig)
