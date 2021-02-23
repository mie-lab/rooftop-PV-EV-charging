import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure


def plot_cumsum_plot(baseline, scenario1, scenario2, scenario3):
    journal = Journal.POWERPOINT_A3

    fig, ax = init_figure(nrows=1,
                          ncols=1,
                          columnes=Columnes.ONE,
                          journal=journal)

    lbase = plot_cumsum_by_scenario(baseline[baseline.is_home], ax, label='Baseline')
    l1 = plot_cumsum_by_scenario(scenario1[scenario1.is_home], ax, label='Scenario 1')
    l2 = plot_cumsum_by_scenario(scenario2, ax, label='Scenario 2')
    l3 = plot_cumsum_by_scenario(scenario3, ax, label='Scenario 3')

    xlim = plt.xlim()
    lxy = plt.plot(np.arange(0, xlim[1]), np.arange(0, xlim[1]), label='x=y (only using pv generation)', color='k',
                   linestyle=':', linewidth=2)
    lyx = plt.plot(np.arange(0, xlim[1]), - np.arange(0, xlim[1]), label='x=-y (only using grid generation)', color='k',
                   linestyle=':', linewidth=2)

    lyx = plt.plot(np.arange(0, xlim[1]), 0 * np.arange(0, xlim[1]), label=None, color='k',
                   linestyle='--', linewidth=2)
    xlim = plt.xlim()

    ylim = plt.ylim()
    ylim = (ylim[0], xlim[1])
    print(ylim)
    ax.set_ylim(ylim)
    ax.set_xlim((0, xlim[1]))
    plt.xlabel('Total energy used by cars (cumulative) [MWh]', labelpad=15)
    plt.ylabel('PV charging - grid charging \n (cumulative) [MWh]', labelpad=15)
    leg = plt.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.15, -0.2), borderpad=0., frameon=False)
    plt.tight_layout()
    bbox_extra_artists = [fig, leg, ax]
    save_figure(os.path.join('plots', 'cumulative_energy_plot', 'cumsum_all_scenarios.png'),
                bbox_extra_artists=bbox_extra_artists)


def plot_cumsum_by_scenario(data_raw, ax, label=None):
    data = data_raw.copy()

    cumsum_saldo = (data['charged_from_pv'] - data['charged_from_outside']).cumsum() / 1000
    cumsum_total_consumption = data['needed_by_car'].cumsum() / 1000

    handle = ax.plot(cumsum_total_consumption, cumsum_saldo, label=label, linewidth=2)
    return handle[0]


def parse_dates(data_raw):
    data = data_raw.copy()
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])

    return data


if __name__ == '__main__':
    output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')

    baseline = pd.read_csv(os.path.join(output_folder, 'results_baseline.csv'))
    baseline = parse_dates(baseline).sort_values('start')

    scenario1 = pd.read_csv(os.path.join(output_folder, 'results_scenario1.csv'))
    scenario1 = parse_dates(scenario1).sort_values('start')

    scenario2 = pd.read_csv(os.path.join(output_folder, 'results_scenario2.csv'))
    scenario2 = parse_dates(scenario2).sort_values('start')

    scenario3 = pd.read_csv(os.path.join(output_folder, 'results_scenario3.csv'))
    scenario3 = parse_dates(scenario3).sort_values('start')

    plot_cumsum_plot(baseline, scenario1, scenario2, scenario3)
