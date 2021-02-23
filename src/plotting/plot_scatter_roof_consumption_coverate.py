import logging
import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from src.methods.PV_interface import get_area_factor_for_user
from src.methods.helpers import get_user_id
from src.methods.pv_swissbuildings_json import PVModel
from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure


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

output_folder = os.path.join('.', 'data', 'output')

def add_roof_area(df):
    all_vins = df.index.tolist()
    for vin in all_vins:
        user_id = get_user_id(vin)
        area_factor = get_area_factor_for_user(user_id)

        warnings.warn("using fixed pv model")
        pv = PVModel(user_id, area_factor=area_factor, scenario='PVMODEL_SPV170')
        df.loc[vin, 'area'] = pv.area
        df.loc[vin, 'user'] = int(user_id)
        df.loc[vin, 'max_kW'] = pv.max_W / 1000

        pvdict = pv.data['PVMODEL_SPV170']
        to_pop_list = ['year_Wh_dc', 'year_Wh_ac', 'year_inv_eff', 'max_W']
        for to_pop in to_pop_list:
            pvdict.pop(to_pop)
        df.loc[vin, 'max_gen_kw'] = max(list(pvdict.values())) * 2 / 1000 * area_factor


def return_ols_values(x, y, scenario):
    xx = sm.add_constant(x, prepend=False)
    mod = sm.OLS(y, xx)
    res = mod.fit()
    pval = res.pvalues
    params = res.params
    rsquared = res.rsquared

    logging.debug('\t\t' + f'pval demand: {pval[0]:.2E}')
    logging.debug('\t\t' + f'pval area: {pval[1]:.2E}')

    logging.debug('\t\t' + f'coef demand: {params[0]:.2E} [%/kWh]')
    logging.debug('\t\t' + f'coef area: {params[1]:.2E}  [%/m2]')

    logging.debug('\t\t' + f'rsquared: {rsquared:.2f}')

    print(res.summary())

    return {'scenario': scenario, 'Coef_D': params[0], 'p_D': pval[0], 'Coef_A': params[1],
            'p_A': pval[1], 'R2': rsquared}


def no_formatter(x):
    return x


def scientific_formatter(x):
    return '%1.2E' % x


def float_formatter(x):
    return '%1.2F' % x


if __name__ == '__main__':
    output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')
    df = pd.read_csv(os.path.join(output_folder, 'coverage_by_scenario.csv'))

    df.set_index('vin', inplace=True)
    df['area'] = 0
    df['user'] = 0
    add_roof_area(df)

    column_names = ['baseline', 'scenario1', 'scenario2', 'scenario3']
    demand_names = ['total_demand', 'total_demand', 'total_demand_s2', 'total_demand_s3']
    df[column_names] = df[column_names] * 100  # in %

    plt.figure()
    plt.hist(df['max_kW'], bins=20)
    plt.xlabel('kwp [kW]')
    plt.ylabel('user count')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'kwp_hist.pdf'))
    plt.close()

    # regression analysis
    results_dict_list = []
    column_labels = ['Baseline', 'Scenario 1', 'Scenario 2', 'Scenario 3']
    for ix, column_name in enumerate(column_names):
        logging.debug("\t" + column_name)
        demand_col = demand_names[ix]
        x = df.loc[:, [demand_col, 'max_kW']].values.reshape(-1, 2)
        y = df[column_name].values.reshape(-1, 1)
        results_dict_list.append(return_ols_values(x, y, scenario=column_labels[ix]))

    results_df = pd.DataFrame(results_dict_list)
    results_df.set_index('scenario', inplace=True)
    formatter_list = 4 * [scientific_formatter] + [float_formatter]

    results_df.to_latex('table.tex', bold_rows=True, formatters=formatter_list)

    journal = Journal.POWERPOINT_A3
    fig, axs = init_figure(nrows=1,
                           ncols=2,
                           columnes=Columnes.ONE,
                           journal=journal,
                           sharey=True)

    ax0 = axs[0]
    ax1 = axs[1]
    size = 50
    ax0.scatter(df['total_demand'], df['baseline'], s=size, label='Baseline')
    ax0.scatter(df['total_demand'], df['scenario1'], s=size, label='Scenario 1')
    ax0.scatter(df['total_demand_s2'], df['scenario2'], s=size, label='Scenario 2')
    ax0.scatter(df['total_demand_s3'], df['scenario3'], s=size, label='Scenario 3')
    ax0.set_xlabel("Total demand [kWh]")
    ax0.set_ylabel("Coverage by PV [\%]")

    ax1.scatter(df['max_kW'], df['baseline'], s=size)
    ax1.scatter(df['max_kW'], df['scenario1'], s=size)
    ax1.scatter(df['max_kW'], df['scenario2'], s=size)
    ax1.scatter(df['max_kW'], df['scenario3'], s=size)
    ax1.set_xlabel("Peak power [kW]")
    ax1.set_xlim(0, 30)
    # ax1.ylabel("coverage by PV [\%]")

    leg = ax0.legend(loc='upper left', frameon=False, bbox_to_anchor=(-0.35, -0.15),
                     ncol=4, markerscale=3)

    bbox_extra_artists = [fig, leg]
    fig_out = os.path.join('plots', 'sensitivity_kwp_bevdemand', 'sensitivity_kwp_bevdemand.png')
    save_figure(fig_out, bbox_extra_artists=bbox_extra_artists, dpi=10)
