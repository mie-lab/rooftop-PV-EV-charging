import datetime
import os
import sys

sys.path.append('.')
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from multiprocessing import Pool
from src.methods.PV_interface import get_area_factor_for_user
from src.methods.helpers import get_user_id
from src.plotting.myplotlib import init_figure, Columnes, Journal, save_figure
import warnings
from src.methods.pv_swissbuildings_json import PVModel
from src.methods.loading_and_preprocessing import load_and_prepare_scenario_raw_data
from src.methods.PV_interface import get_PV_generated_from_pandas_row
from src.constants import project_constants


def get_kwp(df):
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

    return df


if __name__ == "__main__":
    output_folder = os.path.join('.', 'data', 'output', 'PVMODEL_SPV170')
    df_kwp = pd.read_csv(os.path.join(output_folder, 'coverage_by_scenario.csv'))

    data_folder = os.path.join('.', 'data')
    car_is_at_home_file = os.path.join(data_folder, 'Car_is_at_home_table_UTC.csv')
    fig_out = os.path.join('plots', 'energy_generation_pv', 'energy_generation_pv.png')
    pv_model = "PVMODEL_SPV170"

    car_is_at_home_file = os.path.join(data_folder, 'Car_is_at_home_table_UTC.csv')

    data = load_and_prepare_scenario_raw_data(car_is_at_home_file)
    all_vehicles = data['vin'].unique()
    x_dates = pd.date_range(start=datetime.datetime(2017, 2, 1, 0, 0),
                            end=datetime.datetime(2017, 12, 31, 0, 0),
                            freq='D')

    vin_time_combinations = pd.MultiIndex.from_product([all_vehicles, x_dates], names=["vin", "start"])
    daily_pv_per_user = pd.DataFrame(index=vin_time_combinations).reset_index()
    daily_pv_per_user['end'] = daily_pv_per_user['start'] + datetime.timedelta(days=1)
    get_PV_generated_from_pandas_row_partial = partial(get_PV_generated_from_pandas_row, pv_model=pv_model,
                                                       max_power_kw=10000)

    # data['generated_by_pv'] = list(map(get_PV_generated_from_pandas_row_partial, data.iterrows())) # single core for debugging
    with Pool(24) as pool:
        daily_pv_per_user['generated_by_pv'] = pool.map(get_PV_generated_from_pandas_row_partial,
                                                        daily_pv_per_user.iterrows())
        pool.close()
        pool.join()

    ############### get kwp

    df_kwp.set_index('vin', inplace=True)
    df_kwp['area'] = 0
    df_kwp['user'] = 0
    df_kwp = get_kwp(df_kwp)

    journal = Journal.POWERPOINT_A3
    fig, axes = init_figure(nrows=1, ncols=2,
                            columnes=Columnes.ONE, fig_height_mm=120,
                            journal=journal, sharex=False, sharey=False,
                            gridspec_kw={'width_ratios': [2, 1]})
    plt.sca(axes[0])
    ax = axes[0]
    sns.lineplot(data=daily_pv_per_user, x="start", y="generated_by_pv",
                 linewidth=1.5, ax=ax)
    ax.set_xlabel('Days in 2017 [d]')
    ax.set_ylabel('Daily PV generation [kWh]')
    t_start = project_constants['t_start']
    t_end = project_constants['t_end']
    x_dates = pd.date_range(start=t_start,
                            end=t_end,
                            freq='M', ) + datetime.timedelta(days=1)
    axes[0].set_xticks(x_dates[::2])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.sca(axes[1])
    sns.histplot(data=df_kwp, x='max_kW', ax=axes[1], bins=np.arange(0, 40, 5))

    plt.xlabel('kWp [kW]')
    plt.ylabel('User count')
    axes[1].set_xticks([0, 10, 20, 30], minor=False)
    plt.xlim((0, 35))

    save_figure(fig_out)
