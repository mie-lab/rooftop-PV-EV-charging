import logging
import os
from functools import partial
from multiprocessing import Pool
import pandas as pd

from src.methods.PV_interface import get_PV_generated_from_pandas_row
from src.methods.loading_and_preprocessing import load_and_prepare_baseline_data, load_and_prepare_scenario_raw_data, \
    compute_additional_columns
from src.methods.scenarios_for_users import create_scenario_table
import datetime

logging.basicConfig(
    filename='calculate_scenarios.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_cached_csv(filepath, data_raw, pv_model, max_power_kw=11):
    data = data_raw.copy()
    cache_path = os.path.join(os.path.dirname(filepath), 'cache', os.path.basename(filepath) +
                              '_pvgen_maxkw=' + str(max_power_kw) +
                              '_pv_model=' + pv_model + '_cached.csv')

    if os.path.isfile(cache_path):
        logging.debug("\tload from cache")
        data['generated_by_pv'] = pd.read_csv(cache_path, index_col=0)
        # data['start'] = pd.to_datetime(data['start'])
        # data['end'] = pd.to_datetime(data['end'])

    else:
        logging.debug("\tStart pv calculation")
        data['start'] = pd.to_datetime(data['start'])
        data['end'] = pd.to_datetime(data['end'])

        get_PV_generated_from_pandas_row_partial = partial(get_PV_generated_from_pandas_row, pv_model=pv_model,
                                                           max_power_kw=max_power_kw)
        # data['generated_by_pv'] = list(map(get_PV_generated_from_pandas_row_partial, data.iterrows())) # single core for debugging
        with Pool(24) as pool:
            data['generated_by_pv'] = pool.map(get_PV_generated_from_pandas_row_partial, data.iterrows())
            pool.close()
            pool.join()
        logging.debug("\tfinished pv calculation")
        logging.debug("caching {}".format(cache_path))
        data['generated_by_pv'].to_csv(cache_path, index=True)

    assert 'generated_by_pv' in data.columns

    return data

def check_user_plausibilty(baseline_data, data):
    vins_baseline = baseline_data['vin'].unique()
    vins_data = data['vin'].unique()

    for ix, vin_b in enumerate(vins_baseline):
        assert vin_b == vins_data[ix], f"only in data: {set(vins_data) - set(vins_baseline)}," \
                                       f" only in baseline {set(vins_baseline) - set(vins_data)}"

    return True


if __name__ == '__main__':

    battery_capacity = 13.5  # tesla power box.
    battery_charging_power = 4.6 # Dauerbetrieb
    max_power_kw = 11
    # pv_model = "PVMODEL_SPV170"

    all_pv_models = ["PVMODEL_SPV170", "PVMODEL_JA", "PVMODEL_JINKO", "ONLY_SOLAR_PVSYSEFF_5",
     "ONLY_SOLAR_PVSYSEFF_6", "ONLY_SOLAR_PVSYSEFF_7", "ONLY_SOLAR_PVSYSEFF_8",
     "ONLY_SOLAR_PVSYSEFF_9", "ONLY_SOLAR_PVSYSEFF_10", "ONLY_SOLAR_PVSYSEFF_11",
     "ONLY_SOLAR_PVSYSEFF_12", "ONLY_SOLAR_PVSYSEFF_13", "ONLY_SOLAR_PVSYSEFF_14",
     "ONLY_SOLAR_PVSYSEFF_15", "ONLY_SOLAR_PVSYSEFF_16", "ONLY_SOLAR_PVSYSEFF_17",
     "ONLY_SOLAR_PVSYSEFF_18", "ONLY_SOLAR_PVSYSEFF_19", "ONLY_SOLAR_PVSYSEFF_20",
     "ONLY_SOLAR_PVSYSEFF_21", "ONLY_SOLAR_PVSYSEFF_22", "ONLY_SOLAR_PVSYSEFF_23",
     "ONLY_SOLAR_PVSYSEFF_24", "ONLY_SOLAR_PVSYSEFF_25", "ONLY_SOLAR_PVSYSEFF_26",
     "ONLY_SOLAR_PVSYSEFF_27", "ONLY_SOLAR_PVSYSEFF_28", "ONLY_SOLAR_PVSYSEFF_29",
     "ONLY_SOLAR_PVSYSEFF_100"]

    for pv_model in all_pv_models:
        print(pv_model)
        path_to_data_folder = os.path.join('.', 'data')
        output_folder = os.path.join(path_to_data_folder, 'output', pv_model)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filepath = os.path.join(path_to_data_folder, 'car_is_at_home_table_UTC.csv')
        filepath_baseline = os.path.join(path_to_data_folder, 'data_baseline.csv')

        logging.debug("read files")
        data_baseline = load_and_prepare_baseline_data(filepath_baseline)
        data = load_and_prepare_scenario_raw_data(filepath)

        # calculate pv generation (load from cache folder if possible)
        # the calculation does include the charging limitation.
        logging.debug("get pv generation for baseline data")
        data_baseline = get_cached_csv(filepath_baseline, data_baseline, pv_model=pv_model,
                                       max_power_kw=max_power_kw)
        logging.debug("get pv generation for scenario 2 data")
        data = get_cached_csv(filepath, data, pv_model=pv_model, max_power_kw=max_power_kw)
        data_unrestricted = get_cached_csv(filepath, data, pv_model=pv_model, max_power_kw=10000)
        data_joint_restriction = get_cached_csv(filepath, data, pv_model=pv_model,
                                                max_power_kw=max_power_kw + battery_charging_power)
        data_battery_restriction = get_cached_csv(filepath, data, pv_model=pv_model,
                                                  max_power_kw=battery_charging_power)

        data['charged_from_pv_unrestricted'] = data_unrestricted['generated_by_pv']
        data['generated_by_pv_joint_restriction'] = data_joint_restriction['generated_by_pv']
        data['generated_by_pv_battery_restriction'] = data_battery_restriction['generated_by_pv']

        # compute the columns
        #     1) electricity generated by PV in kWH
        #     2) electricity needed by car in kWH
        data = compute_additional_columns(data, drop_debug_columns=False)
        data_baseline = compute_additional_columns(data_baseline)

        # validate that both datasets have exactly the same users
        check_user_plausibilty(data_baseline, data)

        # delete bad week

        results = create_scenario_table(data_baseline, data, battery_capacity,
                                      battery_charging_power, max_power_kw, pv_model, path_to_data_folder)

        table = results['table']
        table.to_csv(os.path.join(output_folder, 'coverage_by_scenario.csv'))

        # # unpack & write results
        # print("write scenario results")
        baseline_results = results['baseline']
        scenario1_results = results['scenario1']
        scenario2_results = results['scenario2']
        scenario3_results = results['scenario3']

        baseline_results.to_csv(os.path.join(output_folder, 'results_baseline.csv'), index=False)
        scenario1_results.to_csv(os.path.join(output_folder, 'results_scenario1.csv'), index=False)
        scenario2_results.to_csv(os.path.join(output_folder, 'results_scenario2.csv'), index=False)
        scenario3_results.to_csv(os.path.join(output_folder, 'results_scenario3.csv'), index=False)



