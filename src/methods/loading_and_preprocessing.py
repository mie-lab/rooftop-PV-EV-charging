import copy
import pandas as pd
import os
import datetime
from src.methods.helpers import soc2remainingCharge, deltasoc2tocharge
import numpy as np
from src.constants import project_constants

def get_id_matching_dict(filepath_matching):
    """Create a dict that has myway_ids as keys and
    vins as values"""
    matching = pd.read_csv(filepath_matching, sep=';')
    myway_id = matching['user_id']
    vin = matching['vin']
    return dict(zip(myway_id, vin))

def get_inverse_id_matching_dict(filepath_matching):
    """Create a dict that has myway_ids as keys and
    vins as values"""
    matching = pd.read_csv(filepath_matching, sep=';')
    myway_id = matching['user_id']
    vin = matching['vin']
    return dict(zip(vin, myway_id))

def filter_temporal_extent(data):

    t_start = project_constants['t_start']
    t_end = project_constants['t_end']
    # print(t_end)
    filter_boolean = ((data['start'] >= t_start) & (data['end'] < t_end))

    return data.loc[filter_boolean].copy()

def filter_good_users(data_raw, attr='vin', path_to_data_folder=os.path.join('.', 'data'),
                      include_multifamiliy_homes=True):
    """
    This function filters good users.
    We exclude:
    - users with zero demand
    - users in a single home with correct roof-matching (manual_validation.csv)
    - users that were only available in the baseline calculation (Address matching problem)

    This function replaces validate_data
    """
    data = data_raw.copy()

    # select a.vin,  count(b.is_home) from ev_homepv.ecar_data as a, ev_homepv.ecarid_is_athome as b
    # where a.id = b.bmw_id  and b.is_home group by a.vin order by count
    vins_with_wrong_home = ['0007f9c8534b7924352bed2b9842b1fc',
                            'e4aac45c1a674b721f2e3edf2b385fca',
                            # 00e5e36b1cd864655a12882d763e9a64, # Home is ok but the user has low usage
                            # 0eb363a4a1b5dce2751bc198d9188a82, # not in the data anyways
                            # 'e0d8c66f3d54112243c037b06c2bac26', # not in the data anyways
                            '003d3821dfaabc96fa1710c2128aeb62']

    # 0656f4967b02fa467e7a85ea85b62cd1 # this one is also likely to have the wrong home

    vins_with_too_big_houses = [
                                '008941e8226d2fa31c16443b852a3e89',
                                '00d98da128fd56261384c5a43da99ecf',
                                '0f0e8871819e36a735fd115b6660ba72']

    vins_only_in_baseline = ['08ef62af248b9208348ad293eab4f1a7']

    other_bad_vins = vins_only_in_baseline + vins_with_too_big_houses + vins_with_wrong_home

    filepath = os.path.join(path_to_data_folder,
                            "manual_validation.csv")
    validation = pd.read_csv(filepath, sep=';', encoding='latin-1')



    matching_dict = get_id_matching_dict(os.path.join(path_to_data_folder,
                                                      "vin_id_matching.csv"))

    # only keep ids of users that live in a single home (validation['EFH'] == 1)
    # and were we were able to correctly recognize the roof area (validation['Brauchbar'] == 1)
    if include_multifamiliy_homes:
        good_userids_ix = (((validation['Brauchbar'] == 1) & (validation['EFH'] == 1)) |
                          ((validation['Brauchbar'] == 1) & (~validation['reduction_factor'].isna() )))
    else:
        good_userids_ix = (validation['Brauchbar'] == 1) & (validation['EFH'] == 1)
    good_userids = validation.loc[good_userids_ix, 'ID']

    # transform user_id to vin
    good_vins = good_userids.map(matching_dict)
    good_vins = good_vins.dropna()
    good_vins = good_vins[~good_vins.isin(other_bad_vins)]
    good_row_ix = data[attr].isin(good_vins)

    return data[good_row_ix]


def load_and_prepare_scenario_raw_data(filepath, path_to_data_folder=os.path.join('.', 'data'),
                                       drop_debug_columns=True):
    """return baseline_data in (almost) the same format as the results from the scenarios

        do the following preprocessing steps for the baseline data
            - load it
            - filter users based on manual validation table
            - transform timestamps to datetime objects
            - filter time to temporal extent used in paper
            - [optional] drops annoying columns used to validate the data generation
    """
    data = pd.read_csv(filepath)
    data = filter_good_users(data, 'vin', path_to_data_folder)

    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])
    data = filter_temporal_extent(data)

    if drop_debug_columns:
        debug_columns = ['delta_soc', 'bmw_id', 'zustand', 'start_end']
        data = data.drop(debug_columns, axis=1)


    return data

def load_and_prepare_baseline_data(filepath_baseline, path_to_data_folder=os.path.join('.', 'data')):
    """return baseline_data in (almost) the same format as the results from the scenarios

        do the following preprocessing steps for the baseline data
            - load it
            - filter users based on manual validation table
            - transform timestamps to datetime objects
            - rename columns
            - filter time to temporal extent used in paper

    """

    data_baseline = pd.read_csv(filepath_baseline)
    data_baseline = filter_good_users(data_baseline, 'vin', path_to_data_folder)

    data_baseline['timestamp_start_utc'] = pd.to_datetime(data_baseline['timestamp_start_utc'])
    data_baseline['timestamp_end_utc'] = pd.to_datetime(data_baseline['timestamp_end_utc'])
    data_baseline = data_baseline.rename(columns={'timestamp_start_utc': 'start', 'timestamp_end_utc': 'end',
                                                  'soc_customer_start': 'soc_start',
                                                  'soc_customer_end': 'soc_end'})

    data_baseline = filter_temporal_extent(data_baseline)

    return data_baseline


def compute_additional_columns(car_data, drop_debug_columns=False):
    """
    adds the column:
    1) electricity generated by PV in kWH
    2) electricity needed by car in kWH

    Parameters
    ----------
    car_data: pandas df
        data_PV_Solar frame containing data_PV_Solar about car movements

    Returns
    -------
    car_data_with_extra_column: pandas df
        car data_PV_Solar with columns added
    """

    car_data_copy = copy.deepcopy(car_data)

    try:
        needed_by_car = car_data_copy['total_segment_charging'].apply(deltasoc2tocharge)

    except KeyError:
        needed_by_car = [soc2remainingCharge(car_data_copy["soc_start"][car_data_copy.index[i]]) -
                         soc2remainingCharge(car_data_copy["soc_end"][car_data_copy.index[i]])
                         for i in range(len(car_data_copy.index))]


    needed_by_car = np.maximum(0, needed_by_car)
    assert (np.all(np.array(needed_by_car) >= 0))
    car_data_copy['needed_by_car'] = needed_by_car

    charged_from_pv = [np.minimum(car_data_copy['generated_by_pv'][car_data_copy.index[i]],
                                  car_data_copy['needed_by_car'][car_data_copy.index[i]])
                       for i in range(len(car_data_copy.index))]

    # exclude negative charged from PV to prevent artificial results
    # charged_from_pv = np.maximum(0, charged_from_pv)

    assert (np.all(np.array(charged_from_pv) >= 0))

    car_data_copy['charged_from_pv'] = charged_from_pv
    car_data_copy.loc[~car_data_copy.is_home, 'charged_from_pv'] = 0

    return car_data_copy