import numpy as np
import pint
import pandas as pd
import os
import datetime

ureg = pint.UnitRegistry()

def soc2remainingCharge(soc):
    "calculates amount of energy that is missing in the car battery"
    # charge = 0.293 * (100.0 - soc) + 0.232 * np.sqrt(100.0 - soc)  # formula to calc charge from soc
    charge = deltasoc2tocharge(100-soc)
    assert charge >= 0.0
    # return charge * ureg.kilowatthour
    return charge

def deltasoc2tocharge(delta_soc):
    "returns energy for a given difference in state of charge"

    charge = 0.293 * (delta_soc) + 0.232 * np.sqrt(delta_soc)  # formula to calc charge from soc
    assert charge >= 0.0
    # return charge * ureg.kilowatthour
    return charge

def remainingCharge2soc(charge):
    # wolfram alpha magic
    # https://www.wolframalpha.com/input/?i=invert+y+%3D+0.293+*+%28100-x%29+%2B+0.232+*+sqrt%28100-x%29
    # old: soc = -3.42674 * (-29.0785 + charge) + 1.6478263810833557*10**-44 * np.sqrt(4.654229268178746*10**86 + 8.972720402977183*10**87 * charge)
    soc = -3.41297 * (-29.2082 + charge) + 5.82418 * (10 ** -6) * np.sqrt(
        6.30817 * (10 ** 10) * charge + 2.89702 * (10 ** 9))

    assert 0 <= soc <= 100.0
    return soc


def get_user_id(vin, path_to_data_folder=os.path.join('.', 'data')):
    """Transforms vin to myway user id"""
    # print(f"vid ID:{vid}")
    filepath_to_table = os.path.join(path_to_data_folder, "vin_id_matching.csv")
    data = pd.read_csv(filepath_to_table, sep=';')
    user_id = data['user_id'].loc[data['vin'] == vin]

    if len(user_id) == 0:
        return None
    return str(int(user_id.iloc[0]))


def get_vin(user_id, path_to_data_folder=os.path.join('.', 'data')):
    """Transforms vin to myway user id"""
    # print(f"vid ID:{vid}")
    filepath_to_table = os.path.join(path_to_data_folder, "matching_bmw_to_address.csv")
    # print(os.path.abspath(filepath_to_table))
    data = pd.read_csv(filepath_to_table, sep=';')
    relevant_columns = ['BMW_vid', 'BMW_userid']
    data = data[relevant_columns]
    # print(data_PV_Solar)
    vin = data['BMW_vid'].loc[data['BMW_userid'] == user_id]

    # print(user_id.iloc[0])
    # print(vid)
    if len(vin) == 0:
        return None
    return vin.iloc[0]



