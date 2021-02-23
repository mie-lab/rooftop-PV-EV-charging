import datetime
import os

import numpy as np
import pint

from src.methods.helpers import get_user_id
from src.methods.pv_swissbuildings_json import PVModel
import pandas as pd
ureg = pint.UnitRegistry()

# TODO
pv_efficency = 0.18
# pv_efficency = 180

import logging

pv_cache = {}


def get_PV_generated_from_pandas_row(ix_row, pv_model, max_power_kw=11):
    """wrapper function for get_PV_generated that can be used with the
    df.iterrows() generator"""

    # pv_model="PVMODEL_SPV170"
    # pv_model is set without default to avoid errors
    ix, row = ix_row
    try:
        return get_PV_generated(start=row['start'],
                                end=row['end'],
                                house_ID=row['vin'],
                                pv_model=pv_model,
                                max_power_kw=max_power_kw)
    except (ValueError, FileNotFoundError):
        return -1

def get_area_factor_for_user(user_id, path_to_data_folder=os.path.join('.', 'data')):
    filepath = os.path.join(path_to_data_folder,
                            "manual_validation.csv")
    validation = pd.read_csv(filepath, sep=';', encoding='latin-1')
    validation['reduction_factor'] = validation['reduction_factor'].fillna(1)
    validation.set_index('ID', inplace=True)
    area_factor = 1 / validation.loc[int(user_id), 'reduction_factor']
    return area_factor


def get_PV_generated(start, end, house_ID, pv_model, path_to_data_folder=os.path.join('.', 'data'), max_power_kw=None):
    # todo: efficiency. If we still have to take into account the efficiency it has to be done in the method
    #  PVModel.get_solar_radiation() because otherwise the max_power_kw restriction would be too strong
    """
    start
    end
    house_ID: vin
    """

    # pv_model="PVMODEL_SPV170"
    # pv_model is set without default to avoid errors

    logging.debug("pv_model: ".format(pv_model))


    assert start.year >= 2017, "solar model is defined starting from Jan 2017"
    assert end.year >= 2017
    if house_ID not in pv_cache:
        user_id = get_user_id(house_ID, path_to_data_folder)

        # get area factor
        area_factor = get_area_factor_for_user(user_id)

        if user_id is None:
            # this means we don't have a  house for this user
            raise ValueError("No vin - myway user_id matching available for {}".format(user_id))

        pv = PVModel(str(user_id), scenario=pv_model, path_to_data_folder=path_to_data_folder,
                     area_factor=area_factor)
        pv_cache[house_ID] = pv

    pv = pv_cache[house_ID]
    start_datetime = datetime.datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S")

    generated_energy = pv.get_solar_radiation(scenario=pv_model,
                                              startts=start_datetime,
                                              endts=end_datetime,
                                              max_power_kw=max_power_kw)
    generated_KWh = generated_energy / 1000
    return generated_KWh


def get_max_pv_charged(start, end, max_charging_power):
    start_datetime = datetime.datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S")
    max_charged = (end_datetime.timestamp() - start_datetime.timestamp()) * max_charging_power / (60 * 60)
    # print(start)
    # print(end)
    # print(max_charged)
    return np.float(max_charged)
