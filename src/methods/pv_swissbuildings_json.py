'''
Created on Dec 12, 2019

@author: rene
'''
import datetime
import gzip
import json
import os
import shutil

import pint

ureg = pint.UnitRegistry()


def unpack_file(filename):
    assert filename[-3:] == '.gz'

    with gzip.open(filename, 'rb') as f_in:
        with open(filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


class PVModel:

    def __init__(self, user_id, scenario, path_to_data_folder=os.path.join('.', 'data'), area_factor=1):

        """Area factor is factor that reduced the power because the roof was recognized as too big.
        E.g., if the person is living in a house for two families, the area factor should be 0.5 so that only half the
        power is returned in the end.
        """

        # with gzip.GzipFile(os.path.join("data_PV_Solar", "solar_rad_{}.json.gz".format(user_id)), 'r') as f:
        #    self.data_PV_Solar = json.loads(f.read()))
        filepath = os.path.join(path_to_data_folder,
                                "data_PV_Solar")
        filepath = os.path.join(filepath,
                                "solar_rad_{}.json".format(user_id))

        if not os.path.isfile(filepath):
            unpack_file(filepath + '.gz')

        self.area_factor = area_factor

        with open(filepath, 'r') as f:
            self.data = json.loads(f.read())
            self.user_id = user_id
            self.area = self.data['area'] * area_factor
            self.max_W = self.data[scenario]['max_W'] * area_factor




    @staticmethod
    def _get_band(dt):
        """
            Returns band from timestamp
        """

        delta = dt - datetime.datetime(2017, 1, 1, 0, 0)
        half_hours = int(delta.total_seconds() / 60.0 / 30.0)

        return half_hours + 1

    @staticmethod
    def _get_datetime(b):
        """
            Returns timestamp from band
        """
        minutes = b * 30.0
        return datetime.datetime(2017, 1, 1, 0, 0) + datetime.timedelta(minutes=minutes)

    def get_solar_radiation(self, scenario, startts, endts, max_power_kw=None):
        """

        max_power: Is the maximum allowed charging power in kw
                 (will be controlled per 30 minutes average)
            Calculates total rooftop solar irradiation between startts and endts timestamp in Wh

            scenarios:
                PVMODEL_SPV170
                PVMODEL_JA
                PVMODEL_JINKO
                ONLY_SOLAR_PVSYSEFF_5
                ONLY_SOLAR_PVSYSEFF_6
                ONLY_SOLAR_PVSYSEFF_7
                ONLY_SOLAR_PVSYSEFF_8
                ONLY_SOLAR_PVSYSEFF_9
                ONLY_SOLAR_PVSYSEFF_10
                ONLY_SOLAR_PVSYSEFF_11
                ONLY_SOLAR_PVSYSEFF_12
                ONLY_SOLAR_PVSYSEFF_13
                ONLY_SOLAR_PVSYSEFF_14
                ONLY_SOLAR_PVSYSEFF_15
                ONLY_SOLAR_PVSYSEFF_16
                ONLY_SOLAR_PVSYSEFF_17
                ONLY_SOLAR_PVSYSEFF_18
                ONLY_SOLAR_PVSYSEFF_19
                ONLY_SOLAR_PVSYSEFF_20
                ONLY_SOLAR_PVSYSEFF_21
                ONLY_SOLAR_PVSYSEFF_22
                ONLY_SOLAR_PVSYSEFF_23
                ONLY_SOLAR_PVSYSEFF_24
                ONLY_SOLAR_PVSYSEFF_25
                ONLY_SOLAR_PVSYSEFF_26
                ONLY_SOLAR_PVSYSEFF_27
                ONLY_SOLAR_PVSYSEFF_28
                ONLY_SOLAR_PVSYSEFF_29
                ONLY_SOLAR_PVSYSEFF_100

        """

        assert endts >= startts

        start_band = self._get_band(startts)
        end_band = self._get_band(endts)

        seconds_in_half_hour = 60.0 * 30.0

        if start_band == end_band:
            percentage_in_start_band = (endts - startts).total_seconds() / seconds_in_half_hour
            percentage_in_end_band = 0.0
        else:
            percentage_in_start_band = (self._get_datetime(start_band) - startts).total_seconds() / seconds_in_half_hour
            percentage_in_end_band = 1.0 - (self._get_datetime(end_band) - endts).total_seconds() / seconds_in_half_hour

        assert 0 <= percentage_in_start_band <= 1.0
        assert 0 <= percentage_in_end_band <= 1.0

        tot_Wh = 0.0
        last_iter = len(range(start_band, end_band + 1))
        for i, b in enumerate(range(start_band, end_band + 1)):

            band_key = "band_{}_Wh".format(b)
            if band_key not in self.data[scenario]:
                _tot_Wh = 0.0
            else:
                _tot_Wh = float(self.data[scenario][band_key]) * self.area_factor

            # take care of partially covered start and end band
            if i == 0:
                _tot_Wh = _tot_Wh * percentage_in_start_band
            if i == last_iter:
                _tot_Wh = _tot_Wh * percentage_in_end_band


            # include max limit here
            if max_power_kw is not None:
                # control for maximum charging power. We can charge maximum of_
                # max_power_kw * 1000 / 2 [WH] per half hour.
                # if _tot_Wh > max_power_kw * 1000 / 2:
                #     print('cut of pv for user ', self.user_id, _tot_Wh, max_power_kw * 1000 / 2)

                tot_Wh += min(_tot_Wh, max_power_kw * 1000 / 2)


            else:
                tot_Wh += _tot_Wh

        return tot_Wh
        # return tot_Wh * ureg.watthour

# pv = PVModel("1761")

# print(pv.get_solar_radiation("PVMODEL_SPV170",
#                             datetime.datetime(2017, 1, 1, 0, 0),
#                             datetime.datetime(2017, 12, 31, 23, 59)))
