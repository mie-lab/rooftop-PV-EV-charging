import numpy as np
import pandas as pd
import os
from src.methods.helpers import soc2remainingCharge


def extract_user(data, vin):
    """
    Extracts part of the preprocessed dataframe that belongs to the selected user.

    Parameters
    ----------
    data: pandas df
        dataframe to be selected from
    user: string
        userID/vin to be selected from

    Returns
    -------
    datacopy: pandas df
        data_PV_Solar with rows selected

    """
    assert isinstance(vin, str)
    return data[data['vin'] == vin].copy()


def baseline(data_baseline_raw):
    """
    Calculate the baseline scenario
    Parameters
    ----------
    data_baseline_raw

    Returns
    -------

    """
    # print(f"\tbaseline called for user {user}")
    data_baseline = data_baseline_raw.copy()

    all_users = data_baseline['vin'].unique()
    coverage_all = []
    user_df_list = []
    for user in all_users:

        user_data = extract_user(data_baseline, user)
        user_data = user_data.sort_values(by=['start'])

        # user_data.to_csv("debug_user_data_baseline.csv")
        user_data_ishome = user_data[user_data['is_home']]

        total_charged = sum(user_data_ishome['charged_from_pv'])
        total_demand = sum(user_data_ishome['needed_by_car'])

        if total_demand != 0:
            coverage = total_charged / total_demand
        else:
            coverage = 1
        assert 0 <= coverage <= 1
        coverage_all.append((user, coverage, total_demand))

        user_df_list.append(user_data)

    all_user_data = pd.concat(user_df_list)
    all_user_data['charged_from_outside'] = all_user_data['needed_by_car'] - all_user_data['charged_from_pv']
    assert (all_user_data['charged_from_outside'] >= (0 - 0.01)).all()
    return coverage_all, all_user_data


def scenario_1(data_raw):
    """
    Computes weighted average fraction of self-produced energy that is being charged.

    Parameters
    ----------
    data: pandas df
        dataframe to be selected from
    user: string
        userID/vin to be selected from

    Returns
    -------
    coverage: float
        computed average charge covered by own PV production
    """


    data = data_raw[data_raw['is_home']].copy()
    data_nothome = data_raw[~ data_raw['is_home']].copy()
    coverage_all = []

    all_users = data['vin'].unique()
    for user in all_users:

        user_data = extract_user(data, user)

        total_charged = sum(user_data['charged_from_pv'])
        total_demand = sum(user_data['needed_by_car'])

        if (total_demand == 0):
            print(f'\t\tuser with zero demand: {user}')

        coverage = total_charged / total_demand

        assert 0 <= coverage <= 1
        coverage_all.append((user, coverage))

    data['charged_from_outside'] = data['needed_by_car'] - data['charged_from_pv']

    assert (data['charged_from_outside'] >= (0 - 0.01)).all()
    data = data.append(data_nothome)
    data['charged_from_outside'] = data['charged_from_outside'].fillna(0)
    return coverage_all, data


def scenario_2(data_raw):
    """
    Computes fraction when Energy can be charged, but does not have to be equal to real user data_PV_Solar.

    Parameters
    ----------
    data: pandas df
        dataframe to be selected from
    user: string
        userID/vin to be selected from

    Returns
    -------
    coverage: float
        computed average charge covered by own PV production
    """
    data = data_raw.copy()
    all_users = data['vin'].unique()
    coverage_all = []
    user_df_list = []
    for user in all_users:

        # filter
        user_data = extract_user(data, user)
        user_data = user_data.sort_values('start')
        # init new columns
        user_data['kWh_start'] = [0.] * len(user_data.index)
        user_data['kWh_end'] = [0.] * len(user_data.index)
        user_data['charged_from_outside'] = [0.] * len(user_data.index)
        user_data['max_kWh'] = [soc2remainingCharge(0)] * len(user_data.index)
        user_data['charged_from_pv'] = [0.] * len(user_data.index)

        # calculcate total consumption per segment in kWh
        user_data['total_segment_consumption_kWh'] = \
            [soc2remainingCharge(0) -
             soc2remainingCharge(user_data['total_segment_consumption'][user_data.index[i]])
             for i in range(len(user_data.index))]

        # drop debug columns:

        # relevant_columns = ['kWh_start', 'start', 'end', 'is_home', 'kWh_end', 'total_segment_consumption_kWh',
        #                     'generated_by_pv', 'max_kWh', 'charged_from_outside']
        # user_data = user_data[relevant_columns]

        user_data.loc[user_data.index[0], 'kWh_start'] = user_data['max_kWh'].iloc[0] # car starts fully charged
        for i, ix in enumerate(user_data.index):
            if i != 0:
                # end has to be updated with the consumptions of the other segments
                user_data.loc[ix, 'kWh_start'] = user_data['kWh_end'].iloc[i - 1]

            # load from outside if necessary
            if user_data.loc[ix, 'kWh_start'] < 0:
                # charged from outside is accounted to the prior timestep when the user was at home (and therefore
                # able to charge). The goal is to immitate a normal charging schedule so that we always end up
                # with min zero at the end of the segment
                if user_data.loc[ix, 'is_home']:
                    # if the user is home now he had to charge the car the last time when he was at home. This is the
                    # before last segment (i-2). We change that he charged the car there and then propagate our change
                    # back to the current time step without adjusting other values.

                    user_data.loc[user_data.index[i - 2], 'charged_from_outside'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 2], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i - 1], 'kWh_start'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 1], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i], 'kWh_start'] += - user_data['kWh_start'].iloc[i]


                else:
                    # if the user is not at home we only propagate 1 timestep back
                    user_data.loc[user_data.index[i - 1], 'charged_from_outside'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 1], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i], 'kWh_start'] += - user_data['kWh_start'].iloc[i]

                assert np.isclose(0, user_data['kWh_start'].iloc[i])

            if user_data.loc[ix, 'is_home']:
                # load from PV, the column 'generated_by_pv' already considers max charging
                max_pv_charging = user_data.loc[ix, 'generated_by_pv']

                user_data.loc[ix, 'kWh_end'] = np.minimum(user_data.loc[ix, 'max_kWh'],
                                                          user_data.loc[ix, 'kWh_start'] +
                                                          max_pv_charging +
                                                          user_data.loc[ix, 'total_segment_consumption_kWh'])

                # we only charge using pv (grid charging is account for in the next iteration). Therefore if
                # - delta SOC is positiv it was done using pv, otherwise 0
                user_data.loc[ix, 'charged_from_pv'] = \
                    np.maximum(-1 * (user_data.loc[ix, 'kWh_start'] - user_data.loc[ix, 'kWh_end']
                                     + user_data.loc[ix, 'total_segment_consumption_kWh'])
                               , 0)

            else:
                # if not at home, just consume
                # consume the maximum amount of home charging is so that the car reaches 100% SOC
                # background is, that the total_segment_consumption can be > 100 % SOC if the user did a long tour
                # with charging in between
                user_data.loc[ix, 'kWh_end'] = np.maximum(user_data.loc[ix, 'kWh_start'] - user_data.loc[ix, 'max_kWh'],
                                                          user_data.loc[ix, 'kWh_start'] +
                                                          user_data.loc[ix, 'total_segment_consumption_kWh'])

        assert np.all(user_data['charged_from_outside'] >= 0)
        assert np.all(user_data['total_segment_consumption_kWh'] <= 0)
        total_charged_from_outside = sum(user_data['charged_from_outside'])
        total_demand = - sum(user_data['total_segment_consumption_kWh'])
        coverage = 1 - (total_charged_from_outside / total_demand)
        assert 0 <= coverage <= 1

        coverage_all.append((user, coverage, total_demand))
        user_df_list.append(user_data)


    all_user_data = pd.concat(user_df_list)
    all_user_data['needed_by_car'] = all_user_data['charged_from_pv'] + all_user_data['charged_from_outside']

    assert (all_user_data['kWh_end'] >= -soc2remainingCharge(0) - 0.01).all()

    # charge at end of segment should never be higher as max possible charge
    assert (all_user_data.kWh_end <= all_user_data.max_kWh + 0.01).all()

    # in the end, the difference in state of charge (all_user_data['kWh_start'] - all_user_data['kWh_end']) musst be
    # explained by  'charged_from_pv' + 'charged_from_outside' + 'total_segment_consumption_kWh' (last one is bounded by
    # the max SOC of the car as we don't account for the consumption of very long trips)

    max_segment_consumption = np.maximum(all_user_data['total_segment_consumption_kWh'], - all_user_data['max_kWh'])

    saldo = (all_user_data['kWh_start'] - all_user_data['kWh_end'] + all_user_data['charged_from_pv'] +
             all_user_data['charged_from_outside'] + max_segment_consumption)

    assert (saldo.abs() < 0.01).all()
    return coverage_all, all_user_data


def scenario_3(data_raw, battery_capacity, battery_power, max_power_kw, path_to_data_folder=os.path.join('.', 'data')):
    """
    Computes fraction when Energy can be charged, but does not have to be equal to real user data_PV_Solar.

    Parameters
    ----------
    data: pandas df
        dataframe to be selected from
    user: string
        userID/vin to be selected from

    Returns
    -------
    coverage: float
        computed average charge covered by own PV production
    """
    data = data_raw.copy()
    all_users = data['vin'].unique()
    coverage_all = []
    user_df_list = []
    for user in all_users:

        # filter
        user_data = extract_user(data, user)
        user_data = user_data.sort_values('start')
        # init new columns
        user_data['kWh_start'] = [0.] * len(user_data.index)
        user_data['kWh_end'] = [0.] * len(user_data.index)
        user_data['max_kWh'] = [soc2remainingCharge(0)] * len(user_data.index)

        user_data['battery_start'] = [0.] * len(user_data.index)
        user_data['battery_end'] = [0.] * len(user_data.index)
        user_data['charged_from_outside'] = [0.] * len(user_data.index)
        user_data['battery_used_by_car'] = [0.] * len(user_data.index)
        # calculcate total consumption per segment in kWh
        user_data['total_segment_consumption_kWh'] = \
                    [soc2remainingCharge(0) -
                     soc2remainingCharge(user_data['total_segment_consumption'][user_data.index[i]])
                     for i in range(len(user_data.index))]

        user_data.loc[user_data.index[0], 'kWh_start'] = user_data['max_kWh'].iloc[0] # car starts fully charged
        for i, ix in enumerate(user_data.index):

            if i != 0:
                # end has to be updated with the consumptions of the other segments
                user_data.loc[ix, 'kWh_start'] = user_data['kWh_end'].iloc[i - 1]
                user_data.loc[ix, 'battery_start'] = user_data['battery_end'].iloc[i - 1]


            if user_data.loc[ix, 'kWh_start'] < 0:
                # charged from outside is accounted to the prior timestep when the user was at home (and therefore
                # able to charge). The goal is to immitate a normal charging schedule so that we always end up
                # with min zero at the end of the segment
                if user_data.loc[ix, 'is_home']:
                    # if the user is home now he had to charge the car the last time when he was at home. This is the
                    # before last segment (i-2). We change that he charged the car there and then propagate our change
                    # back to the current time step without adjusting other values.

                    user_data.loc[user_data.index[i - 2], 'charged_from_outside'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 2], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i - 1], 'kWh_start'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 1], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i], 'kWh_start'] += - user_data['kWh_start'].iloc[i]


                else:
                    # if the user is not at home we only propagate 1 timestep back
                    user_data.loc[user_data.index[i - 1], 'charged_from_outside'] += - user_data['kWh_start'].iloc[i]
                    user_data.loc[user_data.index[i - 1], 'kWh_end'] += - user_data['kWh_start'].iloc[i]

                    user_data.loc[user_data.index[i], 'kWh_start'] += - user_data['kWh_start'].iloc[i]

                assert np.isclose(0, user_data['kWh_start'].iloc[i])


            if user_data.loc[ix, 'is_home']:
                # load from PV, the column 'generated_by_pv' already considers max charging
                max_pv_charging_available_for_car = user_data.loc[ix, 'generated_by_pv']
                max_pv_charging_battery = user_data.loc[ix, 'generated_by_pv']
                max_joint_pv_charging = user_data.loc[ix, 'generated_by_pv_joint_restriction']

                # calculate charging limits
                segment_duration_hours = (user_data.loc[ix, 'end'] - user_data.loc[ix, 'start']).seconds/3600
                max_battery_charging_available_for_car = min(user_data.loc[ix, 'battery_start'],
                                                            segment_duration_hours * battery_power)
                max_energy_through_power_lim_car = segment_duration_hours * max_power_kw

                # charge car with max pv generation available or until full
                user_data.loc[ix, 'kWh_end'] = \
                    min([user_data.loc[ix, 'max_kWh'],  # max capacity
                               user_data.loc[ix, 'kWh_start'] + max_energy_through_power_lim_car,  # max charging power
                               user_data.loc[ix, 'kWh_start'] +
                               user_data.loc[ix, 'total_segment_consumption_kWh'] +
                               max_pv_charging_available_for_car +
                               max_battery_charging_available_for_car])

                # calculate pv charged in car
                # for 'charged_from_pv' we only count pv generation that enters the car.
                # Not pv generation that is charged into the battery
                pv_charged_by_car = np.maximum(-1 * (user_data.loc[ix, 'kWh_start'] -
                                                     user_data.loc[ix, 'kWh_end'] +
                                                     user_data.loc[ix, 'total_segment_consumption_kWh']
                                                     )
                                               , 0)

                user_data.loc[ix, 'charged_from_pv'] = pv_charged_by_car

                # calculate battery usage
                # everything that was charged in the car but that did not come from pv generation came from the battery
                battery_power_used_by_car = np.maximum(pv_charged_by_car - max_pv_charging_available_for_car, 0)
                user_data.loc[ix, 'battery_used_by_car'] = battery_power_used_by_car

                remaining_pv_gen_for_battery = np.maximum(max_joint_pv_charging - pv_charged_by_car, 0)

                user_data.loc[ix, 'battery_end'] = np.minimum(battery_capacity,
                                                              user_data.loc[ix, 'battery_start'] -
                                                              battery_power_used_by_car +
                                                              remaining_pv_gen_for_battery)

            else:
                # if not at home
                # consume the maximum amount of home charging is so that the car reaches 100% SOC
                # background is, that the total_segment_consumption can be > 100 % SOC if the user did a long tour
                # with charging in between
                user_data.loc[ix, 'kWh_end'] = np.maximum(user_data.loc[ix, 'kWh_start'] - user_data.loc[ix, 'max_kWh'],
                                                          user_data.loc[ix, 'kWh_start']
                                                          + user_data.loc[ix, 'total_segment_consumption_kWh'])

                # and charge battery with available pv
                pv_generation_while_away = user_data.loc[ix, 'generated_by_pv_battery_restriction']

                user_data.loc[ix, 'battery_end'] = np.minimum(battery_capacity,
                                                              user_data.loc[ix, 'battery_start'] +
                                                              pv_generation_while_away)

        assert np.all(user_data['charged_from_outside'] >= 0)
        assert np.all(user_data['total_segment_consumption_kWh'] <= 0)
        total_charged_from_outside = sum(user_data['charged_from_outside'])
        total_demand = - sum(user_data['total_segment_consumption_kWh'])
        coverage = 1 - (total_charged_from_outside / total_demand)
        assert 0 <= coverage <= 1

        coverage_all.append((user, coverage, total_demand))
        user_df_list.append(user_data)

    all_user_data = pd.concat(user_df_list)
    all_user_data['needed_by_car'] = all_user_data['charged_from_pv'] + all_user_data['charged_from_outside']
    assert (all_user_data['kWh_end'] >= -soc2remainingCharge(0) - 0.01).all()

    # in the end, the difference in state of charge of the car (all_user_data['kWh_start'] - all_user_data['kWh_end'])
    # musst be explained by  'charged_from_pv' + 'charged_from_outside' + 'total_segment_consumption_kWh'
    # (last one is bounded by the max SOC of the car as we don't account for the consumption of very long trips)

    max_segment_consumption = np.maximum(all_user_data['total_segment_consumption_kWh'], - all_user_data['max_kWh'])

    saldo = (all_user_data['kWh_start'] - all_user_data['kWh_end'] + all_user_data['charged_from_pv'] +
             all_user_data['charged_from_outside'] + max_segment_consumption)
    assert (saldo.abs() < 0.01).all()

    return coverage_all, all_user_data

def create_scenario_table(data_baseline, data, battery_capacity, battery_power, max_power_kw, pv_model,
                          path_to_data_folder=os.path.join('.', 'data')):
    """
    Creates a dataframe that contains coverage in all different scenarios

    Parameters
    ----------
    data: pd-dataframe
        dataframe to extract information from
    capacity: float
        maximum batteriy capacity

    Returns
    -------

    table: pandas-df
        table with the three scenarios
    """

    # for every scenario, calculate the following columns:
    # - 'needed_by_car': How much energy is required by the EV. In the baseline and in scenario 1 this corresponds to
    #       what was actually charged by the user in a segment.
    # - 'charged_from_pv': How much energy could the user charge from the roof-top pv panel
    # - 'charged_from_outside': needed_by_car - charged_from_pv

    print("baseline")
    baseline_coverage, baseline_results = baseline(data_baseline)

    print("scenario 1")
    scenario1_coverage, scenario1_results = scenario_1(data)

    print("scenario 2")
    scenario2_coverage, scenario2_results = scenario_2(data)

    print("scenario 3")
    scenario3_coverage, scenario3_results = scenario_3(data, battery_capacity=battery_capacity,
                                                       battery_power=battery_power, max_power_kw=max_power_kw)
    # transform to coverage information to single dataframe
    baseline_coverage = pd.DataFrame(baseline_coverage, columns=['vin', 'baseline', 'total_demand']).set_index('vin')
    scenario1_coverage = pd.DataFrame(scenario1_coverage, columns=['vin', 'scenario1']).set_index('vin')
    scenario2_coverage = pd.DataFrame(scenario2_coverage, columns=['vin', 'scenario2', 'total_demand_s2']).set_index('vin')
    scenario3_coverage = pd.DataFrame(scenario3_coverage, columns=['vin', 'scenario3', 'total_demand_s3']).set_index('vin')

    table = pd.concat((baseline_coverage, scenario1_coverage, scenario2_coverage, scenario3_coverage), axis=1)

    results = {'table': table,
               'baseline': baseline_results.sort_values(by=['vin', 'start']),
               'scenario1': scenario1_results.sort_values(by=['vin', 'start']),
               'scenario2': scenario2_results.sort_values(by=['vin', 'start']),
               'scenario3': scenario3_results.sort_values(by=['vin', 'start'])}


    return results
