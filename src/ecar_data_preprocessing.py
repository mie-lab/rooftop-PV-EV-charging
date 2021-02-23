# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:52:10 2018

@author: martinhe

Create a table called bmw_homes that relates the home address of sbb-green class participants (as 
geometry) with their user_id and vin/vid of the bmw.  

Deleted Attributes:
    df (TYPE): Description
    engine (TYPE): Description
    p_dest (TYPE): Description
    p_source (TYPE): Description
    rename_columns (dict): Description
"""

import datetime
import logging
import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from pyproj import Transformer
from sqlalchemy import create_engine
import warnings
# import database login information
from src.db_login import DSN
from src.table_information import home_table_info, ecar_table_info, ecarid_athome_table_info

logging.basicConfig(
    filename='value_fillin.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


# create new table
def create_ecar_homes_table(home_table_info, DSN, df):
    """Creates a table with all home locations of ecar users.
    
    Creates a table based on the information in 'home_table_info' that holds all homes of ecar
    users as point geometries, their user_id and their vehicle id.
    
    Args:
        home_table_info (dict): Dictionary with the following structure: 
                                {'schema': schema_name(str),
                                'home_table_name': name_for_table (str)}
        DSN (dict): Dictionary with the database connection information.
                    It has the following structure 
                    {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
        df (pandas dataframe): Pandas dataframe with information about all owners ecar home locations movement.
                Needs the following columns: ['vin' (vehicle id), 'user_id', 'long', 'lat'], coordinates in WGS84
    """
    home_table_name = home_table_info['home_table_name_single']
    schema = home_table_info['schema_single']
    with psycopg2.connect(**DSN) as conn:
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE IF EXISTS {home_table_name};").format(**home_table_info)

        cur.execute(query)

        conn.commit()

        # create bmw_homes table
        query = sql.SQL("""create table {home_table_name}(
                    vin VARCHAR,
                    user_id INT,
                    long FLOAT,
                    lat FLOAT);
        """).format(**home_table_info)
        cur.execute(query)

        query = """SELECT AddGeometryColumn ('{schema_single}', '{home_table_name_single}', 'geometry', 4326, 'POINT',2);""".format(
            **home_table_info)
        cur.execute(query)

        conn.commit()

        # push dataframe to psql
        engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))
        df.loc[:, ['vin', 'user_id', 'long', 'lat']].to_sql(home_table_name,
                                                            engine, schema=schema, index=False, if_exists='append')

        # set geometry and create different indices
        # set geometry
        cur = conn.cursor()
        query = sql.SQL("""UPDATE {home_table_name} SET geometry =
                 ST_GeomFromText('POINT(' || long || ' ' || lat|| ')',4326);""").format(**home_table_info)
        cur.execute(query)

        # indices
        query = sql.SQL("""CREATE INDEX bmwhome_geom_geog_idx ON {home_table_name} USING gist (geography(geometry));
            CREATE INDEX bmwhome_geom_idx ON {home_table_name} USING gist (geometry);""").format(**home_table_info)
        cur.execute(query)
        conn.commit()


def create_ecar_data_table(ecar_table_info, DSN):
    """Creates table with ecar tracking data_PV_Solar

    It is important that the data_PV_Solar has an unique row id.
    
    Args:
        ecar_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecar_table_name': name_for_table (str)}
        DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
    """

    with psycopg2.connect(**DSN) as conn:
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE IF EXISTS {ecar_table_name};").format(**ecar_table_info)
        cur.execute(query)

        query = sql.SQL("CREATE TABLE {ecar_table_name} as select * from gc1.bmw;").format(**ecar_table_info)
        cur.execute(query)

        conn.commit()


def create_ecarid_is_athome_table(ecarid_athome_table_info, DSN):
    """Create table that tells you for every ecar record if it was recorded at home
    
    Args:
        ecarid_athome_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecarid_athome_table_name': name_for_table (str)}
        DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
    """

    with psycopg2.connect(**DSN) as conn:
        cur = conn.cursor()

        # drop if exists
        query = sql.SQL("DROP TABLE IF EXISTS {ecarid_athome_table_name};").format(**ecarid_athome_table_info)
        cur.execute(query)

        # create bmwid_ishome table
        query = sql.SQL("""create table {ecarid_athome_table_name}(
                    vin VARCHAR,
                    user_id INT,
                    bmw_id INT,
                    zustand VARCHAR,
                    start_end VARCHAR,
                    timestamp TIMESTAMP,
                    soc FLOAT,
                    is_home boolean DEFAULT FALSE);
        """).format(**ecarid_athome_table_info)
        cur.execute(query)
        conn.commit()


def fill_ecarid_is_athome_table(ecarid_athome_table_info, ecar_table_info,
                                home_table_info, DSN, start_end_flag):
    """Fill ecarid_athome table.
    The `ecarid_athome` table has the combined information of `home_table` and `ecar_table`.
    All car activities and a flag whether the activity started or ended at home.
    For this purpose we treat the start and the end of a segment seperatly
    
    Args:
        ecarid_athome_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecarid_athome_table_name': name_for_table (str)}
        ecar_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecar_table_name': name_for_table (str)}
        home_table_info (dict): Dictionary with the following structure: 
                                {'schema': schema_name(str),
                                'home_table_name': name_for_table (str)}
        DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
        start_end_flag (TYPE): Can be either `start` or `end` to process start points 
                or end points.
    
    Raises:
        NameError: start_end_flag has to be either `start` or `end`
    """

    # set variable fields depending on if you treat the start or the end of
    # an entry
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))

    start_end_dict = {}
    start_end_dict['start_end_string'] = sql.SQL(start_end_flag.capitalize())

    if start_end_flag == 'start':
        start_end_dict['geom_field'] = sql.Identifier('geom_start')
        start_end_dict['ts_field'] = sql.Identifier('timestamp_start_utc')
        start_end_dict['soc_field'] = sql.Identifier('soc_customer_start')
    elif start_end_flag == 'end':
        start_end_dict['geom_field'] = sql.Identifier('geom_end')
        start_end_dict['ts_field'] = sql.Identifier('timestamp_end_utc')
        start_end_dict['soc_field'] = sql.Identifier('soc_customer_end')
    else:
        raise NameError('start_end_flag has to be "start" or "end" but was {}'.format(start_end_flag))

    with psycopg2.connect(**DSN) as conn:
        cur = conn.cursor()

        # fill bmwid_ishome with base data_PV_Solar (=original BMW data_PV_Solar + tripleg and user ids)
        query = sql.SQL("""INSERT INTO {ecarid_athome_table_name} 
                    (vin, bmw_id, start_end, zustand, timestamp, soc)
                    select vin, id, '{start_end_string}', zustand, {ts_field}, {soc_field}
                    from {ecar_table_name} where {geom_field} is not null
                    """).format(**{**ecar_table_info, **ecarid_athome_table_info, **start_end_dict})
        cur.execute(query)

        # create the information if a BMW was at home in a temporary table.
        # 'is_home' means within 500 meters of its home location. 500 meters was chosen
        # because cars are often parked in a garage which impacts GPS accuracy 

        query = sql.SQL("""drop table if exists temp_bmwid_ishome;
                CREATE temp TABLE temp_bmwid_ishome as 
                SELECT DISTINCT ON(bmw_id) {ecar_table_name}.id as bmw_id, 
                {ecar_table_name}.{geom_field} as bmw_geom,
                {home_table_name}.geometry as home_geom,
                {home_table_name}.vin as home_vin,
                {ecar_table_name}.vin as ecar_vin,
                 ST_DWithin({ecar_table_name}.{geom_field}::geography, 
                           {home_table_name}.geometry::geography, 500) as is_home
                FROM {ecar_table_name}, {home_table_name}
                WHERE {ecar_table_name}.vin = {home_table_name}.vin
                AND {ecar_table_name}.{geom_field} is not null order by bmw_id;""").format(
            **{**home_table_info, **ecar_table_info, **start_end_dict})
        cur.execute(query)
        conn.commit()
        # add/update the boolean is_home to bmwid_ishome table
        query = sql.SQL("""UPDATE {ecarid_athome_table_name} SET is_home = temp_bmwid_ishome.is_home
                    from temp_bmwid_ishome
                    WHERE {ecarid_athome_table_name}.bmw_id = temp_bmwid_ishome.bmw_id 
                    and {ecarid_athome_table_name}.start_end = '{start_end_string}';""").format(
            **{**ecarid_athome_table_info, **start_end_dict})
        cur.execute(query)
        conn.commit()
        #
        # # export bmwid_athome_table
        # query = """select * from {ecarid_athome_table_name};""".format(**ecarid_athome_table_info)
        # return pd.read_sql(query, engine)


def impute_iteration(df):
    """Fills missing data_PV_Solar from adjacent rows

    imputes all locations using the following rules:
        `start location` = `end location` if car did not move.
        `end location` = `start location` if car did not move.
        `start location` = `last end location`
        `last end location` = `start location`
        
        Only missing geometries are imputed. 
    
    Args:
        df (Pandas dataframe): Pandas dataframe created using this query: 
        "SELECT id, vin, zustand, timestamp_start_utc, timestamp_end_utc, 
            km_stand_end - km_stand_start AS delta_km,
            latitude_start, longitude_start, latitude_end, longitude_end
        FROM {schema_single}.{ecar_table_name_single} 
        ORDER BY vin, timestamp_start_utc ASC".format(**ecar_table_info)
    
    Returns:
        TYPE: (pandas dataframe)
    """
    fill_start_from_end = (df['delta_km'] == 0) \
                          & np.isnan(df['latitude_start']) \
                          & np.isnan(df['longitude_start']) \
                          & ~np.isnan(df['latitude_end']) \
                          & ~np.isnan(df['longitude_end'])

    fill_end_from_start = (df['delta_km'] == 0) \
                          & ~np.isnan(df['latitude_start']) \
                          & ~np.isnan(df['longitude_start']) \
                          & np.isnan(df['latitude_end']) \
                          & np.isnan(df['longitude_end'])

    # fill start from end  
    df.loc[fill_start_from_end, 'latitude_start'] = df[fill_start_from_end]['latitude_end']
    df.loc[fill_start_from_end, 'longitude_start'] = df[fill_start_from_end]['longitude_end']

    # fill end from start
    df.loc[fill_end_from_start, 'latitude_end'] = df[fill_end_from_start]['latitude_start']
    df.loc[fill_end_from_start, 'longitude_end'] = df[fill_end_from_start]['longitude_start']

    # shift down
    df[['latitude_end_prev', 'longitude_end_prev', 'vin_prev']] = \
        df[['latitude_end', 'longitude_end', 'vin']].shift(periods=1, fill_value=np.NAN)

    # shift up
    df[['latitude_start_next', 'longitude_start_next', 'vin_next']] = \
        df[['latitude_start', 'longitude_start', 'vin']].shift(periods=-1, fill_value=np.NAN)

    # fill start from previous end
    fill_start_from_prev_end = (df['vin'] == df['vin_prev']) \
                               & np.isnan(df['latitude_start']) \
                               & np.isnan(df['longitude_start']) \
                               & ~np.isnan(df['latitude_end_prev']) \
                               & ~np.isnan(df['longitude_end_prev'])

    # fill end from next start 
    fill_end_from_next_start = (df['vin'] == df['vin_next']) \
                               & ~np.isnan(df['latitude_start_next']) \
                               & ~np.isnan(df['longitude_start_next']) \
                               & np.isnan(df['latitude_end']) \
                               & np.isnan(df['longitude_end'])

    # fill start from prev end  
    df.loc[fill_start_from_prev_end, 'latitude_start'] = df[fill_start_from_prev_end]['latitude_end_prev']
    df.loc[fill_start_from_prev_end, 'longitude_start'] = df[fill_start_from_prev_end]['longitude_end_prev']

    # fill end from start
    df.loc[fill_end_from_next_start, 'latitude_end'] = df[fill_end_from_next_start]['latitude_start_next']
    df.loc[fill_end_from_next_start, 'longitude_end'] = df[fill_end_from_next_start]['longitude_start_next']

    df.drop(['latitude_start_next', 'longitude_start_next',
             'latitude_end_prev', 'longitude_end_prev',
             'vin_prev', 'vin_next'], axis=1, inplace=True)
    return df


def fill_trivial_gaps(DSN, ecar_table_info):
    """This function treats missing geometries that are easy to fill.

    The ecar records have among other data_PV_Solar the starting location and the end location
    of a segment, the activity of the segment, as well as the readings from the
    milage counter. 
    The geometries are often missing, however if we know the location of a car
    and we know from the milage counter that the car did not move, we can 
    impute the missing geometries.

    Iteratively imputes all locations using the following rules:
        `start location` = `end location` if car did not move.
        `end location` = `start location` if car did not move.
        `start location` = `last end location`
        `last end location` = `start location`
        
        Only missing geometries are imputed.

    Args:
        DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
        ecar_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecar_table_name': name_for_table (str)}
    """

    # download ecar data_PV_Solar
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))

    pandas_query = """SELECT id, vin, zustand, timestamp_start_utc, timestamp_end_utc, 
        km_stand_end - km_stand_start AS delta_km,
        latitude_start, longitude_start, latitude_end, longitude_end
        FROM {schema_single}.{ecar_table_name_single} 
        ORDER BY vin, timestamp_start_utc ASC""".format(**ecar_table_info)

    df = pd.read_sql(pandas_query, engine)

    # counters to keep track of imputation progress  
    nb_nans_init = df['latitude_start'].isna().sum() + df['latitude_end'].isna().sum()
    nb_nans_old = nb_nans_init
    delta_nans = 1

    # iteratively impute all rows 
    while delta_nans > 0:
        df = impute_iteration(df)
        nb_nans = df['latitude_start'].isna().sum() + df['latitude_end'].isna().sum()
        delta_nans = nb_nans_old - nb_nans

        nb_nans_old = nb_nans

    logging.info('Total number of values: %s', str(df['latitude_start'].shape[0]))
    logging.info('Intial values missing: %s', str(nb_nans_init))
    logging.info('Values filled: %s', str(nb_nans_init - nb_nans))
    logging.info('Values not filled: %s', str(nb_nans))

    # write to database
    logging.info('Write filled table to database')
    df = df[['id', 'latitude_start', 'longitude_start', 'latitude_end',  'longitude_end']]

    df.to_sql(name='temp_table_trivial_imputation', con=engine, index=False,
              if_exists='replace')

    logging.info('update geometries in ecar data table')
    with psycopg2.connect(**DSN) as conn:
        cur = conn.cursor()
        query = sql.SQL("""UPDATE {ecar_table_name} as ecardata
                SET latitude_start = tt.latitude_start,
                    longitude_start = tt.longitude_start
                FROM temp_table_trivial_imputation as tt
                WHERE ecardata.latitude_start is Null 
                    and ecardata.longitude_start is Null 
                    and tt.latitude_start is not Null 
                    and tt.longitude_start is not Null
                    and ecardata.id = tt.id;""").format(**ecar_table_info)

        cur.execute(query)

        query = sql.SQL("""UPDATE {ecar_table_name} as ecardata
                        SET latitude_end = tt.latitude_end,
                            longitude_end = tt.longitude_end
                        FROM temp_table_trivial_imputation as tt
                        WHERE ecardata.latitude_end is Null 
                            and ecardata.longitude_end is Null 
                            and tt.latitude_end is not Null 
                            and tt.longitude_end is not Null
                            and ecardata.id = tt.id;""").format(**ecar_table_info)
        cur.execute(query)

        query = sql.SQL("""UPDATE {ecar_table_name}
                    SET geom_start = ST_SetSRID(ST_MakePoint(longitude_start, latitude_start), 4326)
                    WHERE latitude_start is not Null and longitude_start is not Null 
                            and geom_start is Null;
                    """).format(**ecar_table_info)

        cur.execute(query)

        query = sql.SQL("""UPDATE {ecar_table_name}
                            SET geom_end = ST_SetSRID(ST_MakePoint(longitude_end, latitude_end), 4326)
                            WHERE latitude_end is not Null and longitude_end is not Null
                                and geom_end is Null;
                            """).format(**ecar_table_info)
        cur.execute(query)
        conn.commit()

        cur.execute("DROP TABLE temp_table_trivial_imputation")
        conn.commit()


def create_dataframe_with_unique_carid_timestamps_combination(ecarid_athome_table_info, DSN):
    """Create dataframe with unique vehcile id and timestamp combinations

    In the ecar tracking data_PV_Solar several events can have the same time stamp. A common example
    is the end of a chargeing segment and the start of an idle segment. This function
    downloads the ecar tracking data_PV_Solar from the database and returns a data_PV_Solar frame where every
    timestamp is unique (for every indiviual vehicle)
    
    Args:
        ecarid_athome_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecarid_athome_table_name': name_for_table (str)}
         DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
    
    Returns:
        TYPE: pandas dataframe
    """
    # sql engine for pandas db connection
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))

    # download all data_PV_Solar from
    pandas_query = """SELECT * from {schema_single}.{ecarid_athome_table_name_single}""".format(
        **ecarid_athome_table_info)

    # set vehicle id as index and sort by index
    df = pd.read_sql_query(pandas_query, con=engine)
    df.set_index(['vin', 'timestamp'], inplace=True, drop=False)
    df.index.rename(['vin_ix', 'timestamp_ix'], inplace=True)
    df.sort_values(['vin', 'timestamp', 'bmw_id'], inplace=True)

    # Every car can have multiple events at the same time.
    # Extract distinct time slots for each car. E.g., 
    # [start_1; end_1]
    # [start_2; end 2]

    # get unique values of vehicle ids and timestamps
    df_unique = df.loc[:, ['vin', 'timestamp']].drop_duplicates()
    df_unique.set_index(['vin', 'timestamp'], inplace=True, drop=False)
    df_unique.index.rename(['vin_ix', 'timestamp_ix'], inplace=True)

    # sort by car id and timestamp, then create time slots
    df_unique.sort_values(['vin', 'timestamp'], inplace=True)

    return df_unique, df


#


def create_segmented_ecar_data(ecarid_athome_table_info, DSN):
    """Returns ecar tracking data_PV_Solar as segments

    The ecar tracking data_PV_Solar can have multiple events with the same timestamp.
    This function aggregates them to have distinct segments for each car. 
    e.g.:
        #[start_1; end_1]
        #[start_2; end 2]
        #[start_3; end 3] 
        
        where end_(i) = start_(i+1)

    Args:
        ecarid_athome_table_info (dict): Dictionary with the following structure: 
                            {'schema': schema_name(str),
                            'ecarid_athome_table_name': name_for_table (str)}
        DSN (dict): Dictionary with the database connection information.
                It has the following structure 
                {'dbname': (str), 'port': (str), 'user':(str), 'password':(str), 'host':(str)}
    
    Returns:
        TYPE: pandas dataframe
    """
    # get unique combinations of start time and vehicle id whenever an ecar
    # was spotted at home or not at home. 

    ecar_unique_timestamps, ecardata_raw = create_dataframe_with_unique_carid_timestamps_combination(
        ecarid_athome_table_info, DSN)

    # We create 1 segment (row) that summarizes all actions (charging, waiting,
    # ect.) that happend while a car is continuously at home and 1 segment
    # that sumarrizes all activity that takes place while a car is continuously 
    # away from home (driving, charging, waiting, driving, ...). 

    ecar_unique_timestamps['start'] = ecar_unique_timestamps['timestamp']
    ecar_unique_timestamps['end'] = ecar_unique_timestamps['start'].shift(-1)
    ecar_unique_timestamps['vin_lag'] = ecar_unique_timestamps['vin'].shift(-1)

    ecar_unique_timestamps['soc_start'] = ecardata_raw['soc'].groupby(by=ecardata_raw.index).mean()
    ecar_unique_timestamps['soc_end'] = ecar_unique_timestamps['soc_start'].shift(-1)

    # delta soc is positiv for charging and negativ for consumption
    ecar_unique_timestamps['delta_soc'] = ecar_unique_timestamps['soc_end'] - ecar_unique_timestamps['soc_start']

    # delete rows where the shift does not make sense because it where different users. This is the
    # last value of every user
    ecar_unique_timestamps = ecar_unique_timestamps.drop(
        ecar_unique_timestamps[ecar_unique_timestamps['vin'] != ecar_unique_timestamps['vin_lag']].index)
    ecar_unique_timestamps = ecar_unique_timestamps.drop(columns='vin_lag')

    # create a list of all values, ids and start_end flags that took place 
    # in one segment (while a user was continously at home). This is mainly 
    # for debugging.
    ecar_unique_timestamps['bmw_id'] = ecardata_raw['bmw_id'].groupby(by=ecardata_raw.index).apply(list)
    ecar_unique_timestamps['zustand'] = ecardata_raw['zustand'].groupby(by=ecardata_raw.index).apply(list)
    ecar_unique_timestamps['start_end'] = ecardata_raw['start_end'].groupby(by=ecardata_raw.index).apply(list)

    return ecar_unique_timestamps, ecardata_raw


def set_is_home_flag(ecar_unique_timestamps, ecardata_raw):
    """Add a flag to every ecar segment that tells if the car was at home.
    
    Args:
        ecar_unique_timestamps (TYPE): Segmented ecar data_PV_Solar from `create_segmented_ecar_data`
        ecardata_raw (TYPE): Raw (and complete) ecar tracking data_PV_Solar
    
    Returns:
        TYPE: pandas dataframe
    """
    # Define the creteria for a segment to be at home:
    fahrt_home_start = {'zustand': ['fahrt'], 'start_end': ['Start'], 'is_home': [True]}
    fahrt = {'zustand': ['fahrt']}

    # per default, set is_home to False
    ecar_unique_timestamps['is_home'] = False

    home = {'is_home': [True]}
    # check the "home" condition.
    # ".all(axis=1) transforms result to series.
    df_home_TRUE_series = ecardata_raw.loc[:, ['is_home']].isin(home).all(axis=1)
    # because we have start and end times appear, every column exists twice 
    # we filter doublons by grouping the columns by their index
    df_home_TRUE = df_home_TRUE_series.groupby(by=ecardata_raw.index).first()

    # update main dataframe without pandas complaining...
    df_home_TRUE = (df_home_TRUE[df_home_TRUE])  # keep only where df_home_TRUE is True

    # only update the entries where is_home is TRUE
    intersecting_ix = ecar_unique_timestamps.index.intersection(df_home_TRUE.index)
    ecar_unique_timestamps.loc[intersecting_ix, 'is_home'] = True

    # remove flag when user starts driving at home (he then won't be at home
    # in the segment)
    # same approach as above with different multi-column condition (=fahrt_home_start)
    df_fahrt_home_start_TRUE_series = ecardata_raw.loc[
                                      :, ['zustand', 'start_end', 'is_home']].isin(fahrt_home_start).all(axis=1)
    df_fahrt_home_start_TRUE = df_fahrt_home_start_TRUE_series.groupby(by=ecardata_raw.index).max()

    # update main dataframe without pandas complaining...
    # only update the entries where is_home is TRUE
    df_fahrt_home_start_TRUE = (df_fahrt_home_start_TRUE[df_fahrt_home_start_TRUE])
    intersecting_ix = ecar_unique_timestamps.index.intersection(df_fahrt_home_start_TRUE.index)
    ecar_unique_timestamps.loc[intersecting_ix, 'is_home'] = False

    # create an id for every segment where a car is continuously at home or not at home
    ecar_unique_timestamps['segment_id'] = ecar_unique_timestamps['is_home'].astype('int').diff().abs()
    ecar_unique_timestamps['segment_id'].fillna(0, inplace=True)
    ecar_unique_timestamps['segment_id'] = ecar_unique_timestamps['segment_id'].cumsum()

    # create an extra column with only the negative power consumption
    neg_con = ecar_unique_timestamps['delta_soc'] < 0

    ecar_unique_timestamps['only_consumption'] = 0
    ecar_unique_timestamps['only_charging'] = 0

    ecar_unique_timestamps.loc[neg_con, 'only_consumption'] = \
        ecar_unique_timestamps.loc[neg_con, 'delta_soc']
    ecar_unique_timestamps.loc[~neg_con, 'only_charging'] = \
        ecar_unique_timestamps.loc[~neg_con, 'delta_soc']



    return ecar_unique_timestamps


def aggregate_home_nothome_segments(ecar_unique_timestamps):
    """Aggregates consecutive segments with same is_home value

    This function aggregates all segments from `create_segmented_ecar_data` 
    and from `set_is_home_flag` that are consecutive and have the 
    same value for the is_home flag (True, False)
    
    Args:
        ecar_unique_timestamps (pandas dataframe): Data from `set_is_home_flag`
    
    Returns:
        TYPE: pandas dataframe
    """
    # aggregate continuous home and not_home segments

    # flatten makes a flat list out of a list of lists
    flatten = lambda l: [item for sublist in l for item in sublist]

    ecar_unique_grouper = ecar_unique_timestamps.groupby(['vin', 'segment_id'])

    df_unique_agg_first = ecar_unique_grouper[['vin', 'start', 'soc_start', 'is_home']].first()
    df_unique_agg_last = ecar_unique_grouper[['end', 'soc_end']].last()
    ecar_unique_agg = pd.concat([df_unique_agg_first, df_unique_agg_last], sort=False, axis=1)

    ecar_unique_agg['total_segment_consumption'] = ecar_unique_grouper['only_consumption'].sum()
    ecar_unique_agg['total_segment_charging'] = ecar_unique_grouper['only_charging'].sum()
    #    #debug
    #    ecar_unique_agg['all_soc_start'] = ecar_unique_grouper['soc_start'].apply(list)
    #    ecar_unique_agg['all_soc_end'] = ecar_unique_grouper['soc_end'].apply(list)
    #    ecar_unique_agg['all_t_start'] = ecar_unique_grouper['start'].apply(list)

    ecar_unique_agg['delta_soc'] = ecar_unique_grouper['delta_soc'].apply(list)
    ecar_unique_agg['bmw_id'] = ecar_unique_grouper['bmw_id'].apply(list).apply(flatten)
    ecar_unique_agg['zustand'] = ecar_unique_grouper['zustand'].apply(list).apply(flatten)
    ecar_unique_agg['start_end'] = ecar_unique_grouper['start_end'].apply(list).apply(flatten)

    ecar_unique_agg.index.rename(['vin_ix', 'segment_id'], inplace=True)
    ecar_unique_agg.sort_values(['vin', 'start'], inplace=True)

    return ecar_unique_agg


# export baseline data
def export_baseline_data(ecar_table_info, ecarid_athome_table_info, DSN,
                         min_date=datetime.datetime(year=2017, month=2, day=1),
                         max_date=datetime.datetime(year=2018, month=12, day=31)):
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**DSN))

    ecar_data_query = """select id, vin, zustand, timestamp_start_utc,
                        timestamp_end_utc, soc_customer_start, soc_customer_end,
                       user_id
                       from {schema_single}.{ecar_table_name_single}""".format(**ecar_table_info)

    ecarid_is_athome_query = """select bmw_id, is_home
                                from {schema_single}.{ecarid_athome_table_name_single}
                                where start_end = 'Start'""".format(**ecarid_athome_table_info)

    ecar_data = pd.read_sql(ecar_data_query, engine)
    ecar_data.set_index('id', inplace=True)

    ecarid_is_athome = pd.read_sql(ecarid_is_athome_query, engine)
    ecarid_is_athome.set_index('bmw_id', inplace=True)

    ecar_data_joined = ecar_data.join(ecarid_is_athome['is_home'])

    ecar_data_joined['is_home'] = ecar_data_joined['is_home'].fillna(False)
    # is_home flag relates to the start of a segment. If a trip (='fahrt') starts at home, 'is_home' is set to False
    ecar_data_joined.loc[ecar_data_joined['zustand'] == 'fahrt', 'is_home'] = False
    ecar_data_joined.sort_values(by=['vin', 'timestamp_start_utc'], inplace=True)

    # drop data where timestamps are not plausible
    ix = ecar_data_joined['timestamp_start_utc'] <= ecar_data_joined['timestamp_end_utc']
    ecar_data_joined = ecar_data_joined[ix]

    # filter timestamps
    ecar_data_joined = ecar_data_joined[ecar_data_joined['timestamp_start_utc'] >= min_date]
    ecar_data_joined = ecar_data_joined[ecar_data_joined['timestamp_end_utc'] <= max_date]

    return ecar_data_joined


if __name__ == '__main__':
    """This script creates an aggregation from ecar tracking data_PV_Solar.

    The ecar data_PV_Solar we process creates event-based entries whenever 
    the car startes or stops charging, moving or resting. This can result in several 
    event-entries per second. For our analysis, we are interested in aggregated segments
    the summarizes all activities when a user is 'at home' or 'not at home'.

    This script performs the necessary steps to create such an aggregated view of the data_PV_Solar.

    The script uses a mix of pandas/geopandas and postgresql/postgis to perform thsi analysis.

    """
    # define projections
    logging.basicConfig(level=logging.DEBUG)
    transformer = Transformer.from_crs('epsg:21781', 'epsg:4326', always_xy=True)

    file_out = os.path.join(".", "data", "car_is_at_home_table_UTC.csv")
    file_out_baseline = os.path.join(".", "data", "data_baseline.csv")

    min_date = datetime.datetime(year=2017, month=2, day=1)
    max_date = datetime.datetime(year=2017, month=12, day=31)

    # a file that matches the ecar data_PV_Solar to home adresses (via an id)
    df = pd.read_csv(os.path.join('.', 'data', 'matching_bmw_to_address.csv'), sep=";")

    # transform to wgs84
    long, lat = transformer.transform(df['GWR_x'].values, df['GWR_y'].values)
    df['long'] = long
    df['lat'] = lat

    rename_columns = {'BMW_vid': 'vin', 'BMW_userid': 'user_id'}
    df.rename(columns=rename_columns, inplace=True)

    # start data_PV_Solar preparation
    # create tables with home locations and raw data_PV_Solar
    logging.info("create_ecar_homes_table")
    create_ecar_homes_table(home_table_info=home_table_info, DSN=DSN, df=df)
    logging.info("create_ecar_data_table")
    create_ecar_data_table(ecar_table_info=ecar_table_info, DSN=DSN)

    # fill gaps in ecar data_PV_Solar table
    logging.info("fill_trivial_gaps")
    fill_trivial_gaps(DSN=DSN, ecar_table_info=ecar_table_info)

    # create table  data_PV_Solar that combines information
    logging.info("create_ecarid_is_athome_table")
    create_ecarid_is_athome_table(ecarid_athome_table_info=ecarid_athome_table_info, DSN=DSN)
    logging.info("fill_ecarid_is_athome_table-start")
    fill_ecarid_is_athome_table(ecarid_athome_table_info=ecarid_athome_table_info,
                                ecar_table_info=ecar_table_info, home_table_info=home_table_info, DSN=DSN,
                                start_end_flag='start')
    logging.info("fill_ecarid_is_athome_table-end")
    fill_ecarid_is_athome_table(ecarid_athome_table_info=ecarid_athome_table_info,
                                ecar_table_info=ecar_table_info, home_table_info=home_table_info,
                                DSN=DSN, start_end_flag='end')

    logging.info("write {}".format(file_out_baseline))

    baseline_data = export_baseline_data(ecar_table_info, ecarid_athome_table_info, DSN,
                                         min_date=min_date,
                                         max_date=max_date)

    baseline_data.to_csv(file_out_baseline)

    logging.info("create_segmented_ecar_data")
    ecar_unique_timestamps, ecardata_raw = create_segmented_ecar_data(
        ecarid_athome_table_info=ecarid_athome_table_info, DSN=DSN)
    logging.info("set_is_home_flag")
    ecar_unique_timestamps = set_is_home_flag(ecar_unique_timestamps, ecardata_raw)
    logging.info("aggregate_home_nothome_segments")
    ecar_unique_agg = aggregate_home_nothome_segments(ecar_unique_timestamps)

    # filter timestamps
    ecar_unique_agg = ecar_unique_agg[ecar_unique_agg['start'] >= min_date]
    ecar_unique_agg = ecar_unique_agg[ecar_unique_agg['end'] <= max_date]

    logging.info("to_csv")
    ecar_unique_agg.to_csv(file_out, index=False)
    logging.info("done")
