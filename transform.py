"""

Our goal for each dataset is to have a datetime column that serves as the index.

LDF:


"""
import pandas as pd
import pytz

pd.options.mode.copy_on_write = True 


def transform_ldf(raw_ercot_df):
    """
    Merge LdfDate with LdfHour to have a datetime column.
    This data appears to only change when an event occurs;
    otherwise, it will continue to have the same hourly factors each day.
    Meaning that missing dates have the same factors as the last posted date.
    So, you can fill in the missing dates with the same values as the latest hourly factors for each substation.
    """
    pass

def pivot_dam_lmp(raw_ercot_df):
    """
    DAM LMP:
        Could you pivot the CSV file to have rows as datetime, columns as the bus, and cell values as the LMP? This would make the dataframe easier to read. And if yo could make it yearly CSVs so the file size can be manageable. (Attached is a sketch of the desired dataframe)
        Same as LDF, merge DeliveryDate and HourEnding to have datetime column.
        DSTFlag is for Daylight Saving Time repeated values.
    """
    pass


# Set timezone to Central Prevailing Time (ERCOT), which is America/Chicago (CST/CDT)
texas_tz = pytz.timezone('America/Chicago')

def convert_to_datetimes(df):
    """
    Convert DeliveryDate and HourEnding columns to a single UTC seconds column.
    Args:
    - df (pd.DataFrame): Input dataframe with 'DeliveryDate', 'HourEnding', and 'DSTFlag'.
    
    Returns:
    - df (pd.DataFrame): Dataframe with a new 'utc_seconds' column.
    """
    # Adjust from hourEnding to HourBeginning so the parser won't choke
    hour_adjusted = (df['HourEnding'].str.split(':').str[0].astype(int) - 1).astype(str) + ":00"
    df['HourBeginning'] = hour_adjusted

    # Now convert to datetime
    dt_combined = pd.to_datetime(df['DeliveryDate'] + ' ' + df['HourBeginning'], format='%m/%d/%Y %H:%M')

    # Localize and convert to UTC without creating extra columns
    df['datetime_local'] = pd.Series(index=df.index, dtype='datetime64[ns, America/Chicago]')

    # Handle non-DST rows (CST)
    non_dst_rows = df['DSTFlag'] == 'N'
    df.loc[non_dst_rows, 'datetime_local'] = (
        dt_combined[non_dst_rows]
        .dt.tz_localize(texas_tz, ambiguous=False)
    )

    # Handle DST rows (CDT)
    dst_rows = df['DSTFlag'] == 'Y'
    df.loc[dst_rows, 'datetime_local'] = (
        dt_combined[dst_rows]
        .dt.tz_localize(texas_tz, ambiguous=True)
    )
    return df


def parse_date_from_csv_filename(fname):
    # E.g.
    # str = 'cdr.00014837.0000000000000000.20190101.003000.LFMODWEATHERNP3565.csv'

    year_month_day = fname.split('.')[3] # 20190101
    time_of_day = fname.split('.')[4] # 003000

    combined_str = year_month_day + time_of_day

    parsed_date = pd.to_datetime(combined_str, format='%Y%m%d%H%M%S')
    return parsed_date

def process_one_weather_csv(path):
    zones = [
        "Coast",
        "East",
        "FarWest",
        "North",
        "NorthCentral",
        "SouthCentral",
        "Southern",
        "West",
    ]

    zone = zones[0]

    df = pd.read_csv(path)
    prediction_time = parse_date_from_csv_filename(path.split('/')[-1])
    # Filter to only include the active prediction model.
    only_in_use = df[df['InUseFlag'] == 'Y']

    with_local_times = convert_to_datetimes(only_in_use).reset_index(drop=True)

    pivoted = pivot_weather_columns(with_local_times)
    pivoted.insert(0, 'PredictionEnd', [prediction_time.floor('D') + pd.Timedelta(days=7)])
    pivoted.insert(0, 'PredictionStart', [prediction_time.floor('D')])
    pivoted.insert(0, 'PredictionTime', [prediction_time])

    pivoted.insert(0, 'Zone', [zone])

    return pivoted



def pivot_weather_columns(df):
    """
    We have additional data we would like to acquire: Seven-Day Load Forecast by Weather Zone.
        This data needs additional processing, we want it as follows:
        Each row has the hourly forecast for the next seven days at each timestep, so we should have 7*24 columns. (Attached is a sketch of the desired dataframe)
        You can have one CSV file for each zone.
        Same timeframe as the rest, 2019-2023.
    """
    zone = "Coast"

    # InUseFlag is kinda a hack here because we're pivoting all the rows to columns.
    reshaped_df = df.pivot_table(index='InUseFlag', columns=df.index, values=zone)

    # Rename columns to reflect the horizon (e.g., Day 1, Day 2, etc.)
    reshaped_df.columns = [
        f'T+{col+1}' for col in reshaped_df.columns
    ]

    # Reset index to make it a flat DataFrame
    reshaped_df.reset_index(drop=True, inplace=True)
    return reshaped_df
