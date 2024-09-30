"""

Our goal for each dataset is to have a datetime column that serves as the index.

LDF:


"""
import pandas as pd
import pytz



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
    df['datetime_local'] = pd.Series(index=df.index, dtype='datetime64[ns]')

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


def process_weather(df):
    """
    We have additional data we would like to acquire: Seven-Day Load Forecast by Weather Zone.
        This data needs additional processing, we want it as follows:
        Each row has the hourly forecast for the next seven days at each timestep, so we should have 7*24 columns. (Attached is a sketch of the desired dataframe)
        You can have one CSV file for each zone.
        Same timeframe as the rest, 2019-2023.
    """
    zone = 'Coast'
    # Step 1: Create a new column that represents the horizon (e.g., 0-hour, 24-hour, etc.)
    df['Horizon'] = df.groupby(['datetime_local']).cumcount()

    # Step 2: Pivot the DataFrame so that the horizons become columns
    pivoted_df = df.pivot(index=['datetime_local'], columns='Horizon', values=zone)

    # Optional: Rename the columns to reflect the horizon
    # pivoted_df.columns = [f'Coast_Horizon_Day{col + 1}' for i, col in enumerate(pivoted_df.columns)]

    # Reset index to make it a flat DataFrame
    pivoted_df.reset_index(inplace=True)
    return pivoted_df
