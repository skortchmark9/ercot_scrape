"""

Our goal for each dataset is to have a datetime column that serves as the index.

"""
from collections import defaultdict
import time
import os
import pandas as pd
import pytz
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from optimization import load_results


pd.options.mode.copy_on_write = True 

WEATHER_ZONES = [
    "Coast",
    "East",
    "FarWest",
    "North",
    "NorthCentral",
    "SouthCentral",
    "Southern",
    "West",
]

# Set timezone to Central Prevailing Time (ERCOT), which is America/Chicago (CST/CDT)
texas_tz = pytz.timezone('America/Chicago')

def convert_interval_ending_to_datetimes(df):
    """Assuming 5minute intervals"""
    dt_combined = pd.to_datetime(df['INTERVAL_ENDING'], format='%m/%d/%Y %H:%M')
    dt_combined = dt_combined - pd.Timedelta(5, 'min')
    return dt_combined

def convert_hour_date_to_datetimes(df):
    keys_lower = [k.lower() for k in df.keys()]
    hour_keys = [k for k in df.keys() if 'hour' in k.lower()]
    date_keys = [k for k in df.keys() if 'date' in k.lower()]
    if len(hour_keys) < 1:
        raise Exception(f"Couldn't find hour keys in {df.keys()}")
    if len(date_keys) < 1:
        raise Exception(f"Couldn't find date keys in {df.keys()}")
    hour_key = hour_keys[0]
    date_key = date_keys[0]

    # Adjust from HourEnding to HourBeginning so the parser won't choke
    hour_adjusted = (df[hour_key].str.split(':').str[0].astype(int) - 1).astype(str) + ":00"
    df['HourBeginning'] = hour_adjusted        
    # Now convert to datetime
    dt_combined = pd.to_datetime(df[date_key] + ' ' + df['HourBeginning'], format='%m/%d/%Y %H:%M')
    del df['HourBeginning']
    return dt_combined

def convert_to_datetimes(df):
    """
    Convert DeliveryDate and HourEnding columns to a single UTC seconds column.
    Args:
    - df (pd.DataFrame): Input dataframe with 'DeliveryDate', hour_key, and 'DSTFlag'.
    
    Returns:
    - df (pd.DataFrame): Dataframe with a new 'utc_seconds' column.
    """
    keys_lower = [k.lower() for k in df.keys()]
    if 'interval_ending' in keys_lower:
        dt_combined = convert_interval_ending_to_datetimes(df)
    else:
        dt_combined = convert_hour_date_to_datetimes(df)

    # Localize and convert to UTC without creating extra columns
    df['datetime_local'] = pd.Series(index=df.index, dtype='datetime64[ns, America/Chicago]')

    # Handle non-DST rows (CST)
    non_dst_rows = df['DSTFlag'] == 'N'
    df.loc[non_dst_rows, 'datetime_local'] = (
        dt_combined[non_dst_rows]
        .dt.tz_localize(texas_tz, ambiguous=False, nonexistent='shift_forward')
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


## WEATHER

def process_one_weather_csv(path):
    df = pd.read_csv(path)
    prediction_time = parse_date_from_csv_filename(path.split('/')[-1])
    # Filter to only include the active prediction model.
    only_in_use = df[df['InUseFlag'] == 'Y']

    with_local_times = convert_to_datetimes(only_in_use).reset_index(drop=True)

    dfs = []
    for zone in WEATHER_ZONES:
        pivoted = pivot_weather_columns(with_local_times, zone)
        pivoted.insert(0, 'PredictionEnd', [prediction_time.floor('D') + pd.Timedelta(days=7)])
        pivoted.insert(0, 'PredictionStart', [prediction_time.floor('D')])
        pivoted.insert(0, 'PredictionTime', [prediction_time])
        pivoted.insert(0, 'Zone', [zone])
        dfs.append((zone, pivoted))
    return dfs


def pivot_weather_columns(df, zone):
    """
    We have additional data we would like to acquire: Seven-Day Load Forecast by Weather Zone.
        This data needs additional processing, we want it as follows:
        Each row has the hourly forecast for the next seven days at each timestep, so we should have 7*24 columns. (Attached is a sketch of the desired dataframe)
        You can have one CSV file for each zone.
        Same timeframe as the rest, 2019-2023.
    """
    # InUseFlag is kinda a hack here because we're pivoting all the rows to columns.
    reshaped_df = df.pivot_table(index='InUseFlag', columns=df.index, values=zone)

    # Rename columns to reflect the horizon (e.g., Day 1, Day 2, etc.)
    reshaped_df.columns = [
        f'{col:02d}-{col+1:02d}H' for col in reshaped_df.columns
    ]

    # Reset index to make it a flat DataFrame
    reshaped_df.reset_index(drop=True, inplace=True)
    return reshaped_df

def process_all_weather_csvs(input_dir, output_dir='weather_by_zone'):
    start = time.time()
    csv_files = sorted(glob.glob(f'{input_dir}/*.csv'))

    by_zone = {}
    for zone in WEATHER_ZONES:
        by_zone[zone] = []

    # Loop over all CSV files and read them into pandas dataframes
    for i, file in enumerate(csv_files):
        if i % 10 == 0:
            print(f'processing file {i}/{len(csv_files)}')
        for zone, df in process_one_weather_csv(file):
            by_zone[zone].append(df)

    t1 = time.time()
    print(f"Read all csvs in {t1 - start}")

    # Concatenate into one df per zone
    merged_by_zone = {}
    for zone, dfs in by_zone.items():
        merged_by_zone[zone] = pd.concat(dfs, ignore_index=True)

    t2 = time.time()
    print(f"Concatenated in in {t2 - t1}")

    os.makedirs(output_dir, exist_ok=True)
    for zone, merged_df in merged_by_zone.items():
        merged_df.to_csv(output_dir + '/' + zone + '.csv', index=False)

    t3 = time.time()
    print(f"Wrote to disk in {t3 - t2}")


## DAM LMP

def process_one_dam_lmp(path):
    """
    DAM LMP:
        Could you pivot the CSV file to have rows as datetime, columns as the bus, and cell values as the LMP? This would make the dataframe easier to read. And if yo could make it yearly CSVs so the file size can be manageable. (Attached is a sketch of the desired dataframe)
        Same as LDF, merge DeliveryDate and HourEnding to have datetime column.
        DSTFlag is for Daylight Saving Time repeated values.
    """
    df = pd.read_csv(path)

    with_local_times = convert_to_datetimes(df).reset_index(drop=True)

    # Pivot so busses are columns
    reshaped_df = with_local_times.pivot_table(index='datetime_local', columns=['BusName'], values='LMP')

    # Reset index to make it a flat DataFrame
    reshaped_df.reset_index(inplace=True)
    return reshaped_df

def process_all_dam_lmps(input_dir, output_dir='dam_lmps'):
    t0 = time.time()
    csv_files = sorted(glob.glob(f'{input_dir}/*.csv'))

    # Loop over all CSV files and read them into pandas dataframes

    dfs = []
    for i, file in enumerate(csv_files):
        if i % 10 == 0:
            print(f'processing file {i}/{len(csv_files)}')
        dfs.append(process_one_dam_lmp(file))

    t1 = time.time()
    print(f"Read all csvs in {t1 - t0}")

    # Concatenate into one df per zone
    merged = pd.concat(dfs, ignore_index=True)
    t2 = time.time()
    print(f"Concatenated in in {t2 - t1}")

    os.makedirs(output_dir, exist_ok=True)

    split_by_year(output_dir + '/' + 'dam_lmp.csv', df=merged)
    t3 = time.time()
    print(f"Wrote to disk in {t3 - t2}")

## Works for wind and solar since they are giving prices over last hour
def process_all_historical(input_dir, output_dir='solar'):
    t0 = time.time()
    csv_files = sorted(glob.glob(f'{input_dir}/*.csv'))

    # Loop over all CSV files and read them into pandas dataframes
    dfs = []
    for i, file in enumerate(csv_files):
        if i % 10 == 0:
            print(f'processing file {i}/{len(csv_files)}')

        # Only add the last row from the file.
        df = pd.read_csv(file)
        df2 = df.tail(1)
        with_local_times = convert_to_datetimes(df2).reset_index(drop=True)

        dfs.append(with_local_times)

    t1 = time.time()
    print(f"Read all csvs in {t1 - t0}")

    merged = pd.concat(dfs, ignore_index=True)
    t2 = time.time()
    print(f"Concatenated in in {t2 - t1}")

    os.makedirs(output_dir, exist_ok=True)

    split_by_year(output_dir + '/' + output_dir + '.csv', df=merged)
    t3 = time.time()
    print(f"Wrote to disk in {t3 - t2}")


def split_by_year(path, df=None):
    filename = path.split('/')[-1]

    if df is None:
        t0 = time.time()
        df = pd.read_csv(filename)
        t1 = time.time()
        print(f"Read df in {t1-t0}s")

    if 'datetime_local' not in df.keys():
        df = convert_to_datetimes(df)

    split_by_year = {year: group for year, group in df.groupby(df['datetime_local'].dt.year)}

    # Verify the result
    for year, subset in split_by_year.items():
        new_filename = filename.replace('.csv', f'-{year}.csv')
        subset.to_csv(path.replace(filename, new_filename), index=False)



def load_settlement_points():
    """Settlement Points (SPs) are the buses that have distinct prices
    (ercot settle prices at these buses), while the other buse prices
    are based on the nearest SP.
    
    This mapping allows us to find the list of nodes which are associated
    with a given SP, and vice versa."""
    path = 'static_data/Settlement_Points_11112024_174625.csv'
    df = pd.read_csv(path)

    bus_to_sp = {}
    sp_to_bus = defaultdict(list)

    for _, row in df.iterrows():
        bus_to_sp[row['ELECTRICAL_BUS']] = row['PSSE_BUS_NAME']
        sp_to_bus[row['PSSE_BUS_NAME']].append(row['ELECTRICAL_BUS'])

    return bus_to_sp, sp_to_bus

### Realtime LMPs

def process_one_realtime_lmp(path):
    df = pd.read_csv(path)
    pivoted = df.pivot_table(index='SCEDTimestamp', columns='SettlementPoint', values='LMP')

    pivoted['datetime_local'] = pivoted.index.map(lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S'))

    # Reorder columns to put datetime_local at the front
    # pivoted = pivoted.reset_index(drop=True)
    d = pivoted.to_dict('records')[0]
    return d

def process_one_yr_realtime_lmp(year, input_dir='realtime_lmps', output_dir='realtime_lmps_by_year'):
    year = int(year)
    t0 = time.time()
    csv_files = glob.glob(f'{input_dir}/*_{year}*.csv')

    dicts = []
    futures = []
    processed = 0
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_one_realtime_lmp, file)
            for file in csv_files
        ]

        for future in as_completed(futures):
            try:
                the_dict = future.result()
                # Can happen due to 2019 as a timestamp, getting picked up by glob
                if the_dict['datetime_local'].year != year:
                    continue

                dicts.append(the_dict)
                processed += 1
                if processed % 10 == 0:
                    print(f"Got results for ({processed}/{len(csv_files)})")
            except Exception as e:
                print(f'Error occurred: {e}')

    t1 = time.time()
    merged = pd.DataFrame(dicts)
    merged.sort_values('datetime_local', inplace=True)
    merged.to_csv(f"{output_dir}/realtime_lmp_{year}.csv", index=False)
    print(f"merged, sorted, wrote in {t1 - t0}s")
    return merged

def avg_hourly_prices(df):
    df['datetime_local'] = pd.to_datetime(df['datetime_local'], format='ISO8601')
    df['hour'] = df['datetime_local'].dt.hour
    df['day'] = df['datetime_local'].dt.day
    df['month'] = df['datetime_local'].dt.month
    df['year'] = df['datetime_local'].dt.year

    hourly_avg = df.groupby(['year', 'month', 'day', 'hour']).mean()
    hourly_avg.reset_index(inplace=True)
    hourly_avg['datetime_local'] = pd.to_datetime(hourly_avg[['year', 'month', 'day', 'hour']])

    result = hourly_avg.drop(columns=['year', 'month', 'day', 'hour'])
    dna = result.dropna(axis='columns')
    return dna

def compare_dam_with_actual(opt_results, actual, node):
    """
    Compare the actual LMPs with the DAM predictions.
    """

    opt_result = opt_results[node]
    realtime_prices = actual[node]

    # Plot the dam and actual against the time
    x_times = range(365 * 24 - 1)

    expected_net_profit = 0
    actual_net_profit = 0

    actual_profit_timesteps = []

    for t in x_times:
        charge_amt = opt_result['charge_schedule'][t]
        discharge_amt = opt_result['discharge_schedule'][t]
        expected_profit_timestep = opt_result['profit_timestep'][t]
        actual_profit_timestep = realtime_prices[t] * (discharge_amt - charge_amt)
        expected_net_profit += expected_profit_timestep
        actual_net_profit += actual_profit_timestep
        actual_profit_timesteps.append(actual_profit_timestep)
        
    print(f"Expected net profit: {expected_net_profit}")
    print(f"Actual net profit: {actual_net_profit}")
    print(f"Net profit difference: {abs(actual_net_profit - expected_net_profit)}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Prices', 'Profits'))

    fig.add_trace(go.Scatter(
        x=list(x_times), 
        y=opt_result['node_lmp'], 
        mode='lines', 
        name='DAM Price',
        line=dict(color='blue')
    ),  row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(x_times), 
        y=realtime_prices, 
        mode='lines', 
        name='Actual Price',
        line=dict(color='red')
    ),  row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(x_times), 
        y=opt_result['profit_timestep'], 
        mode='lines', 
        name='Expected Profit',
        line=dict(color='blue')
    ),  row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(x_times), 
        y=actual_profit_timesteps, 
        mode='lines', 
        name='Actual Profit',
        line=dict(color='red')
    ),  row=2, col=1)

    fig.update_layout(
        title=f'Expected {expected_net_profit:,.2f} vs Actual Profit {actual_net_profit:,.2f}',
        xaxis_title='Time (hours)',
        yaxis_title='Profit',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit", row=2, col=1)

    fig.show()


def compare_dam_with_actual_profits(year):
    """
    Compare the actual LMPs with the DAM predictions.
    """
    actual = pd.read_csv(f'realtime_lmps_by_year/realtime_lmp_{year}_hourly.csv')
    results = load_results(year)

    nodes = set(results.keys()).intersection(set(actual.keys()))

    expected_net_profits = []
    actual_net_profits = []
    for node in nodes:
        opt_result = results[node]
        realtime_prices = actual[node]

        # Plot the dam and actual against the time
        x_times = range(8595)

        expected_net_profit = 0
        actual_net_profit = 0

        try:
            for t in x_times:
                charge_amt = opt_result['charge_schedule'][t]
                discharge_amt = opt_result['discharge_schedule'][t]
                expected_profit_timestep = opt_result['profit_timestep'][t]
                actual_profit_timestep = realtime_prices[t] * (discharge_amt - charge_amt)
                expected_net_profit += expected_profit_timestep
                actual_net_profit += actual_profit_timestep

            expected_net_profits.append(expected_net_profit)
            actual_net_profits.append(actual_net_profit)
        except Exception as e:
            print(f"Error occurred for node {node}: {e}")
            continue

    expected_net_profits = np.array(expected_net_profits)
    actual_net_profits = np.array(actual_net_profits)
    print(f"Across {len(nodes)} nodes in {year}")
    print(f"\tMean optimal profit: {np.mean(expected_net_profits):,.2f}")
    print(f"\tMean actual profit: {np.mean(actual_net_profits):,.2f}")

    diff = abs(np.mean((actual_net_profits - expected_net_profits) / expected_net_profits) * 100)
    print(f"\tActual profit as a percentage of optimal {diff:.2f}%")