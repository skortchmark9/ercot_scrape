from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from transform import load_settlement_points

def load_lmp_data(filepath):
    """
    Read csv where rows are timestemps, columns are nodes, and values are LMPs
    """
    ercot_data = pd.read_csv(filepath)

    _, sp_to_bus = load_settlement_points()

    without_nans = ercot_data.dropna(axis='columns')

    # Drop datetime column
    without_nans.drop(columns=['datetime_local'], inplace=True)

    # Filter out any buses that are not settlement points
    columns = [c for c in without_nans.columns if c not in sp_to_bus]
    only_sps = without_nans.drop(columns=columns)

    return only_sps


def optimize_battery_placement(node_name, node_lmp):
    """
    Solve the optimization problem to maximize profit from battery operations.

    Returns:
    - Optimal profit and decision variables.
    """
    H = len(node_lmp)
    max_battery_size = 100
    max_charge_rate = 50
    efficiency = 0.95
    state_of_charge_initial = 50

    # Initialize model
    model = Model("Battery_Profit_Optimization")
    # Reduce noise
    model.setParam( 'OutputFlag', False )


    # Decision variables
    charge = model.addVars(H, lb=0, ub=max_charge_rate, name="Charge (MW)")
    discharge = model.addVars(H, lb=0, ub=max_charge_rate, name="Discharge (MW)")
    state_of_charge = model.addVars(H, lb=0, ub=max_battery_size, name="StateOfCharge (MW)")

    # Objective: Maximize profit
    model.setObjective(
        quicksum(
            node_lmp[t] * discharge[t] * efficiency - node_lmp[t] * charge[t]
            for t in range(H)
        ),
        GRB.MAXIMIZE
    )

    # Constraints
    # State of charge dynamics
    for t in range(H - 1):  # Exclude the last time step
        model.addConstr(
            state_of_charge[t + 1] == state_of_charge[t] + efficiency * charge[t] - discharge[t],
            name=f"StateOfChargeDynamics_Time{t}"
        )

    # Initial state of charge
    model.addConstr(state_of_charge[0] == state_of_charge_initial, name=f"InitialState_Node")

    # Capacity constraints
    for t in range(H):
        model.addConstr(state_of_charge[t] <= max_battery_size, name=f"CapacityLimit_Time{t}")

    # Charge/discharge limits
    for t in range(H):
        # @TODO: Should these account for efficiency?
        model.addConstr(charge[t] <= max_charge_rate, name=f"ChargeLimit_Time{t}")
        model.addConstr(discharge[t] <= max_charge_rate, name=f"DischargeLimit_Time{t}")

    # Optimize
    model.optimize()

    # Results
    if model.status == GRB.OPTIMAL:
        profit = model.objVal
        charge_schedule = [charge[t].x for t in range(H)]
        discharge_schedule = [discharge[t].x for t in range(H)]
        soc_schedule = [state_of_charge[t].x for t in range(H)]
        profit_timestep = [
            node_lmp[t] * discharge_schedule[t] * efficiency - node_lmp[t] * charge_schedule[t]
            for t in range(H)
        ]
        return {
            "node_name": node_name,
            "node_lmp": node_lmp,
            "total_profit": profit,
            'profit_timestep': profit_timestep,
            "charge_schedule": charge_schedule,
            "discharge_schedule": discharge_schedule,
            "soc_schedule": soc_schedule,
        }
    else:
        model.computeIIS()
        model.write("infeasible.ilp")
        raise Exception("Optimization failed!")

def display_node_results(results):
    """
    Create a dataframe to display the results for a specific node.

    Each parameter is hourly over the course of the year
    """
    df = pd.DataFrame({
        'Charge (MW)': results['charge_schedule'],
        'Discharge (MW)': results['discharge_schedule'],
        'State of Charge (MWh)': results['soc_schedule'],
        'LMP ($/MWh)': results['node_lmp'],
        'Profit Timestep': results['profit_timestep'],
    })

    df['Cumulative Profit'] = df['Profit Timestep'].cumsum()
    print(df)

    return df

def plot_multi_node_results(all_results, nodes):
    """
    Display two plots with a shared time (x) axis. The first should have
    the first should have the LMP prices, and the last should
    have the cumulative profit. Each trace on the plots represents a single
    node.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for node in nodes:
        df = display_node_results(all_results[node])

        # Convert year hours to datetime
        x_times = pd.date_range(start='1/1/2023', periods=len(df), freq='h')

        # Plot LMP prices. Convert hours to datetime        
        axs[0].plot(x_times, df['LMP ($/MWh)'], label=f'{node} LMP')
        
        # Plot cumulative profit
        axs[1].plot(x_times, df['Cumulative Profit'], label=f'{node} Cumulative Profit')

    axs[0].set_ylabel('LMP ($/MWh)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    axs[1].set_ylabel('Cumulative Profit ($)')
    axs[1].set_xlabel('Time (hours)')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_single_node_results(results):
    """
    Display three plots with a shared time (x) axis. The first should have
    state of charge, the second should have the LMP prices, and the last should
    have the cumulative profit.
    """
    df = display_node_results(results)

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Plot state of charge
    axs[0].plot(df.index, df['Discharge (MW)'], label='Discharge (MW)')
    axs[0].set_ylabel('Discharge (MWh)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Plot LMP prices
    axs[1].plot(df.index, df['LMP ($/MWh)'], label='LMP ($/MWh)', color='orange')
    axs[1].set_ylabel('LMP ($/MWh)')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Plot cumulative profit
    axs[2].plot(df.index, df['Cumulative Profit'], label='Cumulative Profit', color='green')
    axs[2].set_ylabel('Cumulative Profit ($)')
    axs[2].set_xlabel('Time (hours)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    

def top_across_all_yrs(all_data):
    """
    Find the top 5 most profitable nodes across all years.
    """
    counter = Counter()
    for yr_data in all_data:
        for node, results in yr_data.items():
            counter[node] += results['total_profit']

    return counter

def find_best_battery_locations(lmps):
    """
    Find the optimal battery placement for each node in the ERCOT grid.
    """
    all_results = {}
    counter = Counter()
    start = time.time()

    processed = 0
    # In parallel, iterate over each node and optimize, saving results.
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(optimize_battery_placement, node, node_lmp)
            for node, node_lmp in lmps.items()
        ]
        
        for future in as_completed(futures):
            try:
                results = future.result()
                node = results['node_name']
                all_results[node] = results
                counter[node] = results['total_profit']
            except Exception as e:
                print(f'Error occurred: {e}')

            processed += 1
            if processed % 10 == 0:
                print(f"Got results for ({processed}/{len(lmps.columns) - 1})")
                print(f"Time elapsed: {time.time() - start:.2f} seconds")

    return counter, all_results


def save_results(all_results, name):
    """
    Save the results to pickle dump.
    """
    dir = 'results'
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(all_results, f)

def load_results(name):
    """
    Load the results from pickle dump.
    """
    with open(f'results/{name}.pkl', 'rb') as f:
        return pickle.load(f)

def optimize_across_years():
    # years = (2019, 2020, 2021, 2022, 2023)
    years = (2022, 2023)
    for year in years:
        t0 = time.time()
        lmp_file = f"data_dir/dam_lmps_by_year/dam_lmp-{year}.csv"
        lmps = load_lmp_data(lmp_file)
        t1 = time.time()
        print(f"Loaded {year} data in: {t1 - t0:.2f} seconds")

        c, a = find_best_battery_locations(lmps)
        print(f"5 most profitable locations in {year}")
        print(c.most_common(5))
        t2 = time.time()
        print(f"Optimized across {year} data in: {t2 - t1:.2f} seconds")

        save_results(a, str(year))
        t3 = time.time()
        print(f"Saved results for {year} in: {t3 - t2:.2f} seconds")

def main():
    # Load dataset
    ercot_filepath = "data_dir/dam_lmps_by_year/dam_lmp-2019.csv"
    lmp = load_lmp_data(ercot_filepath)

    node = 'AMI_AMISTAG1'

    # Optimize
    results = optimize_battery_placement(node, lmp[node])

    # Display results
    print(f"Optimal Profit for Node {node}: ${results['total_profit']:.2f}")
    display_node_results(results)


if __name__ == "__main__":
    main()