from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transform import load_settlement_points

def load_lmps_data(filepath):
    """
    Read csv where rows are timestemps, columns are nodes, and values are LMPs
    """
    ercot_data = pd.read_csv(filepath)

    _, sp_to_bus = load_settlement_points()

    without_nans = ercot_data.dropna(axis='columns')

    # Filter out any buses that are not settlement points
    columns = [c for c in without_nans.columns[1:] if c not in sp_to_bus]
    only_sps = without_nans.drop(columns=columns)

    return only_sps

# Parameters
def setup_parameters(LMP, total_battery_budget=500, default_battery_capacity=100):
    """
    Set up optimization parameters.
    Args:
    - LMP: Locational Marginal Prices matrix (N x H).
    - total_battery_budget: Total battery capacity available for placement.
    - default_battery_capacity: Default maximum battery capacity for each node.

    Returns:
    - Parameters dictionary.
    """
    without_timestamps = LMP.drop(columns='datetime_local')
    H, N = without_timestamps.shape
    return {
        'N': N,  # Number of nodes
        'H': H,  # Number of time periods
        'max_battery_size': np.array([default_battery_capacity] * N),
        'max_charge_rate': np.array([50] * N),
        'efficiency': 0.95,  # Battery single-trip efficiency
        'state_of_charge_initial': np.array([50] * N),
        'total_battery_budget': total_battery_budget,
        'LMP': without_timestamps,
    }

# Optimization model
def optimize_battery_placement(params):
    """
    Solve the optimization problem to maximize profit from battery operations.

    Args:
    - params: Dictionary of parameters.

    Returns:
    - Optimal profit and decision variables.
    """
    # Extract parameters
    N = params['N']
    H = params['H']
    max_battery_size = params['max_battery_size']
    max_charge_rate = params['max_charge_rate']
    efficiency = params['efficiency']
    state_of_charge_initial = params['state_of_charge_initial']
    LMP = params['LMP']
    total_battery_budget = params['total_battery_budget']

    print(LMP.shape, N, H)
    # Initialize model
    model = Model("Battery_Profit_Optimization")

    # Decision variables
    charge = model.addVars(N, H, lb=0, ub=max_charge_rate[0], name="Charge (MW)")
    discharge = model.addVars(N, H, lb=0, ub=max_charge_rate[0], name="Discharge")  # Discharging (MW)
    state_of_charge = model.addVars(N, H, lb=0, ub=max_battery_size[0], name="StateOfCharge")  # State of charge (MWh)
    placed = model.addVars(N, vtype=GRB.BINARY, name="BatteryPlacement")  # Battery placement
    max_battery_size_decision = model.addVars(N, lb=0, ub=max_battery_size[0], name="BatterySize")  # Battery size

    # Objective: Maximize profit
    model.setObjective(
        quicksum(
            placed[n] * (LMP.iloc[t, n] * discharge[n, t] * efficiency - LMP.iloc[t, n] * charge[n, t])
            for n in range(N) for t in range(H)
        ),
        GRB.MAXIMIZE
    )


    # Constraints
    # State of charge dynamics
    for n in range(N):
        for t in range(H - 1):  # Exclude the last time step
            model.addConstr(
                state_of_charge[n, t + 1] == state_of_charge[n, t] + efficiency * charge[n, t] - discharge[n, t],
                name=f"StateOfChargeDynamics_Node{n}_Time{t}"
            )

    # Initial state of charge
    for n in range(N):
        model.addConstr(state_of_charge[n, 0] == state_of_charge_initial[n], name=f"InitialState_Node{n}")

    # Capacity constraints
    for n in range(N):
        for t in range(H):
            model.addConstr(state_of_charge[n, t] <= max_battery_size_decision[n] * placed[n], name=f"CapacityLimit_Node{n}_Time{t}")

    # Charge/discharge limits
    for n in range(N):
        for t in range(H):
            model.addConstr(charge[n, t] <= max_charge_rate[n] * placed[n], name=f"ChargeLimit_Node{n}_Time{t}")
            model.addConstr(discharge[n, t] <= max_charge_rate[n] * placed[n], name=f"DischargeLimit_Node{n}_Time{t}")

    # Total battery size budget
    model.addConstr(quicksum(max_battery_size_decision[n] * placed[n] for n in range(N)) <= total_battery_budget,
                    name="BatterySizeBudget")

    # Optimize
    model.optimize()

    # Results
    if model.status == GRB.OPTIMAL:
        profit = model.objVal
        placement = {n: placed[n].x for n in range(N)}
        charge_schedule = {n: [charge[n, t].x for t in range(H)] for n in range(N)}
        discharge_schedule = {n: [discharge[n, t].x for t in range(H)] for n in range(N)}
        soc_schedule = {n: [state_of_charge[n, t].x for t in range(H)] for n in range(N)}
        profit_timestep = {
            n: [
                (LMP.iloc[t, n] * discharge_schedule[n][t] * efficiency - LMP.iloc[t, n] * charge_schedule[n][t])
                for t in range(H)]
            for n in range(N)
        }
        (LMP.iloc[t, n] * discharge[n, t] * efficiency - LMP.iloc[t, n] * charge[n, t])
        return {
            "LMP": LMP,
            "total_profit": profit,
            'profit_timestep': profit_timestep,
            "placement": placement,
            "charge_schedule": charge_schedule,
            "discharge_schedule": discharge_schedule,
            "soc_schedule": soc_schedule,
        }
    else:
        model.computeIIS()
        model.write("infeasible.ilp")
        raise Exception("Optimization failed!")

def display_node_results(node_idx, results):
    """
    Create a dataframe to display the results for a specific node.

    Each parameter is hourly over the course of the year
    """
    df = pd.DataFrame({
        'Charge (MW)': results['charge_schedule'][node_idx],
        'Discharge (MW)': results['discharge_schedule'][node_idx],
        'State of Charge (MWh)': results['soc_schedule'][node_idx],
        'LMP ($/MWh)': results['LMP'].iloc[:, node_idx],
        'Profit Timestep': results['profit_timestep'][node_idx],
    })

    df['Cumulative Profit'] = df['Profit Timestep'].cumsum()

    return df


def plot_node_results(params, node_idx, results):
    """
    Create 3 strip charts where time is the x axis, and charge/discharge are a time series,
    price is a time series on a different axis, and state of charge is a time series on a third axis.
    """
    df = display_node_results(params, node_idx, results)
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Charge and discharge
    ax[0].plot(df['Net Charge (MW)'], label='Net Charge (MW)', color='blue')
    ax[0].set_ylabel('MW')
    ax[0].legend()

    # LMP
    ax[1].plot(params['LMP'].iloc[:, node_idx], label='LMP ($/MWh)', color='green')
    ax[1].set_ylabel('$/MWh')

    # State of charge
    ax[2].plot(results['soc_schedule'][node_idx], label='State of Charge (MWh)', color='purple')
    ax[2].set_ylabel('MWh')

    plt.xlabel('Hour')
    plt.suptitle(f'Node {node_idx} Results')
    plt.show()


def main():
    # Load dataset
    ercot_filepath = "data_dir/dam_lmps_by_year/dam_lmp-2019.csv"
    LMP = load_lmps_data(ercot_filepath)

    # Set up parameters
    params = setup_parameters(LMP)

    # Optimize
    results = optimize_battery_placement(params)

    # Display results
    print(f"Optimal Profit: ${results['profit']:.2f}")
    print("\nBattery Placement:")
    for node, placed in results["placement"].items():
        print(f"Node {node}: {'Yes' if placed > 0.5 else 'No'}")

    print("\nCharge Schedule (MW):")
    for node, schedule in results["charge_schedule"].items():
        print(f"Node {node}: {schedule}")

    print("\nDischarge Schedule (MW):")
    for node, schedule in results["discharge_schedule"].items():
        print(f"Node {node}: {schedule}")

    print("\nState of Charge (MWh):")
    for node, schedule in results["soc_schedule"].items():
        print(f"Node {node}: {schedule}")

if __name__ == "__main__":
    main()