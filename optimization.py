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


# Optimization model
def optimize_battery_placement(node_lmp):
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