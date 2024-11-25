from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np

# Load ERCOT dataset
def load_ercot_data(filepath):
    """
    Load and preprocess ERCOT dataset for optimization.
    The dataset must contain:
    - LMP (Locational Marginal Prices): A matrix with shape (N, H), where
      N is the number of nodes and H is the number of time steps.
    """
    ercot_data = pd.read_csv(filepath)
    # Assuming the dataset has columns ['Node', 'Time', 'LMP']
    # Pivot to create a matrix of LMPs
    lmp_matrix = ercot_data.pivot(index='Node', columns='Time', values='LMP').values
    return lmp_matrix

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
    N, H = LMP.shape
    return {
        'N': N,  # Number of nodes
        'H': H,  # Number of time periods
        'E_max': np.array([default_battery_capacity] * N),  # Max battery size per node
        'P_max': np.array([50] * N),  # Max charge/discharge rate per node
        'eta': 0.95,  # Battery round-trip efficiency
        'S_initial': np.array([50] * N),  # Initial state of charge
        'total_battery_budget': total_battery_budget,  # Total capacity budget
        'LMP': LMP,  # Locational Marginal Prices
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
    E_max = params['E_max']
    P_max = params['P_max']
    eta = params['eta']
    S_initial = params['S_initial']
    LMP = params['LMP']
    total_battery_budget = params['total_battery_budget']

    # Initialize model
    model = Model("Battery_Profit_Optimization")

    # Decision variables
    c = model.addVars(N, H, lb=0, ub=P_max[0], name="Charge")  # Charging (MW)
    d = model.addVars(N, H, lb=0, ub=P_max[0], name="Discharge")  # Discharging (MW)
    s = model.addVars(N, H, lb=0, ub=E_max[0], name="StateOfCharge")  # State of charge (MWh)
    z = model.addVars(N, vtype=GRB.BINARY, name="BatteryPlacement")  # Battery placement
    E_max_decision = model.addVars(N, lb=0, ub=E_max[0], name="BatterySize")  # Battery size

    # Objective: Maximize profit
    model.setObjective(
        quicksum(
            z[n] * (LMP[n, t] * d[n, t] - LMP[n, t] * c[n, t])
            for n in range(N) for t in range(H)
        ),
        GRB.MAXIMIZE
    )

    # Constraints
    # State of charge dynamics
    for n in range(N):
        for t in range(H - 1):  # Exclude the last time step
            model.addConstr(
                s[n, t + 1] == s[n, t] + eta * c[n, t] - d[n, t] / eta,
                name=f"StateOfChargeDynamics_Node{n}_Time{t}"
            )

    # Initial state of charge
    for n in range(N):
        model.addConstr(s[n, 0] == S_initial[n], name=f"InitialState_Node{n}")

    # Capacity constraints
    for n in range(N):
        for t in range(H):
            model.addConstr(s[n, t] <= E_max_decision[n] * z[n], name=f"CapacityLimit_Node{n}_Time{t}")

    # Charge/discharge limits
    for n in range(N):
        for t in range(H):
            model.addConstr(c[n, t] <= P_max[n] * z[n], name=f"ChargeLimit_Node{n}_Time{t}")
            model.addConstr(d[n, t] <= P_max[n] * z[n], name=f"DischargeLimit_Node{n}_Time{t}")

    # Total battery size budget
    model.addConstr(quicksum(E_max_decision[n] * z[n] for n in range(N)) <= total_battery_budget,
                    name="BatterySizeBudget")

    # Optimize
    model.optimize()

    # Results
    if model.status == GRB.OPTIMAL:
        profit = model.objVal
        placement = {n: z[n].x for n in range(N)}
        charge_schedule = {n: [c[n, t].x for t in range(H)] for n in range(N)}
        discharge_schedule = {n: [d[n, t].x for t in range(H)] for n in range(N)}
        soc_schedule = {n: [s[n, t].x for t in range(H)] for n in range(N)}

        return {
            "profit": profit,
            "placement": placement,
            "charge_schedule": charge_schedule,
            "discharge_schedule": discharge_schedule,
            "soc_schedule": soc_schedule,
        }
    else:
        raise Exception("Optimization failed!")

# Main execution
if __name__ == "__main__":
    # Load dataset
    ercot_filepath = "ercot_dataset.csv"  # Replace with the actual file path
    LMP = load_ercot_data(ercot_filepath)

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