import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from optimization import load_results
import pandas as pd


def compare_dam_with_actual(opt_results, actual, node, year='2020'):
    """
    Compare the actual LMPs with the DAM predictions.
    """

    opt_result = opt_results[node]
    realtime_prices = actual[node]

    n_opt = len(opt_result['charge_schedule'])
    n_rt = len(realtime_prices)

    if n_opt != n_rt:
        print('#' * 80)
        print(f"len optimization results {n_opt} != realtime {n_rt}")
    
    x_extent = min(n_opt, n_rt)
    max_possible = 365 * 24 - 1
    if x_extent < max_possible:
        print(f"Lacking data for end of year - missing {max_possible - x_extent} hours.")

    
    # Plot the dam and actual against the time
    print('#' * 80)
    x_times = range(x_extent)

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

    # Map the x_times into datetimes
    x_times = pd.date_range(start=f'1/1/{year}', periods=350 * 24, freq='h')

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
    actual = pd.read_csv(f'data_dir/realtime_lmps_by_year/realtime_lmp_{year}_hourly.csv')
    results = load_results(year)

    nodes = set(results.keys()).intersection(set(actual.keys()))

    expected_net_profits_per_node = {}
    actual_net_profits_per_node = {}
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

            expected_net_profits_per_node[node] = expected_net_profit
            actual_net_profits_per_node[node] = actual_net_profit
        except Exception as e:
            print(f"Error occurred for node {node}: {e}")
            continue

    expected_net_profits = np.array(list(expected_net_profits_per_node.values()))
    actual_net_profits = np.array(list(actual_net_profits_per_node.values()))
    print(f"Across {len(nodes)} nodes in {year}")
    print(f"\tMean optimal profit across all nodes: {np.mean(expected_net_profits):,.2f}")
    print(f"\tMean actual profit all nodes: {np.mean(actual_net_profits):,.2f}")

    diff = abs(np.mean((actual_net_profits - expected_net_profits) / expected_net_profits) * 100)
    print(f"\tMean difference of prices per-node {diff:.2f}%")

    np.argmin(expected_net_profits_per_node)

    # Min and max of optimal and actual prices
    most_profitable_node_expected = max(expected_net_profits_per_node, key=expected_net_profits_per_node.get)
    most_profitable_node_actual = max(actual_net_profits_per_node, key=actual_net_profits_per_node.get)
    print(f"\tExpected most profitable node: {most_profitable_node_expected}")
    print(f"\t\tExpected profit: {expected_net_profits_per_node[most_profitable_node_expected]:,.2f}")
    print(f"\t\tActual profit: {actual_net_profits_per_node[most_profitable_node_expected]:,.2f}")

    print(f"\tActual most profitable node: {most_profitable_node_actual}")
    print(f"\t\tExpected profit: {expected_net_profits_per_node[most_profitable_node_actual]:,.2f}")
    print(f"\t\tActual profit: {actual_net_profits_per_node[most_profitable_node_actual]:,.2f}")
