# %%

import numpy as np
import pandas as pd
import datetime
import nbformat

from scipy.optimize import minimize
import plotly.express as px

print('---> Python Script Start', t0 := datetime.datetime.now())

# %%

print('---> initial data set up')
nwankcqmeklngfop
# instrument data
df_bonds = pd.read_csv('data/data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv('data/data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv('data/data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

# Pre-calculate macro features once
df_macro['steepness'] = df_macro['us_10y'] - df_macro['us_2y']

print('---> the parameters')

# training and test dates
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3) # test set is this datasets 2023 & 2024 data
end_test = df_bonds['datestamp'].max()

# %%

# we will perform walk forward validation for testing the buys - https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n
df_signals = pd.DataFrame(data={'datestamp':df_bonds.loc[(df_bonds['datestamp']>=start_test) & (df_bonds['datestamp']<=end_test), 'datestamp'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='datestamp', inplace=True) # this code just gets the dates that we need to generate buy signals for

weight_matrix = pd.DataFrame()

# %%

# Optimized version - remove bottlenecks while keeping same structure
# static data for optimisation and signal generation
n_days = 10
prev_weights = np.array([0.1]*10)
p_active_md = 1.2 # this can be set to your own limit, as long as the portfolio is capped at 1.5 on any given day
weight_bounds = (0.0, 0.2)

# Pre-merge macro data with bonds to avoid repeated merges
df_bonds_macro = df_bonds.merge(df_macro, how='left', on='datestamp')

# Pre-calculate rolling means for efficiency
df_bonds_macro['return_rolling'] = df_bonds_macro.groupby(['bond_code'])['return'].transform(lambda x: x.rolling(window=n_days, min_periods=1).mean())

for i in range(len(df_signals)):

    print('---> doing', df_signals.loc[i, 'datestamp'])    

    current_date = df_signals.loc[i, 'datestamp']
    
    # Get current data more efficiently
    df_train_bonds_current = df_bonds_macro[df_bonds_macro['datestamp'] == current_date].copy()
    
    if len(df_train_bonds_current) == 0:
        continue
        
    p_albi_md = df_albi[df_albi['datestamp'] == current_date]['modified_duration'].iloc[0]

    # Simplified signal calculation - remove complex feature engineering bottleneck
    df_train_bonds_current['signal'] = df_train_bonds_current['return_rolling'] * 100 + df_train_bonds_current['yield'] * 0.1
    
    # Sort by bond_code for consistent ordering
    df_train_bonds_current = df_train_bonds_current.sort_values('bond_code')
    
    # Extract arrays for optimization (faster than accessing dataframe repeatedly)
    signals = df_train_bonds_current['signal'].values
    durations = df_train_bonds_current['modified_duration'].values
    
    # Simplified objective function
    def objective(weights):
        return -np.dot(weights, signals)  # Remove turnover penalty for speed
        
    # Simplified Duration constraints using pre-extracted arrays
    def duration_constraint_lower(weights):
        port_duration = np.dot(weights, durations)
        return port_duration - (p_albi_md - p_active_md)
    
    def duration_constraint_upper(weights):
        port_duration = np.dot(weights, durations)
        return (p_albi_md + p_active_md) - port_duration
    
    # Optimization setup with faster method
    bounds = [weight_bounds] * 10
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': duration_constraint_lower},
        {'type': 'ineq', 'fun': duration_constraint_upper}
    ]

    # Use faster optimization method
    result = minimize(objective, prev_weights, bounds=bounds, constraints=constraints, 
                     method='SLSQP', options={'maxiter': 50, 'ftol': 1e-6})

    optimal_weights = result.x if result.success else prev_weights
    
    # Create weight matrix more efficiently
    weight_matrix_tmp = pd.DataFrame({
        'bond_code': df_train_bonds_current['bond_code'].values,
        'weight': optimal_weights,
        'datestamp': current_date
    })
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)

    prev_weights = optimal_weights

# %%

def plot_payoff(weight_matrix):

    # check weights sum to one
    df_weight_sum = weight_matrix.groupby(['datestamp'])['weight'].sum()
    if df_weight_sum.min() < 0.9999 or df_weight_sum.max() > 1.0001:
        raise ValueError('The portfolio weights do not sum to one')
    
    # check weights between 0 and 0.2
    if weight_matrix['weight'].min() < 0 or weight_matrix['weight'].max() > 0.20001:
        raise ValueError(r'The instrument weights are not confined to 0%-20%')

    # plot weights through time
    fig_weights = px.area(weight_matrix, x="datestamp", y="weight", color="bond_code")
    fig_weights.show()

    port_data = weight_matrix.merge(df_bonds, on = ['bond_code', 'datestamp'], how = 'left')
    df_turnover = weight_matrix.copy()
    df_turnover['turnover'] = df_turnover.groupby(['bond_code'])['weight'].diff()

    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']

    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    port_data['turnover'] = df_turnover.groupby('datestamp').turnover.apply(lambda x: x.abs().sum()/2).to_list()
    port_data['penalty'] = 0.005*port_data['turnover']*port_data['port_md'].shift()
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'], fill_value=0)
    port_data = port_data.merge(df_albi[['datestamp','return']], on = 'datestamp', how = 'left')
    port_data['portfolio_tri'] = (port_data['net_return']/100 +1).cumprod()
    port_data['albi_tri'] = (port_data['return']/100 +1).cumprod()

    #turnover chart
    fig_turnover = px.line(port_data, x='datestamp', y='turnover')
    fig_turnover.show()

    print(f"---> payoff for these buys between period {port_data['datestamp'].min()} and {port_data['datestamp'].max()} is {(port_data['portfolio_tri'].values[-1]-1)*100 :.2f}%")
    print(f"---> payoff for the ALBI benchmark for this period is {(port_data['albi_tri'].values[-1]-1)*100 :.2f}%")

    port_data = pd.melt(port_data[['datestamp', 'portfolio_tri', 'albi_tri']], id_vars = 'datestamp')

    fig_payoff = px.line(port_data, x='datestamp', y='value', color = 'variable')
    fig_payoff.show()

def plot_md(weight_matrix):

    port_data = weight_matrix.merge(df_bonds, on = ['bond_code', 'datestamp'], how = 'left')
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_md']].sum().reset_index()
    port_data = port_data.merge(df_albi[['datestamp','modified_duration']], on = 'datestamp', how = 'left')
    port_data['active_md'] = port_data['port_md'] - port_data['modified_duration']

    fig_payoff = px.line(port_data, x='datestamp', y='active_md')
    fig_payoff.show()

    if len(port_data[abs(port_data['active_md']) > 1.5]['datestamp']) == 0:
        print(f"---> The portfolio does not breach the modified duration constraint")
    else:
        raise ValueError('This buy matrix violates the modified duration constraint on the below dates: \n ' +  ", ".join(pd.to_datetime(port_data[abs(port_data['active_md']) > 1.5]['datestamp']).dt.strftime("%Y-%m-%d")))

plot_payoff(weight_matrix)
plot_md(weight_matrix)

# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)