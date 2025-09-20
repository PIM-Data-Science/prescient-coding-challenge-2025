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

# instrument data
df_bonds = pd.read_csv("C:\\Users\\bmahl\\Downloads\\prescient2\\prescient-coding-challenge-2025\\data\\data_bonds.csv")
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv("C:\\Users\\bmahl\\Downloads\\prescient2\\prescient-coding-challenge-2025\\data\\data_albi.csv")
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv("C:\\Users\\bmahl\\Downloads\\prescient2\\prescient-coding-challenge-2025\\data\\data_macro.csv")
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

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

# This cell contains a sample solution
# You are not restricted to the choice of signal, or the portfolio optimisation used to generate weights
# You may modify anything within this cell as long as it produces a weight matrix in the required form, and the solution does not violate any of the rules

# static data for optimisation and signal generation
n_days = df_bonds['bond_code'].nunique()
prev_weights = [0.1]*10
p_active_md = 1.2 # this can be set to your own limit, as long as the portfolio is capped at 1.5 on any given day
weight_bounds = (0.0, 0.2)

for i in range(len(df_signals)):

    print('---> doing', df_signals.loc[i, 'datestamp'])    

    # this iterations training set
    df_train_bonds = df_bonds[df_bonds['datestamp']<df_signals.loc[i, 'datestamp']].copy()
    df_train_albi = df_albi[df_albi['datestamp']<df_signals.loc[i, 'datestamp']].copy()
    df_train_macro = df_macro[df_macro['datestamp']<df_signals.loc[i, 'datestamp']].copy()

    # this iterations test set
    df_test_bonds = df_bonds[df_bonds['datestamp']>=df_signals.loc[i, 'datestamp']].copy()
    df_test_albi = df_albi[df_albi['datestamp']>=df_signals.loc[i, 'datestamp']].copy()
    df_test_macro = df_macro[df_macro['datestamp']>=df_signals.loc[i, 'datestamp']].copy()

    p_albi_md = df_train_albi['modified_duration'].tail(1)

    # feature engineering
    df_train_macro['steepness'] = df_train_macro['us_10y'] - df_train_macro['us_2y'] 
    df_train_bonds['md_per_conv'] = df_train_bonds.groupby(['bond_code'])['return'].transform(lambda x: x.rolling(window=n_days).mean()) * df_train_bonds['convexity'] / df_train_bonds['modified_duration']
    df_train_bonds = df_train_bonds.merge(df_train_macro, how='left', on = 'datestamp')
    df_train_bonds['yield_ma'] = df_train_bonds.groupby('bond_code')['yield'].transform(lambda x: x.rolling(window=n_days).mean()) #Yield momentum signal
    df_train_bonds['signal'] = df_train_bonds['yield'] - df_train_bonds['yield_ma']
    df_train_bonds_current = df_train_bonds[df_train_bonds['datestamp'] == df_train_bonds['datestamp'].max()]
    
    # optimisation objective
    def objective(weights, signal, prev_weights, turnover_lambda=0.5):
        turnover = np.sum(np.abs(weights - prev_weights))
        return -(np.dot(weights, signal) - turnover_lambda * turnover)
        
    # Duration constraints
    def duration_constraint(weights, durations_today):
        port_duration = np.dot(weights, durations_today)
        return [100*(port_duration - (p_albi_md - p_active_md)), 100*((p_albi_md + p_active_md) - port_duration)]
    
    # Optimization setup
    turnover_lambda = 0.5
    bounds = [weight_bounds] * 10
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'])[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'])[1]}
    ]
    
    result = minimize(objective, prev_weights, args=(df_train_bonds_current['signal'], prev_weights, turnover_lambda), bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x if result.success else prev_weights
    weight_matrix_tmp = pd.DataFrame({'bond_code': df_train_bonds_current['bond_code'],
                                      'weight': optimal_weights,
                                      'datestamp': df_signals.loc[i, 'datestamp']})
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp])

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

