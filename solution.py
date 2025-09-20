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
df_bonds = pd.read_csv('data/data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv('data/data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv('data/data_macro.csv')
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
n_days = 10
prev_weights = [0.1]*10
p_active_md = 1.5  # Increased to max allowed for more allocation flexibility
weight_bounds = (0.0, 0.2)

for i in range(len(df_signals)):

    # EVOLVE-BLOCK-START
    print('---> doing', df_signals.loc[i, 'datestamp'])

    # this iterations training set
    df_train_bonds = df_bonds[df_bonds['datestamp']<df_signals.loc[i, 'datestamp']].copy()
    df_train_albi = df_albi[df_albi['datestamp']<df_signals.loc[i, 'datestamp']].copy()
    df_train_macro = df_macro[df_macro['datestamp']<df_signals.loc[i, 'datestamp']].copy()

    # feature engineering
    current_albi_md = df_train_albi['modified_duration'].iloc[-1] if not df_train_albi.empty else 0.0

    # Get last macro steepness (us_10y - us_2y); default 0 if NaN
    last_macro = df_train_macro[df_train_macro['datestamp'] == df_train_macro['datestamp'].max()]
    us_steep = (last_macro['us_10y'].iloc[0] - last_macro['us_2y'].iloc[0]) if not last_macro.empty else 0.0

    # Blended carry (income + roll-down), 10d momentum, yield rel to 20d avg, macro steep adj, convexity
    df_train_bonds['carry_income'] = df_train_bonds['yield'] / 252
    df_train_bonds['carry_rolldown'] = df_train_bonds['yield'] * df_train_bonds['modified_duration'] / 252
    df_train_bonds['carry'] = (df_train_bonds['carry_income'] + df_train_bonds['carry_rolldown']) / 2
    df_train_bonds['momentum_10d'] = df_train_bonds.groupby('bond_code')['return'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df_train_bonds['yield_rel_avg'] = df_train_bonds.groupby('bond_code')['yield'].transform(lambda x: x / x.rolling(20, min_periods=1).mean())

    df_train_bonds_current = df_train_bonds[df_train_bonds['datestamp'] == df_train_bonds['datestamp'].max()].sort_values('bond_code').reset_index(drop=True).fillna(0)
    df_train_bonds_current['macro_steep_adj'] = us_steep * df_train_bonds_current['modified_duration']

    # Z-score normalize
    for feat in ['carry', 'momentum_10d', 'yield_rel_avg', 'macro_steep_adj', 'convexity']:
        mean_val = df_train_bonds_current[feat].mean()
        std_val = df_train_bonds_current[feat].std()
        df_train_bonds_current[feat + '_z'] = (df_train_bonds_current[feat] - mean_val) / std_val if std_val > 0 else 0

    # Signal: carry (0.35), mom (0.25), rel (0.2), macro (0.1), convexity (0.1) for vol protection
    df_train_bonds_current['signal'] = (
        0.35 * df_train_bonds_current['carry_z'] +
        0.25 * df_train_bonds_current['momentum_10d_z'] +
        0.2 * df_train_bonds_current['yield_rel_avg_z'] +
        0.1 * df_train_bonds_current['macro_steep_adj_z'] +
        0.1 * df_train_bonds_current['convexity_z']
    )

    def objective(weights, signal, prev_weights, turnover_lambda=0.1):
        turnover = np.sum(np.abs(weights - prev_weights)) / 2
        return -(np.dot(weights, signal) - turnover_lambda * turnover)

    def duration_constraint(weights, durations_today, albi_md, active_md_limit):
        port_duration = np.dot(weights, durations_today)
        albi_md_scalar = albi_md.item() if isinstance(albi_md, pd.Series) else albi_md
        return [
            port_duration - (albi_md_scalar - active_md_limit),
            (albi_md_scalar + active_md_limit) - port_duration
        ]

    # Optimization setup: Balanced lambda for signal-following with turnover control
    turnover_lambda = 0.12
    bounds = [weight_bounds] * 10
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'].values, current_albi_md, p_active_md)[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'].values, current_albi_md, p_active_md)[1]}
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

