# %%
import numpy as np
import pandas as pd
import datetime
import nbformat 
from scipy.optimize import minimize
import plotly.express as px

# %%
print('---> Python Script Start', t0 := datetime.datetime.now())

# %%
print('---> initial data set up')

# instrument data
df_bonds = pd.read_csv('/Users/admin/Documents/Prescienct_Hackathon/prescient-coding-challenge-2025/data/data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv('/Users/admin/Documents/Prescienct_Hackathon/prescient-coding-challenge-2025/data/data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv('/Users/admin/Documents/Prescienct_Hackathon/prescient-coding-challenge-2025/data/data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

# %%
df_bonds.head()

# %%
num_unique_bonds = df_bonds['bond_code'].nunique()
print("Number of unique bonds:", num_unique_bonds)

# %%
unique_bonds = df_bonds['bond_name'].unique()
print(unique_bonds)
print("Total unique bond names:", len(unique_bonds))



# %%
import pandas as pd

# Example: your bond_name column
df_bonds['bond_name'] = df_bonds['bond_name'].astype(str)  # ensure it's string

# Extract duration using regex: everything after "Bond "
df_bonds['bond_duration'] = df_bonds['bond_name'].str.extract(r'Bond (.+)')

# Check the result
print(df_bonds[['bond_name', 'bond_duration']].drop_duplicates())


# %%
import pandas as pd

# Ensure bond_name is string
df_bonds['bond_name'] = df_bonds['bond_name'].astype(str)

# Extract duration
df_bonds['bond_duration'] = df_bonds['bond_name'].str.extract(r'Bond (.+)')

# Convert duration to numeric years
def duration_to_years(duration):
    if 'Month' in duration:
        # Convert months to years
        num = int(duration.split()[0])
        return num / 12
    elif 'Year' in duration:
        # Keep years as integer/float
        num = int(duration.split()[0])
        return float(num)
    else:
        return None  # or np.nan

df_bonds['bond_duration_years'] = df_bonds['bond_duration'].apply(duration_to_years)

# Check results
print(df_bonds[['bond_name', 'bond_duration', 'bond_duration_years']].drop_duplicates())


# %%
import matplotlib.pyplot as plt

# Make sure the duration and yield columns exist
# df_bonds['bond_duration_years']  (numeric)
# df_bonds['yield']                (numeric)

plt.figure(figsize=(10,6))
plt.scatter(df_bonds['bond_duration_years'], df_bonds['yield'], color='blue')

plt.xlabel('Bond Duration (Years)')
plt.ylabel('Yield')
plt.title('Yield vs Bond Duration')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

# Scatter plot of bond duration vs modified duration
plt.figure(figsize=(10,6))
plt.scatter(df_bonds['bond_duration_years'], df_bonds['modified_duration'], color='green')

plt.xlabel('Bond Duration (Years)')
plt.ylabel('Modified Duration')
plt.title('Modified Duration vs Bond Duration')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%



# %%
df_albi.head()

# %%
df_macro.head()

# %%
# training and test dates
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3) # test set is this datasets 2023 & 2024 data
end_test = df_bonds['datestamp'].max()
start_test = pd.to_datetime(start_test)
end_test   = pd.to_datetime(end_test)

# %%
# Ensure datestamps are Timestamps
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp'])

# Define start and end test dates as Timestamps
start_test = pd.Timestamp('2023-01-03')
end_test   = df_bonds['datestamp'].max()

# Select the dates for signal generation
df_signals = df_bonds.loc[
    (df_bonds['datestamp'] >= start_test) & 
    (df_bonds['datestamp'] <= end_test), 
    ['datestamp']
].drop_duplicates().reset_index(drop=True)

df_signals.sort_values(by='datestamp', inplace=True)

# Initialize empty weight matrix
weight_matrix = pd.DataFrame()


# %%
df_signals.head()

# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Ensure datestamps are Timestamps
df_signals['datestamp'] = pd.to_datetime(df_signals['datestamp'])
df_bonds['datestamp']   = pd.to_datetime(df_bonds['datestamp'])
df_albi['datestamp']    = pd.to_datetime(df_albi['datestamp'])
df_macro['datestamp']   = pd.to_datetime(df_macro['datestamp'])

# Static parameters
n_days = 10
prev_weights = np.array([0.1]*10)
p_active_md = 1.2
weight_bounds = (0.0, 0.2)
weight_matrix = pd.DataFrame()

for i in range(len(df_signals)):
    cutoff_date = pd.Timestamp(df_signals.loc[i, 'datestamp'])
    print('---> processing', cutoff_date)

    # Training sets
    df_train_bonds = df_bonds[df_bonds['datestamp'] < cutoff_date].copy()
    df_train_albi  = df_albi[df_albi['datestamp'] < cutoff_date].copy()
    df_train_macro = df_macro[df_macro['datestamp'] < cutoff_date].copy()

    # Current ALBI modified duration
    p_albi_md = df_train_albi['modified_duration'].iloc[-1]

    # Feature engineering
    df_train_bonds['yield_trend'] = df_train_bonds.groupby('bond_code')['yield'].transform(
        lambda x: x.diff().rolling(n_days).mean())
    df_train_bonds['return_trend'] = df_train_bonds.groupby('bond_code')['return'].transform(
        lambda x: x.rolling(n_days).mean())
    df_train_bonds['duration_trend'] = df_train_bonds.groupby('bond_code')['modified_duration'].transform(
        lambda x: x.diff().rolling(n_days).mean())
    df_train_macro['steepness'] = df_train_macro['us_10y'] - df_train_macro['us_2y']
    df_train_bonds['md_per_conv'] = df_train_bonds.groupby('bond_code')['return'].transform(
        lambda x: x.rolling(n_days).mean()) * df_train_bonds['convexity'] / df_train_bonds['modified_duration']

    # Merge macro features
    df_train_bonds = df_train_bonds.merge(df_train_macro, how='left', on='datestamp')
    df_train_bonds['duration_factor'] = df_train_bonds['bond_duration_years'] / df_train_bonds['bond_duration_years'].max()

    # Normalize features
    features_to_norm = ['md_per_conv', 'yield_trend', 'return_trend', 'duration_trend', 'steepness']
    for col in features_to_norm:
        mean_val = df_train_bonds[col].mean()
        std_val = df_train_bonds[col].std()
        df_train_bonds[col + '_norm'] = (df_train_bonds[col] - mean_val) / (std_val if std_val != 0 else 1)

    # Compute signal with normalized features
    df_train_bonds['signal'] = (
      0.25 * df_train_bonds['md_per_conv_norm']      # slightly lower
    + 0.1  * df_train_bonds['yield_trend_norm']     # reduce
    + 0.35 * df_train_bonds['return_trend_norm']    # increase
    - 0.05 * df_train_bonds['duration_trend_norm']  # smaller negative
    + 0.25 * df_train_bonds['steepness_norm']       # keep high emphasis
    )

    # Current bonds for this iteration
    df_current = df_train_bonds[df_train_bonds['datestamp'] == df_train_bonds['datestamp'].max()]

    # Objective function
    def objective(weights, signal, prev_weights, turnover_lambda=0.1):
        turnover = np.sum(np.abs(weights - prev_weights))
        return -(np.dot(weights, signal) - turnover_lambda * turnover)

    # Duration constraints
    def duration_constraint(weights, durations):
        port_duration = np.dot(weights, durations)
        return [
            100*(port_duration - (p_albi_md - p_active_md)),
            100*((p_albi_md + p_active_md) - port_duration)
        ]

    # Optimization setup
    bounds = [weight_bounds] * len(prev_weights)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_current['modified_duration'])[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_current['modified_duration'])[1]}
    ]

    # Run optimization
    result = minimize(
        objective,
        prev_weights,
        args=(df_current['signal'].values, prev_weights, 0.5),
        bounds=bounds,
        constraints=constraints
    )

    # Final weights: clip & normalize
    optimal_weights = result.x if result.success else prev_weights
    optimal_weights = np.clip(optimal_weights, 0, 0.2)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    # Append to weight matrix
    weight_matrix_tmp = pd.DataFrame({
        'bond_code': df_current['bond_code'],
        'weight': optimal_weights,
        'datestamp': cutoff_date
    })
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)

    # Update previous weights
    prev_weights = optimal_weights


# %%
import matplotlib.pyplot as plt


# Convert the date column to datetime
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp'])

# Example: define training period
train_start = '2005-01-01'
train_end   = '2022-12-31'

# Filter for training period
train_df = df_bonds[(df_bonds['datestamp'] >= train_start) & (df_bonds['datestamp'] <= train_end)]

# Get the first 10 unique bonds
unique_bonds = train_df['bond_code'].unique()

# Plot returns for each bond
plt.figure(figsize=(12,6))

for bond in unique_bonds:
    bond_data = train_df[train_df['bond_code'] == bond]
    plt.plot(bond_data['datestamp'], bond_data['return'], label=str(bond))

plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Returns of 10 Bonds During Training Period')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


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

# %%


# %%



