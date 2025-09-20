import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import plotly.express as px

print('---> Python Script Start', t0 := datetime.datetime.now())

# Load data
df_bonds = pd.read_csv('data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())
df_albi = pd.read_csv('data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())
df_macro = pd.read_csv('data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

# Parameters
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3)
end_test = df_bonds['datestamp'].max()
p_active_md = 1.5  # Max duration deviation
weight_bounds = (0.0, 0.2)
turnover_lambda = 0.1  # Turnover penalty

print('---> Data Preparation')

# Merge datasets
df_albi_renamed = df_albi.rename(columns={'modified_duration': 'albi_modified_duration', 'return': 'albi_return'})
df = pd.merge(df_bonds, df_albi_renamed[['datestamp', 'albi_modified_duration']], on='datestamp', how='left')
df = pd.merge(df, df_macro, on='datestamp', how='left')

# Sort for time-series operations
df.sort_values(by=['bond_code', 'datestamp'], inplace=True)

# Forward-fill missing values per bond
fill_cols = ['yield', 'modified_duration', 'convexity', 'albi_modified_duration', 'us_10y', 'us_2y']
for col in fill_cols:
    if col in df.columns:
        df[col] = df.groupby('bond_code')[col].ffill()

# Drop rows with missing critical data
df.dropna(subset=['yield', 'modified_duration', 'convexity', 'albi_modified_duration'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Engineer features (causal, backward-looking)
n_days_momentum = 5
df['yield_momentum'] = df.groupby('bond_code')['yield'].transform(lambda x: (x - x.shift(n_days_momentum)) / (x.shift(n_days_momentum) + 1e-10))
df['duration_spread'] = df['modified_duration'] - df['albi_modified_duration']
df['steepness'] = df['us_10y'] - df['us_2y'] if 'us_10y' in df.columns and 'us_2y' in df.columns else 0

# Feature list for scaling
feature_names = ['convexity', 'yield_momentum', 'duration_spread', 'steepness']
df[feature_names] = df[feature_names].fillna(0)

# Walk-forward validation dates
df_signals = pd.DataFrame({'datestamp': df.loc[(df['datestamp'] >= start_test) & (df['datestamp'] <= end_test), 'datestamp'].unique()})
df_signals.sort_values(by='datestamp', inplace=True)
df_signals.reset_index(drop=True, inplace=True)

# Initialize weight matrix
weight_matrix = pd.DataFrame()
prev_weights = np.array([0.1] * 10)  # Equal weights start
unique_bonds = sorted(df['bond_code'].unique())
assert len(unique_bonds) == 10, f"Expected 10 bonds, got {len(unique_bonds)}"

def adjust_weights_for_constraints(weights, durations, albi_md, max_iter=50, step_scale=0.05):
    """Adjust weights to satisfy weight and duration constraints."""
    weights = np.clip(weights, 0, 0.2)
    weights /= np.sum(weights) + 1e-10
    port_dur = np.dot(weights, durations)
    diff = port_dur - albi_md
    iteration = 0
    while abs(diff) > 1.5 and iteration < max_iter:
        if diff > 0:
            idx_sort = np.argsort(durations)[::-1][:3]
            for idx in idx_sort:
                if diff <= 0 or weights[idx] <= 0: break
                adjust = min(weights[idx], diff / (durations[idx] + 1e-10) * step_scale)
                weights[idx] -= adjust
                diff -= adjust * durations[idx]
        else:
            idx_sort = np.argsort(durations)[:3]
            for idx in idx_sort:
                if diff >= 0 or weights[idx] >= 0.2: break
                adjust = min(0.2 - weights[idx], abs(diff) / (durations[idx] + 1e-10) * step_scale)
                weights[idx] += adjust
                diff += adjust * durations[idx]
        weights = np.clip(weights, 0, 0.2)
        weights /= np.sum(weights) + 1e-10
        port_dur = np.dot(weights, durations)
        diff = port_dur - albi_md
        iteration += 1
    return weights, abs(diff) <= 1.5

print('---> Generating Signals and Weights')

for i in range(len(df_signals)):
    current_date = df_signals.loc[i, 'datestamp']
    print(f'---> Processing {current_date}')

    # Split data
    df_train = df[df['datestamp'] < current_date].copy()
    df_current = df[df['datestamp'] == current_date].copy()
    df_current_albi = df_albi[df_albi['datestamp'] == current_date].copy()

    # Handle missing data
    if len(df_current) != 10 or df_current_albi.empty:
        print(f"Warning: Missing data for {current_date}. Using adjusted equal weights.")
        optimal_weights = np.array([0.1] * 10)
        p_albi_md = df_albi[df_albi['datestamp'] <= current_date]['modified_duration'].iloc[-1] if not df_albi[df_albi['datestamp'] <= current_date].empty else 5.0
        current_durations = df_current['modified_duration'].values if len(df_current) == 10 else np.array([5.0] * 10)
        optimal_weights, _ = adjust_weights_for_constraints(optimal_weights, current_durations, p_albi_md)
    else:
        # Ensure all bonds
        # Ensure all bonds are present for the current date
        df_current_info = df_current.set_index('bond_code').reindex(unique_bonds).reset_index()
        df_current_info[feature_names] = df_current_info[feature_names].fillna(0)
        df_current_info['modified_duration'] = df_current_info['modified_duration'].fillna(df_current_info['albi_modified_duration'].fillna(5.0))
        df_current_info['return'] = df_current_info['return'].fillna(0.0)

        # Scale features
        scaler = StandardScaler()
        if not df_train.empty:
            scaler.fit(df_train[feature_names])
        else:
            scaler.fit(df_current_info[feature_names])
        features_scaled = scaler.transform(df_current_info[feature_names])
        df_features_scaled = pd.DataFrame(features_scaled, columns=feature_names, index=df_current_info.index)

        # Generate signals (favor high convexity, falling yields, ALBI-aligned duration)
        signal = (
            df_features_scaled['convexity'] * 1.0
            - df_features_scaled['yield_momentum'] * 0.5
            - np.abs(df_features_scaled['duration_spread']) * 1.0
            + df_features_scaled['steepness'] * 0.3
        )

        # Optimization setup
        p_albi_md = df_current_albi['modified_duration'].iloc[0]
        current_durations = df_current_info['modified_duration'].values

        def objective(weights, signal_scores, prev_w, lambda_penalty):
            turnover = np.sum(np.abs(weights - prev_w))
            return -np.dot(weights, signal_scores) + lambda_penalty * turnover

        def duration_constraint(weights, durations, albi_md):
            port_duration = np.dot(weights, durations)
            return [
                (albi_md + p_active_md) - port_duration,  # <= albi + 1.5
                port_duration - (albi_md - p_active_md)   # >= albi - 1.5
            ]

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: duration_constraint(w, current_durations, p_albi_md)[0]},
            {'type': 'ineq', 'fun': lambda w: duration_constraint(w, current_durations, p_albi_md)[1]}
        ]
        bounds = [weight_bounds] * 10
        optimizer_options = {'maxiter': 1000, 'ftol': 1e-14, 'disp': False}

        # Initial guess: equal weights adjusted for duration
        initial_weights = np.array([0.1] * 10)
        initial_weights, _ = adjust_weights_for_constraints(initial_weights, current_durations, p_albi_md)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            args=(signal.values, prev_weights, turnover_lambda),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=optimizer_options
        )

        optimal_weights = result.x if result.success else initial_weights

        # Enforce constraints
        optimal_weights = np.clip(optimal_weights, 0, 0.2)
        optimal_weights /= np.sum(optimal_weights) + 1e-10

        # Adjust for duration
        optimal_weights, success = adjust_weights_for_constraints(optimal_weights, current_durations, p_albi_md)
        if not success:
            print(f"Warning: Duration adjustment failed for {current_date}. Using equal weights.")
            optimal_weights = np.array([0.1] * 10)
            optimal_weights, _ = adjust_weights_for_constraints(optimal_weights, current_durations, p_albi_md)

    # Store weights
    weight_matrix_tmp = pd.DataFrame({
        'datestamp': [current_date] * 10,
        'bond_code': unique_bonds,
        'weight': optimal_weights
    })
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)
    prev_weights = optimal_weights.copy()

def plot_payoff(weight_matrix):
    df_weight_sum = weight_matrix.groupby(['datestamp'])['weight'].sum()
    if df_weight_sum.min() < 0.9999 or df_weight_sum.max() > 1.0001:
        raise ValueError('The portfolio weights do not sum to one')
    if weight_matrix['weight'].min() < -0.00001 or weight_matrix['weight'].max() > 0.20001:
        raise ValueError(r'The instrument weights are not confined to 0%-20%')

    fig_weights = px.area(weight_matrix, x="datestamp", y="weight", color="bond_code", title="Portfolio Weights Over Time")
    fig_weights.show()

    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    df_turnover = weight_matrix.copy()
    df_turnover.sort_values(['bond_code', 'datestamp'], inplace=True)
    df_turnover['prev_weight'] = df_turnover.groupby('bond_code')['weight'].shift(1).fillna(0.1)
    df_turnover['turnover_abs_change'] = (df_turnover['weight'] - df_turnover['prev_weight']).abs()
    daily_turnover = df_turnover.groupby('datestamp')['turnover_abs_change'].sum()

    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    port_data = port_data.merge(daily_turnover.rename('turnover'), on='datestamp', how='left')
    
    port_data['penalty'] = 0.0001 * port_data['turnover'] * port_data['port_md'].shift().fillna(0)
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'].fillna(0))
    port_data = port_data.merge(df_albi[['datestamp', 'return']], on='datestamp', how='left')
    port_data['portfolio_tri'] = (port_data['net_return'] / 100 + 1).cumprod()
    port_data['albi_tri'] = (port_data['return'] / 100 + 1).cumprod()

    fig_turnover = px.line(port_data, x='datestamp', y='turnover', title='Daily Portfolio Turnover')
    fig_turnover.show()

    print(f"---> Payoff for these buys between period {port_data['datestamp'].min()} and {port_data['datestamp'].max()} is {(port_data['portfolio_tri'].values[-1]-1)*100 :.2f}%")
    print(f"---> Payoff for the ALBI benchmark for this period is {(port_data['albi_tri'].values[-1]-1)*100 :.2f}%")

    port_data_melted = pd.melt(port_data[['datestamp', 'portfolio_tri', 'albi_tri']], id_vars='datestamp', var_name='Index', value_name='TRI')
    fig_payoff = px.line(port_data_melted, x='datestamp', y='TRI', color='Index', title='Portfolio vs. ALBI Total Return Index (TRI)')
    fig_payoff.show()

def plot_md(weight_matrix):
    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_md']].sum().reset_index()
    port_data = port_data.merge(df_albi[['datestamp', 'modified_duration']], on='datestamp', how='left')
    port_data['active_md'] = port_data['port_md'] - port_data['modified_duration']

    fig_md = px.line(port_data, x='datestamp', y='active_md', title='Active Modified Duration (Portfolio - ALBI)')
    fig_md.add_hline(y=1.5, line_dash="dash", line_color="red")
    fig_md.add_hline(y=-1.5, line_dash="dash", line_color="red")
    fig_md.show()

    if len(port_data[abs(port_data['active_md']) > 1.5001]['datestamp']) == 0:
        print(f"---> The portfolio does not breach the modified duration constraint")
    else:
        raise ValueError('This buy matrix violates the modified duration constraint on the below dates: \n ' + ", ".join(pd.to_datetime(port_data[abs(port_data['active_md']) > 1.5001]['datestamp']).dt.strftime("%Y-%m-%d")))

plot_payoff(weight_matrix)
plot_md(weight_matrix)

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)
