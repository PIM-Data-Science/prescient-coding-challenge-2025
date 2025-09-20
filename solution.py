# %%

import numpy as np
import pandas as pd
import datetime
import nbformat

from scipy.optimize import minimize
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

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
start_test = datetime.date(2023, 1, 3)
end_test = df_bonds['datestamp'].max()

# %%

# Enhanced feature engineering based on correlation insights
def create_enhanced_features(df_bonds, df_macro, n_days=10):
    """Create enhanced features based on correlation analysis"""

    # Merge macro data
    df_enhanced = df_bonds.merge(df_macro, how='left', on='datestamp')

    # Interest rate features (highly correlated in the matrix)
    df_enhanced['yield_curve_slope'] = df_enhanced['us_10y'] - df_enhanced['us_2y']
    df_enhanced['yield_curve_curvature'] = df_enhanced['us_10y'] - 0.5 * (df_enhanced['us_2y'] + df_enhanced['us_20y'])
    df_enhanced['rate_momentum'] = df_enhanced.groupby('bond_code')['us_10y'].pct_change(5)

    # Bond-specific features
    df_enhanced['yield_spread'] = df_enhanced['yield'] - df_enhanced['us_10y']
    df_enhanced['duration_adjusted_return'] = df_enhanced['return'] / df_enhanced['modified_duration']
    df_enhanced['convexity_yield_ratio'] = df_enhanced['convexity'] / df_enhanced['yield']

    # Rolling features for mean reversion and momentum
    for window in [5, 10, 20]:
        df_enhanced[f'return_ma_{window}'] = df_enhanced.groupby('bond_code')['return'].transform(lambda x: x.rolling(window).mean())
        df_enhanced[f'yield_ma_{window}'] = df_enhanced.groupby('bond_code')['yield'].transform(lambda x: x.rolling(window).mean())
        df_enhanced[f'duration_ma_{window}'] = df_enhanced.groupby('bond_code')['modified_duration'].transform(lambda x: x.rolling(window).mean())

    # Mean reversion signals
    df_enhanced['yield_mean_reversion'] = (df_enhanced['yield'] - df_enhanced['yield_ma_20']) / df_enhanced['yield_ma_20']
    df_enhanced['return_momentum'] = df_enhanced['return_ma_5'] - df_enhanced['return_ma_20']

    # Volatility measures
    df_enhanced['return_volatility'] = df_enhanced.groupby('bond_code')['return'].transform(lambda x: x.rolling(20).std())
    df_enhanced['sharpe_ratio'] = df_enhanced['return_ma_20'] / (df_enhanced['return_volatility'] + 1e-8)

    # Cross-sectional rankings
    df_enhanced['yield_rank'] = df_enhanced.groupby('datestamp')['yield'].rank(ascending=False)
    df_enhanced['return_rank'] = df_enhanced.groupby('datestamp')['return_ma_10'].rank(ascending=False)
    df_enhanced['duration_rank'] = df_enhanced.groupby('datestamp')['modified_duration'].rank()

    return df_enhanced

# %%

# Prepare signals dataframe
df_signals = pd.DataFrame(data={'datestamp': df_bonds.loc[(df_bonds['datestamp'] >= start_test) &
                                                          (df_bonds['datestamp'] <= end_test), 'datestamp'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='datestamp', inplace=True)

weight_matrix = pd.DataFrame()

# %%

# Optimized parameters
n_days = 10
prev_weights = np.array([0.1] * 10)
p_active_md = 1.2
weight_bounds = (0.0, 0.2)
turnover_lambda = 0.3  # Reduced for better performance

# Pre-compute enhanced features for all data
df_bonds_enhanced = create_enhanced_features(df_bonds, df_macro, n_days)

for i in range(len(df_signals)):

    current_date = df_signals.loc[i, 'datestamp']
    print('---> doing', current_date)

    # Training data (more efficient slicing)
    df_train = df_bonds_enhanced[df_bonds_enhanced['datestamp'] < current_date].copy()

    if len(df_train) < 100:  # Ensure minimum training data
        print(f"Insufficient training data for {current_date}")
        continue

    # Current day data for optimization
    df_current = df_bonds_enhanced[df_bonds_enhanced['datestamp'] == current_date].copy()

    if len(df_current) == 0:
        print(f"No current day data for {current_date}")
        continue

    # ALBI duration constraint
    p_albi_md = df_albi[df_albi['datestamp'] < current_date]['modified_duration'].iloc[-1]

    # Enhanced signal generation using multiple approaches
    feature_cols = [
        'yield_spread', 'duration_adjusted_return', 'convexity_yield_ratio',
        'yield_curve_slope', 'yield_curve_curvature', 'rate_momentum',
        'yield_mean_reversion', 'return_momentum', 'sharpe_ratio',
        'yield_rank', 'return_rank', 'duration_rank'
    ]

    # Ensure all features exist and handle missing values
    available_features = [col for col in feature_cols if col in df_current.columns]
    df_current_features = df_current[available_features].fillna(0)

    # Create composite signal
    if len(available_features) > 0:
        # Normalize features
        scaler = StandardScaler()
        try:
            train_features = df_train[available_features].fillna(0)
            if len(train_features) > 0:
                scaler.fit(train_features)
                normalized_features = scaler.transform(df_current_features)

                # Weighted combination of signals
                signal_weights = np.array([
                    0.15,  # yield_spread
                    0.20,  # duration_adjusted_return
                    0.10,  # convexity_yield_ratio
                    0.15,  # yield_curve_slope
                    0.10,  # yield_curve_curvature
                    0.10,  # rate_momentum
                    0.05,  # yield_mean_reversion
                    0.05,  # return_momentum
                    0.05,  # sharpe_ratio
                    0.02,  # yield_rank
                    0.02,  # return_rank
                    0.01   # duration_rank
                ])[:len(available_features)]

                signal = np.dot(normalized_features, signal_weights[:len(available_features)])
            else:
                signal = np.ones(len(df_current)) * 0.1
        except:
            signal = np.ones(len(df_current)) * 0.1
    else:
        signal = np.ones(len(df_current)) * 0.1

    # Enhanced optimization objective
    def objective(weights, signal, prev_weights, current_durations, target_duration, turnover_lambda=0.3):
        portfolio_return = np.dot(weights, signal)
        turnover_cost = turnover_lambda * np.sum(np.abs(weights - prev_weights))

        # Duration tracking penalty
        portfolio_duration = np.dot(weights, current_durations)
        duration_penalty = 0.1 * abs(portfolio_duration - target_duration) ** 2

        # Concentration penalty (encourage diversification)
        concentration_penalty = 0.05 * np.sum(weights ** 2)

        return -(portfolio_return - turnover_cost - duration_penalty - concentration_penalty)

    # Enhanced constraints
    def duration_constraint(weights, durations_today, target_duration, tolerance=1.2):
        port_duration = np.dot(weights, durations_today)
        lower_bound = target_duration - tolerance
        upper_bound = target_duration + tolerance
        return [port_duration - lower_bound, upper_bound - port_duration]

    # Optimization setup
    bounds = [weight_bounds] * len(df_current)
    current_durations = df_current['modified_duration'].values

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, current_durations, p_albi_md)[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, current_durations, p_albi_md)[1]}
    ]

    # Use previous weights as starting point, or equal weights if first iteration
    if i == 0:
        initial_weights = np.ones(len(df_current)) / len(df_current)
    else:
        initial_weights = prev_weights[:len(df_current)] if len(prev_weights) == len(df_current) else np.ones(len(df_current)) / len(df_current)

    # Optimization with error handling
    try:
        result = minimize(
            objective,
            initial_weights,
            args=(signal, initial_weights, current_durations, p_albi_md, turnover_lambda),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if result.success:
            optimal_weights = result.x
        else:
            print(f"Optimization failed for {current_date}, using equal weights")
            optimal_weights = np.ones(len(df_current)) / len(df_current)

    except Exception as e:
        print(f"Optimization error for {current_date}: {e}")
        optimal_weights = np.ones(len(df_current)) / len(df_current)

    # Store results
    weight_matrix_tmp = pd.DataFrame({
        'bond_code': df_current['bond_code'],
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

    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    df_turnover = weight_matrix.copy()
    df_turnover['turnover'] = df_turnover.groupby(['bond_code'])['weight'].diff()

    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']

    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    port_data['turnover'] = df_turnover.groupby('datestamp').turnover.apply(lambda x: x.abs().sum()/2).to_list()
    port_data['penalty'] = 0.005*port_data['turnover']*port_data['port_md'].shift()
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'], fill_value=0)
    port_data = port_data.merge(df_albi[['datestamp','return']], on='datestamp', how='left')
    port_data['portfolio_tri'] = (port_data['net_return']/100 + 1).cumprod()
    port_data['albi_tri'] = (port_data['return']/100 + 1).cumprod()

    # turnover chart
    fig_turnover = px.line(port_data, x='datestamp', y='turnover')
    fig_turnover.show()

    print(f"---> payoff for these buys between period {port_data['datestamp'].min()} and {port_data['datestamp'].max()} is {(port_data['portfolio_tri'].values[-1]-1)*100 :.2f}%")
    print(f"---> payoff for the ALBI benchmark for this period is {(port_data['albi_tri'].values[-1]-1)*100 :.2f}%")

    port_data_melted = pd.melt(port_data[['datestamp', 'portfolio_tri', 'albi_tri']], id_vars='datestamp')
    fig_payoff = px.line(port_data_melted, x='datestamp', y='value', color='variable')
    fig_payoff.show()

def plot_md(weight_matrix):
    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_md']].sum().reset_index()
    port_data = port_data.merge(df_albi[['datestamp','modified_duration']], on='datestamp', how='left')
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