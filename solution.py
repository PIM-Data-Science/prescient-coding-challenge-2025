import numpy as np
import pandas as pd
import datetime
import nbformat

from scipy.optimize import minimize
import plotly.express as px
from scipy.stats import zscore
from plotly.subplots import make_subplots

print('---> Python Script Start', t0 := datetime.datetime.now())

print('---> initial data set up')
nwankcqmeklngfop
# instrument data
df_bonds = pd.read_csv('C:\\Users\\Kamog\\Desktop\\sigma02\\prescient-coding-challenge-2025\\data\\data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

# albi data
df_albi = pd.read_csv('C:\\Users\\Kamog\\Desktop\\sigma02\\prescient-coding-challenge-2025\\data\\data_albi.csv')
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

# macro data
df_macro = pd.read_csv('C:\\Users\\Kamog\\Desktop\\sigma02\\prescient-coding-challenge-2025\\data\\data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

print('---> the parameters')

# training and test dates
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3) # test set is this datasets 2023 & 2024 data
end_test = df_bonds['datestamp'].max()

# 1st manipulation %%
def preprocess_data(df_bonds, df_albi, df_macro):
    """Enhanced data preprocessing with features based on available data"""
    
    # Calculate rolling statistics for bonds
    for bond_code in df_bonds['bond_code'].unique():
        bond_mask = df_bonds['bond_code'] == bond_code
        
        # Price-based features
        df_bonds.loc[bond_mask, 'return_ma_5'] = df_bonds.loc[bond_mask, 'return'].rolling(window=5).mean()
        df_bonds.loc[bond_mask, 'return_ma_20'] = df_bonds.loc[bond_mask, 'return'].rolling(window=20).mean()
        df_bonds.loc[bond_mask, 'return_std_20'] = df_bonds.loc[bond_mask, 'return'].rolling(window=20).std()
        
        # Momentum and trend features
        df_bonds.loc[bond_mask, 'momentum_5'] = df_bonds.loc[bond_mask, 'return'].rolling(window=5).sum()
        df_bonds.loc[bond_mask, 'momentum_20'] = df_bonds.loc[bond_mask, 'return'].rolling(window=20).sum()
        
        # Volatility features
        df_bonds.loc[bond_mask, 'volatility_ratio'] = (
            df_bonds.loc[bond_mask, 'return'].rolling(window=5).std() / 
            df_bonds.loc[bond_mask, 'return'].rolling(window=20).std()
        )
    
    # Add macroeconomic features based on available data
    # Yield curve (typically available in such datasets)
    if 'us_10y' in df_macro.columns and 'us_2y' in df_macro.columns:
        df_macro['yield_curve'] = df_macro['us_10y'] - df_macro['us_2y']
        print("Added yield curve feature")
    
    # Risk-on/risk-off indicator (using equity and commodity data)
    if 'top40_return' in df_macro.columns and 'comdty_fut' in df_macro.columns:
        df_macro['risk_on_off'] = df_macro['top40_return'] - df_macro['comdty_fut']
        print("Added risk on/off feature")
    
    # Merge macro data with bonds
    df_bonds = df_bonds.merge(df_macro, on='datestamp', how='left')
    
    # Create bond-specific features using available macro data
    if 'us_10y' in df_bonds.columns:
        df_bonds['yield_spread'] = df_bonds['yield'] - df_bonds['us_10y']
    
    if 'top40_return' in df_bonds.columns:
        df_bonds['equity_bond_correlation'] = df_bonds.groupby('bond_code')['return'].transform(
            lambda x: x.rolling(20).corr(df_bonds['top40_return'].reindex(x.index))
        )
    
    return df_bonds, df_albi, df_macro

df_bonds, df_albi, df_macro = preprocess_data(df_bonds, df_albi, df_macro)


# we will perform walk forward validation for testing the buys - https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n
df_signals = pd.DataFrame(data={'datestamp':df_bonds.loc[(df_bonds['datestamp']>=start_test) & (df_bonds['datestamp']<=end_test), 'datestamp'].values})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='datestamp', inplace=True) # this code just gets the dates that we need to generate buy signals for

weight_matrix = pd.DataFrame()

#2nd Manipulation %%

# Enhanced optimization with risk management using available data
def calculate_risk_metrics(returns, lookback=60):
    """Calculate risk metrics for portfolio optimization"""
    if len(returns) < lookback:
        lookback = len(returns)
    
    recent_returns = returns[-lookback:]
    volatility = np.std(recent_returns)
    downside_returns = recent_returns[recent_returns < 0]
    downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    return {
        'volatility': volatility,
        'downside_volatility': downside_volatility,
        'var_95': np.percentile(recent_returns, 5) if len(recent_returns) > 0 else 0
    }

def enhanced_objective(weights, signals, prev_weights, risk_metrics, 
                      turnover_lambda=0.1, risk_lambda=0.2, risk_aversion=1.0):
  #  """Enhanced objective function with risk adjustment"""
    # Return component
    return_component = np.dot(weights, signals)
    
    # Turnover penalty
    turnover = np.sum(np.abs(weights - prev_weights))
    turnover_penalty = turnover_lambda * turnover
    
    # Risk penalty
    risk_penalty = risk_lambda * risk_aversion * risk_metrics['volatility']
    
    # Downside risk penalty
    downside_penalty = 0.5 * risk_lambda * risk_aversion * risk_metrics['downside_volatility']
    
    # Value at Risk constraint
    var_penalty = 0 if risk_metrics['var_95'] > -0.05 else 10 * abs(risk_metrics['var_95'] + 0.05)
    
    return -(return_component - turnover_penalty - risk_penalty - downside_penalty - var_penalty)

# Duration constraints
def duration_constraint(weights, durations_today, p_albi_md, p_active_md=1.2):
   # """Duration constraint function"""
    port_duration = np.dot(weights, durations_today)
    return [100 * (port_duration - (p_albi_md - p_active_md)), 
            100 * ((p_albi_md + p_active_md) - port_duration)]

# Concentration constraint
def concentration_constraint(weights, max_weight=0.2):
   # """Ensure no single asset exceeds max weight"""
    return max_weight - np.max(weights)

# This cell contains a sample solution
# You are not restricted to the choice of signal, or the portfolio optimisation used to generate weights
# You may modify anything within this cell as long as it produces a weight matrix in the required form, and the solution does not violate any of the rules

# static data for optimisation and signal generation
n_days = 10
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

    p_albi_md = df_train_albi['modified_duration'].tail(1) #modified 

    # feature engineering
    df_train_macro['steepness'] = df_train_macro['us_10y'] - df_train_macro['us_2y'] 
    df_train_bonds['md_per_conv'] = df_train_bonds.groupby(['bond_code'])['return'].transform(lambda x: x.rolling(window=n_days).mean()) * df_train_bonds['convexity'] / df_train_bonds['modified_duration']
    #3rd Additional features based on available data
    df_train_bonds['momentum'] = df_train_bonds.groupby(['bond_code'])['return'].transform(
        lambda x: x.rolling(window=20).mean())
    
    df_train_bonds['volatility_ratio'] = df_train_bonds.groupby(['bond_code'])['return'].transform(
        lambda x: x.rolling(window=5).std()) / df_train_bonds.groupby(['bond_code'])['return'].transform(
        lambda x: x.rolling(window=20).std())
    
    #df_train_bonds = df_train_bonds.merge(df_train_macro, how='left', on = 'datestamp')
        #5th Merge with available macro data
    available_macro_cols = [col for col in df_train_macro.columns if col not in ['datestamp']]
    df_train_bonds = df_train_bonds.merge(df_train_macro[['datestamp'] + available_macro_cols], 
                                         how='left', on='datestamp')
     # 7th Enhanced signal generation using available features
    signal_components = []
    #------------------
        # Base signal component
    signal_components.append(df_train_bonds['md_per_conv'] * 100)
    
    # Add available macro components to signal
    if 'top40_return' in df_train_bonds.columns:
        signal_components.append(-df_train_bonds['top40_return'] / 10)
    
    if 'comdty_fut' in df_train_bonds.columns:
        signal_components.append(df_train_bonds['comdty_fut'] / 100)
    
    # Add momentum component
    signal_components.append(df_train_bonds['momentum'] * 0.5)
    
    # Add volatility component (lower weight for volatile assets)
    signal_components.append(-df_train_bonds['volatility_ratio'] * 0.3)
    
    # Add yield curve component if available
    
    if 'yield_curve' in df_train_bonds.columns:
        signal_components.append(df_train_bonds['yield_curve'] * 0.2)
    #------------------------
    df_train_bonds['signal'] = sum(signal_components)
    df_train_bonds_current = df_train_bonds[df_train_bonds['datestamp'] == df_train_bonds['datestamp'].max()]

    #
        #8th Calculate risk metrics
    bond_returns = df_train_bonds.pivot(index='datestamp', columns='bond_code', values='return').fillna(0)
    if len(prev_weights) == bond_returns.shape[1]:
        portfolio_returns = bond_returns @ prev_weights
        risk_metrics = calculate_risk_metrics(portfolio_returns.values)
    else:
        risk_metrics = {'volatility': 0.1, 'downside_volatility': 0.05, 'var_95': -0.02}
        
    # optimisation objective
    def objective(weights, signal, prev_weights, turnover_lambda=0.1):
        turnover = np.sum(np.abs(weights - prev_weights))
        return -(np.dot(weights, signal) - turnover_lambda * turnover)
        
    # Duration constraints
    def duration_constraint(weights, durations_today, p_albi_md, p_active=1.2):
        port_duration = np.dot(weights, durations_today)
        return [100*(port_duration - (p_albi_md - p_active_md)), 100*((p_albi_md + p_active_md) - port_duration)]
    
    # Optimization setup
    turnover_lambda = 0.5
    bounds = [weight_bounds] * 10
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'], p_albi_md, p_active_md)[0]},
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w, df_train_bonds_current['modified_duration'], p_albi_md, p_active_md)[1]},
        {'type': 'ineq', 'fun': lambda w: concentration_constraint(w, 0.2)}  # No asset > 20%
    ]

    result = minimize(objective, prev_weights, args=(df_train_bonds_current['signal'], prev_weights, turnover_lambda), bounds=bounds, constraints=constraints)

    optimal_weights = result.x if result.success else prev_weights
    weight_matrix_tmp = pd.DataFrame({'bond_code': df_train_bonds_current['bond_code'],
                                      'weight': optimal_weights,
                                      'datestamp': df_signals.loc[i, 'datestamp']})
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp])

    prev_weights = optimal_weights

    def enhanced_plot_payoff(weight_matrix, df_bonds, df_albi):
    
    # Check weights sum to one
    df_weight_sum = weight_matrix.groupby(['datestamp'])['weight'].sum()
    if df_weight_sum.min() < 0.9999 or df_weight_sum.max() > 1.0001:
        raise ValueError('The portfolio weights do not sum to one')
    
    # Check weights between 0 and 0.2
    if weight_matrix['weight'].min() < 0 or weight_matrix['weight'].max() > 0.20001:
        raise ValueError(r'The instrument weights are not confined to 0%-20%')

    # Check we have 10 bonds
    unique_bonds = weight_matrix['bond_code'].nunique()
    print(f"---> Number of unique bonds in portfolio: {unique_bonds}")
    if unique_bonds != 10:
        print(f"---> WARNING: Expected 10 bonds, found {unique_bonds}")

    # Calculate portfolio performance
    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    
    # FIXED: Calculate turnover correctly
    # Sort by date and bond code to ensure proper diff calculation
    weight_matrix_sorted = weight_matrix.sort_values(['datestamp', 'bond_code'])
    df_turnover = weight_matrix_sorted.copy()
    
    # Calculate absolute weight changes for each bond
    df_turnover['weight_change'] = df_turnover.groupby('bond_code')['weight'].diff().abs()
    
    # Sum weight changes across all bonds for each date
    daily_turnover = df_turnover.groupby('datestamp')['weight_change'].sum() / 2  # Divide by 2 because each trade affects both buy and sell sides
    
    # Create port_data with proper turnover
    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']

    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    
    # Merge turnover data
    port_data = port_data.merge(daily_turnover.reset_index(), on='datestamp', how='left')
    port_data.rename(columns={'weight_change': 'turnover'}, inplace=True)
    
    # Fill NaN values with 0 for the first day
    port_data['turnover'] = port_data['turnover'].fillna(0)
    
    port_data['penalty'] = 0.005 * port_data['turnover'] * port_data['port_md'].shift()
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'], fill_value=0)
    port_data = port_data.merge(df_albi[['datestamp', 'return']], on='datestamp', how='left')
    port_data['portfolio_tri'] = (port_data['net_return'] / 100 + 1).cumprod()
    port_data['albi_tri'] = (port_data['return'] / 100 + 1).cumprod()
    
    # Plot weights through time - check if we have 10 bonds
    bond_count = weight_matrix['bond_code'].nunique()
    print(f"---> Plotting weights for {bond_count} bonds")
    
    fig_weights = px.area(weight_matrix, x="datestamp", y="weight", color="bond_code", 
                         title=f"Portfolio Weights ({bond_count} bonds)")
    fig_weights.show()

    # Plot cumulative returns
    fig_returns = px.line(port_data, x='datestamp', y=['portfolio_tri', 'albi_tri'], 
                         title='Cumulative Returns')
    fig_returns.show()
    
    # Plot turnover - check if we have data
    print(f"---> Turnover data range: {port_data['turnover'].min():.4f} to {port_data['turnover'].max():.4f}")
    fig_turnover = px.line(port_data, x='datestamp', y='turnover', title='Portfolio Turnover')
    fig_turnover.show()
    
    # Plot active modified duration
    port_data = port_data.merge(df_albi[['datestamp', 'modified_duration']], on='datestamp', how='left')
    port_data['active_md'] = port_data['port_md'] - port_data['modified_duration']
    
    fig_md = px.line(port_data, x='datestamp', y='active_md', title='Active Modified Duration')
    fig_md.add_hline(y=1.5, line_dash="dash", line_color="red")
    fig_md.add_hline(y=-1.5, line_dash="dash", line_color="red")
    fig_md.show()
    
    # Calculate additional metrics
    port_data['excess_return'] = port_data['net_return'] - port_data['return']
    port_data['cumulative_excess'] = (port_data['excess_return'] / 100 + 1).cumprod()
    
    # Rolling Sharpe ratio (6-month)
    returns_series = port_data.set_index('datestamp')['net_return'] / 100
    rolling_sharpe = returns_series.rolling(126).mean() / returns_series.rolling(126).std() * np.sqrt(252)
    port_data['rolling_sharpe'] = rolling_sharpe.values
    
    # Plot rolling Sharpe ratio
    fig_sharpe = px.line(port_data, x='datestamp', y='rolling_sharpe', title='Rolling Sharpe Ratio (6M)')
    fig_sharpe.show()
    
    # Drawdown calculation
    port_data['peak'] = port_data['portfolio_tri'].cummax()
    port_data['drawdown'] = (port_data['portfolio_tri'] - port_data['peak']) / port_data['peak']

    # Plot drawdown
    fig_drawdown = px.area(port_data, x='datestamp', y='drawdown', title='Drawdown')
    fig_drawdown.show()

    # Print performance statistics
    total_return = (port_data['portfolio_tri'].values[-1] - 1) * 100
    benchmark_return = (port_data['albi_tri'].values[-1] - 1) * 100
    excess_return = total_return - benchmark_return
    
    volatility = port_data['net_return'].std() * np.sqrt(252)
    sharpe_ratio = (port_data['net_return'].mean() / port_data['net_return'].std()) * np.sqrt(252) if port_data['net_return'].std() > 0 else 0
    
    max_drawdown = port_data['drawdown'].min() * 100
    
    print(f"---> Portfolio return: {total_return:.2f}%")
    print(f"---> Benchmark return: {benchmark_return:.2f}%")
    print(f"---> Excess return: {excess_return:.2f}%")
    print(f"---> Annualized volatility: {volatility:.2f}%")
    print(f"---> Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"---> Maximum drawdown: {max_drawdown:.2f}%")
    print(f"---> Average daily turnover: {port_data['turnover'].mean():.4f}")
    
    return port_data

# Make sure the function is called
print("---> Running enhanced payoff analysis...")
port_data = enhanced_plot_payoff(weight_matrix, df_bonds, df_albi)
print("---> Enhanced analysis completed!")
