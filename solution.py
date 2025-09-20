
#%%
import numpy as np
import pandas as pd
import datetime
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print('---> Enhanced SA Bonds Algorithm Start', t0 := datetime.datetime.now())

# %%
print('---> Initial Data Setup')

# Load data files
df_bonds = pd.read_csv('data/data_bonds.csv')
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).apply(lambda d: d.date())

df_albi = pd.read_csv('data/data_albi.csv')  
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).apply(lambda d: d.date())

df_macro = pd.read_csv('data/data_macro.csv')
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).apply(lambda d: d.date())

print(f"---> Loaded data: {df_bonds['datestamp'].min()} to {df_bonds['datestamp'].max()}")
print(f"---> Available bonds: {df_bonds['bond_code'].unique()}")
print(f"---> Available macro variables: {list(df_macro.columns)}")

# %%
print('---> Parameters')

# Training and test dates  
start_train = datetime.date(2005, 1, 3)
start_test = datetime.date(2023, 1, 3)
end_test = df_bonds['datestamp'].max()

# Portfolio constraints
n_days_lookback = 10
p_active_md = 1.5  # Active duration limit vs ALBI (1.5 as per challenge rules)
weight_bounds = (0.0, 0.2)  # Each bond: 0% to 20%
turnover_penalty = 0.0001  # Transaction cost penalty (0.01% as per challenge)

verbose = True

# %%
print('---> Enhanced Feature Engineering')

def create_advanced_bond_features(df_bonds, df_macro, df_albi, lookback_days=10):
    """Create comprehensive bond trading features"""
    
    # Merge macro data
    df_enhanced = df_bonds.merge(df_macro, on='datestamp', how='left')
    df_enhanced = df_enhanced.merge(df_albi[['datestamp', 'return', 'modified_duration']], 
                                   on='datestamp', how='left', suffixes=('', '_albi'))
    
    # Sort for time series calculations
    df_enhanced = df_enhanced.sort_values(['bond_code', 'datestamp'])
    
    # =================== MACRO FEATURES ===================
    # Yield curve features
    if 'us_10y' in df_enhanced.columns and 'us_2y' in df_enhanced.columns:
        df_enhanced['yield_curve_steepness'] = df_enhanced['us_10y'] - df_enhanced['us_2y']
        df_enhanced['steepness_change'] = df_enhanced.groupby('bond_code')['yield_curve_steepness'].diff(1)
    
    # Economic indicators momentum  
    macro_vars = [col for col in df_enhanced.columns if col in ['cpi', 'repo_rate', 'us_10y']]
    for var in macro_vars:
        if var in df_enhanced.columns:
            df_enhanced[f'{var}_change'] = df_enhanced.groupby('bond_code')[var].diff(1)
            df_enhanced[f'{var}_momentum'] = df_enhanced.groupby('bond_code')[var].pct_change(5)
    
    # =================== BOND-SPECIFIC FEATURES ===================
    # Return and volatility features
    df_enhanced['return_ma'] = df_enhanced.groupby('bond_code')['return'].rolling(lookback_days).mean().reset_index(0, drop=True)
    df_enhanced['return_volatility'] = df_enhanced.groupby('bond_code')['return'].rolling(lookback_days).std().reset_index(0, drop=True)
    df_enhanced['return_momentum'] = df_enhanced.groupby('bond_code')['return'].rolling(5).sum().reset_index(0, drop=True)
    
    # Duration-adjusted metrics
    df_enhanced['duration_adjusted_return'] = df_enhanced['return'] / (df_enhanced['modified_duration'] + 1e-6)
    df_enhanced['convexity_signal'] = df_enhanced['convexity'] / (df_enhanced['modified_duration'] + 1e-6)
    
    # Original signal enhanced
    df_enhanced['md_per_conv'] = (df_enhanced.groupby('bond_code')['return']
                                 .rolling(lookback_days).mean().reset_index(0, drop=True) 
                                 * df_enhanced['convexity'] / (df_enhanced['modified_duration'] + 1e-6))
    
    # =================== CROSS-SECTIONAL FEATURES ===================
    # Relative rankings within each date
    def safe_rank(group):
        if len(group) > 1:
            return group.rank(pct=True)
        else:
            return pd.Series(0.5, index=group.index)
    
    df_enhanced['return_rank'] = df_enhanced.groupby('datestamp')['return'].transform(safe_rank)
    df_enhanced['duration_rank'] = df_enhanced.groupby('datestamp')['modified_duration'].transform(safe_rank) 
    df_enhanced['convexity_rank'] = df_enhanced.groupby('datestamp')['convexity'].transform(safe_rank)
    df_enhanced['volatility_rank'] = df_enhanced.groupby('datestamp')['return_volatility'].transform(safe_rank)
    
    # Fill NaN values
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
    df_enhanced[numeric_cols] = df_enhanced[numeric_cols].fillna(0)
    
    return df_enhanced

# Create enhanced features
print('---> Creating advanced features...')
df_enhanced = create_advanced_bond_features(df_bonds, df_macro, df_albi, n_days_lookback)

# %%
print('---> Advanced Signal Generation')

def create_ensemble_signal(df_train, current_data, method='ensemble'):
    """Generate trading signals using multiple approaches"""
    
    if method == 'original':
        # Original simple signal
        signal = (current_data['md_per_conv'].fillna(0) * 100 - 
                 current_data.get('top40_return', 0).fillna(0) / 10 + 
                 current_data.get('comdty_fut', 0).fillna(0) / 100)
        return signal
    
    elif method == 'enhanced_rule':
        # Enhanced rule-based signal
        signal = current_data['md_per_conv'].fillna(0) * 100  # Base signal
        
        # Macro overlay
        if 'yield_curve_steepness' in current_data.columns:
            steepening = current_data['steepness_change'].fillna(0)
            signal += steepening * 10  # Benefit from steepening
        
        # Cross-sectional factors
        signal += (current_data['return_rank'].fillna(0.5) - 0.5) * 50  # Momentum
        signal += (0.5 - current_data['volatility_rank'].fillna(0.5)) * 30  # Low vol preference
        signal += (current_data['convexity_rank'].fillna(0.5) - 0.5) * 20  # Convexity preference
        
        return signal
    
    elif method == 'ml_enhanced':
        # Machine learning enhanced signal
        feature_cols = [
            'return_ma', 'return_volatility', 'return_momentum',
            'duration_adjusted_return', 'convexity_signal', 'md_per_conv',
            'return_rank', 'duration_rank', 'convexity_rank', 'volatility_rank'
        ]
        
        # Add macro features if available
        macro_features = [col for col in current_data.columns 
                         if 'change' in col or 'momentum' in col or 'steepness' in col]
        feature_cols.extend(macro_features)
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in df_train.columns and col in current_data.columns]
        
        if len(available_features) < 3:
            # Fallback to enhanced rule if insufficient features
            return create_ensemble_signal(df_train, current_data, 'enhanced_rule')
        
        try:
            # Prepare training data
            train_X = df_train[available_features].fillna(0)
            train_y = df_train['return'].shift(-1)  # Next day return as target
            
            # Remove NaN targets
            valid_idx = ~train_y.isna()
            train_X = train_X[valid_idx]
            train_y = train_y[valid_idx]
            
            if len(train_X) < 50:  # Insufficient data
                return create_ensemble_signal(df_train, current_data, 'enhanced_rule')
            
            # Train simple model
            model = LinearRegression()
            model.fit(train_X, train_y)
            
            # Predict for current data
            current_X = current_data[available_features].fillna(0)
            signals = model.predict(current_X)
            
            return pd.Series(signals, index=current_data.index)
            
        except Exception as e:
            if verbose:
                print(f"    ML model failed: {e}, using enhanced rule")
            return create_ensemble_signal(df_train, current_data, 'enhanced_rule')
    
    elif method == 'ensemble':
        # Combine multiple signals
        signal_rule = create_ensemble_signal(df_train, current_data, 'enhanced_rule')
        signal_ml = create_ensemble_signal(df_train, current_data, 'ml_enhanced')
        
        # Weighted combination
        return 0.6 * signal_rule + 0.4 * signal_ml

def advanced_portfolio_optimization(signals, durations, prev_weights, albi_duration, 
                                   turnover_lambda=0.5, risk_aversion=0.1):
    """Enhanced portfolio optimization with multiple objectives"""
    
    def objective(weights):
        # Expected return
        expected_return = np.dot(weights, signals)
        
        # Turnover penalty
        turnover = np.sum(np.abs(weights - prev_weights))
        
        # Risk penalty (concentration risk)
        concentration_risk = np.sum(weights**2)  # Encourages diversification
        
        # Active duration penalty (prefer closer to benchmark)
        port_duration = np.dot(weights, durations)
        duration_penalty = (port_duration - albi_duration)**2
        
        return -(expected_return - 
                turnover_lambda * turnover - 
                risk_aversion * concentration_risk -
                0.1 * duration_penalty)
    
    def duration_constraint(weights):
        port_duration = np.dot(weights, durations)
        lower_bound = port_duration - (albi_duration - p_active_md)
        upper_bound = (albi_duration + p_active_md) - port_duration
        return [lower_bound, upper_bound]
    
    # Optimization setup
    bounds = [weight_bounds] * len(signals)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w)[0]},  # Duration lower
        {'type': 'ineq', 'fun': lambda w: duration_constraint(w)[1]}   # Duration upper
    ]
    
    # Try different starting points for robustness
    best_result = None
    best_value = -np.inf
    
    starting_points = [
        prev_weights.copy(),  # Previous weights
        np.ones(len(signals)) / len(signals),  # Equal weights
    ]
    
    # Add random starting points
    for _ in range(3):
        starting_points.append(np.random.dirichlet(np.ones(len(signals)), 1)[0])
    
    for start_weights in starting_points:
        try:
            result = minimize(objective, start_weights, bounds=bounds, 
                            constraints=constraints, method='SLSQP',
                            options={'maxiter': 1000, 'ftol': 1e-8})
            if result.success and -result.fun > best_value:
                best_result = result
                best_value = -result.fun
        except:
            continue
    
    if best_result and best_result.success:
        optimal_weights = best_result.x
        # Ensure weights sum to exactly 1 and respect bounds
        optimal_weights = np.clip(optimal_weights, weight_bounds[0], weight_bounds[1])
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        return optimal_weights
    else:
        return prev_weights

# %%
print('---> Walk-Forward Optimization')

# Setup signals dataframe
test_dates = df_bonds[(df_bonds['datestamp'] >= start_test) & (df_bonds['datestamp'] <= end_test)]['datestamp'].unique()
df_signals = pd.DataFrame(data={'datestamp': test_dates})
df_signals.drop_duplicates(inplace=True)
df_signals.reset_index(drop=True, inplace=True)
df_signals.sort_values(by='datestamp', inplace=True)

print(f"---> Processing {len(df_signals)} trading days")

# Initialize
weight_matrix = pd.DataFrame()
prev_weights = np.array([0.1] * 10)  # Start with equal weights

# Main optimization loop
for i in range(len(df_signals)):
    current_date = df_signals.loc[i, 'datestamp']
    
    if verbose and i % 20 == 0:
        print(f'---> Processing {current_date} ({i+1}/{len(df_signals)})')
    
    # Current day data
    df_current_bonds = df_enhanced[df_enhanced['datestamp'] == current_date].copy()
    current_albi = df_albi[df_albi['datestamp'] == current_date]
    
    if len(df_current_bonds) == 0 or len(current_albi) == 0:
        # Use previous weights if no data
        weight_matrix_tmp = pd.DataFrame({
            'bond_code': df_enhanced['bond_code'].unique(),
            'weight': prev_weights,
            'datestamp': current_date
        })
        weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)
        continue
    
    # Get ALBI duration for constraint
    albi_duration = current_albi['modified_duration'].iloc[0]
    
    # Training data (all data before current date)
    df_train_bonds = df_enhanced[df_enhanced['datestamp'] < current_date].copy()
    
    # Generate signals using ensemble approach
    signals = create_ensemble_signal(df_train_bonds, df_current_bonds, 'ensemble')
    durations = df_current_bonds['modified_duration'].values
    
    # Ensure we have the right number of bonds
    if len(signals) != len(prev_weights):
        # Reset to equal weights if bond count changes
        prev_weights = np.ones(len(signals)) / len(signals)
    
    # Portfolio optimization
    optimal_weights = advanced_portfolio_optimization(
        signals.values, durations, prev_weights, albi_duration,
        turnover_lambda=0.3,  # Lower turnover penalty for more active management
        risk_aversion=0.05    # Moderate risk aversion
    )
    
    # Store results
    weight_matrix_tmp = pd.DataFrame({
        'bond_code': df_current_bonds['bond_code'],
        'weight': optimal_weights,
        'datestamp': current_date
    })
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)
    
    # Update for next iteration
    prev_weights = optimal_weights

print(f"---> Generated weights for {len(weight_matrix['datestamp'].unique())} days")

# %%
print('---> Enhanced Performance Analysis')

def plot_comprehensive_bond_analysis(weight_matrix):
    """Comprehensive bond portfolio analysis"""
    
    # Weight validation
    df_weight_sum = weight_matrix.groupby(['datestamp'])['weight'].sum()
    if df_weight_sum.min() < 0.9999 or df_weight_sum.max() > 1.0001:
        raise ValueError('Portfolio weights do not sum to one')
    
    if weight_matrix['weight'].min() < -1e-6 or weight_matrix['weight'].max() > 0.20001:
        raise ValueError('Instrument weights not confined to 0%-20%')
    
    # Merge portfolio data
    port_data = weight_matrix.merge(df_bonds, on=['bond_code', 'datestamp'], how='left')
    
    # Calculate turnover
    df_turnover = weight_matrix.copy()
    df_turnover['turnover'] = df_turnover.groupby(['bond_code'])['weight'].diff().fillna(0)
    
    # Portfolio metrics
    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    
    # Aggregate by date
    daily_port = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    daily_port['turnover'] = df_turnover.groupby('datestamp')['turnover'].apply(lambda x: x.abs().sum()/2).values
    daily_port['penalty'] = turnover_penalty * daily_port['turnover'] * daily_port['port_md'].shift(1).fillna(0)
    daily_port['net_return'] = daily_port['port_return'] - daily_port['penalty']
    
    # Merge benchmark
    daily_port = daily_port.merge(df_albi[['datestamp', 'return', 'modified_duration']], 
                                 on='datestamp', how='left', suffixes=('', '_albi'))
    
    # Calculate cumulative performance
    daily_port['portfolio_tri'] = (daily_port['net_return']/100 + 1).cumprod()
    daily_port['albi_tri'] = (daily_port['return']/100 + 1).cumprod()
    daily_port['active_md'] = daily_port['port_md'] - daily_port['modified_duration']
    daily_port['excess_return'] = daily_port['net_return'] - daily_port['return']
    
    # Performance metrics
    total_portfolio_return = (daily_port['portfolio_tri'].iloc[-1] - 1) * 100
    total_albi_return = (daily_port['albi_tri'].iloc[-1] - 1) * 100
    excess_return = total_portfolio_return - total_albi_return
    
    tracking_error = daily_port['excess_return'].std() * np.sqrt(252)
    information_ratio = daily_port['excess_return'].mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    avg_turnover = daily_port['turnover'].mean() * 252  # Annualized
    max_active_duration = daily_port['active_md'].abs().max()
    
    print(f"\n=== ENHANCED BOND PORTFOLIO PERFORMANCE ===")
    print(f"Period: {daily_port['datestamp'].min()} to {daily_port['datestamp'].max()}")
    print(f"Portfolio Total Return: {total_portfolio_return:.2f}%")
    print(f"ALBI Benchmark Return: {total_albi_return:.2f}%") 
    print(f"Excess Return (Alpha): {excess_return:.2f}%")
    print(f"Tracking Error: {tracking_error:.2f}%")
    print(f"Information Ratio: {information_ratio:.3f}")
    print(f"Average Annual Turnover: {avg_turnover:.1f}%")
    print(f"Max Active Duration: {max_active_duration:.2f}")
    
    # Check duration constraint violations
    duration_violations = len(daily_port[daily_port['active_md'].abs() > p_active_md])
    print(f"Duration Constraint Violations: {duration_violations}")
    

    
    # =================== VISUALIZATIONS ===================
    
    # 1. Performance comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_port['datestamp'], y=daily_port['portfolio_tri'],
                             mode='lines', name='Enhanced Portfolio', line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=daily_port['datestamp'], y=daily_port['albi_tri'],
                             mode='lines', name='ALBI Benchmark', line=dict(color='red', width=2, dash='dash')))
    fig1.update_layout(title=f'Portfolio vs ALBI Performance<br>Excess Return: {excess_return:.2f}% | Info Ratio: {information_ratio:.3f}',
                      xaxis_title='Date', yaxis_title='Total Return Index', template='plotly_white')
    fig1.show()
    
    # 2. Weight evolution
    fig2 = px.area(weight_matrix, x="datestamp", y="weight", color="bond_code",
                   title="Portfolio Weights Evolution")
    fig2.show()
    
    # 3. Active duration tracking
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily_port['datestamp'], y=daily_port['active_md'],
                             mode='lines', name='Active Duration', line=dict(color='green')))
    fig3.add_hline(y=p_active_md, line_dash="dash", line_color="red", 
                   annotation_text=f"Upper Limit (+{p_active_md})")
    fig3.add_hline(y=-p_active_md, line_dash="dash", line_color="red",
                   annotation_text=f"Lower Limit (-{p_active_md})")
    fig3.update_layout(title='Active Duration Management', xaxis_title='Date', 
                      yaxis_title='Active Duration', template='plotly_white')
    fig3.show()
    
    # 4. Turnover analysis
    fig4 = px.line(daily_port, x='datestamp', y='turnover', 
                   title=f'Portfolio Turnover<br>Average Annual: {avg_turnover:.1f}%')
    fig4.show()
    
    return daily_port

# Run comprehensive analysis
daily_performance = plot_comprehensive_bond_analysis(weight_matrix)

print('\n---> Enhanced SA Bonds Algorithm Complete', t1 := datetime.datetime.now())
print(f'---> Total time taken: {t1 - t0}')

if (t1 - t0).total_seconds() > 600:  # 10 minutes
    print(" WARNING: Algorithm took longer than 10 minutes!")
else:
    print(" Algorithm completed within time limit")

print('\n=== ENHANCED SA GOVERNMENT BONDS SOLUTION COMPLETE ===')
# %%
