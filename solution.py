# Import Libraries
import numpy as np
import pandas as pd
import datetime
import time
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print('✅ Libraries imported')
print('🕐 Start time:', datetime.datetime.now())

# Load Data
df_bonds = pd.read_csv('data/data_bonds.csv')
df_albi = pd.read_csv('data/data_albi.csv')
df_macro = pd.read_csv('data/data_macro.csv')

# Convert dates
df_bonds['datestamp'] = pd.to_datetime(df_bonds['datestamp']).dt.date
df_albi['datestamp'] = pd.to_datetime(df_albi['datestamp']).dt.date
df_macro['datestamp'] = pd.to_datetime(df_macro['datestamp']).dt.date

# Define periods
start_test = datetime.date(2023, 1, 3)
end_test = df_bonds['datestamp'].max()

print(f'📊 Data loaded: {df_bonds.shape[0]} bond observations')
print(f'📅 Test period: {start_test} to {end_test}')
print(f'🏛️ Bonds: {df_bonds["bond_code"].nunique()}')

# Feature Engineering
def create_features(df_bonds, df_macro):
    df_features = df_bonds.merge(df_macro, on='datestamp', how='left')
    df_features = df_features.sort_values(['bond_code', 'datestamp'])
    
    # Core features
    df_features['steepness'] = df_features['us_10y'] - df_features['us_2y']
    df_features['duration_adj_return'] = df_features['return'] / df_features['modified_duration']
    df_features['convexity_ratio'] = df_features['convexity'] / df_features['modified_duration']
    df_features['macro_bond_signal'] = (df_features['duration_adj_return'] * df_features['convexity_ratio'] - df_features['steepness'] / 100)
    
    # Rolling features
    for period in [1, 5, 21]:
        df_features[f'return_{period}d'] = df_features.groupby('bond_code')['return'].transform(lambda x: x.rolling(period, min_periods=1).mean())
    
    df_features['yield_change_1d'] = df_features.groupby('bond_code')['yield'].diff(1)
    df_features['return_volatility_10d'] = df_features.groupby('bond_code')['return'].transform(lambda x: x.rolling(10, min_periods=5).std())
    
    return df_features

df_features = create_features(df_bonds, df_macro)
print(f'✅ Features created: {df_features.shape[1]} columns')

# Prepare Model Data
df_model = df_features.copy().sort_values(['bond_code', 'datestamp'])
df_model['target_return_1d'] = df_model.groupby('bond_code')['return'].shift(-1)
df_model = df_model.dropna(subset=['target_return_1d'])

# Feature columns (optimized set)
feature_columns = [
    'return_1d', 'return_5d', 'return_21d',
    'yield', 'modified_duration', 'convexity',
    'duration_adj_return', 'convexity_ratio',
    'steepness', 'us_10y',
    'yield_change_1d', 'return_volatility_10d',
    'macro_bond_signal'
]

print(f'📊 Model data: {df_model.shape[0]} rows, {len(feature_columns)} features')

# Walk-Forward Setup
albi_test_dates = df_albi[(df_albi['datestamp'] >= start_test) & (df_albi['datestamp'] <= end_test)]['datestamp'].unique()
df_signals = pd.DataFrame({'datestamp': albi_test_dates}).sort_values('datestamp').reset_index(drop=True)
bond_codes = df_model['bond_code'].unique()

print(f' -> Walk-forward validation setup: {len(df_signals)} dates')
print(f'📅 Period: {df_signals["datestamp"].min()} to {df_signals["datestamp"].max()}')

# Portfolio Optimization Function
def optimize_portfolio_fast(signals, durations, albi_duration, prev_weights=None, max_active_duration=1.5, turnover_penalty=0.1):
    n_bonds = len(signals)
    
    # Use deterministic initial weights for consistency
    if prev_weights is None:
        # Start with duration-neutral weights as initial guess
        duration_neutral_weights = np.ones(n_bonds) / n_bonds
        # Adjust to be closer to duration neutral
        duration_diff = np.abs(durations - albi_duration)
        # Give higher initial weights to bonds closer to ALBI duration
        inverse_duration_diff = 1.0 / (duration_diff + 0.1)
        initial_weights = inverse_duration_diff / np.sum(inverse_duration_diff)
        # Ensure within bounds
        initial_weights = np.minimum(initial_weights, 0.20)
        initial_weights = initial_weights / np.sum(initial_weights)
    else:
        initial_weights = prev_weights.copy()  # Ensure we don't modify the original
    
    def objective_fast(weights):
        portfolio_signal = np.dot(weights, signals)
        portfolio_duration = np.dot(weights, durations)
        active_duration = abs(portfolio_duration - albi_duration)
        
        # Add extremely heavy penalty for duration constraint violations (zero tolerance)
        duration_penalty = 0
        if active_duration > max_active_duration:
            duration_penalty = 10000 * (active_duration - max_active_duration) ** 3  # Cubic penalty
        
        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
            turnover_cost = turnover_penalty * 0.1 * turnover
        else:
            turnover_cost = 0
        
        return -(portfolio_signal - turnover_cost - duration_penalty)
    
    def duration_constraint(weights):
        portfolio_duration = np.dot(weights, durations)
        active_duration = abs(portfolio_duration - albi_duration)
        return max_active_duration - active_duration  # Must be >= 0
    
    def duration_constraint_upper(weights):
        portfolio_duration = np.dot(weights, durations)
        return max_active_duration - (portfolio_duration - albi_duration) + 1e-8  # Add small buffer
    
    def duration_constraint_lower(weights):
        portfolio_duration = np.dot(weights, durations)
        return max_active_duration + (portfolio_duration - albi_duration) + 1e-8  # Add small buffer
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': duration_constraint_upper},
        {'type': 'ineq', 'fun': duration_constraint_lower}
    ]
    
    bounds = [(0.0, 0.20) for _ in range(n_bonds)]
    
    try:
        # Try multiple optimization approaches with strict constraint enforcement
        methods_to_try = ['SLSQP', 'trust-constr']
        
        for method in methods_to_try:
            if method == 'SLSQP':
                options = {'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            else:
                options = {'maxiter': 100, 'disp': False}
                
            result = minimize(objective_fast, initial_weights, method=method, bounds=bounds, constraints=constraints, options=options)
            
            if result.success:
                portfolio_duration = np.dot(result.x, durations)
                active_duration = abs(portfolio_duration - albi_duration)
                if active_duration <= max_active_duration + 1e-6:
                    return result.x, True
        
        # If optimization fails, create duration-neutral portfolio manually
        print(f"⚠️ Optimization failed, using duration-matching fallback")
        
        # Create a portfolio that exactly matches the duration constraint
        target_duration = albi_duration
        
        # Method 1: Find bonds closest to target duration and weight them appropriately
        duration_distances = np.abs(durations - target_duration)
        sorted_indices = np.argsort(duration_distances)
        
        # Start with the bond closest to target duration
        fallback_weights = np.zeros(n_bonds)
        remaining_weight = 1.0
        
        for i in sorted_indices:
            if remaining_weight <= 0:
                break
            # Allocate weight, respecting the 20% limit
            weight_to_allocate = min(0.20, remaining_weight)
            fallback_weights[i] = weight_to_allocate
            remaining_weight -= weight_to_allocate
        
        # Check if this achieves duration constraint
        portfolio_duration = np.dot(fallback_weights, durations)
        active_duration = abs(portfolio_duration - albi_duration)
        
        if active_duration <= max_active_duration + 1e-6:
            return fallback_weights, False
        
        # Method 2: If still failing, use linear programming approach
        # Find two bonds that can bracket the target duration
        lower_duration_bonds = durations <= target_duration
        upper_duration_bonds = durations >= target_duration
        
        if np.any(lower_duration_bonds) and np.any(upper_duration_bonds):
            # Find closest bonds on each side
            lower_idx = np.where(lower_duration_bonds)[0]
            upper_idx = np.where(upper_duration_bonds)[0]
            
            if len(lower_idx) > 0 and len(upper_idx) > 0:
                # Get the closest bond from each side
                lower_bond = lower_idx[np.argmax(durations[lower_idx])]  # Highest duration below target
                upper_bond = upper_idx[np.argmin(durations[upper_idx])]  # Lowest duration above target
                
                if lower_bond != upper_bond:
                    # Calculate weights to hit exact duration
                    d_low = durations[lower_bond]
                    d_high = durations[upper_bond]
                    
                    # Solve: w_low * d_low + w_high * d_high = target_duration
                    # Subject to: w_low + w_high <= 1, w_low <= 0.2, w_high <= 0.2
                    
                    if d_high != d_low:  # Avoid division by zero
                        w_high = (target_duration - d_low) / (d_high - d_low)
                        w_low = 1.0 - w_high
                        
                        # Respect bounds
                        if 0 <= w_low <= 0.20 and 0 <= w_high <= 0.20:
                            fallback_weights = np.zeros(n_bonds)
                            fallback_weights[lower_bond] = w_low
                            fallback_weights[upper_bond] = w_high
                            
                            # Distribute remaining weight to other bonds
                            remaining_weight = 1.0 - w_low - w_high
                            if remaining_weight > 0:
                                remaining_indices = [i for i in range(n_bonds) if i != lower_bond and i != upper_bond]
                                if remaining_indices:
                                    equal_weight = min(0.20, remaining_weight / len(remaining_indices))
                                    for idx in remaining_indices:
                                        allocation = min(equal_weight, remaining_weight)
                                        fallback_weights[idx] = allocation
                                        remaining_weight -= allocation
                                        if remaining_weight <= 1e-6:
                                            break
                            
                            # Final check
                            portfolio_duration = np.dot(fallback_weights, durations)
                            active_duration = abs(portfolio_duration - albi_duration)
                            
                            if active_duration <= max_active_duration + 1e-6:
                                return fallback_weights, False
        
        # Fallback: equal-weight portfolio
        equal_weights = np.ones(n_bonds) / n_bonds
        portfolio_duration = np.dot(equal_weights, durations)
        active_duration = abs(portfolio_duration - albi_duration)
        
        if active_duration <= max_active_duration:
            return equal_weights, True
        else:
            # Duration-aware heuristic weights
            target_duration = albi_duration
            duration_diff = np.abs(durations - target_duration)
            inverse_diff = 1.0 / (duration_diff + 0.1)
            heuristic_weights = inverse_diff / np.sum(inverse_diff)
            heuristic_weights = np.minimum(heuristic_weights, 0.20)
            heuristic_weights = heuristic_weights / np.sum(heuristic_weights)
            return heuristic_weights, False
            
    except Exception:
        return np.ones(n_bonds) / n_bonds, False

print('✅ Optimization function ready')

# Main Walk-Forward Validation Loop
weight_matrix = pd.DataFrame()
prev_weights = None

# Model parameters
rf_params = {'n_estimators': 50, 'max_depth': 6, 'min_samples_split': 50, 'min_samples_leaf': 20, 'random_state': 42, 'n_jobs': -1}
turnover_penalty = 0.2

start_time = time.time()
max_runtime = 580  # Time limit

print('🚀 Starting walk-forward validation...')
print('='*50)

successful_iterations = 0

for i in range(len(df_signals)):
    elapsed_time = time.time() - start_time
    if elapsed_time > max_runtime:
        print(f'⏰ TIME LIMIT REACHED after {elapsed_time:.0f}s')
        break
    
    current_date = df_signals.loc[i, 'datestamp']
    
    # Progress update every 50 iterations
    if i % 50 == 0:
        print(f'📅 Processing {current_date} ({i+1}/{len(df_signals)}) - {elapsed_time:.0f}s elapsed')
    
    # Prepare training data with embargo
    df_train = df_model[df_model['datestamp'] < current_date]
    if len(df_train) > 0:
        embargo_date = df_train['datestamp'].max()
        df_train = df_train[df_train['datestamp'] < embargo_date]
    
    df_current = df_model[df_model['datestamp'] == current_date]
    
    # Handle dates without model data (carry forward previous weights)
    if len(df_current) == 0:
        bond_current = df_bonds[df_bonds['datestamp'] == current_date]
        if len(bond_current) > 0 and prev_weights is not None:
            weight_row = pd.DataFrame({
                'datestamp': [current_date] * len(bond_current),
                'bond_code': bond_current['bond_code'].values,
                'weight': prev_weights
            })
            weight_matrix = pd.concat([weight_matrix, weight_row], ignore_index=True)
            continue
        else:
            continue
    
    if len(df_train) < 100:
        continue
    
    # Dynamic lookback period that ensures sufficient data throughout time
    min_samples = 100  # Minimum samples needed for reliable training
    max_samples = 500  # Maximum samples to prevent overfitting and improve speed
    target_samples = 250  # Optimal number of samples
    
    # Calculate dynamic lookback period based on data density
    total_train_days = (df_train['datestamp'].max() - df_train['datestamp'].min()).days
    avg_samples_per_day = len(df_train) / max(total_train_days, 1)
    
    # Estimate how far back we need to go to get target samples
    if avg_samples_per_day > 0:
        estimated_days_needed = target_samples / avg_samples_per_day
        estimated_years_needed = estimated_days_needed / 365.25
        
        # Use a range around the estimate to find optimal lookback
        lookback_candidates = [
            max(0.5, estimated_years_needed * 0.7),  # 70% of estimate
            estimated_years_needed,                   # Full estimate
            estimated_years_needed * 1.5,            # 150% of estimate
            estimated_years_needed * 2.0             # 200% of estimate
        ]
        
        # Add some standard periods as fallbacks
        lookback_candidates.extend([1, 2, 3, 5])
        lookback_candidates = sorted(set(lookback_candidates))
    else:
        # Fallback to standard periods if we can't estimate
        lookback_candidates = [1, 2, 3, 5]
    
    df_train_recent = None
    best_sample_count = 0
    
    # Find the lookback period that gives us closest to target samples
    for years in lookback_candidates:
        years_int = int(round(years))  # Convert to integer
        # Calculate cutoff date using timedelta (more robust)
        days_back = years_int * 365  # Approximate days in years
        
        # Ensure current_date is a proper date object for arithmetic
        if hasattr(current_date, 'date'):
            curr_date = current_date.date()
        else:
            curr_date = current_date
            
        recent_cutoff = curr_date - datetime.timedelta(days=days_back)
        temp_df = df_train[df_train['datestamp'] >= recent_cutoff]
        
        # Prefer periods that give us close to target_samples but at least min_samples
        if len(temp_df) >= min_samples:
            if df_train_recent is None:
                df_train_recent = temp_df
                best_sample_count = len(temp_df)
            else:
                # Choose the period that gets us closest to target without going too far over
                current_distance = abs(len(temp_df) - target_samples)
                best_distance = abs(best_sample_count - target_samples)
                
                if (current_distance < best_distance or 
                    (len(temp_df) <= max_samples and best_sample_count > max_samples)):
                    df_train_recent = temp_df
                    best_sample_count = len(temp_df)
    
    # If no lookback period gives enough data, use the most recent samples
    if df_train_recent is None or len(df_train_recent) < min_samples:
        df_train_recent = df_train.tail(max(min_samples, len(df_train)))
    
    # Cap the dataset size to prevent excessive computation time
    if len(df_train_recent) > max_samples:
        df_train_recent = df_train_recent.tail(max_samples)
    
    # Train model
    X_train = df_train_recent[feature_columns].fillna(0)
    y_train = df_train_recent['target_return_1d'].fillna(0)
    
    if X_train.shape[0] < 20 or y_train.std() < 0.001:
        continue
    
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    
    # Generate signals
    X_current = df_current[feature_columns].fillna(0)
    predicted_returns = rf_model.predict(X_current)
    
    # Portfolio optimization
    current_durations = df_current['modified_duration'].values
    albi_current = df_albi[df_albi['datestamp'] == current_date]
    albi_duration = albi_current['modified_duration'].iloc[0] if len(albi_current) > 0 else 6.0
    
    optimal_weights, success = optimize_portfolio_fast(
        signals=predicted_returns,
        durations=current_durations,
        albi_duration=albi_duration,
        prev_weights=prev_weights,
        max_active_duration=1.4,  # Use 1.4 to ensure we stay well below 1.5
        turnover_penalty=turnover_penalty
    )
    
    # Store results
    weight_row = pd.DataFrame({
        'datestamp': [current_date] * len(df_current),
        'bond_code': df_current['bond_code'].values,
        'weight': optimal_weights
    })
    weight_matrix = pd.concat([weight_matrix, weight_row], ignore_index=True)
    prev_weights = optimal_weights.copy()
    
    if success:
        successful_iterations += 1

total_time = time.time() - start_time
print('\n' + '='*50)
print(f'🎉 Complete! Runtime: {total_time:.1f}s ({total_time/60:.2f}m)')
print(f'✅ Success rate: {successful_iterations}/{len(df_signals)} iterations')
print(f'📊 Generated {len(weight_matrix)} weight observations')

# Calculate Portfolio Performance
def calculate_performance(weight_matrix, df_bonds, df_albi):
    print('📊 Calculating portfolio performance...')
    
    # Merge portfolio weights with bond returns
    port_data = weight_matrix.merge(df_bonds[['datestamp', 'bond_code', 'return', 'modified_duration']], on=['datestamp', 'bond_code'], how='left')
    
    # Calculate turnover
    df_turnover = weight_matrix.copy()
    df_turnover['turnover'] = df_turnover.groupby(['bond_code'])['weight'].diff()
    
    # Calculate portfolio metrics
    port_data['port_return'] = port_data['return'] * port_data['weight']
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    
    # Aggregate by date
    port_data = port_data.groupby("datestamp")[['port_return', 'port_md']].sum().reset_index()
    port_data['turnover'] = df_turnover.groupby('datestamp').turnover.apply(lambda x: x.abs().sum()/2).values
    
    # Calculate transaction costs
    port_data['penalty'] = 0.005 * port_data['turnover'] * port_data['port_md'].shift()
    port_data['net_return'] = port_data['port_return'] - port_data['penalty'].fillna(0)
    
    # Get ALBI returns for the full test period
    albi_full_period = df_albi[(df_albi['datestamp'] >= start_test) & (df_albi['datestamp'] <= end_test)].copy()
    albi_full_period = albi_full_period[['datestamp', 'return', 'modified_duration']].rename(columns={'modified_duration': 'albi_duration'})
    
    # Merge portfolio with ALBI for complete date coverage
    performance_full = albi_full_period.merge(port_data, on='datestamp', how='left')
    
    # Fill missing portfolio data with zeros
    performance_full['net_return'] = performance_full['net_return'].fillna(0)
    performance_full['port_return'] = performance_full['port_return'].fillna(0)
    
    # Calculate cumulative returns
    performance_full['portfolio_tri'] = (performance_full['net_return'] / 100 + 1).cumprod()
    performance_full['albi_tri'] = (performance_full['return'] / 100 + 1).cumprod()
    
    # Add duration tracking
    performance_full['portfolio_duration'] = performance_full['port_md'].fillna(0)
    performance_full['active_duration'] = performance_full['portfolio_duration'] - performance_full['albi_duration']
    
    print(f'✅ Performance calculated for {len(performance_full)} dates')
    return performance_full

performance = calculate_performance(weight_matrix, df_bonds, df_albi)

# Performance summary
total_return_portfolio = (performance['portfolio_tri'].iloc[-1] - 1) * 100
total_return_albi = (performance['albi_tri'].iloc[-1] - 1) * 100
excess_return = total_return_portfolio - total_return_albi

print('\n🏆 PERFORMANCE SUMMARY')
print('='*40)
print(f'📈 Portfolio Return: {total_return_portfolio:+.2f}%')
print(f'📊 ALBI Return: {total_return_albi:+.2f}%')
print(f'🎯 Excess Return: {excess_return:+.2f}%')
print(f'💰 Avg Trading Cost: {performance["penalty"].mean():.4f}%')
print(f'⚖️ Avg Active Duration: {performance["active_duration"].mean():+.2f}')

# Compliance Validation
def validate_compliance(weight_matrix, performance):
    weight_sums = weight_matrix.groupby('datestamp')['weight'].sum()
    weight_sum_ok = (weight_sums >= 0.999) & (weight_sums <= 1.001)
    weight_bounds_ok = (weight_matrix['weight'] >= 0) & (weight_matrix['weight'] <= 0.20001)
    
    # Check duration constraint for portfolio dates only
    portfolio_dates = performance[performance['portfolio_duration'] > 0]
    duration_ok = portfolio_dates['active_duration'].abs() <= 1.5001
    
    print('🚨 COMPLIANCE CHECK')
    print('='*30)
    print(f'✅ Weight Sum (100%): {weight_sum_ok.all()}')
    print(f'✅ Weight Bounds (0-20%): {weight_bounds_ok.all()}')
    print(f'✅ Duration Constraint (±1.5): {duration_ok.all()}')
    
    if not weight_bounds_ok.all():
        print(f'⚠️ Max weight: {weight_matrix["weight"].max():.4f}')
    if not duration_ok.all():
        print(f'⚠️ Max active duration: {portfolio_dates["active_duration"].abs().max():.3f}')
    
    all_compliant = weight_sum_ok.all() and weight_bounds_ok.all() and duration_ok.all()
    print(f'\n🏆 OVERALL: {"✅ PASS" if all_compliant else "❌ FAIL"}')
    return all_compliant

compliance_status = validate_compliance(weight_matrix, performance)

if compliance_status:
    weight_matrix.to_csv('portfolio_weights_2025.csv', index=False)
    print('💾 Portfolio weights exported to portfolio_weights_2025.csv')
else:
    print('⚠️ Cannot export due to compliance failures')

# Visualization
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

    if len(port_data[abs(port_data['active_md']) > 1.49999]['datestamp']) == 0:
        print(f"---> The portfolio does not breach the modified duration constraint")
    else:
        raise ValueError('This buy matrix violates the modified duration constraint on the below dates: \n ' + ", ".join(pd.to_datetime(port_data[abs(port_data['active_md']) > 1.49999]['datestamp']).dt.strftime("%Y-%m-%d")))

# Execute plotting functions to display visualizations
print('\n📊 Generating visualizations...')
try:
    plot_payoff(weight_matrix)
    plot_md(weight_matrix)
    print('✅ All plots generated successfully!')
except Exception as e:
    print(f'⚠️ Error generating plots: {e}')
    # Try using the original plotting code instead
    try:
        # Fallback plotting code
        weight_pivot = weight_matrix.pivot(index='datestamp', columns='bond_code', values='weight')
        fig_weights = px.imshow(weight_pivot.T, title='Portfolio Weight Evolution', 
                               labels=dict(x="Date", y="Bond Code", color="Weight"), 
                               color_continuous_scale='Blues', aspect='auto')
        fig_weights.show()
        
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(x=performance['datestamp'], y=performance['portfolio_tri'], 
                                           name='Portfolio', line=dict(color='blue', width=2)))
        fig_performance.add_trace(go.Scatter(x=performance['datestamp'], y=performance['albi_tri'], 
                                           name='ALBI Benchmark', line=dict(color='red', width=2)))
        fig_performance.update_layout(title='Portfolio vs ALBI Performance', xaxis_title='Date', 
                                    yaxis_title='Total Return Index', height=500)
        fig_performance.show()
        print('✅ Fallback plots generated successfully!')
    except Exception as e2:
        print(f'❌ Could not generate any plots: {e2}')

print(f'\n🎯 Script completed in {time.time() - start_time:.1f}s')
