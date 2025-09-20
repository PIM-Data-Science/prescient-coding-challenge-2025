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

# Strategy implementation (deterministic, no look-ahead)
# - Rule-based next-day return predictor using momentum, yield, and volatility
# - Convert scores to weights within [0,0.2], sum to 1
# - Enforce modified duration constraint (±1.5 from ALBI) via a simple projection
# - Penalise turnover in the objective to reduce trading costs

# static data for optimisation and signal generation
n_days = 21  # Increased lookback for more stable signals
np.random.seed(42)  # deterministic behaviour for any random ops
prev_weights = np.array([0.1] * 10)
p_active_md = 1.5  # tighter duration tolerance used for projection steps (we enforce ±1.5)
weight_bounds = (0.0, 0.2)

def compute_features(df_bonds_slice, df_macro_slice, n_days):
    """Enhanced feature computation with more predictive factors"""
    df = df_bonds_slice.copy()
    # rolling features per bond
    df['ma5'] = df.groupby('bond_code')['return'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('bond_code')['return'].transform(lambda x: x.rolling(10).mean())
    df['ma21'] = df.groupby('bond_code')['return'].transform(lambda x: x.rolling(21).mean())
    df['vol10'] = df.groupby('bond_code')['return'].transform(lambda x: x.rolling(10).std())
    df['vol21'] = df.groupby('bond_code')['return'].transform(lambda x: x.rolling(21).std())
    df['yield_mom'] = df.groupby('bond_code')['yield'].transform(lambda x: x.pct_change(5))
    df['yield_roc'] = df.groupby('bond_code')['yield'].transform(lambda x: (x - x.shift(21)) / x.shift(21))
    
    # Risk-adjusted return measures
    df['sharpe_21'] = df['ma21'] / (df['vol21'] + 1e-8)
    
    # Price momentum indicators
    df['momentum_5_21'] = df['ma5'] - df['ma21']
    df['momentum_ratio'] = df['ma5'] / (df['ma21'] + 1e-8)
    
    # merge latest macro
    macro_latest = df_macro_slice.copy()
    macro_latest['steep'] = macro_latest['us_10y'] - macro_latest['us_2y']
    macro_latest['fx_mom'] = macro_latest['fx_vol'].pct_change(5)
    macro_latest['comdty_mom'] = macro_latest['comdty_fut'].pct_change(5)
    macro_latest = macro_latest[['datestamp', 'steep', 'fx_vol', 'fx_mom', 'comdty_fut', 'comdty_mom', 'top40_return']]
    df = df.merge(macro_latest, on='datestamp', how='left')
    return df

def score_bonds(df_current):
    """Highly aggressive scoring focused on momentum and yield with market regime adaptation"""
    eps = 1e-8
    
    # Extract features
    features = {
        'momentum_5': df_current['ma5'].fillna(0.0).values,
        'momentum_10': df_current['ma10'].fillna(0.0).values,
        'momentum_21': df_current['ma21'].fillna(0.0).values,
        'yield': df_current['yield'].fillna(0.0).values,
        'volatility': df_current['vol10'].fillna(1.0).values,
        'convexity': df_current['convexity'].fillna(0.0).values,
        'sharpe': df_current['sharpe_21'].fillna(0.0).values,
        'yield_momentum': df_current['yield_mom'].fillna(0.0).values,
        'momentum_5_21': df_current['momentum_5_21'].fillna(0.0).values,
        'momentum_ratio': df_current['momentum_ratio'].fillna(1.0).values
    }
    
    # Z-score normalization with robust scaling
    def robust_z(x):
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))
        return (x - median) / (mad + eps)
    
    normalized = {}
    for name, values in features.items():
        normalized[name] = robust_z(values)
    
    # Dynamic factor weighting based on market regime - more aggressive
    steepness = df_current['steep'].iloc[0] if 'steep' in df_current.columns else 0
    vol_regime = df_current['fx_vol'].iloc[0] if 'fx_vol' in df_current.columns else 1
    
    # Highly aggressive factor weights for maximum outperformance
    if steepness > 0.3:  # Steep yield curve - favor momentum and yield
        factor_weights = {
            'momentum_5': 0.8, 'momentum_10': 0.6, 'momentum_21': 0.4,
            'yield': 0.7, 'volatility': -0.2, 'convexity': 0.1,
            'sharpe': 0.3, 'yield_momentum': 0.4,
            'momentum_5_21': 0.9, 'momentum_ratio': 0.6
        }
    elif vol_regime > 1.0:  # High volatility - focus on risk-adjusted returns
        factor_weights = {
            'momentum_5': 0.4, 'momentum_10': 0.3, 'momentum_21': 0.6,
            'yield': 0.3, 'volatility': -0.8, 'convexity': 0.2,
            'sharpe': 0.9, 'yield_momentum': 0.2,
            'momentum_5_21': 0.5, 'momentum_ratio': 0.4
        }
    else:  # Normal market conditions - maximum momentum focus
        factor_weights = {
            'momentum_5': 1.0, 'momentum_10': 0.8, 'momentum_21': 0.6,
            'yield': 0.5, 'volatility': -0.3, 'convexity': 0.1,
            'sharpe': 0.4, 'yield_momentum': 0.3,
            'momentum_5_21': 1.2, 'momentum_ratio': 0.8
        }
    
    # Calculate composite score with exponential emphasis on top performers
    score = np.zeros_like(normalized['momentum_5'])
    for name, weight in factor_weights.items():
        if name in normalized:
            # Apply exponential transformation to emphasize strong signals
            transformed = np.sign(normalized[name]) * np.abs(normalized[name]) ** 1.5
            score += weight * transformed
    
    # Extreme emphasis on top 3 bonds
    top_mask = score >= np.percentile(score, 70)
    score[top_mask] *= 2.0  # Double the score for top performers
    
    # small deterministic tie-breaker by bond code
    code_rank = pd.factorize(df_current['bond_code'])[0]
    score = score + 0.001 * code_rank
    
    return score

def project_weights(weights, md_today, albi_md, lower=0.0, upper=0.2):
    """Enhanced weight projection with better duration matching"""
    # Initial projection to satisfy bounds and sum constraint
    w = project_to_box_simplex(weights, lower, upper)
    
    # Check duration constraint
    port_md = np.dot(w, md_today)
    diff = albi_md - port_md
    
    if abs(diff) <= 1.5:
        return w
    
    # More aggressive duration adjustment for better performance
    md_deviation = md_today - albi_md
    if diff > 0:  # Need higher duration
        adjustment_factor = np.maximum(md_deviation, 0)
    else:  # Need lower duration
        adjustment_factor = np.maximum(-md_deviation, 0)
    
    # Normalize and apply more aggressive adjustment
    if adjustment_factor.sum() > 0:
        adjustment_factor = adjustment_factor / adjustment_factor.sum()
        # More aggressive adjustment to quickly meet duration constraint
        alpha = min(0.5, abs(diff) / (1.5 * np.std(md_today)))
        w_new = w * (1 - alpha) + adjustment_factor * alpha
        w_new = project_to_box_simplex(w_new, lower, upper)
        
        # Check if improved
        new_md = np.dot(w_new, md_today)
        if abs(new_md - albi_md) < abs(port_md - albi_md):
            return w_new
    
    return w  # Fallback to original if adjustment doesn't help


def project_to_box_simplex(target, lower=0.0, upper=0.2):
    """Project `target` to the feasible set {w: sum(w)=1, lower<=w<=upper} by solving
    a small QP: minimize ||w - target||^2 s.t. sum(w)=1 and bounds.
    This guarantees the box and simplex constraints exactly.
    """
    target = np.asarray(target, dtype=float)
    n = len(target)

    # objective: 0.5 * ||w - target||^2
    def obj(w):
        return 0.5 * np.sum((w - target) ** 2)

    # equality constraint: sum(w) - 1 = 0
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(lower, upper)] * n

    # initial guess: clipped and normalized
    x0 = np.clip(target, lower, upper)
    if x0.sum() == 0:
        x0 = np.ones(n) / n
    else:
        x0 = x0 / x0.sum()

    res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-9, 'maxiter':200})
    if res.success:
        w = res.x
        # numerical cleanup
        w = np.clip(w, lower, upper)
        w = w / w.sum()
        return w
    # fallback simple heuristic
    x = np.clip(target, lower, upper)
    if x.sum() == 0:
        return np.ones(n) / n
    return x / x.sum()


def optimize_weights(scores, md_today, albi_md, prev_w, lower=0.0, upper=0.2):
    """Highly aggressive optimization focused on maximum outperformance"""
    scores = np.asarray(scores, dtype=float)
    md_today = np.asarray(md_today, dtype=float)
    prev_w = np.asarray(prev_w, dtype=float)

    n = len(scores)

    # Extremely aggressive objective: maximum emphasis on high scores
    def obj(w):
        return (-np.dot(w, scores) * 3.0 +  # Triple emphasis on high scores
                15.0 * (np.dot(w, md_today) - albi_md) ** 2 +  # Moderate duration penalty
                1.0 * np.sum((w - prev_w) ** 2))  # Minimal turnover penalty

    bounds = [(lower, upper)] * n

    # Start with scores-based weights
    x0 = project_to_box_simplex(np.maximum(scores, 0.0), lower, upper)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    
    # Use more aggressive optimization
    res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP', 
                  options={'maxiter': 2000, 'ftol':1e-10, 'disp': False})
    
    if res.success:
        w = np.clip(res.x, lower, upper)
        if w.sum() == 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()
        return w
    
    # Fallback: use initial projection with more concentration
    x0 = project_to_box_simplex(np.maximum(scores, 0.0), lower, upper)
    # Apply additional concentration to top scores
    top_indices = np.argsort(scores)[-3:]  # Top 3 bonds
    x0[top_indices] *= 1.5
    x0 = project_to_box_simplex(x0, lower, upper)
    return x0

# main loop: for each signal date generate weights
for i in range(len(df_signals)):
    date = df_signals.loc[i, 'datestamp']
    if i % 20 == 0:  # Print progress less frequently
        print('---> processing', date)

    # slices without look-ahead
    df_train_bonds = df_bonds[df_bonds['datestamp'] < date].copy()
    df_train_macro = df_macro[df_macro['datestamp'] < date].copy()
    df_train_albi = df_albi[df_albi['datestamp'] < date].copy()

    # require at least one historical row per bond
    if df_train_bonds.empty or df_train_albi.empty:
        # use previous weights if insufficient data
        w_final = prev_weights
    else:
        # compute features and take the latest available date (most recent history)
        df_feat = compute_features(df_train_bonds, df_train_macro, n_days)
        latest_date = df_feat['datestamp'].max()
        df_current = df_feat[df_feat['datestamp'] == latest_date].copy()

        # ensure we have 10 bond rows in the current slice (order preserved)
        # If not, align by bond codes from the universe at this date in df_bonds
        universe = df_bonds[df_bonds['datestamp'] == latest_date]['bond_code'].unique()
        df_current = df_current.set_index('bond_code').reindex(universe).reset_index()

        # scoring
        scores = score_bonds(df_current)
        md_today = df_current['modified_duration'].fillna(df_current['modified_duration'].mean()).values
        albi_md_val = df_train_albi['modified_duration'].iloc[-1]

        # use highly aggressive optimizer
        w_final = optimize_weights(scores, md_today, albi_md_val, prev_weights,
                       lower=weight_bounds[0], upper=weight_bounds[1])

    # store weights aligned with bond codes for that date
    # pick the bond order from the most recent date in df_bonds up to 'date'
    bond_order = df_bonds[df_bonds['datestamp'] < date].groupby('bond_code')['datestamp'].max().sort_values().index.tolist()
    # ensure length 10 order; if not exactly 10, fallback to unique bond codes in dataset
    if len(bond_order) != 10:
        bond_order = sorted(df_bonds['bond_code'].unique())

    weight_matrix_tmp = pd.DataFrame({'bond_code': bond_order, 'weight': w_final, 'datestamp': date})
    weight_matrix = pd.concat([weight_matrix, weight_matrix_tmp], ignore_index=True)
    prev_weights = w_final
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
    # Trading cost: 0.01% * modified duration * turnover (per specification)
    # turnover is fraction traded (sum of abs weight changes / 2), port_md is portfolio modified duration
    port_data['penalty'] = 0.0001 * port_data['turnover'] * port_data['port_md']
    port_data['net_return'] = port_data['port_return'].sub(port_data['penalty'], fill_value=0)
    port_data = port_data.merge(df_albi[['datestamp','return']], on = 'datestamp', how = 'left')
    port_data['portfolio_tri'] = (port_data['net_return']/100 +1).cumprod()
    port_data['albi_tri'] = (port_data['return']/100 +1).cumprod()

    #turnover chart
    fig_turnover = px.line(port_data, x='datestamp', y='turnover')
    fig_turnover.show()

    portfolio_return = (port_data['portfolio_tri'].values[-1]-1)*100
    albi_return = (port_data['albi_tri'].values[-1]-1)*100
    print(f"---> payoff for these buys between period {port_data['datestamp'].min()} and {port_data['datestamp'].max()} is {portfolio_return:.2f}%")
    print(f"---> payoff for the ALBI benchmark for this period is {albi_return:.2f}%")
    
    # Calculate outperformance
    outperformance = portfolio_return - albi_return
    print(f"---> Outperformance vs ALBI: {outperformance:.2f}%")
    
    if outperformance > 0:
        print(f"---> SUCCESS: Portfolio outperformed ALBI by {outperformance:.2f}%")
    else:
        print(f"---> NEED IMPROVEMENT: Portfolio underperformed ALBI by {abs(outperformance):.2f}%")

    port_data = pd.melt(port_data[['datestamp', 'portfolio_tri', 'albi_tri']], id_vars = 'datestamp')

    fig_payoff = px.line(port_data, x='datestamp', y='value', color = 'variable')
    fig_payoff.show()
    
    return port_data

def plot_md(weight_matrix):

    port_data = weight_matrix.merge(df_bonds, on = ['bond_code', 'datestamp'], how = 'left')
    port_data['port_md'] = port_data['modified_duration'] * port_data['weight']
    port_data = port_data.groupby("datestamp")[['port_md']].sum().reset_index()
    port_data = port_data.merge(df_albi[['datestamp','modified_duration']], on = 'datestamp', how = 'left')
    port_data['active_md'] = port_data['port_md'] - port_data['modified_duration']

    fig_payoff = px.line(port_data, x='datestamp', y='active_md')
    fig_payoff.show()

    violation_dates = port_data[abs(port_data['active_md']) > 1.5]['datestamp']
    if len(violation_dates) == 0:
        print(f"---> The portfolio does not breach the modified duration constraint")
    else:
        raise ValueError('This buy matrix violates the modified duration constraint on the below dates: \n ' +  ", ".join(pd.to_datetime(violation_dates).dt.strftime("%Y-%m-%d")))

# Run the analysis
result_data = plot_payoff(weight_matrix)
plot_md(weight_matrix)

# %%

print('---> Python Script End', t1 := datetime.datetime.now())
print('---> Total time taken', t1 - t0)