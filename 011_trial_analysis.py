# Task 2: Experimentation and uplift testing

"""
trial_analysis.py
Run: python trial_analysis.py
Outputs:
 - outputs/match_table.csv
 - outputs/store_results.csv
 - outputs/<store_id>_timeseries.png
 - outputs/<store_id>_prepost_bar.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import math

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ------------- Parameters to set -------------
TX_CSV = "tx_chips_clean.csv"   # cleaned chips transactions
TRIAL_STORES = [101, 102]       # replace with actual trial store ids
TRIAL_START = pd.to_datetime("2019-03-01")  # change to actual trial start
TRIAL_END   = pd.to_datetime("2019-03-31")  # trial end
PRE_PERIOD_WEEKS = 8
N_CONTROLS = 2                   # number of matched controls per trial store
ALPHA = 0.05                      # significance level
# --------------------------------------------

# Load data
tx = pd.read_csv(TX_CSV, parse_dates=['date'])
tx = tx[tx['is_chips'].astype(bool)]   # ensure chips only

# create daily store-level aggregates
daily = tx.groupby(['store_id','date']).agg(
    daily_revenue=('line_total','sum'),
    daily_units=('quantity','sum'),
    daily_txns=('transaction_id','nunique')
).reset_index()

# Helper: pre and post masks
trial_start = TRIAL_START
pre_start = TRIAL_START - pd.Timedelta(weeks=PRE_PERIOD_WEEKS)
pre_mask = (daily['date'] >= pre_start) & (daily['date'] < trial_start)
post_mask = (daily['date'] >= TRIAL_START) & (daily['date'] <= TRIAL_END)

# compute pre-period summary features per store for matching
pre = daily[pre_mask].copy()
store_features = pre.groupby('store_id').agg(
    pre_mean_rev=('daily_revenue','mean'),
    pre_median_rev=('daily_revenue','median'),
    pre_std_rev=('daily_revenue','std'),
    pre_mean_units=('daily_units','mean'),
    pre_weekday_pattern = ('daily_revenue', lambda s: s.groupby(s.index % 7).mean().mean()) # simple
).reset_index()

# more robust: weekday pattern vector (mon-sun) - build matrix
# We'll compute mean revenue by weekday for each store (0..6)
weekday = pre.copy()
weekday['weekday'] = weekday['date'].dt.weekday
wk_pivot = weekday.pivot_table(index='store_id', columns='weekday', values='daily_revenue', aggfunc='mean', fill_value=0)
# Merge pivot into store_features
store_features = store_features.merge(wk_pivot.reset_index(), on='store_id', how='left')

# Matching: scale features & use nearest neighbors
feature_cols = [c for c in store_features.columns if c not in ['store_id']]
X = store_features[feature_cols].fillna(0).values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=N_CONTROLS+1, algorithm='auto').fit(Xs)  # +1 because neighbor includes itself
distances, indices = nbrs.kneighbors(Xs)

# Build match table
match_rows = []
for i, sid in enumerate(store_features['store_id'].values):
    neigh = indices[i,1:1+N_CONTROLS]  # skip index 0 = itself
    controls = store_features['store_id'].values[neigh].tolist()
    match_rows.append({'store_id': sid, 'controls': controls, 'distances': distances[i,1:1+N_CONTROLS].tolist()})
match_df = pd.DataFrame(match_rows)
match_df.to_csv(OUT / "match_table.csv", index=False)

# For each trial store, pick its matched controls and run DiD
results = []
for trial in TRIAL_STORES:
    # find controls from match_df
    ctrls = match_df.loc[match_df['store_id']==trial,'controls'].iloc[0]
    print("Trial", trial, "controls", ctrls)

    # subset daily data for trial + controls for pre+post window
    mask = (daily['store_id'].isin([trial]+ctrls)) & (daily['date'] >= pre_start) & (daily['date'] <= TRIAL_END)
    df = daily[mask].copy()

    # label store type and period
    df['is_trial'] = (df['store_id'] == trial).astype(int)
    df['period'] = np.where(df['date'] >= trial_start, 'post', 'pre')
    df['post'] = (df['period']=='post').astype(int)

    # DID regression: daily_revenue ~ is_trial + post + is_trial:post + store_fe + date_fe(optional)
    # add store fixed effects via store dummies
    df['store_id_str'] = df['store_id'].astype(str)
    model = smf.ols('daily_revenue ~ is_trial + post + is_trial:post + C(store_id_str)', data=df).fit(cov_type='HC1')
    coef = model.params.get('is_trial:post', np.nan)
    se = model.bse.get('is_trial:post', np.nan)
    pval = model.pvalues.get('is_trial:post', np.nan)
    # compute agg means
    mean_pre_trial = df[(df['is_trial']==1) & (df['post']==0)]['daily_revenue'].mean()
    mean_post_trial = df[(df['is_trial']==1) & (df['post']==1)]['daily_revenue'].mean()
    mean_pre_ctrl = df[(df['is_trial']==0) & (df['post']==0)]['daily_revenue'].mean()
    mean_post_ctrl = df[(df['is_trial']==0) & (df['post']==1)]['daily_revenue'].mean()
    # DiD naive estimate:
    did_point = (mean_post_trial - mean_pre_trial) - (mean_post_ctrl - mean_pre_ctrl)
    # Save results
    results.append({
        'trial_store': trial,
        'controls': ctrls,
        'did_coef': coef,
        'did_se': se,
        'did_pval': pval,
        'did_point_agg': did_point,
        'mean_pre_trial': mean_pre_trial,
        'mean_post_trial': mean_post_trial,
        'mean_pre_ctrl': mean_pre_ctrl,
        'mean_post_ctrl': mean_post_ctrl
    })

    # bootstrap CI for agg point estimate (store-level daily diffs)
    # compute daily diff series: diff = (trial_daily_rev - avg_ctrl_daily_rev) for days
    trial_series = df[df['is_trial']==1][['date','daily_revenue']].set_index('date').sort_index()
    ctrl_series = df[df['is_trial']==0].groupby('date')['daily_revenue'].mean().reset_index().set_index('date').sort_index()
    merged_series = trial_series.join(ctrl_series, lsuffix='_trial', rsuffix='_ctrl', how='inner').dropna()
    merged_series['diff'] = merged_series['daily_revenue_trial'] - merged_series['daily_revenue_ctrl']

    # bootstrap on days
    nboot = 2000
    boot_means = []
    merged_vals = merged_series['diff'].values
    rng = np.random.default_rng(42)
    for _ in range(nboot):
        sample = rng.choice(merged_vals, size=len(merged_vals), replace=True)
        boot_means.append(sample.mean())
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    # viz: time series with pre/post shaded
    fig, ax = plt.subplots(figsize=(10,4))
    # plot trial and avg control
    df_plot = df.groupby(['date','is_trial']).daily_revenue.mean().reset_index()
    for is_t, group in df_plot.groupby('is_trial'):
        label = 'Trial' if is_t==1 else 'Controls (avg)'
        ax.plot(group['date'], group['daily_revenue'], label=label)
    ax.axvline(trial_start, color='k', linestyle='--', label='Trial start')
    ax.set_title(f"Store {trial} - Daily revenue (trial vs controls)")
    ax.legend()
    fig.savefig(OUT / f"{trial}_timeseries.png")
    plt.close(fig)

    # bar: pre vs post mean (trial vs controls)
    fig, ax = plt.subplots(figsize=(6,4))
    bars = [mean_pre_trial, mean_post_trial, mean_pre_ctrl, mean_post_ctrl]
    labels = ['Trial Pre','Trial Post','Ctrl Pre','Ctrl Post']
    ax.bar(labels, bars)
    ax.set_title(f"Store {trial} - Pre/Post avg daily revenue")
    fig.savefig(OUT / f"{trial}_prepost_bar.png")
    plt.close(fig)

    # append bootstrap CI
    results[-1].update({'boot_mean_diff': merged_series['diff'].mean(), 'boot_ci_low': ci_low, 'boot_ci_high': ci_high})

res_df = pd.DataFrame(results)
# multiple testing correction (BH)
from statsmodels.stats.multitest import multipletests
pvals = res_df['did_pval'].fillna(1).values
rej, pvals_corr, _, _ = multipletests(pvals, alpha=ALPHA, method='fdr_bh')
res_df['pval_adj_bh'] = pvals_corr
res_df['reject_bh'] = rej
res_df.to_csv(OUT / "store_results.csv", index=False)

print("Done. Outputs in", OUT)
