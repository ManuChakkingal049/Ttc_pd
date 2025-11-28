import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
np.random.seed(42)
N_CLIENTS = 50
YEARS = [2020, 2021, 2022, 2023, 2024]
RATINGS = ['1','2','3','4','5','D']
HORIZON_MONTHS = 12

# ---------------------------
# Generate dummy annual December snapshot data
# ---------------------------
rows = []
client_ids = [f"C{str(i).zfill(3)}" for i in range(1, N_CLIENTS+1)]

for year in YEARS:
    snapshot_date = pd.Timestamp(f"{year}-12-31")
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.08, 0.02])
    probs = probs / probs.sum()
    ratings = np.random.choice(RATINGS, size=N_CLIENTS, p=probs)
    for cid, r in zip(client_ids, ratings):
        rows.append({'snapshot_date': snapshot_date, 'client_id': cid, 'rating': r})

df = pd.DataFrame(rows)

# ---------------------------
# Utility functions
# ---------------------------
def compute_migrations(df, horizon_months=12):
    results = []
    snapshot_dates = sorted(df['snapshot_date'].unique())

    for start_snap in snapshot_dates:
        start_slice = df[df['snapshot_date']==start_snap].set_index('client_id')
        end_snap = start_snap + pd.DateOffset(months=horizon_months)
        if end_snap not in snapshot_dates:
            continue
        end_slice = df[df['snapshot_date']==end_snap].set_index('client_id')

        merged = start_slice[['rating']].rename(columns={'rating':'rating_start'}).join(
            end_slice[['rating']].rename(columns={'rating':'rating_end'}),
            how='left')
        merged['EXIT_FLAG'] = merged['rating_end'].isna()

        counts_start = merged['rating_start'].value_counts().reindex(RATINGS, fill_value=0)
        exits_total = int(merged['EXIT_FLAG'].sum())
        total_start = len(merged)
        nonexit_start = total_start - exits_total

        cols = RATINGS + ['EXIT']
        mat_counts = pd.DataFrame(0, index=RATINGS, columns=cols)
        for r in RATINGS:
            grp = merged[merged['rating_start']==r]
            for r_end in RATINGS:
                mat_counts.loc[r, r_end] = (grp['rating_end']==r_end).sum()
            mat_counts.loc[r,'EXIT'] = grp['EXIT_FLAG'].sum()

        mat_pct = mat_counts.copy().astype(float)
        denom = mat_counts.sum(axis=1) - mat_counts['EXIT']
        for r in RATINGS:
            if denom.loc[r] > 0:
                mat_pct.loc[r,:] = mat_counts.loc[r,:] / denom.loc[r]
            else:
                mat_pct.loc[r,:] = np.nan

        results.append({
            'snapshot': start_snap,
            'end_date': end_snap,
            'counts_start': counts_start,
            'exits_total': exits_total,
            'total_start': total_start,
            'nonexit_start': nonexit_start,
            'mat_counts': mat_counts,
            'mat_pct_excl_exit': mat_pct,
            'per_client': merged.reset_index()
        })
    return results

def average_pct_matrices(results):
    mats = [r['mat_pct_excl_exit'] for r in results]
    keys = [r['snapshot'] for r in results]
    stacked = pd.concat(mats, keys=keys)
    return stacked.groupby(level=1).mean()

# ---------------------------
# Streamlit app
# ---------------------------
st.title("Roll-rate / Migration Analysis (Fast Demo)")

st.markdown("""
**Nature of demo data:**
- Annual December snapshots
- 5 years: 2020–2024
- Maximum 50 clients
- Random dummy ratings
- Overlapping horizons of 12 months (each snapshot compares to next year)
""")

# Show raw data
with st.expander("View raw snapshot data (initial dataset)"):
    st.markdown("""
This shows the **raw input data** used for the migration analysis.  
Columns:
- `snapshot_date`: snapshot month/year
- `client_id`: unique client identifier
- `rating`: assigned rating (1–5 or D)
""")
    st.dataframe(df)

# Compute migrations
migrations = compute_migrations(df)

# Snapshot summary
st.subheader("Snapshot summary")
summary_rows = []
for r in migrations:
    row = {
        'snapshot': r['snapshot'].date(),
        'end_date': r['end_date'].date(),
        'total_start': r['total_start'],
        'nonexit_start': r['nonexit_start'],
        'exits_total': r['exits_total']
    }
    for rt in RATINGS:
        row[f"count_{rt}"] = int(r['counts_start'][rt])
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df)

# Compare three snapshots side by side
st.subheader("Client ratings across snapshots (common clients only)")
sel = st.multiselect(
    "Select up to 3 snapshots",
    options=[r['snapshot'].date() for r in migrations],
    default=[r['snapshot'].date() for r in migrations[:3]]
)
if len(sel)>=2:
    dfs = []
    for s in sel:
        tmp = df[df['snapshot_date'].dt.date==s][['client_id','rating']].rename(columns={'rating':f'rating_{s}'})
        dfs.append(tmp.set_index('client_id'))
    common_idx = set.intersection(*[set(d.index) for d in dfs])
    if common_idx:
        merged = pd.concat([d.loc[sorted(common_idx)] for d in dfs], axis=1)
        st.dataframe(merged)
    else:
        st.write("No common clients found between selected snapshots")

# Migration matrices
st.subheader("Migration matrices per snapshot")
snap_choice = st.selectbox("Pick snapshot", options=[r['snapshot'].date() for r in migrations])
sel_mig = next(r for r in migrations if r['snapshot'].date()==snap_choice)
matrix_type = st.radio("Matrix type", options=['Counts','Percent (excl exits)'])
if matrix_type=='Counts':
    st.dataframe(sel_mig['mat_counts'])
else:
    st.dataframe(sel_mig['mat_pct_excl_exit'].round(4))

# Average TTC matrix
st.subheader("Average (TTC) migration matrix")
st.dataframe(average_pct_matrices(migrations).round(4))

# Histogram of absolute moves
st.subheader("Histogram of client absolute rating moves (demo)")
data = sel_mig['per_client'].copy()
data['num_start'] = data['rating_start'].map(lambda x: 6 if x=='D' else int(x))
data['num_end'] = data['rating_end'].map(lambda x: 6 if x=='D' else int(x))
data['abs_move'] = (data['num_end'] - data['num_start']).abs()
plt.figure()
plt.hist(data['abs_move'].dropna(), bins=range(0,8), edgecolor='k')
plt.xlabel("Absolute rating move ('D'=6)")
plt.ylabel("Count of clients")
st.pyplot(plt)

# Interactive explanations
with st.expander("How exit is inferred (click to expand)"):
    st.markdown("""
We **do not** require an 'exited' column.  
- A client present at the snapshot but missing in the horizon-end snapshot is considered **exited/unobserved**.  
- EXIT clients appear in the 'EXIT' column in migration matrices.  
- When computing **percent migration**, we exclude EXIT clients from the denominator.  
- EXIT counts are still reported.
""")

with st.expander("Overlapping vs Non-overlapping snapshots (summary)"):
    st.markdown("""
**Nature in this demo:** Annual snapshots with 12-month horizon → overlapping if quarterly data were used.  
**Overlapping snapshots:**  
- More observations, smoother estimates, but not independent → may bias variance/backtests.  
**Non-overlapping snapshots:**  
- Approx independent observations, better for backtesting/statistics, but fewer observations → higher sampling noise.
""")

with st.expander("Choosing snapshot month (if annual)"):
    st.markdown("""
- Corporates: choose month after year-end reporting (Dec/Mar-Apr).  
- Retail: avoid months with strong seasonality (Feb/Aug).  
- Quarterly snapshots: month choice less critical.
""")
