# streamlit_rollrate_explainer.py
"""
Roll-rate / Migration Explorer (Streamlit)
Input format required:
  snapshot_date | client_id | rating
- Only these 3 columns required (snapshot_date must be parseable as YYYY-MM-DD month-end).
- No 'exited' column: exit is inferred if the client is missing at the horizon-end snapshot.

Features:
- Demo synthetic data (5 years monthly) if you don't upload a file.
- Snapshot frequency: Monthly, Quarterly, Annual.
- Overlapping vs Non-overlapping windows.
- Per-snapshot: counts per rating, migration counts (with EXIT), migration % excluding EXIT in denominators.
- TTC matrix: mean of percentage matrices across snapshots.
- Side-by-side view of three snapshots showing only common clients (visual migration).
- Histogram of per-client absolute moves (demo data).
- Interactive collapsible explanations at every stage (overlapping logic, exit logic, calculation steps).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Roll-rate / Migration Explorer")

# ----------------------------
# Helper functions
# ----------------------------
RATINGS = ['1','2','3','4','5','D']  # use strings consistently; 'D' = default/worst

def generate_demo_data(start='2020-01-31', months=60, n_clients=1200, seed=42):
    """Generate demo monthly data with only snapshot_date, client_id, rating."""
    np.random.seed(seed)
    start = pd.to_datetime(start)
    dates = pd.date_range(start=start, periods=months, freq='M')
    client_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_clients+1)]
    rows = []
    for dt in dates:
        # simple time-varying probabilities to add variation
        base_probs = np.array([0.30,0.28,0.20,0.12,0.08,0.02])
        shift = (dt.year - start.year) * 0.005
        base_probs = base_probs + np.array([-shift, -shift/2, 0, shift/2, shift, shift*0.2])
        base_probs = np.clip(base_probs, 0.0001, None)
        base_probs = base_probs / base_probs.sum()
        sampled = np.random.choice(RATINGS, size=n_clients, p=base_probs)
        for cid, rt in zip(client_ids, sampled):
            rows.append({'snapshot_date': dt, 'client_id': cid, 'rating': rt})
    return pd.DataFrame(rows)

def get_snapshot_dates(all_dates, freq='Q'):
    """Return available snapshot dates aligned to freq: 'M','Q','A'."""
    ds = pd.DatetimeIndex(all_dates)
    if freq == 'Q':
        snaps = ds.to_period('Q').to_timestamp('M')
    elif freq == 'A':
        snaps = ds.to_period('A').to_timestamp('M')
    elif freq == 'M':
        snaps = ds
    else:
        raise ValueError("freq must be one of 'M','Q','A'")
    snaps = pd.DatetimeIndex([s for s in snaps.unique() if s in set(all_dates)])
    return snaps.sort_values()

def compute_migrations(df, snapshot_dates, horizon_months=12, use_overlapping=True, ratings=RATINGS):
    """
    Core migration logic.
    - df: DataFrame with snapshot_date, client_id, rating (no exit column).
    - snapshot_dates: candidate snapshot month-ends (DatetimeIndex).
    - horizon_months: migration horizon.
    - use_overlapping: if False, choose non-overlapping snapshots with step = horizon_months.
    Returns a list of dicts per snapshot with counts, matrices, per-client info.
    """
    snapshot_dates = pd.DatetimeIndex(snapshot_dates)
    if not use_overlapping:
        # choose non-overlapping snapshots so that windows don't overlap
        non_overlap = []
        i = 0
        snaps = list(snapshot_dates)
        while i < len(snaps):
            non_overlap.append(snaps[i])
            target = snaps[i] + pd.DateOffset(months=horizon_months)
            j = i + 1
            while j < len(snaps) and snaps[j] < target:
                j += 1
            i = j
        snapshot_dates = pd.DatetimeIndex(non_overlap)

    results = []
    available_dates = sorted(df['snapshot_date'].unique())

    for snap in snapshot_dates:
        start_date = snap
        end_candidate = snap + pd.DateOffset(months=horizon_months)
        # align end_date to latest available month-end <= end_candidate
        candidates = [d for d in available_dates if d <= end_candidate]
        if not candidates:
            continue
        end_date = max(candidates)

        # slice start and end snapshots
        start_slice = df[df['snapshot_date'] == start_date].set_index('client_id')
        end_slice = df[df['snapshot_date'] == end_date].set_index('client_id')

        merged = start_slice[['rating']].rename(columns={'rating':'rating_start'}).join(
            end_slice[['rating']].rename(columns={'rating':'rating_end'}),
            how='left'
        )

        # infer exit: rating_end is NaN -> EXIT
        merged['EXIT_FLAG'] = merged['rating_end'].isna()

        # counts at snapshot
        counts_start = merged['rating_start'].value_counts().reindex(ratings, fill_value=0)

        n_total_start = len(merged)
        n_exit = int(merged['EXIT_FLAG'].sum())
        n_nonexit = n_total_start - n_exit

        # migration count matrix (rows=start rating; cols=end ratings + 'EXIT')
        cols = ratings + ['EXIT']
        mat_counts = pd.DataFrame(0, index=ratings, columns=cols)
        for r in ratings:
            grp = merged[merged['rating_start'] == r]
            for r_end in ratings:
                mat_counts.loc[r, r_end] = (grp['rating_end'] == r_end).sum()
            mat_counts.loc[r, 'EXIT'] = grp['rating_end'].isna().sum()

        # migration % matrix excluding EXIT in denominator
        mat_pct = mat_counts.astype(float)
        row_tot = mat_counts.sum(axis=1)
        row_exit = mat_counts['EXIT']
        denom = row_tot - row_exit  # observed-to-observed denominator
        for r in ratings:
            if denom.loc[r] > 0:
                mat_pct.loc[r, :] = mat_counts.loc[r, :] / denom.loc[r]
            else:
                mat_pct.loc[r, :] = np.nan

        # per-client metric for histogram: numeric distance (treat 'D' as worst=6)
        def rating_to_num(x):
            if pd.isna(x):
                return np.nan
            if x == 'D':
                return 6
            return int(x)
        per_client = merged.reset_index().copy()
        per_client['num_start'] = per_client['rating_start'].map(rating_to_num)
        per_client['num_end'] = per_client['rating_end'].map(rating_to_num)
        per_client['abs_move'] = (per_client['num_end'] - per_client['num_start']).abs()

        results.append({
            'snapshot': snap,
            'start_date': start_date,
            'end_date': end_date,
            'counts_start': counts_start,
            'n_total_start': n_total_start,
            'n_nonexit': n_nonexit,
            'n_exit': n_exit,
            'mat_counts': mat_counts,
            'mat_pct_excl_exit': mat_pct,
            'per_client': per_client
        })

    return results

def average_pct_matrices(results, ratings=RATINGS):
    """Arithmetic mean of percentage matrices across results (TTC)."""
    mats = []
    keys = []
    for r in results:
        mm = r['mat_pct_excl_exit'].reindex(index=ratings, columns=ratings+['EXIT'])
        mats.append(mm)
        keys.append(r['snapshot'])
    if not mats:
        return pd.DataFrame()
    stacked = pd.concat(mats, keys=keys)
    mean_mat = stacked.groupby(level=1).mean()
    return mean_mat

# ----------------------------
# UI - Sidebar controls & data load
# ----------------------------
st.title("Roll-rate / Migration Explorer (Interactive)")

st.sidebar.header("Data & snapshot controls")

uploaded = st.sidebar.file_uploader("Upload CSV (columns: snapshot_date, client_id, rating)", type=['csv'])
use_demo = st.sidebar.checkbox("Use demo data (5 years monthly)", value= True if uploaded is None else False)

if uploaded is not None and not use_demo:
    df = pd.read_csv(uploaded, parse_dates=['snapshot_date'])
    # Minimal validation
    expected_cols = {'snapshot_date','client_id','rating'}
    if not expected_cols.issubset(set(df.columns)):
        st.error(f"CSV must contain columns: {expected_cols}. Found: {set(df.columns)}")
        st.stop()
    # ensure snapshot_date is month-end (if not convert to month-end)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    # Optionally coerce to month-end to avoid mismatches (we keep as provided but warn)
    # If user provided mid-month dates, we'll align them to month-end for matching snapshots:
    df['snapshot_date'] = df['snapshot_date'].dt.to_period('M').dt.to_timestamp('M')
else:
    df = generate_demo_data()

# allow user to inspect first rows
with st.expander("Preview raw input (first 100 rows)"):
    st.write("Input should have exactly these columns: snapshot_date (month-end), client_id, rating")
    st.dataframe(df.head(100))

# Snapshot controls
freq = st.sidebar.selectbox("Snapshot frequency", options=['Q','A','M'], index=0,
                            help="Q = quarterly, A = annual, M = monthly")
horizon_months = st.sidebar.number_input("Horizon (months)", min_value=1, max_value=60, value=12)
use_overlapping = st.sidebar.checkbox("Use overlapping snapshots", value=True)

# Interactive explanation: overlapping vs non-overlapping
with st.sidebar.expander("Why overlapping vs non-overlapping matters (click to expand)"):
    st.markdown("""
    **Overlapping snapshots**
    - Snapshots are at a cadence shorter than the horizon (e.g., quarterly snapshots with 12-month horizon).
    - Pros: more observations, smoother migration estimates (useful when portfolio small).
    - Cons: same client may appear in multiple windows → observations are correlated (not independent). This can:
        - Understate variance,
        - Bias backtests if independence assumed,
        - Double-count transitions (if client moves multiple times within overlapping windows).
    - Use: exploratory analysis, smoothing.

    **Non-overlapping snapshots**
    - Choose snapshots so their horizon windows do not overlap (e.g., annual snapshots with a 12-month horizon).
    - Pros: windows approximately independent → better for statistical tests and backtesting.
    - Cons: fewer observations → more sampling noise.
    - Use: model validation, regulatory backtesting.
    """)

# ----------------------------
# Compute snapshots and migrations
# ----------------------------
snapshot_dates = get_snapshot_dates(df['snapshot_date'].unique(), freq=freq)
if len(snapshot_dates) == 0:
    st.error("No valid snapshot dates found in the data after aligning to the chosen frequency.")
    st.stop()

migrations = compute_migrations(df, snapshot_dates, horizon_months=horizon_months, use_overlapping=use_overlapping)

# Snapshot summary table
summary_rows = []
for r in migrations:
    row = {
        'snapshot': r['snapshot'],
        'end_date': r['end_date'],
        'n_start': r['n_total_start'],
        'n_nonexit': r['n_nonexit'],
        'n_exit': r['n_exit']
    }
    for rt in RATINGS:
        row[f'count_{rt}'] = int(r['counts_start'].get(rt, 0))
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows).sort_values('snapshot').reset_index(drop=True)

st.subheader("Snapshot summary")
st.caption("Counts at snapshot (total clients), non-exit clients at horizon-end, exit counts inferred as missing at horizon-end.")
st.dataframe(summary_df)

with st.expander("How exit is inferred (click to expand)"):
    st.markdown("""
    We **do not** require an 'exited' column in the input. Instead:
    - A client present at the snapshot (snapshot_date = t) who **does not appear** in the horizon-end snapshot (t + H months)
      is considered **exited / unobserved** for that horizon.
    - Those clients are placed in the 'EXIT' column in migration count matrices.
    - When computing migration **percentages**, we **exclude** EXIT clients from denominators because the rating outcome is unobserved.
      This avoids biasing observed migration rates toward zero.
    - We still report the EXIT counts so you can assess how many outcomes are missing.
    """)

# ----------------------------
# Side-by-side snapshot comparison (show only common clients)
# ----------------------------
st.subheader("Compare snapshots (side-by-side) — show only common clients")
with st.expander("How this works (click to expand)"):
    st.markdown("""
    - Pick up to 3 snapshots to compare side-by-side.
    - We join client lists and show only those clients that are **present in all selected snapshots** (common clients).
    - This is useful to visually inspect migrations for the same clients across time.
    - Note: if many clients exit between snapshots, the common set can be small.
    """)

snap_choices = st.multiselect("Select up to 3 snapshots to compare", options=list(snapshot_dates), format_func=lambda d: d.date(), default=list(snapshot_dates[:3]))
if len(snap_choices) >= 2:
    snap_choices = snap_choices[:3]
    frames = []
    for s in snap_choices:
        tmp = df[df['snapshot_date'] == s][['client_id','rating']].rename(columns={'rating': f'rating_{s.date()}'})
        frames.append(tmp.set_index('client_id'))
    # compute common clients
    common_idx = set.intersection(*[set(f.index) for f in frames])
    if len(common_idx) == 0:
        st.warning("No common clients between selected snapshots.")
    else:
        merged_common = pd.concat([f.loc[sorted(common_idx)] for f in frames], axis=1)
        st.write(f"Showing {len(merged_common)} common clients (first 500 rows).")
        st.dataframe(merged_common.head(500))
else:
    st.info("Select 2 or 3 snapshots to show common clients side-by-side.")

# ----------------------------
# Migration matrices (select snapshot via dropdown)
# ----------------------------
st.subheader("Per-snapshot migration matrix (counts & percent)")
snap_map = {r['snapshot']: r for r in migrations}
snap_display = [s for s in snap_map.keys()]
selected_snapshot = st.selectbox("Pick snapshot to inspect", options=snap_display, format_func=lambda d: d.date())

if selected_snapshot is not None:
    sel = snap_map[selected_snapshot]
    matrix_type = st.radio("Matrix type", options=["Counts", "Percent (excluding EXIT)"])
    if matrix_type == "Counts":
        st.write(f"Migration counts from {sel['start_date'].date()} to {sel['end_date'].date()}")
        st.dataframe(sel['mat_counts'])
    else:
        st.write(f"Migration percentages (denominator excludes EXIT) from {sel['start_date'].date()} to {sel['end_date'].date()}")
        st.dataframe(sel['mat_pct_excl_exit'].round(4))

    with st.expander("Explain: how this matrix is built and why we exclude EXIT from % denominators"):
        st.markdown("""
        Steps:
        1. For snapshot at time t, take all clients present and their rating (rating_start).
        2. Find the same client at time t + H (aligned to nearest month-end <= t+H). If the client is not present at t+H => we flag EXIT.
        3. Count transitions rating_start -> rating_end (these are observed transitions).
        4. Place missing outcomes in the 'EXIT' column (these are unobserved outcomes).
        5. For percentage matrices we divide observed transitions by (row_total - exits_in_row), i.e., exclude EXIT from denominator, because EXITs yield no rating outcome.
        Why exclude EXIT?
        - If we included EXIT in denominators, we would dilute observed migration rates toward zero artificially (since many EXITs have unknown outcomes).
        - Excluding EXIT gives the conditional migration probability given the client remained observable.
        """)

# ----------------------------
# TTC (averaged) migration matrix
# ----------------------------
st.subheader("Average (TTC) migration matrix across snapshots")
mean_mat = average_pct_matrices(migrations)
if mean_mat.empty:
    st.write("No matrices to average.")
else:
    st.dataframe(mean_mat.round(4))
    with st.expander("Explain: how we compute TTC (mean) and limitations"):
        st.markdown("""
        - We compute a simple arithmetic mean of the per-snapshot percentage matrices (element-wise mean).
        - This yields a time-aggregated TTC-style migration probability matrix.
        - Limitations:
            - Overlapping windows cause correlated matrices (same client contributes multiple times).
            - If you need variance estimates or confidence intervals, consider block bootstrap or use non-overlapping snapshots.
            - Weighting: you might want to weight snapshots by exposure size instead of simple mean.
        """)

# ----------------------------
# Histogram of per-client migration magnitude
# ----------------------------
st.subheader("Histogram: distribution of client migration magnitudes (demo-style)")
hist_snapshot_choice = st.selectbox("Pick snapshot for histogram", options=snap_display, format_func=lambda d: d.date(), index=0)
if hist_snapshot_choice:
    perclient = snap_map[hist_snapshot_choice]['per_client']
    # exclude exits for histogram of observed moves, but allow user to toggle
    show_exits = st.checkbox("Include EXITs (they appear as NaN moves) in histogram", value=False)
    data = perclient['abs_move']
    if not show_exits:
        data = data.dropna()
    # plot using matplotlib (no seaborn)
    fig, ax = plt.subplots(figsize=(6,3))
    bins = np.arange(-0.5, 7.5, 1)  # 0..6 moves (6 representing D)
    ax.hist(data, bins=bins)
    ax.set_xlabel("Absolute rating move (numeric proxy; 'D' treated as 6)")
    ax.set_ylabel("Number of clients")
    ax.set_title(f"Histogram of absolute moves: {hist_snapshot_choice.date()} → {snap_map[hist_snapshot_choice]['end_date'].date()}")
    st.pyplot(fig)

    with st.expander("Interpretation & real-life differences"):
        st.markdown("""
        - The demo random data often shows many clients with zero move (no change), a moderate number with small moves (±1), and fewer with large moves.
        - In real portfolios:
            - The mass at zero is usually larger for stable portfolios.
            - There may be seasonal patterns (e.g., more downgrades following an economic downturn).
            - Defaults are relatively rare; real-world default spikes coincide with stress periods.
            - Sample size, portfolio composition (corporate vs retail), and rating policy (point-in-time vs through-the-cycle) all affect the shape.
        """)

# ----------------------------
# Final notes and downloads
# ----------------------------
st.markdown("---")
st.subheader("Notes & recommended next steps")
st.markdown("""
- Replace demo data by uploading your CSV (snapshot_date as month-end).
- If you plan to use these matrices for capital or regulatory calculations, prefer **non-overlapping snapshots** for backtesting or perform robust statistical adjustments for overlap.
- Consider weighting snapshot matrices by exposure amounts (EAD) if rating counts do not reflect exposure differences.
""")

with st.expander("Full step-by-step logic (collapsed walkthrough)"):
    st.markdown("""
    1. Input: a panel of (snapshot_date, client_id, rating) with monthly frequency (month-end).
    2. Choose snapshot frequency (M/Q/A) to define snapshot dates present in the data.
    3. For each snapshot t:
        a. Identify clients present at t.
        b. Find the observation of the same client at t+H (align horizon to nearest month-end <= t+H).
        c. If client absent at t+H -> mark EXIT (unobserved outcome).
        d. Count rating_start -> rating_end transitions; put exits in 'EXIT' column.
        e. For % matrices, compute conditional probabilities given the client did NOT exit (exclude EXIT from denominator).
    4. Average these per-snapshot % matrices (arithmetic mean) for a TTC-style migration matrix. Optionally weight snapshots or exposures.
    5. Use non-overlapping snapshots or bootstrapping for inference if independence required.
    """)

st.caption("Script by the assistant. Replace demo data with your real snapshot CSV to run on production data.")
