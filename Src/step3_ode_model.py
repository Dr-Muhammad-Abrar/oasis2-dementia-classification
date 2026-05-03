# =============================================================
# step3_ode_model.py  —  ODE progression modeling + pipeline
# Run AFTER step2_train_model.py
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import *


# ── 1. Load and prepare longitudinal data ────────────────────
def load_longitudinal():
    print("\n" + "="*60)
    print("STEP 3A: Loading longitudinal data for ODE fitting")
    print("="*60)

    df = pd.read_excel(CSV_PATH)

    # standardise column names (OASIS-2 sometimes has slight variations)
    df.columns = df.columns.str.strip()

    # keep only patients with enough visits
    visit_counts = df.groupby('Subject ID').size()
    valid_subs   = visit_counts[visit_counts >= MIN_VISITS].index
    df_long      = df[df['Subject ID'].isin(valid_subs)].copy()

    # drop rows with missing CDR or MR Delay
    df_long = df_long.dropna(subset=['CDR', 'MR Delay'])

    # sort by subject and time
    df_long = df_long.sort_values(['Subject ID', 'MR Delay'])

    print(f"Patients with {MIN_VISITS}+ visits: {df_long['Subject ID'].nunique()}")
    print(f"Total visit records: {len(df_long)}")
    print(f"CDR range: {df_long['CDR'].min()} — {df_long['CDR'].max()}")
    print(f"MR Delay range: {df_long['MR Delay'].min()} — {df_long['MR Delay'].max()} days")

    return df_long


# ── 2. ODE model definition ───────────────────────────────────
# We model CDR progression as a logistic growth ODE:
#
#   dC/dt = r * C * (1 - C / K)
#
# where:
#   C = CDR score (disease severity)
#   r = progression rate (fitted per patient)
#   K = carrying capacity = 3.0 (max CDR)
#
# This is biologically motivated: progression accelerates
# initially then slows as it approaches maximum severity.

CDR_MAX = 3.0   # maximum possible CDR score

def logistic_ode(C, t, r, K=CDR_MAX):
    """Logistic growth ODE for CDR progression"""
    dCdt = r * C * (1.0 - C / K)
    return dCdt


def solve_ode(t_span, C0, r, K=CDR_MAX):
    """Solve ODE given initial condition and rate"""
    # avoid zero initial condition (ODE stays at 0)
    C0 = max(C0, 0.01)
    t_eval = np.linspace(t_span[0], t_span[-1], 200)
    C = odeint(logistic_ode, C0, t_eval, args=(r, K))
    return t_eval, C.flatten()


def fit_patient_ode(times, cdrs):
    """
    Fit the logistic ODE to a single patient's CDR trajectory.
    Returns fitted rate r, and R² score.
    """
    C0 = cdrs[0]

    def model_func(t, r):
        t_arr = np.array(t)
        C = odeint(logistic_ode, max(C0, 0.01),
                   [0] + list(t_arr), args=(r, CDR_MAX))
        return C[1:].flatten()

    try:
        popt, _ = curve_fit(
            model_func, times, cdrs,
            p0=[0.001],
            bounds=(0, 0.1),
            maxfev=5000
        )
        r_fitted = popt[0]

        # compute R²
        predicted = model_func(times, r_fitted)
        ss_res = np.sum((cdrs - predicted) ** 2)
        ss_tot = np.sum((cdrs - cdrs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return r_fitted, r2

    except Exception:
        return None, None


# ── 3. Fit ODE for all patients ───────────────────────────────
def fit_all_patients(df_long):
    print("\n" + "="*60)
    print("STEP 3B: Fitting ODE to each patient trajectory")
    print("="*60)

    results = []
    subjects = df_long['Subject ID'].unique()

    for subj in subjects:
        sub_df = df_long[df_long['Subject ID'] == subj].sort_values('MR Delay')
        times  = sub_df['MR Delay'].values.astype(float)
        cdrs   = sub_df['CDR'].values.astype(float)
        group  = sub_df['Group'].iloc[-1]   # final diagnosis

        # normalise times to start at 0
        times = times - times[0]

        r_fitted, r2 = fit_patient_ode(times, cdrs)

        results.append({
            'Subject ID':    subj,
            'Group':         group,
            'n_visits':      len(times),
            'baseline_CDR':  cdrs[0],
            'final_CDR':     cdrs[-1],
            'CDR_change':    cdrs[-1] - cdrs[0],
            'r_fitted':      r_fitted,
            'R2':            r2,
            'followup_days': times[-1]
        })

    ode_df = pd.DataFrame(results)
    ode_df = ode_df.dropna(subset=['r_fitted'])

    print(f"\nSuccessfully fitted ODE for {len(ode_df)} patients")
    print(f"\nProgression rate (r) by group:")
    print(ode_df.groupby('Group')['r_fitted'].describe().round(6))
    print(f"\nMean R² (goodness of fit): {ode_df['R2'].mean():.4f}")

    # save
    ode_path = os.path.join(ODE_DIR, "ode_fitted_params.csv")
    ode_df.to_csv(ode_path, index=False)
    print(f"\nODE parameters saved: {ode_path}")

    return ode_df


# ── 4. Visualise ODE fits ─────────────────────────────────────
def visualise_ode_fits(df_long, ode_df, n_per_group=3):
    print("\nGenerating ODE trajectory visualisations...")

    fig, axes = plt.subplots(3, n_per_group,
                             figsize=(n_per_group * 4, 10))
    fig.suptitle("ODE-Fitted CDR Progression Trajectories", fontsize=14)

    colors = {'Nondemented': 'green',
              'Converted':   'orange',
              'Demented':    'red'}

    for row_i, group in enumerate(['Nondemented', 'Converted', 'Demented']):
        group_df = ode_df[ode_df['Group'] == group].head(n_per_group)

        for col_i, (_, params) in enumerate(group_df.iterrows()):
            ax     = axes[row_i][col_i]
            subj   = params['Subject ID']
            r      = params['r_fitted']
            r2     = params['R2']

            sub_df = df_long[df_long['Subject ID'] == subj].sort_values('MR Delay')
            times  = sub_df['MR Delay'].values.astype(float)
            cdrs   = sub_df['CDR'].values.astype(float)
            times  = times - times[0]

            # plot actual data
            ax.scatter(times, cdrs, color=colors[group],
                       zorder=5, s=60, label='Observed CDR')

            # plot ODE solution
            if times[-1] > 0:
                t_fit, C_fit = solve_ode([0, times[-1]],
                                         max(cdrs[0], 0.01), r)
                ax.plot(t_fit, C_fit, '--', color=colors[group],
                        alpha=0.8, label=f'ODE fit (r={r:.5f})')

            ax.set_ylim(-0.1, CDR_MAX + 0.2)
            ax.set_xlabel("Days since baseline")
            ax.set_ylabel("CDR Score")
            ax.set_title(f"{group}\n{subj}\nR²={r2:.3f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ODE_DIR, "ode_trajectories.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"ODE trajectories saved: {save_path}")


# ── 5. Group comparison of progression rates ─────────────────
def plot_progression_rates(ode_df):
    print("\nGenerating progression rate comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ODE Progression Rates by Diagnostic Group", fontsize=13)

    # boxplot of r by group
    groups = ['Nondemented', 'Converted', 'Demented']
    colors = ['green', 'orange', 'red']
    data   = [ode_df[ode_df['Group'] == g]['r_fitted'].dropna().values
              for g in groups]

    bp = axes[0].boxplot(data, labels=groups, patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Fitted progression rate (r)")
    axes[0].set_title("Distribution of r by Group")
    axes[0].grid(True, alpha=0.3)

    # scatter: baseline CDR vs progression rate
    color_map = {'Nondemented': 'green',
                 'Converted':   'orange',
                 'Demented':    'red'}
    for group in groups:
        sub = ode_df[ode_df['Group'] == group]
        axes[1].scatter(sub['baseline_CDR'], sub['r_fitted'],
                        c=color_map[group], label=group, alpha=0.7, s=60)

    axes[1].set_xlabel("Baseline CDR Score")
    axes[1].set_ylabel("Fitted progression rate (r)")
    axes[1].set_title("Baseline CDR vs Progression Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ODE_DIR, "progression_rates.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Progression rates plot saved: {save_path}")


# ── 6. Integrated pipeline: DL prediction → ODE projection ───
def integrated_pipeline(ode_df):
    print("\n" + "="*60)
    print("STEP 3C: Integrated Pipeline — DL Stage → ODE Projection")
    print("="*60)

    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        print("test_predictions.csv not found — skipping integration step.")
        print("Run step2_train_model.py first to generate predictions.")
        return

    pred_df = pd.read_csv(pred_path)
    # merge with ODE parameters on Subject ID
    merged = pred_df.merge(
        ode_df[['Subject ID', 'r_fitted', 'R2', 'Group']],
        on='Subject ID', how='inner'
    )
    print(f"Matched {len(merged)} records with ODE parameters")

    if len(merged) == 0:
        print("No matches found between test predictions and ODE params.")
        return

    # for each matched patient:
    # use DL predicted CDR class → map to CDR score → use as ODE initial condition
    # then project 2 years (730 days) forward
    label_to_cdr = {0: 0.0, 1: 0.5, 2: 1.0}   # rough mapping
    projection_days = 730

    fig, axes = plt.subplots(1, min(6, len(merged)),
                             figsize=(min(6, len(merged)) * 3.5, 5))
    if len(merged) == 1:
        axes = [axes]
    fig.suptitle("Integrated Pipeline: DL Stage → ODE 2-Year Projection",
                 fontsize=12)

    for i, (_, row) in enumerate(merged.head(6).iterrows()):
        ax = axes[i]

        # DL predicted initial CDR
        pred_label = int(row['Predicted_Label'])
        C0_dl      = label_to_cdr[pred_label]
        r          = row['r_fitted']

        # true CDR from data
        true_label = int(row['Label'])
        C0_true    = label_to_cdr[true_label]

        # ODE projection from DL prediction
        t_proj, C_proj = solve_ode([0, projection_days],
                                   max(C0_dl, 0.01), r)

        # ODE projection from true label (for comparison)
        _, C_true_proj = solve_ode([0, projection_days],
                                   max(C0_true, 0.01), r)

        ax.plot(t_proj, C_proj, 'b-',
                label=f'DL init (pred={LABEL_NAMES[pred_label]})',
                linewidth=2)
        ax.plot(t_proj, C_true_proj, 'g--',
                label=f'True init ({LABEL_NAMES[true_label]})',
                linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5,
                   label='MCI threshold')
        ax.axhline(y=1.0, color='red',    linestyle=':', alpha=0.5,
                   label='Dementia threshold')
        ax.set_ylim(0, CDR_MAX)
        ax.set_xlabel("Days from baseline")
        ax.set_ylabel("Projected CDR")
        ax.set_title(f"{row['Subject ID']}\nr={r:.5f}", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(ODE_DIR, "integrated_pipeline.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Integrated pipeline visualisation saved: {save_path}")


# ── 7. Summary statistics table ───────────────────────────────
def print_summary(ode_df):
    print("\n" + "="*60)
    print("FINAL SUMMARY — ODE MODEL RESULTS")
    print("="*60)

    summary = ode_df.groupby('Group').agg(
        n_patients  = ('Subject ID', 'count'),
        mean_r      = ('r_fitted',   'mean'),
        std_r       = ('r_fitted',   'std'),
        mean_R2     = ('R2',         'mean'),
        mean_CDR_change = ('CDR_change', 'mean')
    ).round(6)

    print(summary.to_string())

    summary_path = os.path.join(ODE_DIR, "ode_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved: {summary_path}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df_long = load_longitudinal()
    ode_df  = fit_all_patients(df_long)
    visualise_ode_fits(df_long, ode_df)
    plot_progression_rates(ode_df)
    integrated_pipeline(ode_df)
    print_summary(ode_df)
    print("\n✓ Step 3 complete. Full pipeline finished!")
    print(f"\nAll results saved in: {OUTPUT_DIR}")
