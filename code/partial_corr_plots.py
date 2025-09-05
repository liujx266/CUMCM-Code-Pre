#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Partial-correlation (去混杂) analysis and plotting

- Input: D:\\CUMCM-Code-Pre\\code\\合并结果去重-手动去掉异常值并补零.xlsx
  (workspace-relative: code/合并结果去重-手动去掉异常值并补零.xlsx)

- What it does (following the screenshot's method):
  1) Optional preprocessing for Y: map to (0,1) if it looks like a percent, then y_asin = arcsin(sqrt(y)).
  2) Build decimal gestational week `week_dec = 周 + 天/7` when `周` and `天` columns exist; otherwise use a single gestational-week column.
  3) Compute partial correlations via residual method:
       r_{Y,周 | BMI + QC} and r_{Y,BMI | 周 + QC}
     For each pair, regress both variables on the controls (with intercept),
     correlate the residuals, and plot residual vs residual with a fitted line.
  4) Save plots as PNG with Chinese font 微软雅黑, 10pt.

Usage (from repository root):
  python code/partial_corr_plots.py

Notes:
  - The script tries to auto-detect column names in Chinese/English.
    If auto-detection fails or is wrong, update COLUMN_OVERRIDES below or
    use interactive prompts to select columns by index.
  - Dependencies: pandas, numpy, matplotlib, scipy (for pearsonr p-value),
                  statsmodels, patsy.
"""

from __future__ import annotations

import math
import os
import sys
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from patsy import bs


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Excel input path (workspace-relative). You may also pass a path as argv[1].
EXCEL_PATH = r"合并结果去重-手动去掉异常值并补零.xlsx"

# If you know the exact columns, set them here to bypass auto-detection.
# Keys: y, week, day, bmi, sex, qc (where qc is a list of column names)
COLUMN_OVERRIDES: Dict[str, object] = {
    # Example (uncomment and edit):
    # "y": "Y含量",
    # "week": "孕周",
    # "day": "天",
    # "bmi": "BMI",
    # "sex": "胎儿性别",  # used if only_male=True
    # "qc": ["uniq_reads", "AA", "gc_total", "dup_rate"],
}

# Whether to filter to male fetus only (if a sex column exists)
ONLY_MALE = False

# Output directory for figures and exports (created next to the Excel file)
def _default_outdir(excel_path: str) -> str:
    d = os.path.dirname(os.path.abspath(excel_path))
    out = os.path.join(d, "partialcorr_output")
    os.makedirs(out, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm_str(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def list_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def autodetect_columns(df: pd.DataFrame) -> Dict[str, object]:
    """Heuristic detection for common Chinese/English names.

    Returns a dict with keys possibly including: y, week, day, bmi, sex, qc
    """
    cols = list(df.columns)
    norm_map = {c: _norm_str(c) for c in cols}

    # Candidate patterns
    y_keys = ["y", "y含量", "y染色体", "y比例", "胎儿y", "chr_y", "fetal_y"]
    week_keys = ["孕周", "周龄", "周", "gestationalweek", "ga", "week"]
    day_keys = ["天", "day", "days"]
    bmi_keys = ["bmi", "体质指数", "身高体重指数"]
    sex_keys = ["胎儿性别", "性别", "sex", "gender"]

    def pick(keys: Sequence[str]) -> Optional[str]:
        for k in keys:
            k_n = _norm_str(k)
            for c, cn in norm_map.items():
                if k_n == cn or k_n in cn:
                    return c
        return None

    y_col = pick(y_keys)
    week_col = pick(week_keys)
    day_col = pick(day_keys)
    bmi_col = pick(bmi_keys)
    sex_col = pick(sex_keys)

    # QC: include numeric columns with typical QC patterns
    qc_patterns = [
        "neff", "uniq", "unique", "reads", "aa", "gc", "filt",
        "dup", "duplicate", "mapped", "depth", "coverage", "nratio",
    ]
    qc_cols: List[str] = []
    for c in list_numeric_columns(df):
        cn = norm_map[c]
        if c in {y_col, week_col, day_col, bmi_col}:
            continue
        if any(p in cn for p in qc_patterns):
            qc_cols.append(c)

    # Compute log_neff if uniq_reads and AA exist
    uniq_col = None
    aa_col = None
    for c, cn in norm_map.items():
        if uniq_col is None and ("uniq" in cn or "unique" in cn) and c in df.columns:
            uniq_col = c
        if aa_col is None and (cn == "aa" or cn.endswith("_aa") or "aa" in cn) and c in df.columns:
            aa_col = c
    if uniq_col is not None and aa_col is not None:
        qc_cols.append("log_neff(auto)")  # placeholder; computed later

    detected: Dict[str, object] = {}
    if y_col: detected["y"] = y_col
    if week_col: detected["week"] = week_col
    if day_col: detected["day"] = day_col
    if bmi_col: detected["bmi"] = bmi_col
    if sex_col: detected["sex"] = sex_col
    if qc_cols:
        # unique preserving order
        seen = set()
        qcl = []
        for q in qc_cols:
            if q not in seen:
                seen.add(q)
                qcl.append(q)
        detected["qc"] = qcl
    return detected


def interactive_pick(df: pd.DataFrame, prompt: str, allow_skip: bool = False) -> Optional[str]:
    print("\n" + prompt)
    for i, c in enumerate(df.columns):
        print(f"  [{i}] {c}")
    while True:
        s = input("Enter index" + (" (or blank to skip)" if allow_skip else "") + ": ").strip()
        if allow_skip and s == "":
            return None
        if s.isdigit() and int(s) in range(len(df.columns)):
            return df.columns[int(s)]
        print("Invalid input. Try again.")


def ensure_columns(df: pd.DataFrame) -> Dict[str, object]:
    detected = autodetect_columns(df)
    print("Auto-detected columns:")
    for k in ["y", "week", "day", "bmi", "sex", "qc"]:
        print(f"  {k}: {detected.get(k)}")

    # Apply overrides if provided
    final = dict(detected)
    for k, v in COLUMN_OVERRIDES.items():
        final[k] = v
    print("\nAfter applying overrides:")
    for k in ["y", "week", "day", "bmi", "sex", "qc"]:
        print(f"  {k}: {final.get(k)}")

    # Minimal required: y and at least one of week or (week+day), and bmi
    needs_interactive = (final.get("y") is None) or (final.get("bmi") is None) or (final.get("week") is None and final.get("day") is None)

    if needs_interactive:
        print("\nInteractive selection (one-time):")
        if final.get("y") is None:
            final["y"] = interactive_pick(df, "Pick Y column (Y比例/含量等)")
        if final.get("week") is None and final.get("day") is None:
            wk = interactive_pick(df, "Pick gestational week column (孕周/周龄). If you have separate 周 and 天, pick 周 here.")
            final["week"] = wk
            # Optional day
            try:
                day = interactive_pick(df, "Optional: pick Day column (天), or press Enter to skip", allow_skip=True)
            except EOFError:
                day = None
            if day:
                final["day"] = day
        if final.get("bmi") is None:
            final["bmi"] = interactive_pick(df, "Pick BMI column")
        # QC optional interactive selection
        try:
            ans = input("Pick QC columns? (y/N): ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            qcs: List[str] = []
            print("Select QC columns by indices separated with commas, or blank to skip.")
            for i, c in enumerate(df.columns):
                print(f"  [{i}] {c}")
            s = input("Indices: ").strip()
            if s:
                for part in s.split(','):
                    part = part.strip()
                    if part.isdigit():
                        qcs.append(df.columns[int(part)])
            if qcs:
                final["qc"] = qcs

    return final


def asinsqrt_transform(y: pd.Series) -> pd.Series:
    y = y.astype(float)
    # If it looks like percent (0-100), scale to 0-1
    ymax = y.max(skipna=True)
    if pd.notna(ymax) and ymax > 1.5:
        y = y / 100.0
    # Avoid exact 0/1 for asin(sqrt())
    n = y.notna().sum()
    eps_adj = (0.5 / max(n, 1))
    y = y.clip(eps_adj, 1 - eps_adj)
    return np.arcsin(np.sqrt(y))


def compute_week_decimal(df: pd.DataFrame, week_col: str, day_col: Optional[str]) -> pd.Series:
    if day_col is not None and day_col in df.columns:
        return df[week_col].astype(float) + df[day_col].astype(float) / 7.0
    return df[week_col].astype(float)


def add_log_neff_if_possible(df: pd.DataFrame, qc_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    # If a placeholder present, attempt to compute from uniq_reads & AA
    if "log_neff(auto)" not in qc_cols:
        return df, qc_cols
    qc_cols = [c for c in qc_cols if c != "log_neff(auto)"]
    cols_lower = {_norm_str(c): c for c in df.columns}
    uniq = None
    aa = None
    for k, v in cols_lower.items():
        if uniq is None and ("uniq" in k or "unique" in k) and pd.api.types.is_numeric_dtype(df[v]):
            uniq = v
        if aa is None and (k == "aa" or k.endswith("_aa") or "aa" in k) and pd.api.types.is_numeric_dtype(df[v]):
            aa = v
    if uniq is not None and aa is not None:
        neff = df[uniq].astype(float) * (1.0 - df[aa].astype(float))
        df = df.copy()
        df["log_neff"] = np.log10(neff.replace(0, np.nan))
        qc_cols.append("log_neff")
        print("Computed QC column: log_neff = log10(uniq_reads*(1-AA))")
    return df, qc_cols


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after linear regression on X (with intercept)."""
    # Add intercept
    X_ = np.column_stack([np.ones(len(X)), X])
    # Solve least squares
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    y_hat = X_.dot(beta)
    return y - y_hat


def partial_corr_residual_method(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    y_res = residualize(y, Z)
    x_res = residualize(x, Z)
    r, p = pearsonr(x_res, y_res)
    return r, p, x_res, y_res


# --------------------------- Nonlinear controls (splines) --------------------
def _sanitize_names(names: Sequence[str]) -> Dict[str, str]:
    """Return a mapping from original names to patsy-safe tokens.

    Replaces non [0-9a-zA-Z_] with '_', ensures uniqueness.
    """
    mapping: Dict[str, str] = {}
    used = set()
    for c in names:
        tok = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
        # Ensure leading char is valid for patsy/python identifiers
        if tok and not re.match(r"[A-Za-z_]", tok[0]):
            tok = "v_" + tok
        if not tok:
            tok = "col"
        base = tok
        k = 1
        while tok in used:
            k += 1
            tok = f"{base}_{k}"
        mapping[c] = tok
        used.add(tok)
    return mapping


def partial_corr_with_spline(df: pd.DataFrame, y_var: str, x_var: str, nonlinear_control: str, qc_vars: Sequence[str], spline_df: int = 4) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Residual method with nonlinear control using B-splines.

    Computes r(Y, X | bs(nonlinear_control, df) + QC) by:
      - Regressing Y on bs(nonlinear_control, df) + QC
      - Regressing X on bs(nonlinear_control, df) + QC
      - Correlating residuals
    Returns (r, p, x_residuals, y_residuals)
    """
    cols = [y_var, x_var, nonlinear_control] + list(qc_vars)
    df3 = df[cols].dropna().copy()
    name_map = _sanitize_names(cols)
    df3 = df3.rename(columns=name_map)

    y = name_map[y_var]
    x = name_map[x_var]
    nl = name_map[nonlinear_control]
    qc_tokens = [name_map[q] for q in qc_vars]

    rhs = f"bs({nl}, df={int(spline_df)})"
    if qc_tokens:
        rhs += " + " + " + ".join(qc_tokens)

    y_formula = f"{y} ~ {rhs}"
    x_formula = f"{x} ~ {rhs}"

    y_model = smf.ols(y_formula, data=df3).fit()
    x_model = smf.ols(x_formula, data=df3).fit()

    # Align indices (should already match since same df3)
    idx = y_model.resid.index.intersection(x_model.resid.index)
    r, p = pearsonr(x_model.resid.loc[idx].values, y_model.resid.loc[idx].values)
    return r, p, x_model.resid.loc[idx].values, y_model.resid.loc[idx].values


def bootstrap_ci_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 12345) -> Tuple[float, float]:
    """Bootstrap percentile CI for Pearson r."""
    n = len(x)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = pearsonr(x[idx], y[idx])[0]
    alpha = 1 - ci
    lo, hi = np.quantile(rs, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)


def scatter_with_fit(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, out_png: str, r: float, p: float, n: Optional[int] = None, ci95: Optional[Tuple[float, float]] = None, qc_note: Optional[str] = None, annotate: bool = True, show_title: bool = True):
    # Font: 微软雅黑, 10pt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.scatter(x, y, s=12, alpha=0.7, edgecolor='none')
    # Fit line
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)  # slope, intercept
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, b1*xs + b0, color='tab:red', linewidth=1.2)
    if show_title and title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)
    # Annotation at top-left
    if annotate:
        text = f"r = {r:.3f}\np = {p:.3g}"
        if n is not None:
            text += f"\nn = {n}"
        if ci95 is not None:
            text += f"\n95% CI = [{ci95[0]:.3f}, {ci95[1]:.3f}]"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
        # QC footnote at bottom-left
        if qc_note:
            ax.text(0.02, 0.02, qc_note, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=9, color='dimgray')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    excel_path = EXCEL_PATH
    if len(sys.argv) >= 2:
        excel_path = sys.argv[1]
    if not os.path.exists(excel_path):
        print(f"ERROR: Excel not found: {excel_path}")
        sys.exit(2)

    # Load first sheet by default
    xls = pd.ExcelFile(excel_path)
    sheet_name = xls.sheet_names[0]
    df = xls.parse(sheet_name)
    print(f"Loaded sheet: {sheet_name}, rows={len(df)} cols={len(df.columns)}")

    cols = ensure_columns(df)
    y_col = cols.get("y")
    week_col = cols.get("week")
    day_col = cols.get("day")
    bmi_col = cols.get("bmi")
    sex_col = cols.get("sex")
    qc_cols: List[str] = list(cols.get("qc", []))

    # Optional male-only filter
    if ONLY_MALE and sex_col in df.columns:
        mask_male = df[sex_col].astype(str).str.contains(r"^(m|male|男)$", case=False, regex=True, na=False)
        df = df.loc[mask_male].copy()
        print(f"Filtered to male: n={len(df)}")

    # Compute week_dec
    df["week_dec"] = compute_week_decimal(df, week_col, day_col)

    # Build QC features (include only numeric and not NaN-only)
    df, qc_cols = add_log_neff_if_possible(df, qc_cols)
    qc_cols = [c for c in qc_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Transform y
    df["y_asin"] = asinsqrt_transform(df[y_col])

    # Drop rows with missing essentials
    essential = ["y_asin", "week_dec", bmi_col] + qc_cols
    df2 = df.dropna(subset=essential).copy()
    n0 = len(df)
    n1 = len(df2)
    if n1 < n0:
        print(f"Dropped rows with NA in essentials: {n0 - n1}")

    # Build control matrices
    Z_week_ctrl = df2[[bmi_col] + qc_cols].astype(float).to_numpy()
    Z_bmi_ctrl = df2[["week_dec"] + qc_cols].astype(float).to_numpy()

    y_vec = df2["y_asin"].to_numpy(dtype=float)
    week_vec = df2["week_dec"].to_numpy(dtype=float)
    bmi_vec = df2[bmi_col].to_numpy(dtype=float)

    # r_{Y,周 | BMI + QC}
    r1, p1, week_res1, y_res1 = partial_corr_residual_method(y_vec, week_vec, Z_week_ctrl)
    # r_{Y,BMI | 周 + QC}
    r2, p2, bmi_res2, y_res2 = partial_corr_residual_method(y_vec, bmi_vec, Z_bmi_ctrl)

    outdir = _default_outdir(excel_path)
    QC_NOTE = "QC=log_neff, GC_total, AA, dup 等"
    # Plots
    # CI and n for linear control (week)
    ci1 = bootstrap_ci_corr(week_res1, y_res1)
    scatter_with_fit(
        x=week_res1,
        y=y_res1,
        title="残差(周|BMI+QC) vs 残差(Y|BMI+QC)",
        xlabel="周(去混杂残差)",
        ylabel="Y(去混杂残差)",
        out_png=os.path.join(outdir, "partialcorr_Y_vs_week.png"),
        r=r1,
        p=p1,
        n=len(week_res1),
        ci95=ci1,
        qc_note=QC_NOTE,
    )
    # no-header version
    scatter_with_fit(
        x=week_res1,
        y=y_res1,
        title="",
        xlabel="周(去混杂残差)",
        ylabel="Y(去混杂残差)",
        out_png=os.path.join(outdir, "partialcorr_Y_vs_week_noheader.png"),
        r=r1,
        p=p1,
        annotate=True,
        show_title=False,
        n=len(week_res1),
        ci95=ci1,
        qc_note=QC_NOTE,
    )

    # CI and n for linear control (BMI)
    ci2 = bootstrap_ci_corr(bmi_res2, y_res2)
    scatter_with_fit(
        x=bmi_res2,
        y=y_res2,
        title="残差(BMI|周+QC) vs 残差(Y|周+QC)",
        xlabel="BMI(去混杂残差)",
        ylabel="Y(去混杂残差)",
        out_png=os.path.join(outdir, "partialcorr_Y_vs_BMI.png"),
        r=r2,
        p=p2,
        n=len(bmi_res2),
        ci95=ci2,
        qc_note=QC_NOTE,
    )
    # no-header version
    scatter_with_fit(
        x=bmi_res2,
        y=y_res2,
        title="",
        xlabel="BMI(去混杂残差)",
        ylabel="Y(去混杂残差)",
        out_png=os.path.join(outdir, "partialcorr_Y_vs_BMI_noheader.png"),
        r=r2,
        p=p2,
        annotate=True,
        show_title=False,
        n=len(bmi_res2),
        ci95=ci2,
        qc_note=QC_NOTE,
    )

    # Nonlinear control: control Y and 周 by bs(BMI) + QC, then correlate residuals
    if True:
        try:
            r3, p3, week_res3, y_res3 = partial_corr_with_spline(
                df=df2.assign(Y=df2["y_asin"], W=df2["week_dec"], BMI=df2[bmi_col]),
                y_var="y_asin",
                x_var="week_dec",
                nonlinear_control=bmi_col,
                qc_vars=qc_cols,
                spline_df=4,
            )
            ci3 = bootstrap_ci_corr(week_res3, y_res3)
            scatter_with_fit(
                x=week_res3,
                y=y_res3,
                title="非线性控制: 残差(周|bs(BMI)+QC) vs 残差(Y|bs(BMI)+QC)",
                xlabel="周(去混杂残差, 非线性BMI)",
                ylabel="Y(去混杂残差, 非线性BMI)",
                out_png=os.path.join(outdir, "partialcorr_Y_vs_week_splineBMI.png"),
                r=r3,
                p=p3,
                n=len(week_res3),
                ci95=ci3,
                qc_note=QC_NOTE,
            )
            # no-header version
            scatter_with_fit(
                x=week_res3,
                y=y_res3,
                title="",
                xlabel="周(去混杂残差, 非线性BMI)",
                ylabel="Y(去混杂残差, 非线性BMI)",
                out_png=os.path.join(outdir, "partialcorr_Y_vs_week_splineBMI_noheader.png"),
                r=r3,
                p=p3,
                annotate=True,
                show_title=False,
                n=len(week_res3),
                ci95=ci3,
                qc_note=QC_NOTE,
            )
        except Exception as e:
            print("Nonlinear control (BMI spline) failed:", f"{type(e).__name__}: {e}")

    # Nonlinear control: control Y and BMI by bs(周) + QC, then correlate residuals
    if True:
        try:
            r4, p4, bmi_res4, y_res4 = partial_corr_with_spline(
                df=df2,
                y_var="y_asin",
                x_var=bmi_col,
                nonlinear_control="week_dec",
                qc_vars=qc_cols,
                spline_df=4,
            )
            ci4 = bootstrap_ci_corr(bmi_res4, y_res4)
            scatter_with_fit(
                x=bmi_res4,
                y=y_res4,
                title="非线性控制: 残差(BMI|bs(周)+QC) vs 残差(Y|bs(周)+QC)",
                xlabel="BMI(去混杂残差, 非线性周)",
                ylabel="Y(去混杂残差, 非线性周)",
                out_png=os.path.join(outdir, "partialcorr_Y_vs_BMI_splineWeek.png"),
                r=r4,
                p=p4,
                n=len(bmi_res4),
                ci95=ci4,
                qc_note=QC_NOTE,
            )
            # no-header version
            scatter_with_fit(
                x=bmi_res4,
                y=y_res4,
                title="",
                xlabel="BMI(去混杂残差, 非线性周)",
                ylabel="Y(去混杂残差, 非线性周)",
                out_png=os.path.join(outdir, "partialcorr_Y_vs_BMI_splineWeek_noheader.png"),
                r=r4,
                p=p4,
                annotate=True,
                show_title=False,
                n=len(bmi_res4),
                ci95=ci4,
                qc_note=QC_NOTE,
            )
        except Exception as e:
            print("Nonlinear control (Week spline) failed:", f"{type(e).__name__}: {e}")

    # Export a small summary CSV
    rows = [
        ("r_{Y,周|BMI+QC}", r1),
        ("p_{Y,周|BMI+QC}", p1),
        ("r_{Y,BMI|周+QC}", r2),
        ("p_{Y,BMI|周+QC}", p2),
    ]
    if 'r3' in locals():
        rows.extend([
            ("r_{Y,周|bs(BMI)+QC}", r3),
            ("p_{Y,周|bs(BMI)+QC}", p3),
        ])
    if 'r4' in locals():
        rows.extend([
            ("r_{Y,BMI|bs(周)+QC}", r4),
            ("p_{Y,BMI|bs(周)+QC}", p4),
        ])
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    csv_path = os.path.join(outdir, "partialcorr_summary.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_path}")

    # Also export residuals used for the plots
    resid_payload = {
        "week_res": week_res1,
        "y_res_week": y_res1,
        "bmi_res": bmi_res2,
        "y_res_bmi": y_res2,
    }
    if 'r3' in locals():
        resid_payload.update({
            "week_res_nlBMI": week_res3,
            "y_res_week_nlBMI": y_res3,
        })
    if 'r4' in locals():
        resid_payload.update({
            "bmi_res_nlWeek": bmi_res4,
            "y_res_bmi_nlWeek": y_res4,
        })
    resid_df = pd.DataFrame(resid_payload)
    resid_csv = os.path.join(outdir, "partialcorr_residuals.csv")
    resid_df.to_csv(resid_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {resid_csv}")

    print("\nDone. Figures are in:")
    print(outdir)
    print("\nStats:")
    print(f"r(Y,周|BMI+QC) = {r1:.4f}, p = {p1:.3g}")
    print(f"r(Y,BMI|周+QC) = {r2:.4f}, p = {p2:.3g}")
    if 'r3' in locals():
        print(f"r(Y,周|bs(BMI)+QC) = {r3:.4f}, p = {p3:.3g}")
    if 'r4' in locals():
        print(f"r(Y,BMI|bs(周)+QC) = {r4:.4f}, p = {p4:.3g}")


if __name__ == "__main__":
    main()
