# -*- coding: utf-8 -*-
"""
GAM 建模（连续 + Logistic）— 健壮版
- 读取同目录 Excel: 合并结果去重-手动去掉异常值并补零.xlsx
- 连续: asin√(Y) ~ s(孕周)+s(BMI)+线性QC
- Logistic: hit=1(Y>=0.04) ~ s(孕周)+s(BMI)+线性QC
输出：gam_output\ 下若干图表&CSV

注：pyGAM 的单项 p 值有偏乐观；主要看 EDF（若可得）+ 置信带与曲线形状。
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from pygam import LinearGAM, LogisticGAM, s, l

# ---------------- 配置 ----------------
EXCEL_FILE = "合并结果去重-手动去掉异常值并补零.xlsx"
SHEET_NAME = 0
Y_THRESHOLD = 0.04
BMI_BINS = [20, 28, 32, 36, 40, 50]
ONLY_MALE = False

OVERRIDES = {
    "y":   "Y染色体浓度",
    "week":"检测孕周",
    "day": "间隔天数",   # 若不是 0~6 的“天”，脚本会自动忽略
    "bmi": "孕妇BMI",
    "sex": None,
    "qc":  ["GC含量","13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量"]
}

# --------------- 工具函数 ---------------
def _norm(s): return re.sub(r"\s+", "", str(s)).lower()

# 通过关键词自动检测所需要的处理的列
def autodetect_columns(df: pd.DataFrame):
    nm = {_norm(c): c for c in df.columns}
    def find(cands):
        for k in cands:
            kn = _norm(k)
            for c, raw in nm.items():
                if kn == c or kn in c: return raw
        return None
    y   = find(["y染色体浓度","y含量","胎儿y","y比例","fetal_y","chr_y","y"])
    week= find(["检测孕周","孕周","周","week","gestationalweek","ga"])
    day = find(["天","day","days","间隔天数"])
    bmi = find(["孕妇bmi","bmi","体质指数"])
    sex = find(["胎儿性别","性别","sex","gender"])
    qc_patterns = ["uniq","unique","reads","aa","gc","filt","dup","repeat","mapped","coverage","depth","nratio"]
    qc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and
          any(p in _norm(c) for p in qc_patterns)]
    return {"y":y,"week":week,"day":day,"bmi":bmi,"sex":sex,"qc":qc}

def compute_week_decimal(df, week_col, day_col=None):
    week = pd.to_numeric(df[week_col], errors='coerce').astype(float)
    if day_col and day_col in df.columns:
        day = pd.to_numeric(df[day_col], errors='coerce')
        ok = day.dropna()
        if len(ok) and ( (ok%1==0).mean()>0.95 ) and ( ok.between(0,6).mean()>0.95 ):
            week = week + ok.astype(float)/7.0
    return week

def ensure_log_neff(df, qc_cols):
    nm = {_norm(c): c for c in df.columns}
    uniq = None; aa = None
    for k,v in nm.items():
        if uniq is None and ("uniq" in k or "unique" in k) and pd.api.types.is_numeric_dtype(df[v]): uniq=v
        if aa   is None and (k=="aa" or k.endswith("_aa") or "aa" in k) and pd.api.types.is_numeric_dtype(df[v]): aa=v
    if uniq and aa:
        neff = pd.to_numeric(df[uniq], errors='coerce') * (1 - pd.to_numeric(df[aa], errors='coerce'))
        df["log_neff"] = np.log10(neff.replace(0, np.nan))
        if "log_neff" not in qc_cols: qc_cols.append("log_neff")
    return df, qc_cols

def asinsqrt_y(y: pd.Series):
    y = pd.to_numeric(y, errors='coerce').astype(float)
    if y.max(skipna=True) > 1.5: y = y/100.0
    n = y.notna().sum()
    eps = 0.5/max(n,1)
    return np.arcsin(np.sqrt(y.clip(eps, 1-eps))), y

def make_outdir(base):
    out = os.path.join(base, "gam_output"); os.makedirs(out, exist_ok=True); return out

def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); print("Saved:", path)

# --- 关键：兼容不同版本 pyGAM 的 EDF 取法 ---
def get_edf_per_term(gam_obj, n_s_terms=2):
    """返回长度为 n_s_terms 的 EDF 数组；若不可得则用 np.nan 填充。"""
    edf_arr = None
    stats = getattr(gam_obj, "statistics_", {}) or {}
    # 常见键
    for k in ["edof_per_term", "edof_by_term", "edf_per_term", "edf_by_term"]:
        if isinstance(stats, dict) and (k in stats):
            try:
                arr = np.asarray(stats[k]).ravel()
                if arr.size >= n_s_terms:
                    edf_arr = arr[:n_s_terms]
                    break
            except Exception:
                pass
    if edf_arr is None:
        # 实在取不到：用 nan 占位
        edf_arr = np.full(n_s_terms, np.nan, dtype=float)
    # 也取一个总 EDF 以备导出
    total_edf = None
    for k in ["edof", "edf", "effective_dof"]:
        if isinstance(stats, dict) and (k in stats):
            try:
                total_edf = float(np.asarray(stats[k]).squeeze())
                break
            except Exception:
                pass
    return edf_arr, total_edf

# --------------- 主流程 ---------------
def main():
    # 字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fpath = os.path.join(os.getcwd(), EXCEL_FILE)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"未找到 Excel：{fpath}")

    xls = pd.ExcelFile(fpath)
    df  = xls.parse(SHEET_NAME)
    print(f"Loaded: {EXCEL_FILE} / sheet={SHEET_NAME}, n={len(df)}")

    det = autodetect_columns(df)
    for k,v in OVERRIDES.items():
        if v is not None: det[k]=v
    print("列映射：", det)

    y_col, week_col, day_col, bmi_col, sex_col = det["y"], det["week"], det.get("day"), det["bmi"], det.get("sex")
    qc_cols = list(det.get("qc", []))
    if not y_col or not week_col or not bmi_col:
        raise RuntimeError("至少需要 y / 孕周 / BMI 三列，请在 OVERRIDES 指定。")

    df = df.copy()
    df["week_dec"] = compute_week_decimal(df, week_col, day_col)
    df["bmi"]      = pd.to_numeric(df[bmi_col], errors='coerce').astype(float)
    df["y_raw"]    = pd.to_numeric(df[y_col], errors='coerce').astype(float)

    df, qc_cols = ensure_log_neff(df, qc_cols)
    qc_cols = [c for c in qc_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    print("使用的 QC 列：", qc_cols)

    keep = ["week_dec","bmi","y_raw"] + qc_cols
    df = df[keep].dropna().copy()
    print("清洗后 n =", len(df))

    if ONLY_MALE and sex_col and sex_col in df.columns:
        mask_male = df[sex_col].astype(str).str.contains(r"^(m|male|男)$", case=False, regex=True, na=False)
        df = df.loc[mask_male].copy()

    y_asin, y01 = asinsqrt_y(df["y_raw"])
    df["y_asin"] = y_asin
    df["y01"]    = y01
    df["hit"]    = (df["y01"] >= Y_THRESHOLD).astype(int)

    outdir = make_outdir(os.getcwd())

    # ---------- 连续版 ----------
    X_core = df[["week_dec","bmi"]].to_numpy()
    X_qc   = df[qc_cols].to_numpy() if len(qc_cols) else None
    X = X_core if X_qc is None else np.c_[X_core, X_qc]
    y = df["y_asin"].to_numpy()

    terms = s(0, n_splines=20) + s(1, n_splines=20)
    if len(qc_cols):
        for i in range(X_qc.shape[1]): terms += l(2+i)

    print("\n正在 gridsearch 拟合 LinearGAM ...")
    gam = LinearGAM(terms); gam.gridsearch(X, y)
    print("LinearGAM 拟合完成。")
    # 打印摘要（不要再 print(gam.summary()) 以免多打印一个 None）
    gam.summary()

    # 取 EDF（兼容不同版本）
    edf_terms, edf_total = get_edf_per_term(gam, n_s_terms=2)
    pvals = np.asarray(gam.statistics_.get("p_values", [np.nan]*(2+len(qc_cols)+1))).ravel()

    # 画部分依赖
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["检测孕周", "孕妇BMI"]
    for i, ax in enumerate(axes):
        XX = gam.generate_X_grid(term=i)
        pdep, ci = gam.partial_dependence(term=i, X=XX, width=.95)
        ci = np.asarray(ci)
        lo, hi = (ci[:,0], ci[:,1]) if (ci.ndim==2 and ci.shape[1]==2) else (ci[0], ci[1])
        ax.plot(XX[:,i], pdep, lw=1.8, label="平滑样条效应")
        ax.fill_between(XX[:,i], lo, hi, alpha=.25, label="95% 置信带")
        edf_txt = f"{edf_terms[i]:.1f}" if np.isfinite(edf_terms[i]) else "N/A"
        p_txt   = f"{pvals[i]:.3g}" if np.isfinite(pvals[i]) else "N/A"
        # 避免与左图例（upper left）重叠：左图把文字放到右上角
        if i == 0:
            ax.text(.98, .98, f"EDF={edf_txt}\np={p_txt}", transform=ax.transAxes,
                    ha='right', va='top', bbox=dict(fc='white', ec='none', alpha=.8))
        else:
            ax.text(.02, .98, f"EDF={edf_txt}\np={p_txt}", transform=ax.transAxes,
                    ha='left', va='top', bbox=dict(fc='white', ec='none', alpha=.8))
        raw = df["week_dec"].to_numpy() if i==0 else df["bmi"].to_numpy()
        ax.plot(raw, np.full_like(raw, ax.get_ylim()[0]), '|', ms=8, alpha=.18)
        ax.set_xlabel(labels[i])
        if i==0: ax.set_ylabel("对 asin√(Y浓度) 的效应")
        if i==0: ax.legend(loc='upper left', fontsize=9)
    axes[0].set_title("孕周的部分效应")
    axes[1].set_title("BMI 的部分效应")
    savefig(fig, os.path.join(outdir, "GAM_partial_dependence.png"))

    # 导出连续版摘要
    expl_dev = float(gam.statistics_.get('pseudo_r2', {}).get('explained_deviance', np.nan))
    pd.DataFrame({
        "term":["s(week)","s(BMI)","TOTAL_EDF","Explained_Deviance"],
        "edf":[edf_terms[0] if np.isfinite(edf_terms[0]) else np.nan,
               edf_terms[1] if np.isfinite(edf_terms[1]) else np.nan,
               edf_total if (edf_total is not None) else np.nan,
               expl_dev]
    }).to_csv(os.path.join(outdir,"GAM_continuous_summary.csv"), index=False, encoding="utf-8-sig")

    # ---------- Logistic ----------
    print("\n正在 gridsearch 拟合 LogisticGAM ...")
    lg = LogisticGAM(terms); lg.gridsearch(X, df["hit"].to_numpy())
    print("LogisticGAM 拟合完成。")
    lg.summary()

    edf_terms_lg, edf_total_lg = get_edf_per_term(lg, n_s_terms=2)
    pvals_lg = np.asarray(lg.statistics_.get("p_values", [np.nan]*(2+len(qc_cols)+1))).ravel()

    phat = lg.predict_proba(X)
    auc  = roc_auc_score(df["hit"], phat)
    ap   = average_precision_score(df["hit"], phat)
    print(f"ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

    bins = pd.IntervalIndex.from_breaks(BMI_BINS, closed="left")
    df["bmi_grp"] = pd.cut(df["bmi"], bins=bins)
    bmi_reprs = df.groupby("bmi_grp")["bmi"].median().dropna()
    week_grid = np.linspace(df["week_dec"].min(), df["week_dec"].max(), 240)

    fig, ax = plt.subplots(figsize=(7,4))
    rows = []  # 收集达到阈值的孕周，单独导出为表格
    for grp, bmi0 in bmi_reprs.items():
        Xg = np.c_[week_grid, np.full_like(week_grid, bmi0)]
        if len(qc_cols):
            Qmed = df[qc_cols].median().to_numpy()
            Xg = np.c_[Xg, np.tile(Qmed, (len(week_grid),1))]
        p = lg.predict_mu(Xg)
        ax.plot(week_grid, p, label=f'BMI {int(grp.left)}-{int(grp.right)}')
        for tar in (0.90, 0.95):
            idx = int(np.argmax(p >= tar))
            if p[idx] >= tar:
                t_star = float(week_grid[idx])
                ax.axvline(t_star, ls='--', alpha=.25)  # 仅保留参考竖线，不加文字
                rows.append({"bmi_group": f"{int(grp.left)}-{int(grp.right)}",
                             "target": tar, "t_star_week": t_star})
    ax.axhline(0.90, ls=':', color='gray'); ax.axhline(0.95, ls=':', color='gray')
    ax.set_ylim(0,1); ax.set_xlabel("孕周"); ax.set_ylabel(f"Pr(Y≥{Y_THRESHOLD*100:.0f}%)")
    ax.legend(ncol=2, fontsize=8)
    savefig(fig, os.path.join(outdir, "LogisticGAM_Prob_vs_Week_byBMI.png"))

    # 导出达到阈值的孕周（长表）
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(os.path.join(outdir,"t_at_targets_by_BMI.csv"),
                   index=False, encoding="utf-8-sig")
    # 透视为宽表，便于阅读：每个 BMI 组一行，包含 t@90%, t@95%
    wide = df_rows.pivot(index='bmi_group', columns='target', values='t_star_week')
    # 列重命名
    rename_map = {}
    for c in wide.columns:
        try:
            val = float(c)
        except Exception:
            val = c
        if val == 0.90:
            rename_map[c] = 't_at_90'
        elif val == 0.95:
            rename_map[c] = 't_at_95'
    wide = wide.rename(columns=rename_map).reset_index()
    # 加入样本量 n（该 BMI 组中样本数）
    bin_str = pd.cut(df["bmi"], bins=bins)
    bin_str = bin_str.map(lambda iv: f"{int(iv.left)}-{int(iv.right)}" if pd.notna(iv) else None)
    counts = bin_str.value_counts().to_dict()
    wide['n_samples'] = wide['bmi_group'].map(lambda k: counts.get(k, 0))
    wide = wide[['bmi_group','n_samples'] + [c for c in ['t_at_90','t_at_95'] if c in wide.columns]]
    wide.to_csv(os.path.join(outdir, "t_at_targets_by_BMI_wide.csv"),
                index=False, encoding="utf-8-sig")
    # 同时输出一个 PNG 表格图，便于报告直接引用
    try:
        fig_tbl, ax_tbl = plt.subplots(figsize=(6, 0.6 + 0.35*len(wide)))
        ax_tbl.axis('off')
        col_labels = list(wide.columns)
        cell_text = [[f"{x:.1f}" if isinstance(x, (int, float)) and pd.notna(x) else ("" if x is None else str(x)) for x in row]
                     for row in wide.to_numpy()]
        table = ax_tbl.table(cellText=cell_text, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        savefig(fig_tbl, os.path.join(outdir, "t_at_targets_by_BMI_table.png"))
    except Exception as e:
        print("生成 PNG 表失败：", e)
    pd.DataFrame({
        "term":["s(week)","s(BMI)","TOTAL_EDF","ROC_AUC","PR_AUC"],
        "edf":[edf_terms_lg[0] if np.isfinite(edf_terms_lg[0]) else np.nan,
               edf_terms_lg[1] if np.isfinite(edf_terms_lg[1]) else np.nan,
               edf_total_lg if (edf_total_lg is not None) else np.nan,
               auc, ap]
    }).to_csv(os.path.join(outdir,"GAM_logistic_summary.csv"),
              index=False, encoding="utf-8-sig")

    print("\n全部完成。输出目录：", outdir)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
