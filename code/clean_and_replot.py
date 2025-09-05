from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker


def find_date_columns(df: pd.DataFrame):
    # 通过列名关键词猜测“检测日期/时间”和“末次月经”列
    det_cols_keys = ["检测日期", "检测时间", "采样日期", "采样时间", "检测", "采样"]
    lmp_cols_keys = ["末次月经", "末次月经日期", "末次经期", "LMP"]

    def pick(colnames, keys):
        for key in keys:
            for c in colnames:
                name = str(c)
                if key.lower() in name.lower():
                    return c
        return None

    det_col = pick(df.columns, det_cols_keys)
    lmp_col = pick(df.columns, lmp_cols_keys)
    return det_col, lmp_col


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """稳健地将列解析为日期时间：
    - 先按字符串智能解析（含 2023/9/1、2023-09-01 等）
    - 针对 8 位纯数字（如 20230602）按 %Y%m%d 解析
    - 针对 Excel 序列号（大致 20000~80000 天）按 origin=1899-12-30 解析
    返回 pandas.DatetimeIndex 对应的 Series（NaT 表示无法解析）。
    """
    s_str = series.astype(str).str.strip()
    # 1) 智能解析常见格式
    dt = pd.to_datetime(s_str, errors="coerce", infer_datetime_format=True)

    # 2) 8位纯数字按 %Y%m%d 解析
    mask_8d = s_str.str.match(r"^\d{8}$", na=False)
    if mask_8d.any():
        dt_8d = pd.to_datetime(s_str[mask_8d], format="%Y%m%d", errors="coerce")
        dt.loc[mask_8d] = dt.loc[mask_8d].fillna(dt_8d)

    # 3) Excel 序列号解析（单位：天）
    s_num = pd.to_numeric(series, errors="coerce")
    mask_serial = s_num.between(20000, 80000, inclusive="both")
    if mask_serial.any():
        dt_serial = pd.to_datetime(s_num[mask_serial], unit="D", origin="1899-12-30", errors="coerce")
        dt.loc[mask_serial] = dt.loc[mask_serial].fillna(dt_serial)

    return dt


def filter_invalid_date_rows(df: pd.DataFrame):
    det_col, lmp_col = find_date_columns(df)
    if det_col is None or lmp_col is None:
        return df.copy(), pd.DataFrame(), det_col, lmp_col

    det = to_datetime_safe(df[det_col])
    lmp = to_datetime_safe(df[lmp_col])
    # 异常判定：检测日期 < 末次月经（LMP 晚于检测）才算异常；若 LMP 为空则视为正常
    bad_mask = (det.notna() & lmp.notna()) & (det < lmp)
    bad_rows = df.loc[bad_mask].copy()
    cleaned = df.loc[~bad_mask].copy()
    return cleaned, bad_rows, det_col, lmp_col


def ensure_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "Microsoft YaHei UI Light",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


def plot_bmi(df: pd.DataFrame, output_path: Path):
    # 列: V (Y浓度, 0基=21), K (BMI, 0基=10)
    try:
        if df.shape[1] >= 22:
            y_series = df.iloc[:, 21]
            x_series = df.iloc[:, 10]
        else:
            # 兜底：直接使用最后两列
            y_series = df.iloc[:, -1]
            x_series = df.iloc[:, -2]
    except Exception:
        # 再兜底：若仍异常，交换最后两列
        y_series = df.iloc[:, -1]
        x_series = df.iloc[:, -2]

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(x, y, s=16, color="#1f77b4", alpha=0.75, edgecolors="none", label="样本")
    ax.set_xlabel("孕妇BMI")
    ax.set_ylabel("Y染色体浓度")

    # LOWESS中心线
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        frac = 0.3
        xn = np.asarray(x, dtype=float)
        yn = np.asarray(y, dtype=float)
        low = lowess(yn, xn, frac=frac, it=0, return_sorted=True)
        ax.plot(low[:, 0], low[:, 1], color="#ff7f0e", linewidth=2.0, label=f"LOWESS (frac={frac})")
    except Exception:
        print("[BMI] 提示：缺少 statsmodels，未绘制LOWESS趋势线。")

    # 分箱中位数 ±95% CI（自助法）
    def binned_median_ci(xv, yv, n_bins=8, n_boot=800, seed=42, min_count=5):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if xv.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
        centers, meds, los, his = [], [], [], []
        rng = np.random.default_rng(seed)
        for i in range(n_bins):
            if i < n_bins - 1:
                m = (xv >= edges[i]) & (xv < edges[i + 1])
            else:
                m = (xv >= edges[i]) & (xv <= edges[i + 1])
            y_bin = yv[m]
            if y_bin.size == 0:
                continue
            med = np.median(y_bin)
            if y_bin.size >= min_count:
                n = y_bin.size
                boots = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    boots[b] = np.median(y_bin[idx])
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(med)
            los.append(lo)
            his.append(hi)
        return np.array(centers), np.array(meds), np.array(los), np.array(his)

    c, m, lo, hi = binned_median_ci(x, y)
    if c.size > 0:
        yerr = np.vstack([
            np.where(np.isfinite(lo), m - lo, np.nan),
            np.where(np.isfinite(hi), hi - m, np.nan),
        ])
        ax.errorbar(c, m, yerr=yerr, fmt="o-", color="#2ca02c", linewidth=1.8, markersize=4, capsize=3,
                    label="分箱中位数 ±95% CI")

    # 分界线 0.04
    boundary = 0.04
    ax.axhline(boundary, color="crimson", linestyle="--", linewidth=1.5, label="分界线 0.04")

    # y 轴刻度包含 0.04
    fig.canvas.draw()
    yticks = ax.get_yticks()
    if not np.any(np.isclose(yticks, boundary, atol=1e-9)):
        yticks = np.append(yticks, boundary)
        yticks = np.array(sorted(set(np.round(yticks.astype(float), 10))))
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    fig.canvas.draw()
    for label in ax.get_yticklabels():
        try:
            val = float(label.get_text())
            if np.isclose(val, boundary, atol=1e-9):
                label.set_color("crimson")
                label.set_fontweight("bold")
        except Exception:
            pass

    # 相关标注（左上角）
    try:
        from scipy.stats import spearmanr

        def spearman_ci(xv, yv, n_boot=1000, seed=42):
            xv = np.asarray(xv, dtype=float)
            yv = np.asarray(yv, dtype=float)
            mask = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[mask], yv[mask]
            if xv.size < 3:
                return np.nan, (np.nan, np.nan), np.nan
            rho, p = spearmanr(xv, yv)
            rng = np.random.default_rng(seed)
            n = xv.size
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                r_b, _ = spearmanr(xv[idx], yv[idx])
                if np.isfinite(r_b):
                    boots.append(r_b)
            if len(boots) >= 2:
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo, hi = (np.nan, np.nan)
            return rho, (lo, hi), p

        rho, (lo_, hi_), p = spearman_ci(x, y, n_boot=1000, seed=42)
        annot = f"Spearman ρ = {rho:.3f}\n95% CI [{lo_:.3f}, {hi_:.3f}]\np = {p:.3g}"
        ax.text(0.02, 0.98, annot, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))
    except Exception:
        print("[BMI] 提示：缺少 SciPy，未添加Spearman标注。")

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_week(df: pd.DataFrame, output_path: Path):
    # 列: V (Y浓度, 0基=21), AF(检孕周, 0基=31)
    try:
        if df.shape[1] >= 32:
            y_series = df.iloc[:, 21]
            x_series = df.iloc[:, 31]
        else:
            # 兜底：直接使用最后两列
            y_series = df.iloc[:, -1]
            x_series = df.iloc[:, -2]
    except Exception:
        y_series = df.iloc[:, -1]
        x_series = df.iloc[:, -2]

    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & y.notna()
    data = pd.DataFrame({"week": x[mask], "y": y[mask]}).sort_values("week")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(data["week"], data["y"], s=9, color="#1f77b4", label="Y染色体浓度")
    ax.set_xlabel("检孕周")
    ax.set_ylabel("Y染色体浓度")

    # LOWESS中心线
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        frac = 0.3
        xn = data["week"].to_numpy(dtype=float)
        yn = data["y"].to_numpy(dtype=float)
        low = lowess(yn, xn, frac=frac, it=0, return_sorted=True)
        ax.plot(low[:, 0], low[:, 1], color="#ff7f0e", linewidth=2.0, label=f"LOWESS (frac={frac})")
    except Exception:
        print("[AF] 提示：缺少 statsmodels，未绘制LOWESS趋势线。")

    # 分箱中位数 ±95% CI（自助法）
    def binned_median_ci(xv, yv, n_bins=8, n_boot=800, seed=42, min_count=5):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if xv.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
        centers, meds, los, his = [], [], [], []
        rng = np.random.default_rng(seed)
        for i in range(n_bins):
            if i < n_bins - 1:
                m = (xv >= edges[i]) & (xv < edges[i + 1])
            else:
                m = (xv >= edges[i]) & (xv <= edges[i + 1])
            y_bin = yv[m]
            if y_bin.size == 0:
                continue
            med = np.median(y_bin)
            if y_bin.size >= min_count:
                n = y_bin.size
                boots = np.empty(n_boot)
                for b in range(n_boot):
                    idx = np.random.default_rng(seed + b).integers(0, n, size=n)
                    boots[b] = np.median(y_bin[idx])
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(med)
            los.append(lo)
            his.append(hi)
        return np.array(centers), np.array(meds), np.array(los), np.array(his)

    c, m, lo, hi = binned_median_ci(data["week"].to_numpy(), data["y"].to_numpy())
    if c.size > 0:
        yerr = np.vstack([
            np.where(np.isfinite(lo), m - lo, np.nan),
            np.where(np.isfinite(hi), hi - m, np.nan),
        ])
        ax.errorbar(c, m, yerr=yerr, fmt="o-", color="#2ca02c", linewidth=1.8, markersize=4, capsize=3,
                    label="分箱中位数 ±95% CI")

    # 额外：P90 / P10 分位数趋势线
    def binned_quantile(xv, yv, q=0.9, n_bins=12, min_count=5):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if xv.size == 0:
            return np.array([]), np.array([])
        edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
        centers, qs = [], []
        for i in range(n_bins):
            if i < n_bins - 1:
                m = (xv >= edges[i]) & (xv < edges[i + 1])
            else:
                m = (xv >= edges[i]) & (xv <= edges[i + 1])
            y_bin = yv[m]
            if y_bin.size >= min_count:
                centers.append(0.5 * (edges[i] + edges[i + 1]))
                qs.append(np.quantile(y_bin, q))
        return np.array(centers), np.array(qs)

    xv = data["week"].to_numpy(dtype=float)
    yv = data["y"].to_numpy(dtype=float)
    c90, q90 = binned_quantile(xv, yv, q=0.9)
    c10, q10 = binned_quantile(xv, yv, q=0.1)
    if c90.size > 0:
        ax.plot(c90, q90, linestyle="--", color="#9467bd", linewidth=2.0, label="P90 趋势")
    if c10.size > 0:
        ax.plot(c10, q10, linestyle="--", color="#8c564b", linewidth=2.0, label="P10 趋势")

    # 分界线 0.04
    boundary = 0.04
    ax.axhline(boundary, color="crimson", linestyle="--", linewidth=1.5, label="分界线 0.04")

    # y 轴刻度包含 0.04
    fig.canvas.draw()
    yticks = ax.get_yticks()
    if not np.any(np.isclose(yticks, boundary, atol=1e-9)):
        yticks = np.append(yticks, boundary)
        yticks = np.array(sorted(set(np.round(yticks.astype(float), 10))))
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    fig.canvas.draw()
    for label in ax.get_yticklabels():
        try:
            val = float(label.get_text())
            if np.isclose(val, boundary, atol=1e-9):
                label.set_color("crimson")
                label.set_fontweight("bold")
        except Exception:
            pass

    # 相关标注（左上角）
    try:
        from scipy.stats import spearmanr

        def spearman_ci(xv_, yv_, n_boot=1000, seed=42):
            xv_ = np.asarray(xv_, dtype=float)
            yv_ = np.asarray(yv_, dtype=float)
            mask = np.isfinite(xv_) & np.isfinite(yv_)
            xv_, yv_ = xv_[mask], yv_[mask]
            if xv_.size < 3:
                return np.nan, (np.nan, np.nan), np.nan
            rho, p = spearmanr(xv_, yv_)
            rng = np.random.default_rng(seed)
            n = xv_.size
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                r_b, _ = spearmanr(xv_[idx], yv_[idx])
                if np.isfinite(r_b):
                    boots.append(r_b)
            if len(boots) >= 2:
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo, hi = (np.nan, np.nan)
            return rho, (lo, hi), p

        rho, (lo_, hi_), p = spearman_ci(data["week"].to_numpy(), data["y"].to_numpy(), n_boot=1000, seed=42)
        annot = f"Spearman ρ = {rho:.3f}\n95% CI [{lo_:.3f}, {hi_:.3f}]\np = {p:.3g}"
        ax.text(0.02, 0.98, annot, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))
    except Exception:
        print("[AF] 提示：缺少 SciPy，未添加Spearman标注。")

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    ensure_fonts()
    script_dir = Path(__file__).resolve().parent
    excel_path = script_dir / "fujian.xlsx"
    out_bmi = script_dir / "fujian_Y_vs_BMI_scatter.png"
    out_week = script_dir / "fujian_Y_vs_AF_line.png"
    report_csv = script_dir / "invalid_date_rows.csv"

    if not excel_path.exists():
        raise FileNotFoundError(f"未找到Excel文件: {excel_path}")

    # 读取Excel
    try:
        df0 = pd.read_excel(excel_path, sheet_name=0)
    except Exception:
        df0 = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")

    # 过滤异常日期行（检测日期 < 末次月经）
    df_clean, bad_rows, det_col, lmp_col = filter_invalid_date_rows(df0)
    if det_col is None or lmp_col is None:
        print("未自动识别到日期列，跳过过滤，仅重新绘图。")
    else:
        print(f"识别到日期列：检测列=‘{det_col}’，末次月经列=‘{lmp_col}’")
        print(f"发现异常行数：{len(bad_rows)}（检测日期 < 末次月经）")
        try:
            bad_rows.to_csv(report_csv, index=False)
            print(f"已导出异常行到: {report_csv}")
        except Exception as e:
            print(f"导出异常行失败: {e}")

    # 使用清洗后的数据绘图
    plot_bmi(df_clean, out_bmi)
    print(f"已保存图像: {out_bmi}")
    plot_week(df_clean, out_week)
    print(f"已保存图像: {out_week}")
