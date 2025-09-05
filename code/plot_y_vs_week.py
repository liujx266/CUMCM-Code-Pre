from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker


def main():
    # Configure fonts and sizes (Microsoft YaHei, 10pt)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Microsoft YaHei UI", "Microsoft YaHei UI Light", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    excel_path = script_dir / "fujian.xlsx"
    output_path = script_dir / "fujian_Y_vs_AF_line.png"

    if not excel_path.exists():
        raise FileNotFoundError(f"未找到Excel文件: {excel_path}")

    # Read Excel: prefer positional indices for columns V(22) and AF(32)
    # Fallback to usecols if width is smaller or headers are unusual.
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except Exception:
        # Retry with openpyxl if available
        df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")

    try:
        if df.shape[1] >= 32:
            # Zero-based positions: V=21, AF=31
            y_series = df.iloc[:, 21]
            x_series = df.iloc[:, 31]
        else:
            raise IndexError("列数不足，尝试按列字母读取")
    except Exception:
        # Fallback: explicitly pull columns by Excel letters
        slim = pd.read_excel(excel_path, sheet_name=0, usecols="V,AF")
        # First is V (Y浓度), second is AF (检孕周)
        y_series = slim.iloc[:, 0]
        x_series = slim.iloc[:, 1]

    # Coerce to numeric, drop NaNs
    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & y.notna()
    data = pd.DataFrame({"week": x[mask], "y": y[mask]})
    # Sort by gestational week for a clean line
    data = data.sort_values("week")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(
        data["week"],
        data["y"],
        s=9,  # approximately markersize=3**2
        color="#1f77b4",
        label="Y染色体浓度",
    )

    ax.set_xlabel("检孕周")
    ax.set_ylabel("Y染色体浓度")
    # 不设置标题，按需仅显示坐标轴与图例

    # LOWESS 趋势线（若安装了 statsmodels）
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        frac = 0.3  # 平滑窗口比例(0~1)，可按需要调整
        xn = data["week"].to_numpy(dtype=float)
        yn = data["y"].to_numpy(dtype=float)
        low = lowess(yn, xn, frac=frac, it=0, return_sorted=True)
        ax.plot(low[:, 0], low[:, 1], color="#ff7f0e", linewidth=2.0, label=f"LOWESS (frac={frac})")
    except Exception:
        print("未能绘制LOWESS趋势线，请先安装 statsmodels：pip install statsmodels")

    # 分箱中位数 + 95% CI（自助法）
    def binned_median_ci(xv, yv, n_bins=8, n_boot=1000, seed=42, min_count=5):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if xv.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
        centers = []
        meds = []
        los = []
        his = []
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

    c, m, lo, hi = binned_median_ci(data["week"].to_numpy(), data["y"].to_numpy(), n_bins=8, n_boot=800, seed=42, min_count=5)
    if c.size > 0:
        yerr = np.vstack([
            np.where(np.isfinite(lo), m - lo, np.nan),
            np.where(np.isfinite(hi), hi - m, np.nan),
        ])
        ax.errorbar(
            c,
            m,
            yerr=yerr,
            fmt="o-",
            color="#2ca02c",
            linewidth=1.8,
            markersize=4,
            capsize=3,
            label="分箱中位数 ±95% CI",
        )

    # 额外：按孕周分箱绘制Y浓度的 90% 与 10% 分位数趋势
    def binned_quantile(xv, yv, q=0.9, n_bins=12, min_count=5):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if xv.size == 0:
            return np.array([]), np.array([])
        edges = np.linspace(np.min(xv), np.max(xv), n_bins + 1)
        centers = []
        qs = []
        for i in range(n_bins):
            if i < n_bins - 1:
                m = (xv >= edges[i]) & (xv < edges[i + 1])
            else:
                m = (xv >= edges[i]) & (xv <= edges[i + 1])
            y_bin = yv[m]
            if y_bin.size >= min_count:
                centers.append(0.5 * (edges[i] + edges[i + 1]))
                qs.append(np.quantile(y_bin, q))
        if len(centers) == 0:
            return np.array([]), np.array([])
        return np.array(centers), np.array(qs)

    x_arr = data["week"].to_numpy(dtype=float)
    y_arr = data["y"].to_numpy(dtype=float)
    c90, q90 = binned_quantile(x_arr, y_arr, q=0.9, n_bins=12, min_count=5)
    c10, q10 = binned_quantile(x_arr, y_arr, q=0.1, n_bins=12, min_count=5)
    if c90.size > 0:
        ax.plot(c90, q90, linestyle="--", color="#9467bd", linewidth=2.0, label="P90 趋势")
    if c10.size > 0:
        ax.plot(c10, q10, linestyle="--", color="#8c564b", linewidth=2.0, label="P10 趋势")

    # Emphasize y=0.04 boundary
    boundary = 0.04
    ax.axhline(boundary, color="crimson", linestyle="--", linewidth=1.5, label="分界线 0.04")

    # Ensure the 0.04 tick is present and visually emphasized
    fig.canvas.draw()  # initialize ticks
    yticks = ax.get_yticks()
    if not np.any(np.isclose(yticks, boundary, rtol=0, atol=1e-9)):
        yticks = np.append(yticks, boundary)
        yticks = np.array(sorted(set(np.round(yticks.astype(float), 10))))
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))

    # After setting ticks, color the 0.04 label
    fig.canvas.draw()
    for label in ax.get_yticklabels():
        try:
            val = float(label.get_text())
            if np.isclose(val, boundary, atol=1e-9):
                label.set_color("crimson")
                label.set_fontweight("bold")
        except Exception:
            pass

    # Light grid for readability
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    # Legend: 固定右上角，避免与左上角统计标注冲突
    ax.legend(frameon=False, loc="upper right")

    # 计算并标注 Spearman ρ、95% CI 和 p 值
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

        xv = data["week"].to_numpy(dtype=float)
        yv = data["y"].to_numpy(dtype=float)
        rho, (lo, hi), p = spearman_ci(xv, yv, n_boot=1000, seed=42)
        annot = f"Spearman ρ = {rho:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]\np = {p:.3g}"
        ax.text(
            0.02,
            0.98,
            annot,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )
    except Exception:
        print("未能计算Spearman相关/置信区间，请先安装 SciPy：pip install scipy")

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"已保存图像: {output_path}")

    # Optional: show the plot interactively (comment out if running headless)
    # plt.show()


if __name__ == "__main__":
    main()
