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

    # Paths
    script_dir = Path(__file__).resolve().parent
    excel_path = script_dir / "fujian.xlsx"
    output_path = script_dir / "fujian_Y_vs_BMI_scatter.png"

    if not excel_path.exists():
        raise FileNotFoundError(f"未找到Excel文件: {excel_path}")

    # Read Excel
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except Exception:
        df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")

    # Columns: V (Y浓度, zero-based 21), K (BMI, zero-based 10)
    try:
        if df.shape[1] >= 22:  # need at least up to V
            y_series = df.iloc[:, 21]
            x_series = df.iloc[:, 10]
        else:
            raise IndexError("列数不足，尝试按列字母读取")
    except Exception:
        slim = pd.read_excel(excel_path, sheet_name=0, usecols="V,K")
        # First is V (Y浓度), second is K (BMI)
        y_series = slim.iloc[:, 0]
        x_series = slim.iloc[:, 1]

    # Coerce to numeric and drop NaNs
    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    # Plot scatter
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(
        x,
        y,
        s=16,
        color="#1f77b4",
        alpha=0.75,
        edgecolors="none",
        label="样本",
    )

    ax.set_xlabel("孕妇BMI")
    ax.set_ylabel("Y染色体浓度")
    # 不设置标题，按需仅显示坐标轴与图例

    # LOWESS 趋势线 + 95%置信带（若安装了 statsmodels）
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        def lowess_with_ci(xv, yv, frac=0.3, it=0, grid=None, n_boot=200, seed=42):
            xv = np.asarray(xv, dtype=float)
            yv = np.asarray(yv, dtype=float)
            if grid is None:
                grid = np.linspace(np.nanmin(xv), np.nanmax(xv), 200)
            # 中心曲线（基于全体样本）
            base = lowess(yv, xv, frac=frac, it=it, return_sorted=True)
            y_base = np.interp(grid, base[:, 0], base[:, 1])
            # 自助法估计置信区间
            rng = np.random.default_rng(seed)
            preds = np.empty((n_boot, grid.size), dtype=float)
            n = xv.size
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                xb, yb = xv[idx], yv[idx]
                boot = lowess(yb, xb, frac=frac, it=it, return_sorted=True)
                preds[b] = np.interp(grid, boot[:, 0], boot[:, 1])
            lower = np.percentile(preds, 2.5, axis=0)
            upper = np.percentile(preds, 97.5, axis=0)
            return grid, y_base, lower, upper

        frac = 0.3  # 平滑窗口比例(0~1)，可根据数据调整
        xn = np.asarray(x, dtype=float)
        yn = np.asarray(y, dtype=float)
        grid, y_hat, y_lo, y_hi = lowess_with_ci(xn, yn, frac=frac, it=0, n_boot=200, seed=42)
        ax.fill_between(grid, y_lo, y_hi, color="#ff7f0e", alpha=0.2, label="95% 置信带")
        ax.plot(grid, y_hat, color="#ff7f0e", linewidth=2.0, label=f"LOWESS (frac={frac})")
    except Exception:
        print("未能绘制LOWESS趋势线/置信带，请先安装 statsmodels：pip install statsmodels")

    # Emphasize y=0.04 boundary
    boundary = 0.04
    ax.axhline(boundary, color="crimson", linestyle="--", linewidth=1.5, label="分界线 0.04")

    # Ensure the 0.04 tick is present and emphasized
    fig.canvas.draw()
    yticks = ax.get_yticks()
    if not np.any(np.isclose(yticks, boundary, rtol=0, atol=1e-9)):
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

    # Grid
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
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

        rho, (lo, hi), p = spearman_ci(x, y, n_boot=1000, seed=42)
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

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"已保存图像: {output_path}")

    # plt.show()


if __name__ == "__main__":
    main()
