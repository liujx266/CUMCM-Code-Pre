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
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"已保存图像: {output_path}")

    # plt.show()


if __name__ == "__main__":
    main()
