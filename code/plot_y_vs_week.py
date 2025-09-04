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
    ax.plot(
        data["week"],
        data["y"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        color="#1f77b4",
        label="Y染色体浓度",
    )

    ax.set_xlabel("检孕周")
    ax.set_ylabel("Y染色体浓度")
    # 不设置标题，按需仅显示坐标轴与图例

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

    # Legend
    ax.legend(frameon=False)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"已保存图像: {output_path}")

    # Optional: show the plot interactively (comment out if running headless)
    # plt.show()


if __name__ == "__main__":
    main()
