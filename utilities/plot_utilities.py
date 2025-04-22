import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Iterable, List
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency
from typing import Iterable, Optional
import re


def to_percent(y, position):
    return f"{y:.0f}%"


def is_light_or_dark(rgbColor):
    if len(rgbColor) == 3:
        r, g, b = rgbColor
    else:
        r, g, b, _ = rgbColor

    brightness = r * 0.299 + g * 0.587 + b * 0.114

    return "light" if brightness > 0.5 else "dark"


def plot_triangle_corr_matrix(
    corr: pd.DataFrame,
    ax: mpl.axes,
    mask_half: str = "upper",
    annotation: bool = False,
    label_rotation: float = 55,
    annot_fs: float = 16,
    ticks_fs: float = 18,
    cbar_fs: float = 18,
    highlight: bool = False,
    high_threshold: float = 0.5,
    show_nan: bool = True,
    cramers: bool = False,
):
    """
    This functions aims at plotting the correlation heatmap
    """

    if cramers:
        cmap = plt.get_cmap("Reds")
        vmin, vmax = [0, 1]
    else:
        cmap = plt.get_cmap("seismic")
        vmin, vmax = [-1, 1]

    if mask_half == "upper":
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_tri = np.ma.masked_where(mask, corr)
        colorbar_loc = "right"
        xlabel_pos = "bottom"
        xlabel_rotation = label_rotation
        ylabel_pos = "left"

    elif mask_half == "lower":
        mask = np.tril(np.ones_like(corr_tri, dtype=bool))
        corr_tri = np.ma.masked_where(mask, corr)
        colorbar_loc = "left"
        xlabel_pos = "top"
        xlabel_rotation = -label_rotation
        ylabel_pos = "right"

    cax = ax.matshow(corr_tri, cmap=cmap, vmin=-1, vmax=1)

    cbar = plt.gcf().colorbar(cax, fraction=0.046, pad=0.04, location=colorbar_loc)
    cbar.ax.tick_params(labelsize=cbar_fs)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.xaxis.set_ticks_position(xlabel_pos)
    ax.yaxis.set_ticks_position(ylabel_pos)
    ax.set_xticklabels(
        corr.columns, rotation=xlabel_rotation, ha="right", fontsize=ticks_fs
    )
    ax.set_yticklabels(corr.columns, fontsize=ticks_fs)

    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)

    if annotation:
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if not mask[i, j] and not np.isnan(corr.iloc[i, j]):
                    val = corr.iloc[i, j]
                    color = (
                        "w"
                        if is_light_or_dark(cmap((val - vmin) / (vmax - vmin)))
                        == "dark"
                        else "k"
                    )
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=annot_fs,
                    )

    if highlight:
        highlight_mask = np.abs(corr) >= high_threshold

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if highlight_mask.iloc[i, j] and i > j:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            lw=3,
                        )
                    )

    if show_nan:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if np.isnan(corr.iloc[i, j]) and i > j:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            edgecolor="black",
                            facecolor="black",
                            lw=1,
                        )
                    )


def Rstyle_spines(
    ax: mpl.axes,
    spines_left: Iterable[str] = ["left", "bottom"],
    offset: float = 0,
    lw: float = 2,
):
    """
    This function changes the graph spines to make them
    look like R-styled
    """

    for loc, spine in ax.spines.items():
        if loc in spines_left:
            spine.set_position(("outward", offset))
            spine.set_linewidth(lw)
        else:
            spine.set_color("none")

    if "left" in spines_left:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines_left:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def custom_format_func(value: float, pos):
    return f"{value:.2f}"

def multicolumn_barplot(
    df: pd.DataFrame,
    fig: mpl.figure,
    width: str,
    y: str,
    spacing=0.4,
    num_col: int = 2,
    tick_fs: int = 20,
):
    """This functions produces a multicolumn horizontal bar plot."""

    n_features = len(df)
    xrange = (df[width].min(), df[width].max())
    axs = []
    parts = 1 / (num_col + (num_col - 1) * spacing)
    for i in range(num_col):
        axs.append(fig.add_axes([i * (parts + spacing), 0, parts, 0.95]))

        start_idx = i * n_features // num_col
        end_idx = min((i + 1) * n_features // num_col, n_features)

        axs[i].barh(
            df[y].iloc[start_idx:end_idx][::-1],
            df[width].iloc[start_idx:end_idx][::-1],
            color="blue",
        )

        axs[i].set_xlim(xrange)
        axs[i].tick_params(axis="both", which="major", labelsize=tick_fs)
        Rstyle_spines(axs[i], lw=1)

    fig.subplots_adjust(wspace=spacing)
    return axs


def plot_num_vs_target(fig: mpl.figure, df: pd.DataFrame, cols_to_plot: List[str]):
    """This function aims at plotting a long twwo columns plot
    where each numerical features is plot:
    * entirely (left)
    * just the instances falling into the minor class
    """

    num_rows = len(cols_to_plot)
    subfigs = fig.subfigures(num_rows, 1, wspace=0.07)

    for i, feature in enumerate(cols_to_plot):
        axs = subfigs[i].subplots(1, 2)
        subfigs[i].suptitle(feature, fontsize=22)
        feature_range = np.abs(df[feature].max() - df[feature].min())
        if df[feature].nunique() > 30 or feature_range > 30:
            axs[0].hist(df[df["TARGET"] == 0][feature], edgecolor="w", color="C2")
            axs[1].hist(df[df["TARGET"] == 1][feature], edgecolor="w", color="C1")
            Rstyle_spines(axs[0], lw=1)
            Rstyle_spines(axs[1], lw=1)
            axs[0].set_yscale("log")
            axs[1].set_yscale("log")
            axs[0].tick_params(axis="both", which="major", labelsize=8)
            axs[1].tick_params(axis="both", which="major", labelsize=8)
            axs[0].set_title(f"{feature} distribution", fontsize=10)
            axs[1].set_title(f"{feature}'s minor class distribution", fontsize=10)
        else:
            width = 0.8
            counts = df[feature].value_counts()
            global_percent = 100 * counts / counts.sum()
            target_percent = (
                100 * df[df["TARGET"] == 1][feature].value_counts() / counts
            )
            axs[0].bar(x=counts.index.values, height=global_percent, color="C2")
            axs[0].set_xlim(
                (counts.index.min() - width / 2, counts.index.max() + width / 2)
            )
            axs[0].set_xticks(
                np.arange(counts.index.min(), counts.index.max()).astype("int")
            )
            axs[0].tick_params(axis="both", which="major", labelsize=8)
            Rstyle_spines(axs[0], lw=1)
            axs[0].yaxis.set_major_formatter(FuncFormatter(to_percent))

            axs[1].bar(x=counts.index.values, height=target_percent, color="C1")
            axs[1].set_xlim(
                (counts.index.min() - width / 2, counts.index.max() + width / 2)
            )
            axs[1].set_xticks(
                np.arange(counts.index.min(), counts.index.max()).astype("int")
            )
            Rstyle_spines(axs[1], lw=1)
            axs[1].axhline(y=8.1, ls="--", lw=2, color="k")
            axs[1].tick_params(axis="both", which="major", labelsize=8)
            axs[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
            axs[0].set_title(f"{feature} distribution", fontsize=10)
            axs[1].set_title(f"{feature}'s minor class percentage", fontsize=10)


def set_to_percent(pct, allvals):
    """Set the printing format to percent"""
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"
