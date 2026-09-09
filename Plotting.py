"""External Imports (Libraries and APIs)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
 
 
def plot_fitness_surface(surfaces: list, saveloc: str = "tenet_surface.png", label_measure: str = "Fitness"):
    n = len(surfaces)
    fig, axes = plt.subplots(1, n, figsize = (3.6 * n, 3.9), squeeze = False)
 
 
    for ax, s in zip(axes[0], surfaces):
        a1, a2 = s["a1_grid"], s["a2_grid"]
 
        im = ax.imshow(
            s["Z"], origin = "lower", cmap = "RdBu_r", vmin = -1, vmax = 1,
            extent = [a1[0], a1[-1], a2[0], a2[-1]], aspect = "auto"
        )
        ax.axhline(0, color = "k", lw = 0.5, alpha = 0.4)
        ax.axvline(0, color = "k", lw = 0.5, alpha = 0.4)
 
        ax.set_title(f"magnitude $Y$ = {s['Y']:g}", fontsize = 10)
        ax.set_xlabel("Tenet 1:  cos(B, $SS^T$)")
 
 
        if ax is axes[0][0]:
            ax.set_ylabel("Tenet 2:  cos(G, S)")
 
        else:
            ax.set_yticklabels([])
 
 
    fig.colorbar(im, ax = axes[0].tolist(), fraction = 0.025, pad = 0.02, label = label_measure)
    fig.suptitle(f"{label_measure} over the two-tenet space (Number of seeds: {n})", fontsize = 11)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
 
 
 
 
def plot_interaction_trajectories(result: dict, saveloc: str, title: str = None):
    fig, ax = plt.subplots(figsize = (6, 4))
 
    gens = np.asarray(result["recorded_gens"])
 
 
    for trajectory in result["trajectories"]:
        ax.plot(gens, trajectory, lw = 0.8)
 
 
    ax.set_title(title)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Regulation coefficient")
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi = 150)
    plt.close(fig)
 
 
 
 
def plot_measure_surfaces(
        surfaces: list,
        saveloc: str = "tenet_measure.png",
        label_measure: str = "measure",
        scale: str = "sequential",
        subtitle: str = None,
        colour_scale: str = "percentile",
        clip_percentile: float = 98.0
    ):
    """
    Panels of heatmaps over the two-tenet space, one panel per magnitude.
 
    colour_scale:
      "percentile" (default) -- clip the range at clip_percentile of |Z| so a
                                single extreme cell cannot flatten the rest of
                                the map to one colour.  Cells beyond the clip
                                are drawn at the end colour and the colourbar
                                is marked as clipped.
      "symlog"                -- symmetric log scaling, for signed measures
                                spanning several orders of magnitude.
      "linear"                -- raw min/max (the old behaviour).
 
    A fixed [-1, 1] range, or a raw min/max dominated by one outlier, hides the
    genuine gradient across the rest of the space.
    """
    import matplotlib.colors as mcolors
 
    n = len(surfaces)
    allZ = np.concatenate([np.asarray(s["Z"]).ravel() for s in surfaces])
 
    diverging = (scale == "diverging")
    cmap = "RdBu_r" if diverging else "viridis"
 
    lo, hi = float(np.nanmin(allZ)), float(np.nanmax(allZ))
    clipped = False
    norm = None
 
 
    if colour_scale == "symlog":
        m = float(np.nanmax(np.abs(allZ))) or 1.0
        small = np.abs(allZ)[np.abs(allZ) > 0]
        lin = float(np.percentile(small, 25)) if small.size else m / 1e3
        norm = mcolors.SymLogNorm(linthresh = max(lin, m / 1e6), vmin = -m if diverging else lo, vmax = m)
 
    elif colour_scale == "percentile":
 
        if diverging:
            m = float(np.percentile(np.abs(allZ), clip_percentile)) or 1.0
            vmin, vmax = -m, m
 
        else:
            vmin = float(np.percentile(allZ, 100.0 - clip_percentile))
            vmax = float(np.percentile(allZ, clip_percentile))
 
 
            if vmax == vmin:
                vmax = vmin + 1e-12
 
 
        clipped = (lo < vmin) or (hi > vmax)
 
    else:
        if diverging:
            m = max(abs(lo), abs(hi)) or 1.0
            vmin, vmax = -m, m
 
        else:
            vmin, vmax = lo, (hi if hi != lo else lo + 1e-12)
 
 
    fig, axes = plt.subplots(1, n, figsize = (3.9 * n, 4.2), squeeze = False)
 
 
    for ax, s in zip(axes[0], surfaces):
        a1, a2 = s["a1_grid"], s["a2_grid"]
 
        kw = dict(origin = "lower", cmap = cmap,
                  extent = [a1[0], a1[-1], a2[0], a2[-1]], aspect = "auto")
 
 
        if norm is not None:
            im = ax.imshow(s["Z"], norm = norm, **kw)
 
        else:
            im = ax.imshow(s["Z"], vmin = vmin, vmax = vmax, **kw)
 
 
        ax.axhline(0, color = "k", lw = 0.5, alpha = 0.4)
        ax.axvline(0, color = "k", lw = 0.5, alpha = 0.4)
 
        ax.set_title(f"magnitude $Y$ = {s['Y']:g}", fontsize = 10)
        ax.set_xlabel("Tenet 1:  cos(B, $SS^T$)")
 
 
        if ax is axes[0][0]:
            ax.set_ylabel("Tenet 2:  cos(G, S)")
 
        else:
            ax.set_yticklabels([])
 
 
    cbar_label = label_measure
 
 
    if clipped:
        cbar_label = f"{label_measure}\n(colour clipped at {clip_percentile:g}th pct; full range {lo:+.3g} to {hi:+.3g})"
 
    elif colour_scale == "symlog":
        cbar_label = f"{label_measure}  (symlog)"
 
 
    fig.colorbar(im, ax = axes[0].tolist(), fraction = 0.025, pad = 0.02,
                 label = cbar_label)
 
    title = f"{label_measure} over the two-tenet space"
 
 
    if subtitle:
        title = f"{title}\n{subtitle}"
 
 
    fig.suptitle(title, fontsize = 12, y = 1.06)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
 
 
 
 
def _fmt(v, sig: int = 4, lo: float = 1e-6, hi: float = 1e7):
    """
    Table-cell formatting that keeps `sig` SIGNIFICANT digits.
 
    The previous rule used "%.2e" outside a narrow band, which keeps only three
    significant digits, so 1.234e-5 printed as "1.23e-05" and 12345.7 as
    "1.23e+04".  Here anything in [lo, hi) is written in plain decimal with
    however many places are needed for `sig` significant digits (trailing
    zeros trimmed), and only genuinely extreme values fall back to
    scientific notation.
    """
 
    if isinstance(v, str):
 
        return v
 
 
    v = float(v)
 
 
    if v == 0.0 or not np.isfinite(v):
 
        return "0" if v == 0.0 else str(v)
 
 
    a = abs(v)
 
 
    if lo <= a < hi:
        # decimals needed so that `sig` significant digits survive
        import math
        decimals = max(0, sig - 1 - int(math.floor(math.log10(a))))
        text = f"{v:.{decimals}f}"
 
 
        if "." in text:
            text = text.rstrip("0").rstrip(".")
 
 
        return text or "0"
 
 
    return f"{v:.{max(sig - 1, 0)}e}"
 
 
 
 
def create_search_table(column_tites: np.ndarray, row_results: np.ndarray, save_loc: str, title: str):
 
    n_rows = len(row_results)
    n_cols = len(column_tites)
 
    # Wide enough for the headers, tall enough for every row.
    fig, ax = plt.subplots(figsize = (2.05 * n_cols, 0.62 * n_rows + 1.6))
 
    tbl = ax.table(
        cellText = [[_fmt(v) for v in r] for r in row_results],
        colLabels = column_tites,
        loc = "center",
        cellLoc = "center"
    )
 
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.3)
 
 
    # Grey backdrop on the header and on the two repeated key columns
    # (magnitude and energy gate), which label the block structure rather
    # than carrying results.
    KEY_COLS = (0, 1)
    HEADER_BG = "#d0d0d0"
    KEY_BG = "#ececec"
 
    for (row, col), cell in tbl.get_celld().items():
 
        cell.set_linewidth(0.6)
        cell.PAD = 0.32                      # more breathing room inside a cell
 
        # Slight bold everywhere; the header and key columns stay full bold.
        cell.set_text_props(weight = "semibold")
 
 
        if row == 0:
            cell.set_text_props(weight = "bold", wrap = True)
            cell.set_facecolor(HEADER_BG)
            cell.set_height(cell.get_height() * 2.6)
 
        elif col in KEY_COLS:
            cell.set_facecolor(KEY_BG) 
            cell.set_text_props(weight = "bold")
 
 
    tbl.auto_set_column_width(col = list(range(n_cols)))
 
    # auto_set_column_width packs the columns tight; widen them for legibility.
    COL_PAD = 1.55
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(cell.get_width() * COL_PAD)
 
 
    ax.set_title(title, pad = 8, fontweight = "bold", fontsize = 16) 
 
    ax.axis("off")
    fig.subplots_adjust(top = 0.9)
 
    # Underline the magnitude entries.  matplotlib text has no underline
    # attribute and mathtext has no underline command, so use the Unicode
    # combining low line: reliable, and needs no cell geometry (which is not
    # settled until after layout and shifts when the figure is re-laid out).
    for row in range(1, n_rows + 1):
        cell = tbl[row, 0]
        label = cell.get_text().get_text()
 
        if label and "\u0332" not in label:
            cell.get_text().set_text("".join(ch + "\u0332" for ch in label))
 
 
    fig.savefig(save_loc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)