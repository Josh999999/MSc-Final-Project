"""External Imports (Libraries and APIs)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
 
 
 
 
def plot_fitness_surface(
        surfaces: list,
        saveloc: str = "tenet_surface.png",
        label_measure: str = "Fitness",
        scale: str = "sequential",
        vmin: float = None,
        vmax: float = None,
        n_seeds: int = None
    ):
    """
    Panels of one measure over the two-tenet space, one panel per magnitude.

    The colour range is taken from the DATA unless vmin/vmax are given.  It
    used to be hard-coded to [0, 1], which silently clipped every signed
    measure (energy and diff_energy run negative) to the bottom colour.
    """
    n = len(surfaces)
    allZ = np.concatenate([np.asarray(s["Z"]).ravel() for s in surfaces])

    diverging = (scale == "diverging")
    cmap = "coolwarm" if diverging else "viridis"

    lo, hi = float(np.nanmin(allZ)), float(np.nanmax(allZ))


    if vmin is None or vmax is None:

        if diverging:
            m = max(abs(lo), abs(hi)) or 1.0
            auto_min, auto_max = -m, m

        else:
            auto_min, auto_max = lo, (hi if hi != lo else lo + 1e-12)

        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax


    fig, axes = plt.subplots(1, n, figsize = (3.6 * n, 3.9), squeeze = False)


    for ax, s in zip(axes[0], surfaces):
        a1, a2 = s["a1_grid"], s["a2_grid"]

        im = ax.imshow(
            s["Z"], origin = "lower", cmap = cmap, vmin = vmin, vmax = vmax,
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


    fig.colorbar(im, ax = axes[0].tolist(), fraction = 0.025, pad = 0.02,
                 label = label_measure)

    # n is the number of PANELS (one per magnitude), not the seed count, so the
    # seed count has to be passed in if it is to be reported.
    title = f"{label_measure} over the two-tenet space"


    if n_seeds is not None:
        title = f"{title}  (seeds per cell: {n_seeds})"


    fig.suptitle(title, fontsize = 11)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)




def plot_measure_surfaces(
        surfaces: list,
        saveloc: str = "tenet_measure.png",
        label_measure: str = "measure",
        scale: str = "sequential",
        subtitle: str = None,
        colour_scale: str = "rank",
        clip_percentile: float = 98.0,
        cmap: str = None,
        n_levels: int = None
    ):
    
    n = len(surfaces)
    allZ = np.concatenate([np.asarray(s["Z"]).ravel() for s in surfaces])
    finite = allZ[np.isfinite(allZ)]
 
    diverging = (scale == "diverging")
 
    # viridis varies mainly in LIGHTNESS, so small differences are hard to see.
    # turbo sweeps through many hues over the same range, which makes the same
    # value difference far more visible.  Pass cmap= to override.
    if cmap is None:
        cmap = "coolwarm" if diverging else "turbo"
 
    lo, hi = float(np.nanmin(allZ)), float(np.nanmax(allZ))
    norm = None
    vmin, vmax = lo, (hi if hi != lo else lo + 1e-12)
    note = ""
 
 
    if colour_scale == "rank":
        # Map each value to its quantile among all cells.  Equal numbers of
        # cells per colour step, so dense regions of the distribution are
        # spread out and sparse ones compressed.
        qs = np.linspace(0.0, 1.0, 256)
        levels = np.unique(np.quantile(finite, qs)) if finite.size else np.array([0.0, 1.0])
 
 
        if levels.size < 2:
            levels = np.array([float(levels[0]), float(levels[0]) + 1e-12])
 
 
        norm = mcolors.BoundaryNorm(levels, ncolors = 256, clip = True)
        note = "rank / histogram-equalised colour"
 
    elif colour_scale == "symlog":
        m = float(np.nanmax(np.abs(allZ))) or 1.0
        small = np.abs(finite)[np.abs(finite) > 0]
        lin = float(np.percentile(small, 25)) if small.size else m / 1e3
        norm = mcolors.SymLogNorm(linthresh = max(lin, m / 1e6),
                                  vmin = -m if diverging else lo, vmax = m)
        note = "symlog colour"
 
    elif colour_scale == "percentile":
 
        if diverging:
            m = float(np.percentile(np.abs(finite), clip_percentile)) or 1.0
            vmin, vmax = -m, m
 
        else:
            vmin = float(np.percentile(finite, 100.0 - clip_percentile))
            vmax = float(np.percentile(finite, clip_percentile))
 
 
        if vmax == vmin:
            vmax = vmin + 1e-12
 
 
        if lo < vmin or hi > vmax:
            note = f"clipped at {clip_percentile:g}th pct; full range {lo:+.3g} to {hi:+.3g}"
 
    else:
 
        if diverging:
            m = max(abs(lo), abs(hi)) or 1.0
            vmin, vmax = -m, m
 
 
    # Optional: quantise into n_levels discrete bands so steps are countable.
    if n_levels:
        base_cmap = plt.get_cmap(cmap)
        cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, n_levels)))
 
 
        if norm is None:
            norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, n_levels + 1),
                                        ncolors = n_levels)
 
 
    fig, axes = plt.subplots(1, n, figsize = (3.9 * n, 4.2), squeeze = False)
 
 
    for ax, s in zip(axes[0], surfaces):
        a1, a2 = s["a1_grid"], s["a2_grid"]
 
        kw = dict(origin = "lower", cmap = cmap,
                  extent = [a1[0], a1[-1], a2[0], a2[-1]], aspect = "auto")
 
        im = ax.imshow(s["Z"], norm = norm, **kw) if norm is not None \
             else ax.imshow(s["Z"], vmin = vmin, vmax = vmax, **kw)
 
        ax.axhline(0, color = "k", lw = 0.5, alpha = 0.4)
        ax.axvline(0, color = "k", lw = 0.5, alpha = 0.4)
        ax.set_title(f"magnitude $Y$ = {s['Y']:g}", fontsize = 10)
        ax.set_xlabel("Tenet 1:  cos(B, $SS^T$)")
 
 
        if ax is axes[0][0]:
            ax.set_ylabel("Tenet 2:  cos(G, S)")
 
        else:
            ax.set_yticklabels([])
 
 
    cbar_label = f"{label_measure}\n({note})" if note else label_measure
    fig.colorbar(im, ax = axes[0].tolist(), fraction = 0.025, pad = 0.02,
                 label = cbar_label)
 
    title = f"{label_measure} over the two-tenet space"
 
 
    if subtitle:
        title = f"{title}\n{subtitle}"
 
 
    fig.suptitle(title, fontsize = 12, y = 1.06)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
 
 
 
 
def _fmt(v, sig: int = 6, lo: float = 1e-9, hi: float = 1e9):
    """
    Table-cell formatting.  `sig` SIGNIFICANT digits are kept (not decimal
    places), and the plain-decimal band is wide so small values are not forced
    into exponent form.  Display rounding only -- the stored values are full
    float64 either way.
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
 
 
 
 
def create_search_table(column_tites: np.ndarray, row_results: np.ndarray, save_loc: str, title: str, sig: int = 6): 
    n_rows = len(row_results)
    n_cols = len(column_tites)
 
    # Wide enough for the headers, tall enough for every row.
    fig, ax = plt.subplots(figsize = (2.05 * n_cols, 0.62 * n_rows + 1.6))
 
    tbl = ax.table(
        cellText = [[_fmt(v, sig = sig) for v in r] for r in row_results],
        colLabels = column_tites,
        loc = "center",
        cellLoc = "center"
    )
 
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.3)
 
 
    # Grey backdrop on the header and on the two repeated key columns (magnitude and energy gate), which label the block structure rather than carrying results.
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
 
    # Underline the magnitude entries
    for row in range(1, n_rows + 1):
        cell = tbl[row, 0]
        label = cell.get_text().get_text()
 
        if label and "\u0332" not in label:
            cell.get_text().set_text("".join(ch + "\u0332" for ch in label))
 
 
    fig.savefig(save_loc, dpi = 150, bbox_inches = "tight")
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
