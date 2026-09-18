"""External Imports (Libraries and APIs)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
 
 
 
 
def plot_fitness_surface(
        surfaces: list,
        saveloc: str = "tenet_surface.png",
        label_measure: str = "Fitness",
        scale: str = "sequential",
        vmin: float = None,
        vmax: float = None,
        n_seeds: int = None
    ):

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




def plot_trajectories(
        trajectories: np.ndarray, 
        keys: np.ndarray, 
        linestyle: np.ndarray, 
        marker: str, 
        x_label = "x",
        y_label = "y",
        x_scale: str = "linear",
        y_scale: str = "linear",
        x_limits: tuple = None,
        y_limits: tuple = None,
        grid: bool = True,
        title: str = None,        
        subtitle: str = None,
        saveloc: str = "./", 
        marker_size: int = 1,
        alpha: float = 1.0,
        moving_average: bool = False,
        smooth: int = 0,
        band: np.ndarray = None,
        bands: bool = True,
        band_alpha: float = 0.18,         
        dpi: int = 150,
        figsize: tuple = (16, 10), # (6, 4) For interaction trajectories
    ):
    
    fig, ax = plt.subplots(figsize = figsize)
    n = len(trajectories)

    for trajectory, key, ls in zip(trajectories, keys, linestyle):

        trajectory_plot = trajectory


        if moving_average and not (smooth is None or smooth < 2 or trajectory_plot.size < smooth):
    
                # centred, edge-padded so the smoothed line keeps its length
                pad = smooth // 2
                padded = np.pad(trajectory_plot, (pad, smooth - 1 - pad), mode = "edge")
        
                trajectory_plot = np.convolve(padded, np.ones(smooth) / smooth, mode = "valid")

        
        xs = np.arange(trajectory_plot.size)

        ax.plot(
            xs,
            trajectory_plot, 
            label = key, 
            lw = 1 + 0.1 * np.log(n), 
            linestyle = ls, 
            marker = marker,
            marker_size = marker_size,
            alpha = alpha
        )


        if bands and band is not None:
 
            if isinstance(band, (tuple, list)) and len(band) == 2 and np.ndim(band[0]) > 0:
                lower, upper = np.asarray(band[0], float), np.asarray(band[1], float)
 
            else:
                half = np.asarray(band, dtype = float)
                lower, upper = trajectory - half, trajectory + half
 
 
            ax.fill_between(xs, lower, upper, alpha = band_alpha, linewidth = 0)


    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
 
 
    if x_limits: 
        ax.set_xlim(*x_limits)

    if y_limits: 
        ax.set_ylim(*y_limits)

    if grid:     
        ax.grid(alpha = 0.25, linewidth = 0.6)


    ax.legend(frameon = False, fontsize = 9)


    full_title = title or ""
 
    if subtitle:
        full_title = f"{full_title}\n{subtitle}" if full_title else subtitle
 
 
    if full_title:
        ax.set_title(full_title, fontsize = 11)
 
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)




def plot_binary_strip(
        values,
        saveloc: str = "strip.png",
        true_colour: str = "#2c7fb8",
        false_colour: str = "#f0f0f0",
        true_label: str = "True",
        false_label: str = "False",
        height: float = 1.0,
        y_base: float = 0.0,
        title: str = None,
        subtitle: str = None,
        x_label: str = "",
        strip_label: str = "",
        alpha: float = 1.0,
        edge: bool = False,
        legend: bool = True,
        legend_loc: str = "upper right",
        figsize: tuple = (9.0, 1.9),
        dpi: int = 150
    ):
 
    v = np.asarray(values, dtype = bool)
    n = v.size
    edges = np.arange(n + 1, dtype = float)
 
 
    change = np.flatnonzero(np.diff(v)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
 
    fig, ax = plt.subplots(figsize = figsize)
 
 
    for a, b in zip(starts, ends):
        colour = true_colour if v[a] else false_colour
 
        ax.broken_barh(
            [(edges[a], edges[b] - edges[a])],
            (y_base, height),
            facecolors = colour,
            alpha = alpha,
            edgecolor = "white" if edge else "none",
            linewidth = 0.4 if edge else 0.0
        )
 
 
    ax.set_xlim(edges[0], edges[-1]) 
    ax.set_ylim(y_base, y_base + height)
    ax.set_yticks([])
    ax.set_xlabel(x_label)


    if strip_label:
        ax.set_ylabel(strip_label, rotation = 0, ha = "right", va = "center")


    if legend:
        ax.legend(handles = [Patch(facecolor = true_colour, label = true_label),
                                Patch(facecolor = false_colour, label = false_label)],
                    loc = legend_loc, frameon = False, fontsize = 9, ncols = 2)


    full = title or ""

    if subtitle:
        full = f"{full}\n{subtitle}" if full else subtitle


    if full:
        ax.set_title(full, fontsize = 11)


    fig.tight_layout()
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)

 
 
 
def plot_binary_bar(
        values,
        saveloc: str = "binary_bar.png",
        true_colour: str = "#2b7bba",
        false_colour: str = "#d9d9d9",
        true_label: str = "True",
        false_label: str = "False",
        height: float = 1.0,
        y_base: float = 0.0,
        title: str = None,
        subtitle: str = None,
        x_label: str = "",
        row_labels: list = None,
        figsize: tuple = (9.0, 1.6),
        legend: bool = True,
        edge: bool = False,
        max_patches: int = 2000,
        dpi: int = 150
    ):
    rows = values if isinstance(values, (list, tuple)) and np.ndim(values[0]) > 0 else [values]
    rows = [np.asarray(r).astype(bool).ravel() for r in rows]
    n = max(r.size for r in rows) 
    edges = np.arange(n + 1, dtype = float)
 
 
    fig, ax = plt.subplots(figsize = (figsize[0], figsize[1] * max(1, len(rows))))
 
 
    row_h = height / len(rows) 
 
    strip_cmap = mcolors.ListedColormap([false_colour, true_colour])
 
 
    for r, row in enumerate(rows):
        base = y_base + (len(rows) - 1 - r) * row_h
        changes = np.flatnonzero(np.diff(row)) + 1
 
 
        if changes.size + 1 > max_patches:
            ax.imshow(row.reshape(1, -1), aspect = "auto", cmap = strip_cmap,
                      vmin = 0, vmax = 1, interpolation = "nearest",
                      extent = [edges[0], edges[-1], base, base + row_h],
                      origin = "lower", zorder = 1)
            continue
 
 
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [row.size]))
 
        for s0, s1 in zip(starts, stops):
            ax.add_patch(plt.Rectangle(
                (edges[s0], base), edges[s1] - edges[s0], row_h,
                facecolor = true_colour if row[s0] else false_colour,
                edgecolor = "white" if edge else "none",
                linewidth = 0.3 if edge else 0.0))
 
 
    ax.set_xlim(edges[0], edges[-1]) 
    ax.set_ylim(y_base, y_base + height)

    if row_labels:
        ax.set_yticks([y_base + (len(rows) - 1 - r) * row_h + row_h / 2
                        for r in range(len(rows))])
        ax.set_yticklabels(row_labels, fontsize = 9)

    else:
        ax.set_yticks([])


    ax.set_xlabel(x_label)


    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor = true_colour),
                    plt.Rectangle((0, 0), 1, 1, facecolor = false_colour)]
        ax.legend(handles, [true_label, false_label], loc = "upper right",
                    ncol = 2, frameon = False, fontsize = 9,
                    bbox_to_anchor = (1.0, 1.35))


    full = title or ""

    if subtitle:
        full = f"{full}\n{subtitle}" if full else subtitle


    if full:
        ax.set_title(full, fontsize = 11)


    fig.tight_layout()
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)