"""External Imports (Libraries and APIs)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from GRN import develop
from Config import Config
 
 
 
 
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
 
    if cmap is None:
        cmap = "coolwarm" if diverging else "turbo"
 
    lo, hi = float(np.nanmin(allZ)), float(np.nanmax(allZ))
    norm = None
    vmin, vmax = lo, (hi if hi != lo else lo + 1e-12)
    note = ""
 
 
    if colour_scale == "rank":
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
 
 
 
 
def plot_switches(
        ax_strip: plt.Axes,
        x_switches: np.ndarray = None
    ):
    x_switches = np.asarray(x_switches)
 
    if x_switches.dtype == bool:
        x_switches = np.flatnonzero(x_switches)
 
 
    # The strip: one rectangle per switch generation.
    for x_s in x_switches:
        ax_strip.add_patch(plt.Rectangle((x_s, 0), 1, 1, facecolor = "#cc4444", edgecolor = "none"))
 
 
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("generation")
    ax_strip.set_ylabel("switch", rotation = 0, ha = "right", va = "center", fontsize = 8)
    
 
 
 
def plot_trajectories(
        trajectories: np.ndarray, 
        keys: np.ndarray = None, 
        x: np.ndarray = None,
        linestyle: np.ndarray = None, 
        marker: str = "none", 
        x_label = "x",
        y_label = "y",
        x_scale: str = "linear",
        y_scale: str = "linear",
        x_limits: tuple = None,
        y_limits: tuple = None,
        grid: bool = False,
        title: str = None,        
        subtitle: str = None,
        saveloc: str = "./", 
        marker_size: int = 1,
        alpha: float = 1.0,
        moving_average: bool = False,
        smooth: int = 0,
        band: np.ndarray = None,
        bands: bool = False,
        band_alpha: float = 0.18,         
        dpi: int = 150,
        figsize: tuple = (16, 10), # (6, 4) For interaction trajectories
        switch_plot: bool = False,
        switch_lines: bool = False,
        x_switches: np.ndarray = None
    ):
    fig = None
    ax = None
    ax_strip = None
 
    if switch_plot:
        fig, (ax, ax_strip) = plt.subplots(
                2, 1, figsize = (9.0, 4.6), sharex = True,
                gridspec_kw = {"height_ratios": [4, 1], "hspace": 0.08}
            )
        
    else:
        fig, ax = plt.subplots(figsize = figsize)
 
 
    n = len(trajectories)
 
    keys = [None] * n if not keys else keys
    linestyle = ["solid"] * n if not linestyle else linestyle
    marker = "none" if not marker else marker
 
 
    for trajectory, key, ls in zip(trajectories, keys, linestyle):
 
        trajectory_plot = trajectory
 
 
        if moving_average and not (smooth is None or smooth < 2 or trajectory_plot.size < smooth):
    
                # centred, edge-padded so the smoothed line keeps its length
                pad = smooth // 2
                padded = np.pad(trajectory_plot, (pad, smooth - 1 - pad), mode = "edge")
        
                trajectory_plot = np.convolve(padded, np.ones(smooth) / smooth, mode = "valid")
 
 
        # `x` lets a SUBSAMPLED series be plotted against the generations it was
        # actually recorded at (recorded_gens), instead of 0..len-1.  Smoothing
        # does not change the length, so the same x applies either way.
        if x is None:
            xs = np.arange(trajectory_plot.size)
 
        else:
            xs = np.asarray(x, dtype = float)
 
            if xs.size != trajectory_plot.size:
                raise ValueError(
                    f"x has {xs.size} points but the series has {trajectory_plot.size}; "
                    "a subsampled series must be plotted against recorded_gens")
 
        ax.plot(
            xs,
            trajectory_plot, 
            label = key, 
            lw = 1 + 0.1 * np.log(n), 
            linestyle = ls, 
            marker = marker,
            markersize = marker_size,
            alpha = alpha
        )
 
 
        if bands and band is not None:
 
            if isinstance(band, (tuple, list)) and len(band) == 2 and np.ndim(band[0]) > 0:
                lower, upper = np.asarray(band[0], float), np.asarray(band[1], float)
 
            else:
                half = np.asarray(band, dtype = float)
                lower, upper = trajectory - half, trajectory + half
 
 
            ax.fill_between(xs, lower, upper, alpha = band_alpha, linewidth = 0)
 
 
 
 
    # Place the Vertical lines for the switch points
    if switch_lines and x_switches is not None and len(x_switches):
        x_lines = np.asarray(x_switches)
 
        if x_lines.dtype == bool:
            x_lines = np.flatnonzero(x_lines)
 
 
        for x_s in x_lines:
            ax.axvline(x_s, color = "k", ls = "--", lw = 0.7, alpha = 0.45)
 
 
    if switch_plot:
        plot_switches(ax_strip, x_switches)
 
 
 
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
        dpi: int = 150,
        switch_plot: bool = False,
        x_switches: np.ndarray = None
    ):
    fig = None
    ax = None
    ax_strip = None
 
    if switch_plot:
        fig, (ax, ax_strip) = plt.subplots(
                2, 1, figsize = (9.0, 4.6), sharex = True,
                gridspec_kw = {"height_ratios": [4, 1], "hspace": 0.08}
            )
        
    else:
        fig, ax = plt.subplots(figsize = figsize)
 
 
    v = np.asarray(values, dtype = bool)
    n = v.size
    edges = np.arange(n + 1, dtype = float)
 
 
    change = np.flatnonzero(np.diff(v)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
 
 
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
 
 
    if switch_plot:
        plot_switches(ax_strip, x_switches)
 
 
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
 
 
 
 
def plot_transfer(
        transfer_fitness: np.ndarray, 
        fitnesses: np.ndarray, 
        transfer_loss: np.ndarray, 
        switches: np.ndarray, 
        saveloc: str,
        x: np.ndarray = None
    ):
    switch_gens = np.flatnonzero(np.asarray(switches, dtype = bool))
 
    fig, (ax_fit, ax_loss) = plt.subplots(
        2, 1, figsize = (9.0, 5.2), sharex = True,
        gridspec_kw = {"height_ratios": [3, 2], "hspace": 0.10}
    )
 
    # Subsampled series carry their own x (recorded_gens); without it they would
    # be plotted at 0..len-1 and compressed against the left of the axis.
    xs = np.arange(np.size(fitnesses)) if x is None else np.asarray(x, dtype = float)
 
    ax_fit.plot(xs, fitnesses, lw = 1.3, color = "#2b7bba", label = "Selected Fitness")
    ax_fit.plot(xs, transfer_fitness, "o", ms = 7, color = "#cc4444", label = "incumbent on NEW target (switch only)")
 
 
    for g in switch_gens:
        ax_fit.axvline(g, color = "k", ls = "--", lw = 0.7, alpha = 0.4)
 
 
    ax_fit.set_ylabel("Fitness")
    ax_fit.legend(frameon = False, fontsize = 8)
    ax_fit.grid(alpha = 0.25)
 
    ax_loss.plot(xs, transfer_loss, "o-", ms = 6, color = "#cc4444")
    ax_loss.axhline(0, color = "k", lw = 0.8, alpha = 0.5)
 
 
    for g in switch_gens:
        ax_loss.axvline(g, color = "k", ls = "--", lw = 0.7, alpha = 0.4)
 
 
    ax_loss.set_ylabel("Fitness lost\nat the Switch")
    ax_loss.set_xlabel("Generation")
    ax_loss.grid(alpha = 0.25)
 
    fig.suptitle("Transfer to a new Target, recorded in-line with NaN padding", fontsize = 11)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
 
 
 
 
def show_interaction_heatmap(B: np.array, saveloc: str, title = None):
    fig, ax = plt.subplots(figsize=(5, 4))
 
    vmax = np.max(np.abs(B)) or 1.0
    im = ax.imshow(B, cmap="bone", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Gene i")
    ax.set_ylabel("Gene j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)
 
 
 
 
def random_profiles(R: int, N: int, rng: np.random.Generator) -> np.ndarray:
    if N is None: 
        N = Global.N
 
        
    G = rng.uniform(low = -1.0, high = 1.0, size = (R, N))
 
    return G
 
 
 
def show_phenotypes(
        B: np.array, 
        R: int,      
        cfg: Config,
        rng: np.random.Generator,
        saveloc: str, 
        title = None,   
    ):
    phenotypes = []
    initial_profiles = random_profiles(R = R, N = cfg.N, rng = rng)
 
    for G in initial_profiles:
        P = develop(G, B, cfg, T = cfg.T)
        phenotypes.append(P)
 
 
    fig, ax = plt.subplots(figsize=(5, 5))
 
    vmax = np.max(np.abs(phenotypes)) or 1.0
    im = ax.imshow(phenotypes, cmap="bone", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Phenotype samples")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)
 
 
 
def plot_matrices(
        matrices: list,
        keys: list = None,
        saveloc: str = "matrices.png",
        title: str = None,
        subtitle: str = None,
        cmap: str = "RdBu_r",
        symmetric: bool = True,
        colour_label: str = "weight",
        reference: np.ndarray = None,
        reference_key: str = "reference",
        figsize_per: tuple = (3.2, 3.4),
        dpi: int = 150
    ):
    mats = [np.asarray(m, dtype = float) for m in matrices]
    n = len(mats)
    keys = keys if keys is not None else [f"{i}" for i in range(n)]
 
    allv = np.concatenate([m.ravel() for m in mats])
 
 
    if symmetric:
        extent = float(np.nanmax(np.abs(allv))) or 1.0
        vmin, vmax = -extent, extent
 
    else:
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
 
 
    n_panels = n + (1 if reference is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize = (figsize_per[0] * n_panels, figsize_per[1]), squeeze = False)
    panels = list(axes[0])
 
 
    if reference is not None:
        ref = np.asarray(reference, dtype = float)
        r = float(np.nanmax(np.abs(ref))) or 1.0
        ax0 = panels.pop(0)
        im_ref = ax0.imshow(ref, cmap = cmap, vmin = -r, vmax = r, interpolation = "nearest")
        ax0.set_title(str(reference_key), fontsize = 9)
        ax0.set_xticks([]); ax0.set_yticks([])
        fig.colorbar(im_ref, ax = ax0, fraction = 0.046, pad = 0.04)
 
 
    for ax, m, key in zip(panels, mats, keys):
        im = ax.imshow(m, cmap = cmap, vmin = vmin, vmax = vmax, interpolation = "nearest")
        ax.set_title(str(key), fontsize = 9)
        ax.set_xticks([]); ax.set_yticks([])
 
 
    fig.colorbar(im, ax = panels, fraction = 0.025, pad = 0.02, label = colour_label)
 
    full = title or ""
 
 
    if subtitle:
        full = f"{full}\n{subtitle}" if full else subtitle
 
 
    if full:
        fig.suptitle(full, fontsize = 11)
 
 
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)
 
 
 
def plot_round_curves(
        round_curves: list,
        keys: list = None,
        saveloc: str = "round_curves.png",
        separate: str = None,
        title: str = None,
        subtitle: str = None,
        x_label: str = "step",
        y_label: str = "fitness",
        cmap: str = "viridis",
        figsize_per: tuple = (3.6, 3.6),
        share_y: bool = True,
        dpi: int = 150
    ):
    colours = plt.get_cmap(cmap)
 
    if separate == "grid":
        # rows = runs, columns = rounds
        n_runs = len(round_curves)
        n_rounds = max(len(c) for c in round_curves)
        keys = keys if keys is not None else [f"{i}" for i in range(n_runs)]
 
        fig, axes = plt.subplots(n_runs, n_rounds, squeeze = False, sharey = share_y, sharex = True,
                                 figsize = (figsize_per[0] * n_rounds, figsize_per[1] * n_runs))
 
 
        for row, (curves, key) in enumerate(zip(round_curves, keys)):
 
            for col in range(n_rounds):
                ax = axes[row][col]
 
 
                if col < len(curves):
                    c = np.asarray(curves[col], dtype = float)
                    ax.plot(np.arange(c.size), c, lw = 1.4,
                            color = colours(0.15 + 0.75 * col / max(1, n_rounds - 1)))
 
                else:
                    ax.axis("off"); continue
 
 
                ax.grid(alpha = 0.25, linewidth = 0.6)
 
 
                if row == 0:
                    ax.set_title(f"round {col + 1}", fontsize = 10)
 
 
                if col == 0:
                    ax.set_ylabel(f"{key}\n{y_label}", fontsize = 9)
 
 
                if row == n_runs - 1:
                    ax.set_xlabel(x_label)
 
 
        full = title or ""
 
 
        if subtitle:
            full = f"{full}\n{subtitle}" if full else subtitle
 
 
        if full:
            fig.suptitle(full, fontsize = 12)
 
 
        fig.tight_layout()
        fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
        plt.close(fig)
 
 
        return
 
 
    if separate in ("run", "curve"):
        stem, ext = os.path.splitext(saveloc)
        ext = ext or ".png"
        keys = keys if keys is not None else [f"{i}" for i in range(len(round_curves))]
 
 
        for curves, key in zip(round_curves, keys):
            tag = str(key).replace(" ", "")
 
 
            if separate == "run":
                # each ROUND becomes its own panel, so one seed gives a row of
                # R plots read left to right
                plot_round_curves([[c] for c in curves],
                                  [f"round {i + 1}" for i in range(len(curves))],
                                  f"{stem}_{tag}{ext}", None,
                                  title, f"{subtitle} \u2014 {key}" if subtitle else str(key),
                                  x_label, y_label, cmap, figsize_per, share_y, dpi)
 
            else:
 
                for i, c in enumerate(curves):
                    plot_round_curves([[c]], [f"{key}, round {i + 1}"],
                                      f"{stem}_{tag}_round{i + 1}{ext}", None,
                                      title, subtitle, x_label, y_label, cmap,
                                      figsize_per, share_y, dpi)
 
 
        return
 
    n = len(round_curves)
    keys = keys if keys is not None else [f"{i}" for i in range(n)]
 
    fig, axes = plt.subplots(1, n, figsize = (figsize_per[0] * n, figsize_per[1]),
                             squeeze = False, sharey = share_y)
 
 
    for ax, curves, key in zip(axes[0], round_curves, keys):
        r = max(1, len(curves))
 
        for i, c in enumerate(curves):
            c = np.asarray(c, dtype = float)
            ax.plot(np.arange(c.size), c, lw = 1.4,
                    color = colours(0.15 + 0.75 * i / max(1, r - 1)),
                    label = f"round {i + 1}")
 
 
        ax.set_title(str(key), fontsize = 10)
        ax.set_xlabel(x_label)
        ax.grid(alpha = 0.25, linewidth = 0.6)
 
 
        if ax is axes[0][0]:
            ax.set_ylabel(y_label)
 
            if r > 1:                       # a one-curve panel needs no legend
                ax.legend(frameon = False, fontsize = 8)
 
 
    full = title or ""
 
 
    if subtitle:
        full = f"{full}\n{subtitle}" if full else subtitle
 
 
    if full:
        fig.suptitle(full, fontsize = 11)
 
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)
 
 
 
def plot_counts(
        labels: list,
        counts: list,
        total: int = None,
        saveloc: str = "counts.png",
        title: str = None,
        subtitle: str = None,
        y_label: str = "count",
        colour: str = "#2b7bba",
        figsize: tuple = (10.0, 4.6),
        dpi: int = 150
    ):
    counts = np.asarray(counts, dtype = float)
    fig, ax = plt.subplots(figsize = figsize)
 
    bars = ax.bar(range(len(labels)), counts, color = colour)
 
 
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + (total or counts.max()) * 0.02,
                f"{int(c)}", ha = "center", va = "bottom", fontsize = 9)
 
 
    if total is not None:
        ax.set_ylim(0, total * 1.15)
        ax.axhline(total / 2, color = "k", ls = "--", lw = 0.8, alpha = 0.4)
        ax.set_yticks(range(0, total + 1))
 
 
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation = 30, ha = "right", fontsize = 9)
    ax.set_ylabel(y_label)
    ax.grid(axis = "y", alpha = 0.25, linewidth = 0.6)
 
    full = title or ""
 
 
    if subtitle:
        full = f"{full}\n{subtitle}" if full else subtitle
 
 
    if full:
        ax.set_title(full, fontsize = 11)
 
 
    fig.tight_layout()
    fig.savefig(saveloc, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)