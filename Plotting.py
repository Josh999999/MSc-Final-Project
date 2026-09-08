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
        subtitle: str = None
    ):
    """
    Panels of heatmaps over the two-tenet space, one panel per magnitude.
 
    The colour range is taken from the DATA, shared across the panels so they
    stay comparable.  A fixed [-1, 1] range would flatten measures such as
    align_change (order 1e-3) into a single colour.
    """
    n = len(surfaces)
 
    lo = min(float(np.nanmin(s["Z"])) for s in surfaces)
    hi = max(float(np.nanmax(s["Z"])) for s in surfaces)
 
 
    if scale == "diverging":
        # Centre a signed quantity on zero so the sign is readable.
        m = max(abs(lo), abs(hi)) or 1.0
        vmin, vmax, cmap = -m, m, "RdBu_r"
 
    else:
        if hi == lo:
            hi = lo + 1e-12
 
        vmin, vmax, cmap = lo, hi, "viridis"
 
 
    fig, axes = plt.subplots(1, n, figsize = (3.9 * n, 4.2), squeeze = False)
 
 
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
 
    title = f"{label_measure} over the two-tenet space"
 
    if subtitle:
        title = f"{title}\n{subtitle}"
 
 
    fig.suptitle(title, fontsize = 12, y = 1.06)
    fig.savefig(saveloc, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
 