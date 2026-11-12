"""Figures for analyze.py. One PNG per property."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# One fixed colour per group, in every figure.
COLOR = {"reference": "#2a78d6", "predicted": "#eb6834", "native": "#1baf7a"}
LABEL = {"reference": "mdCATH reference", "predicted": "Predicted sequence",
         "native": "Native sequence (control)"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Blue - neutral grey - red for signed quantities; light to dark blue for magnitudes.
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#2a78d6", "#f0efec",
                                                      "#e34948", "#a32a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
})


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def rmsf(out, feats):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for g, group in feats.items():
        vals = np.stack([f.rmsf for f in group])
        x = np.arange(1, vals.shape[1] + 1)
        ax.plot(x, vals.mean(axis=0), color=COLOR[g], label=LABEL[g])
        if len(group) > 1:
            ax.fill_between(x, vals.min(axis=0), vals.max(axis=0), color=COLOR[g], alpha=0.18,
                            linewidth=0)
    ax.set(title="CA fluctuation per residue", xlabel="Residue", ylabel="RMSF (Å)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Line: mean over replicas. Band: replica range.", title_fontsize=7,
              alignment="left")
    _save(fig, out / "rmsf.png")


def rmsd(out, series, frame_ns):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for g, runs in series.items():
        for i, r in enumerate(runs):
            ax.plot(np.arange(len(r)) * frame_ns, r, color=COLOR[g], linewidth=1.4,
                    alpha=0.9, label=LABEL[g] if i == 0 else None)
    ax.set(title="Distance from the first reference frame", xlabel="Time in window (ns)",
           ylabel="CA RMSD (Å)")
    ax.set_ylim(bottom=0)
    ax.legend(title="One line per trajectory", title_fontsize=7, alignment="left")
    _save(fig, out / "rmsd.png")


def rama_jsd(out, arrays):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for name, g in (("predicted_vs_reference", "predicted"), ("native_vs_reference", "native")):
        key = f"{name}__rama_jsd"
        if key in arrays:
            y = arrays[key]
            ax.plot(np.arange(1, len(y) + 1), y, color=COLOR[g], label=f"{LABEL[g]} vs reference")
    ax.set(title="Backbone dihedral (φ, ψ) distributions: divergence from the reference",
           xlabel="Residue", ylabel="Jensen–Shannon divergence")
    ax.set_ylim(0, 1)
    ax.legend()
    _save(fig, out / "rama_jsd.png")


def _maps(out, name, panels, title, vmax, cbar_label):
    """Row of square residue-by-residue maps that share one diverging colour scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 0.8, 3.9),
                             squeeze=False)
    for ax, (label, m) in zip(axes[0], panels):
        n = m.shape[0]
        im = ax.imshow(m, cmap=DIVERGING, vmin=-vmax, vmax=vmax, origin="lower",
                       extent=(0.5, n + 0.5, 0.5, n + 0.5), interpolation="nearest")
        ax.set_title(label, fontsize=9, fontweight="normal")
        ax.set_xlabel("Residue")
        ax.grid(False)
    axes[0][0].set_ylabel("Residue")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, color=MUTED)
    cbar.outline.set_visible(False)
    _save(fig, out / name)


def dccm(out, arrays):
    panels = [(LABEL[g], arrays[f"dccm_{g}"]) for g in ("reference", "predicted", "native")
              if f"dccm_{g}" in arrays]
    _maps(out, "dccm.png", panels, "Dynamic cross-correlation of CA motions", 1.0,
          "Correlation")
    diffs = [(f"{LABEL[g]} − reference", arrays[f"{g}_vs_reference__dccm_diff"])
             for g in ("predicted", "native") if f"{g}_vs_reference__dccm_diff" in arrays]
    if diffs:
        _maps(out, "dccm_diff.png", diffs, "Cross-correlation: difference from the reference",
              1.0, "Correlation difference")


def contacts(out, arrays):
    diffs = [(f"{LABEL[g]} − reference", arrays[f"{g}_vs_reference__contact_freq_diff"])
             for g in ("predicted", "native")
             if f"{g}_vs_reference__contact_freq_diff" in arrays]
    if diffs:
        _maps(out, "contacts.png", diffs,
              "Contact frequency (CA within 8 Å): difference from the reference", 1.0,
              "Frequency difference")


def states(out, feats, projections, group_stats, k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    groups = list(feats)
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        occ = np.array(group_stats[g]["occupancy"])
        ax1.bar(np.arange(k) + (i - (len(groups) - 1) / 2) * width, occ, width * 0.88,
                color=COLOR[g], label=LABEL[g])
    ax1.set(title="State occupancy", xlabel="State (largest first)",
            ylabel="Fraction of frames")
    ax1.set_xticks(np.arange(k), [str(s + 1) for s in range(k)])
    ax1.grid(axis="x", visible=False)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right")

    it = iter(projections)
    for g in groups:
        pts = np.concatenate([next(it) for _ in feats[g]])
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        ax2.scatter(pts[:, 0], y, s=14, color=COLOR[g], edgecolors=SURFACE, linewidths=0.4,
                    alpha=0.85, label=LABEL[g])
    ax2.set(title="Frames in the shared slow coordinates", xlabel="Coordinate 1",
            ylabel="Coordinate 2")
    _save(fig, out / "states.png")


def make_all(out, feats, rmsd_series, arrays, projections, group_stats, frame_ns, k):
    rmsf(out, feats)
    rmsd(out, rmsd_series, frame_ns)
    rama_jsd(out, arrays)
    dccm(out, arrays)
    contacts(out, arrays)
    states(out, feats, projections, group_stats, k)
