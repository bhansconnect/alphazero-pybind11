"""Shared visualization helpers for training-time distribution figures.

Board-renderer and data-loading utilities used by both the Star Gambit
(hexagonal) and tafl (square) visualization drivers in game_runner.py. Kept
free of training-loop state so it can be imported lazily at visualization time.

matplotlib is imported lazily inside the drawing functions so importing this
module never hard-requires a plotting backend.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# History sample loading
# ---------------------------------------------------------------------------


def load_history_samples(paths, iteration, max_samples=32_000, want_v=False):
    """Load a subsample of (canonical, pi[, v]) tensors from one iteration.

    Reads that iteration's history `.ptz` triples, taking an even per-file slice
    up to `max_samples` total. Returns ``(c_data, pi_np)`` (or
    ``(c_data, pi_np, v_np)`` when ``want_v``) where ``c_data`` is a float
    torch.Tensor and the others are float numpy arrays. Returns ``None`` when no
    data is available.
    """
    # Lazy import to avoid a circular dependency (game_runner imports nothing
    # from here at module load; this is only called during visualization).
    from game_runner import glob_file_triples, load_compressed

    hist_location = paths["history"]
    file_triples = glob_file_triples(
        hist_location, f"{iteration:04d}-*-canonical-*.ptz")
    if not file_triples:
        return None

    all_c, all_pi, all_v = [], [], []
    total = 0
    per_file = max(1, max_samples // len(file_triples))
    for c_path, v_path, pi_path, size in file_triples:
        take = min(per_file, size, max_samples - total)
        if take <= 0:
            continue
        c = load_compressed(c_path).float()
        pi = load_compressed(pi_path).float()
        v = load_compressed(v_path).float() if want_v else None
        if take < size:
            idx = torch.randperm(size)[:take]
            all_c.append(c[idx])
            all_pi.append(pi[idx])
            if want_v:
                all_v.append(v[idx])
        else:
            all_c.append(c)
            all_pi.append(pi)
            if want_v:
                all_v.append(v)
        total += take

    if not all_c:
        return None

    c_data = torch.cat(all_c)
    pi_np = torch.cat(all_pi).numpy()
    if want_v:
        v_np = torch.cat(all_v).numpy()
        return c_data, pi_np, v_np
    return c_data, pi_np


# ---------------------------------------------------------------------------
# Phase segmentation
# ---------------------------------------------------------------------------


def quartile_masks(progress, n_bins=4):
    """Split samples into `n_bins` equal-count groups by a progress scalar.

    `progress` is a per-sample 1-D array (e.g. game-progress fraction). Returns
    a list of ``(label, bool_mask)`` ordered earliest→latest. Uses quantile
    edges so each bin has ~equal sample count; if the quantile edges tie (e.g.
    many identical values), falls back to an argsort split so every bin is
    non-empty.
    """
    progress = np.asarray(progress, dtype=np.float64)
    n = progress.shape[0]
    labels = _bin_labels(n_bins)

    if n == 0:
        return [(lab, np.zeros(0, dtype=bool)) for lab in labels]

    edges = np.quantile(progress, np.linspace(0, 1, n_bins + 1))
    # Assign by quantile edges, but verify every bin is populated; ties in the
    # progress distribution can collapse edges and starve a bin.
    masks = []
    ok = True
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            m = (progress >= lo) & (progress <= hi)
        else:
            m = (progress >= lo) & (progress < hi)
        masks.append(m)
        if not m.any():
            ok = False
    if ok:
        return list(zip(labels, masks))

    # Fallback: equal-count split of the sorted order.
    order = np.argsort(progress, kind="stable")
    masks = [np.zeros(n, dtype=bool) for _ in range(n_bins)]
    for b, chunk in enumerate(np.array_split(order, n_bins)):
        masks[b][chunk] = True
    return list(zip(labels, masks))


def _bin_labels(n_bins):
    if n_bins == 4:
        return ["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"]
    return [f"Q{b + 1}" for b in range(n_bins)]


# ---------------------------------------------------------------------------
# Board renderers
# ---------------------------------------------------------------------------


def draw_na(ax, label="N/A"):
    """Render a grey 'N/A' / insufficient-data placeholder panel."""
    ax.set_facecolor("#dddddd")
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_square_heatmap(ax, grid, cmap="YlOrRd", vmin=0, vmax=None,
                        special=None, show_colorbar=True, title=None):
    """Render an (H, W) grid as a square-cell heatmap.

    Orientation matches the C++ ``dump()`` ASCII board: row ``h=0`` at the top,
    column ``w=0`` at the left (``imshow`` with ``origin="upper"``). ``special``
    is an optional iterable of ``(h, w)`` cells to mark (corners + throne in
    tafl) with a small outline. NaN cells render as the colormap's "bad" color.
    Returns the image handle, or ``None`` (after drawing an N/A panel) when the
    grid has no finite values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    grid = np.asarray(grid, dtype=float)
    if grid.size == 0 or not np.isfinite(grid).any():
        draw_na(ax)
        return None

    if vmax is None:
        finite = grid[np.isfinite(grid)]
        vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#dddddd")
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest", aspect="equal")

    H, W = grid.shape
    if special:
        for (h, w) in special:
            ax.add_patch(Rectangle((w - 0.5, h - 0.5), 1, 1, fill=False,
                                   edgecolor="#2222aa", linewidth=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# ---- Hex-canvas rendering ---------------------------------------------------
# A board_side hull rendered as a 13×13 grid indexed [q+6, r+6]. The game's hex
# layout (matching StarGambitGS::dump and play.py) places P0 deploy at the
# bottom-right corner and P1 deploy at the top-left. Heatmaps render via
# axial→pixel coords so peaks land at the corners a player expects.

_HEX_DY = math.sqrt(3) / 2
_hex_unit_verts = np.array([
    (math.cos(math.radians(60 * k + 30)),
     math.sin(math.radians(60 * k + 30)))
    for k in range(6)
])  # pointy-top hexagon, circumradius=1


def _axial_to_xy(q, r):
    """Axial (q, r) → screen (x, y) for pointy-top hexes, +y points down."""
    x = q + r * 0.5
    y = r * _HEX_DY
    return x, y


def _hex_in_bounds(q, r, side=6):
    return abs(q) <= side and abs(r) <= side and abs(-q - r) <= side


def draw_hex_heatmap(ax, grid, cmap="YlOrRd", vmin=0, vmax=None,
                     board_side_hint=None, show_colorbar=True,
                     show_compass=True):
    """Render a 13×13 grid (indexed [q+6, r+6]) as a hex polygon collection.

    board_side_hint: int marks the P0 deploy hex for that inner board size AND
        clips the rendered hexes to that variant's board (so the outer empty
        ring is hidden for board_side=5 variants). None renders all hexes within
        the unified board_side=6 hull and marks both 5- and 6-side spawn hexes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    side = board_side_hint if board_side_hint is not None else 6
    polys = []
    values = []
    for q in range(-side, side + 1):
        for r in range(-side, side + 1):
            if not _hex_in_bounds(q, r, side):
                continue
            val = grid[q + 6, r + 6]
            if np.isnan(val):
                continue
            cx, cy = _axial_to_xy(q, r)
            polys.append(_hex_unit_verts * 0.95 + (cx, cy))
            values.append(val)
    if not polys:
        draw_na(ax)
        return None
    if vmax is None:
        vmax = max(values) if values else 1.0
    if vmax <= 0:
        vmax = 1.0
    coll = PolyCollection(polys, array=np.asarray(values),
                          cmap=cmap, edgecolors="white", linewidths=0.3)
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)

    # Spawn markers.
    spawn_sides = [board_side_hint] if board_side_hint is not None else [5, 6]
    for s in spawn_sides:
        cx, cy = _axial_to_xy(0, s)
        ax.plot(cx, cy, marker="x", markersize=8, mew=1.5,
                color="#222222", linestyle="none")
    # Axis cosmetics: equal aspect, +y down, no ticks; small compass.
    x_min = -side - 0.5 * side - 1
    x_max =  side + 0.5 * side + 1
    y_min = -side * _HEX_DY - 1
    y_max =  side * _HEX_DY + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # invert y so +y is down (matches dump())
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if show_compass:
        ax.text(x_max - 0.4, 0, "E", ha="right", va="center",
                fontsize=7, color="#888888")
        ax.text(x_min + 0.4, 0, "W", ha="left",  va="center",
                fontsize=7, color="#888888")
        ax.text(0, y_min + 0.4, "↑P1", ha="center", va="top",
                fontsize=7, color="#888888")
        ax.text(0, y_max - 0.4, "P0↓", ha="center", va="bottom",
                fontsize=7, color="#888888")
    if show_colorbar:
        plt.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
    return coll


# ---------------------------------------------------------------------------
# Value calibration
# ---------------------------------------------------------------------------


def value_calibration_figure(buckets, iteration, title="Value Calibration"):
    """Predicted vs. actual win-rate calibration curves, one subplot per bucket.

    `buckets` is an ordered dict ``{label: {"v_pred": arr, "v_actual": arr}}``
    (e.g. one entry per Star Gambit variant, or a single "overall" entry for
    tafl). Returns the figure, or ``None`` when `buckets` is empty.
    """
    import matplotlib.pyplot as plt

    if not buckets:
        return None
    n = len(buckets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Iteration {iteration} — {title}", fontsize=12)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    MIN_BIN_COUNT = 10
    for ax, (vname, stats) in zip(axes, buckets.items()):
        vp = np.asarray(stats["v_pred"])
        va = np.asarray(stats["v_actual"])
        bin_idx = np.digitize(vp, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        counts = np.array([(bin_idx == b).sum() for b in range(10)])
        actual_means = np.array([
            va[bin_idx == b].mean() if counts[b] >= MIN_BIN_COUNT else np.nan
            for b in range(10)
        ])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        valid = ~np.isnan(actual_means)
        sc = ax.scatter(bin_centers[valid], actual_means[valid],
                        c=counts[valid], cmap="YlOrRd", s=60, zorder=3)
        ax.plot(bin_centers[valid], actual_means[valid], alpha=0.7)
        for b in range(10):
            if counts[b] > 0:
                ax.text(bin_centers[b], 0.02, f"{counts[b]}",
                        ha="center", va="bottom", fontsize=6,
                        color="#666666",
                        alpha=1.0 if counts[b] >= MIN_BIN_COUNT else 0.4)
        plt.colorbar(sc, ax=ax, label="n samples")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted win prob"); ax.set_ylabel("Actual win rate")
        ax.set_title(f"{vname.capitalize()}\n(n={len(vp)}, "
                     f"bins≥{MIN_BIN_COUNT}: {int(valid.sum())})",
                     fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    return fig
