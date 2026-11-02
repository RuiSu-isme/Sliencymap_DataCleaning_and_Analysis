"""Correctness checks for the AUC implementation in auc_score_new.py.

Two independent references are used:

  * the rank-based Mann-Whitney U statistic (mathematically equal to the
    trapezoid ROC-AUC, but computed from ranks instead of thresholds)
  * a brute-force weighted sweep that builds the full-resolution kernel weight
    map and thresholds the whole map at every 8-bit level
"""

import numpy as np

import auc_score_new as A
import gaze_kernel as gk


def check(label, got, want, tol=1e-12):
    if got is None or want is None:
        ok = got is None and want is None
    else:
        ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:56s} mine={got} ref={want}")
    return ok


def mann_whitney_auc(score_map, fix_mask):
    """Rank-based AUC with average ranks for ties (independent reference)."""
    scores = score_map.ravel().astype(np.float64)
    positive = fix_mask.ravel().astype(bool)
    n_pos = int(positive.sum())
    n_neg = int(scores.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average rank, 1-based
        i = j + 1

    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brute_force_weighted_auc(saliency_map, kernel, ix, iy, fp_scope="all"):
    """Threshold the whole map at every level with an explicit weight map."""
    height, width = saliency_map.shape
    weights = np.zeros((height, width), dtype=np.float64)

    y0 = max(0, iy - kernel.radius)
    y1 = min(height, iy + kernel.radius + 1)
    x0 = max(0, ix - kernel.radius)
    x1 = min(width, ix + kernel.radius + 1)
    ky = y0 - (iy - kernel.radius)
    kx = x0 - (ix - kernel.radius)
    weights[y0:y1, x0:x1] = kernel.weights[ky:ky + (y1 - y0), kx:kx + (x1 - x0)]

    total = weights.sum()
    support = weights > 0
    n_support = int(support.sum())
    denom = saliency_map.size - n_support

    tp, fp = [], []
    for t in range(255, -1, -1):
        above = saliency_map >= t
        tp.append(float(weights[above].sum() / total))
        if fp_scope == "all":
            fp.append(float(above.sum() / saliency_map.size))
        else:
            fp.append(float((above & ~support).sum() / denom))

    return float(np.trapezoid([0.0] + tp + [1.0], [0.0] + fp + [1.0]))


def main():
    rng = np.random.default_rng(2024)
    ok = True

    print("1) point kernel, fp over NON-FIXATED pixels == rank-based Mann-Whitney")
    for trial in range(5):
        smap = rng.integers(0, 256, size=(60, 80), dtype=np.uint8)
        ix, iy = int(rng.integers(0, 80)), int(rng.integers(0, 60))
        kernel = gk.GazeKernel(80, 60, kind="point")
        lookup = A.FrameLookup(smap)
        mass, support = kernel.weighted_histograms(smap, ix, iy, with_counts=True)
        mine = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels,
                                   support_counts=support,
                                   support_pixels=int(support.sum()))

        mask = np.zeros(smap.shape, dtype=bool)
        mask[iy, ix] = True
        ok &= check(f"trial {trial}", mine, mann_whitney_auc(smap, mask), tol=1e-12)

    print("2) point kernel, several fixations, non-fixated fp = Mann-Whitney")
    for trial in range(3):
        smap = rng.integers(0, 256, size=(50, 70), dtype=np.uint8)
        points = list({(int(rng.integers(0, 70)), int(rng.integers(0, 50))) for _ in range(6)})
        lookup = A.FrameLookup(smap)
        mass = np.zeros(256)
        support = np.zeros(256)
        for ix, iy in points:
            m, c = gk.GazeKernel(70, 50, kind="point").weighted_histograms(smap, ix, iy, True)
            mass += m
            support += c
        mine = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels,
                                   support_counts=support,
                                   support_pixels=int(support.sum()))

        mask = np.zeros(smap.shape, dtype=bool)
        for ix, iy in points:
            mask[iy, ix] = True
        ok &= check(f"trial {trial} ({len(points)} fixations)", mine,
                    mann_whitney_auc(smap, mask), tol=1e-12)

    print("3) gaussian kernel, fp over NON-KERNEL pixels = brute-force weighted sweep")
    for trial in range(3):
        smap = rng.integers(0, 256, size=(300, 400), dtype=np.uint8)
        ix, iy = 200, 150
        kernel = gk.GazeKernel(400, 300, kind="gaussian", sigma_deg=1.0, px_per_deg=12.0)
        lookup = A.FrameLookup(smap)
        mass, support = kernel.weighted_histograms(smap, ix, iy, with_counts=True)
        mine = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels,
                                   support_counts=support,
                                   support_pixels=int(support.sum()))
        ok &= check(f"trial {trial} ({kernel.radius} px radius)", mine,
                    brute_force_weighted_auc(smap, kernel, ix, iy, "nonfixation"), tol=1e-12)

    print("4) gaussian kernel, fp over ALL pixels = brute force")
    for trial in range(2):
        smap = rng.integers(0, 256, size=(300, 400), dtype=np.uint8)
        ix, iy = 200, 150
        kernel = gk.GazeKernel(400, 300, kind="gaussian", sigma_deg=1.0, px_per_deg=12.0)
        lookup = A.FrameLookup(smap)
        mass, _ = kernel.weighted_histograms(smap, ix, iy)
        mine = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels)
        ok &= check(f"trial {trial}", mine,
                    brute_force_weighted_auc(smap, kernel, ix, iy, "all"), tol=1e-12)

    print("5) disk kernel, fp over NON-KERNEL pixels = brute force")
    smap = rng.integers(0, 256, size=(200, 260), dtype=np.uint8)
    kernel = gk.GazeKernel(260, 200, kind="disk", radius_px=12)
    lookup = A.FrameLookup(smap)
    mass, support = kernel.weighted_histograms(smap, 130, 100, with_counts=True)
    mine = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels,
                               support_counts=support, support_pixels=int(support.sum()))
    ok &= check("disk radius 12", mine,
                brute_force_weighted_auc(smap, kernel, 130, 100, "nonfixation"), tol=1e-12)

    print("6) degenerate maps")
    flat = np.zeros((40, 50), dtype=np.uint8)
    kernel = gk.GazeKernel(50, 40, kind="gaussian", sigma_deg=1.0, px_per_deg=6.0)
    lookup = A.FrameLookup(flat)
    mass, _ = kernel.weighted_histograms(flat, 25, 20)
    ok &= check("flat map == chance 0.5",
                A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels), 0.5)

    peak = np.zeros((40, 50), dtype=np.uint8)
    peak[20, 25] = 255
    pk = gk.GazeKernel(50, 40, kind="point")
    kmass, ksupport = pk.weighted_histograms(peak, 25, 20, with_counts=True)
    klook = A.FrameLookup(peak)
    ok &= check("point on the only salient pixel == 1.0",
                A.roc_auc_from_mass(kmass, klook.count_suffix, klook.n_pixels,
                                    support_counts=ksupport,
                                    support_pixels=int(ksupport.sum())), 1.0)

    print("7) the ROC is monotone even with heavy ties")
    tied = np.zeros((60, 80), dtype=np.uint8)
    tied[10:40, 20:60] = 200
    tied[5:8, 5:8] = 255
    lookup = A.FrameLookup(tied)
    mass = np.zeros(256)
    for ix, iy in [(30, 20), (31, 20), (30, 21), (50, 50), (5, 6)]:
        m, _ = gk.GazeKernel(80, 60, kind="gaussian", sigma_deg=1.5, px_per_deg=8.0)\
            .weighted_histograms(tied, ix, iy)
        mass += m
    mass_above = np.cumsum(mass[::-1])[::-1]
    tp = mass_above[::-1] / mass.sum()
    fp = lookup.count_suffix[::-1] / lookup.n_pixels
    ok &= check("TP non-decreasing as the threshold drops",
                float(np.all(np.diff(tp) >= -1e-15)), 1.0)
    ok &= check("FP non-decreasing as the threshold drops",
                float(np.all(np.diff(fp) >= -1e-15)), 1.0)
    score = A.roc_auc_from_mass(mass, lookup.count_suffix, lookup.n_pixels)
    ok &= check("score is inside [0, 1]", float(0.0 <= score <= 1.0), 1.0)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
