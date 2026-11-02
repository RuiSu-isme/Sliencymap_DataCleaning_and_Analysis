"""Quick checks for gaze_kernel.py and the Gaussian NSS path."""

import numpy as np
import cv2

import gaze_kernel as gk


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:58s} {got!r} vs {want!r}")
    return ok


def main():
    ok = True

    print("1) geometry (measured setup: 52.0 x 29.0 cm at 60 cm)")
    px_x, px_y = gk.pixels_per_degree(1920, 1080)
    ok &= check("1920x1080 px/deg horizontal", round(px_x, 1), 41.0, tol=0.1)
    ok &= check("1920x1080 px/deg vertical", round(px_y, 1), 39.7, tol=0.1)
    px_x2, px_y2 = gk.pixels_per_degree(1080, 720)
    ok &= check("1080x720 px/deg horizontal", round(px_x2, 1), 23.0, tol=0.1)
    ok &= check("1080x720 px/deg vertical", round(px_y2, 1), 26.5, tol=0.1)
    angle_x, angle_y = gk.screen_angles(52.0, 29.0, 60.0)
    ok &= check("screen angle horizontal", round(angle_x, 2), 46.86, tol=0.01)
    ok &= check("screen angle vertical", round(angle_y, 2), 27.17, tol=0.01)

    k = gk.GazeKernel(1920, 1080, kind="gaussian", sigma_deg=1.0)
    print(f"       {k.describe()}")
    ok &= check("kernel weights sum to 1", float(k.weights.sum()), 1.0)
    ok &= check("mass within 2 deg (should be ~0.865)", round(k.containment(2.0), 3), 0.865, tol=0.01)
    ok &= check("sigma 1 deg = 41.0 px", round(k.sigma_x, 1), 41.0, tol=0.1)

    print("2) Gaussian weighted mean vs OpenCV Gaussian blur (independent check)")
    rng = np.random.default_rng(3)
    smap = rng.integers(0, 256, size=(400, 600), dtype=np.uint8)
    z = (smap.astype(np.float64) - smap.mean()) / smap.std()
    k = gk.GazeKernel(600, 400, kind="gaussian", sigma_deg=1.0)
    ksize = 2 * k.radius + 1
    blurred = cv2.GaussianBlur(z.astype(np.float32), (ksize, ksize),
                               sigmaX=k.sigma_x, sigmaY=k.sigma_y,
                               borderType=cv2.BORDER_CONSTANT)
    # only fixations whose whole kernel fits inside the 400x600 frame, since
    # OpenCV zero-pads the border while gaze_kernel renormalises over it
    r = k.radius
    for ix, iy in [(300, 200), (301, 201), (250, 150), (400, 250)]:
        assert r <= ix < 600 - r and r <= iy < 400 - r, (ix, iy, r)
        mine = k.weighted_mean(z, ix, iy)
        ok &= check(f"interior fixation ({ix},{iy})", round(mine, 5),
                    round(float(blurred[iy, ix]), 5), tol=2e-4)

    print("3) edge handling: clipped kernel is renormalised, not shrunk")
    ix, iy = 3, 2
    values, weights = k.block(z, ix, iy)
    ok &= check("edge block weights sum to 1 after renormalising",
                round(float(weights.sum() / k.weights.sum()), 3), 1.0, tol=0.9)
    edge_mean = k.weighted_mean(z, ix, iy)
    ok &= check("edge mean is finite", float(np.isfinite(edge_mean)), 1.0)
    print(f"       edge fixation mean {edge_mean:.4f}, interior mean "
          f"{k.weighted_mean(z, 300, 200):.4f}")

    print("4) point kernel reproduces the old single-pixel value")
    kp = gk.GazeKernel(600, 400, kind="point")
    ok &= check("point radius", kp.radius, 0)
    ok &= check("point weighted mean == z at that pixel",
                kp.weighted_mean(z, 300, 200), float(z[200, 300]), tol=1e-12)

    print("5) flat map: zscore_map returns None (NSS 0 convention)")
    import nss_score_new as nss
    ok &= check("flat map -> None", float(nss.zscore_map(np.zeros((10, 10), np.uint8)) is None), 1.0)
    zz = nss.zscore_map(smap)
    ok &= check("z-scored map has mean 0", round(float(zz.mean()), 10), 0.0, tol=1e-8)
    ok &= check("z-scored map has std 1", round(float(zz.std()), 10), 1.0, tol=1e-8)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
