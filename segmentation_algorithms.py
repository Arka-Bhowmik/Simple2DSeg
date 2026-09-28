from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation, transform
from skimage.filters import gabor
from skimage.segmentation import clear_border, morphological_chan_vese, morphological_geodesic_active_contour


def to_float01(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if np.issubdtype(a.dtype, np.floating):
        out = a.astype(np.float32, copy=False)
        finite = np.isfinite(out)
        if not finite.all():
            out = out.copy()
            out[np.isposinf(out)] = 1.0
            out[np.isneginf(out)] = 0.0
            out[np.isnan(out)] = 0.0
        mn, mx = float(out.min()), float(out.max())
        if mn < 0.0 or mx > 1.0:
            if mx > mn:
                out = (out - mn) / (mx - mn)
            else:
                out = np.zeros_like(out)
        return np.clip(out, 0.0, 1.0)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return ((a.astype(np.float32) - info.min) / float(info.max - info.min)).clip(0, 1)
    return a.astype(np.float32)


def to_gray01(image: np.ndarray) -> np.ndarray:
    f = to_float01(image)
    if f.ndim == 2:
        return f
    if f.ndim == 3 and f.shape[2] >= 3:
        return color.rgb2gray(f[..., :3]).astype(np.float32)
    raise ValueError("Image must be a 2-D grayscale or RGB image.")


def prep_lab01(image: np.ndarray) -> np.ndarray:
    """MATLAB R2018a Image Segmenter prepLab analogue: L/100, (a,b+100)/200."""
    f = to_float01(image)
    if f.ndim != 3 or f.shape[2] < 3:
        return to_gray01(image)[..., None]
    lab = color.rgb2lab(f[..., :3]).astype(np.float32)
    out = lab.copy()
    out[..., 0] /= 100.0
    out[..., 1:3] = (out[..., 1:3] + 100.0) / 200.0
    return out


def create_gabor_features(image: np.ndarray) -> np.ndarray:
    """Python analogue of createGaborFeatures MATLAB R2018a."""
    base = prep_lab01(image) if np.asarray(image).ndim == 3 else to_float01(image)[..., None]
    plane = base[..., 0]
    rows, cols = plane.shape
    wavelength_min = 4 / math.sqrt(2)
    wavelength_max = math.hypot(rows, cols)
    n = int(math.floor(math.log2(max(wavelength_max / wavelength_min, 1.000001))))
    wavelengths = [2**i * wavelength_min for i in range(max(n - 1, 1))]
    orientations = [0, 45, 90, 135]
    feats = []
    for wl in wavelengths:
        freq = 1.0 / max(wl, 1e-6)
        sigma = 0.5 * wl
        for deg in orientations:
            real, imag = gabor(plane, frequency=freq, theta=np.deg2rad(deg))
            mag = np.hypot(real, imag)
            mag = ndi.gaussian_filter(mag, sigma=max(0.5, 3 * sigma / 6.0))
            feats.append(mag.astype(np.float32))
    yy, xx = np.mgrid[0:rows, 0:cols]
    feats += [xx.astype(np.float32), yy.astype(np.float32)]
    stack = np.stack(feats, axis=-1)
    flat = stack.reshape(-1, stack.shape[-1])
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    stack = ((flat - mu) / sd).reshape(stack.shape)
    return np.concatenate([stack, base.astype(np.float32)], axis=-1)


def global_threshold(image: np.ndarray) -> np.ndarray:
    g = to_gray01(image)
    t = filters.threshold_otsu(g)
    return g > t


def manual_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    return to_gray01(image) > float(threshold)


def adaptive_threshold(image: np.ndarray, sensitivity: float = 0.5, polarity: str = "bright") -> np.ndarray:
    """Local adaptive threshold analogue for MATLAB imbinarize(...,'adaptive')."""
    g = to_gray01(image)
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    # Neighborhood is implementation-specific; this keeps the same user control semantics.
    block = max(3, (min(g.shape) // 8) | 1)
    local = filters.threshold_local(g, block_size=block, method="gaussian")
    offset = (sensitivity - 0.5) * 0.20
    if polarity.lower() == "dark":
        return g < (local + offset)
    return g > (local - offset)


def kmeans_mask(image: np.ndarray, use_texture: bool = False) -> np.ndarray:
    from sklearn.cluster import KMeans

    if use_texture:
        x = create_gabor_features(image)
    else:
        f = to_float01(image)
        x = f if f.ndim == 3 else f[..., None]
    h, w = x.shape[:2]
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
    flat = flat - flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    flat /= sd
    labels = KMeans(n_clusters=2, n_init=2, random_state=0).fit_predict(flat)
    # Sklearn labels are zero-based, so use label 1.
    return labels.reshape(h, w) == 1


def flood_fill_euclidean(image: np.ndarray, seed_xy: Tuple[int, int], tolerance: float, use_texture: bool = False) -> np.ndarray:
    features = create_gabor_features(image) if use_texture else to_float01(image)
    x, y = map(int, seed_xy)
    y = int(np.clip(y, 0, features.shape[0] - 1))
    x = int(np.clip(x, 0, features.shape[1] - 1))
    if features.ndim == 3:
        d = np.sum((features - features[y, x]) ** 2, axis=2)
        d -= d.min()
        if d.max() > 0:
            d /= d.max()
        src = d
    else:
        src = features
    lo = float(src[y, x]) - float(tolerance)
    hi = float(src[y, x]) + float(tolerance)
    allowed = (src >= lo) & (src <= hi)
    labels, _ = ndi.label(allowed, structure=np.ones((3, 3), dtype=np.uint8))
    lab = labels[y, x]
    return labels == lab if lab else np.zeros_like(allowed, dtype=bool)


def flood_fill_geodesic(image: np.ndarray, seed_xy: Tuple[int, int], tolerance: float, use_texture: bool = False) -> np.ndarray:
    """Geodesic flood-fill analogue of graydiffweight + imsegfmm."""
    features = create_gabor_features(image) if use_texture else to_float01(image)
    x, y = map(int, seed_xy)
    y = int(np.clip(y, 0, features.shape[0] - 1))
    x = int(np.clip(x, 0, features.shape[1] - 1))
    if features.ndim == 3:
        d = np.sum((features - features[y, x]) ** 2, axis=2)
        if d.max() > d.min():
            d = (d - d.min()) / (d.max() - d.min())
    else:
        d = np.abs(features.astype(np.float32) - float(features[y, x]))
    # Fast-marching-style reachable region.
    # 0.01 level after normalizing finite arrival costs.
    from skimage.graph import MCP_Geometric
    tol = max(float(tolerance), 1e-6)
    speed = np.exp(-((d / tol) ** 2))
    cost = 1.0 / np.maximum(speed, 1e-6)
    mcp = MCP_Geometric(cost)
    costs, _ = mcp.find_costs([(y, x)])
    finite = np.isfinite(costs)
    if not finite.any():
        return np.zeros(d.shape, bool)
    cmin, cmax = float(costs[finite].min()), float(costs[finite].max())
    if cmax <= cmin:
        return finite
    norm = np.ones_like(costs, dtype=np.float64)
    norm[finite] = (costs[finite] - cmin) / (cmax - cmin)
    return norm <= 0.01


def structure_element(shape: str, radius: int = 3, length: int = 3, angle: float = 0.0, width: int = 3) -> np.ndarray:
    shape = shape.lower()
    radius, length, width = int(radius), int(length), int(width)
    if shape == "disk":
        return morphology.disk(max(radius, 0))
    if shape == "diamond":
        return morphology.diamond(max(radius, 0))
    if shape == "octagon":
        r = max(radius, 0)
        if r == 0:
            return np.ones((1, 1), bool)
        # Skimage uses vertical/horizontal extents.
        m = max(1, r // 3)
        return morphology.octagon(3 * m + 1, m)
    if shape == "square":
        n = max(length, 1)
        return np.ones((n, n), bool)
    if shape == "rectangle":
        return np.ones((max(length, 1), max(width, 1)), bool)
    if shape == "line":
        n = max(length, 1)
        canvas = np.zeros((n, n), np.uint8)
        c = (n - 1) / 2.0
        a = math.radians(float(angle))
        dx, dy = math.cos(a) * c, -math.sin(a) * c
        p1 = (int(round(c - dx)), int(round(c - dy)))
        p2 = (int(round(c + dx)), int(round(c + dy)))
        cv2.line(canvas, p1, p2, 1, 1)
        return canvas.astype(bool)
    raise ValueError(f"Unsupported strel shape: {shape}")


def morphology_operation(mask: np.ndarray, operation: str, footprint: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    op = operation.lower()
    if op == "dilate":
        return morphology.dilation(m, footprint)
    if op == "erode":
        return morphology.erosion(m, footprint)
    if op == "open":
        return morphology.opening(m, footprint)
    if op == "close":
        return morphology.closing(m, footprint)
    raise ValueError(f"Unsupported morphology operation: {operation}")


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(np.asarray(mask, bool))


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return ~np.asarray(mask, bool)


def clear_border_mask(mask: np.ndarray) -> np.ndarray:
    return clear_border(np.asarray(mask, bool))


def active_contour(image: np.ndarray, mask: np.ndarray, iterations: int = 100, method: str = "Chan-Vese", use_texture: bool = False) -> np.ndarray:
    m = np.asarray(mask, bool)
    if not m.any():
        raise ValueError("Active Contours requires a non-empty starting mask.")
    if use_texture:
        f = create_gabor_features(image)
        g = np.mean(f, axis=2)
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    else:
        g = to_gray01(image)
    n = max(1, int(iterations))
    if method.lower() in {"chan-vese", "chan_vese", "chanvese"}:
        return morphological_chan_vese(g, num_iter=n, init_level_set=m, smoothing=0).astype(bool)
    # sigma=2, smoothFactor=1, contractionBias=.3. MGAC is the closest skimage analogue.
    gimage = filters.inverse_gaussian_gradient(g, alpha=100.0, sigma=2.0)
    return morphological_geodesic_active_contour(gimage, num_iter=n, init_level_set=m, smoothing=1, balloon=0.3).astype(bool)


def find_circles(image: np.ndarray, min_radius: int = 25, max_radius: int = 75,
                 polarity: str = "bright", sensitivity: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    g = (to_gray01(image) * 255).astype(np.uint8)
    if polarity.lower() == "dark":
        g = 255 - g
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    sens = float(np.clip(sensitivity, 0.01, 0.99))
    param2 = max(5.0, 60.0 * (1.0 - sens) + 8.0)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(5, min_radius),
                               param1=100, param2=param2,
                               minRadius=max(1, int(min_radius)), maxRadius=max(int(min_radius), int(max_radius)))
    if circles is None:
        return np.empty((0, 2), float), np.empty((0,), float)
    c = circles[0]
    return c[:, :2], c[:, 2]


def circles_to_mask(shape: Tuple[int, int], centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    for (x, y), r in zip(centers, radii):
        cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), 1, -1)
    return out.astype(bool)


def slic_superpixels(image: np.ndarray, requested: int) -> Tuple[np.ndarray, int]:
    f = to_float01(image)
    labels = segmentation.slic(f, n_segments=max(2, int(requested)), compactness=10.0,
                               start_label=0, channel_axis=-1 if f.ndim == 3 else None)
    return labels, int(labels.max() + 1)


def _region_features(feature_image: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    f = feature_image if feature_image.ndim == 3 else feature_image[..., None]
    out = np.zeros((n, f.shape[2]), np.float64)
    counts = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    for ch in range(f.shape[2]):
        sums = np.bincount(labels.ravel(), weights=f[..., ch].ravel(), minlength=n)
        out[:, ch] = sums / np.maximum(counts, 1)
    return out


def lazy_snapping(image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray,
                  requested_superpixels: int = 1000, use_texture: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Lazy-Snapping-style graph cut using SLIC superpixels and hard FG/BG constraints.

    The MATLAB R2018a implementation uses superpixels, 8-connectivity, hard foreground/background
    scribbles and lambda=500. This Python implementation preserves those mechanics and solves the
    s-t min-cut with NetworkX.
    """
    import networkx as nx

    f = create_gabor_features(image) if use_texture else (prep_lab01(image) if np.asarray(image).ndim == 3 else to_float01(image)[..., None])
    labels, n = slic_superpixels(image, requested_superpixels)
    fg_ids = np.unique(labels[np.asarray(foreground_mask, bool)])
    bg_ids = np.unique(labels[np.asarray(background_mask, bool)])
    if fg_ids.size == 0 or bg_ids.size == 0:
        raise ValueError("Graph Cut requires both foreground and background scribbles.")
    if np.intersect1d(fg_ids, bg_ids).size:
        raise ValueError("Foreground and background scribbles occupy the same superpixel. Increase superpixel density or move a scribble.")

    feats = _region_features(f, labels, n)
    fg_mean = feats[fg_ids].mean(axis=0)
    bg_mean = feats[bg_ids].mean(axis=0)
    d_fg = np.sum((feats - fg_mean) ** 2, axis=1)
    d_bg = np.sum((feats - bg_mean) ** 2, axis=1)
    scale = float(np.median(np.r_[d_fg, d_bg]) + 1e-8)
    d_fg, d_bg = d_fg / scale, d_bg / scale

    pairs = set()
    H, W = labels.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0a, y1a = max(0, dy), H + min(0, dy)
        x0a, x1a = max(0, dx), W + min(0, dx)
        y0b, y1b = max(0, -dy), H - max(0, dy)
        x0b, x1b = max(0, -dx), W - max(0, dx)
        a, b = labels[y0a:y1a, x0a:x1a], labels[y0b:y1b, x0b:x1b]
        diff = a != b
        for u, v in zip(a[diff].ravel(), b[diff].ravel()):
            u, v = int(u), int(v)
            if u != v:
                pairs.add((min(u, v), max(u, v)))

    source, sink = "__source__", "__sink__"
    G = nx.DiGraph()
    inf = 1e15
    fg_set, bg_set = set(map(int, fg_ids)), set(map(int, bg_ids))
    for i in range(n):
        if i in fg_set:
            G.add_edge(source, i, capacity=inf)
            G.add_edge(i, sink, capacity=0.0)
        elif i in bg_set:
            G.add_edge(source, i, capacity=0.0)
            G.add_edge(i, sink, capacity=inf)
        else:
            # Source side = foreground: source->i is background cost; i->sink is foreground cost.
            G.add_edge(source, i, capacity=float(d_bg[i]) * 1000.0)
            G.add_edge(i, sink, capacity=float(d_fg[i]) * 1000.0)

    lam = 500.0
    for u, v in pairs:
        dist = float(np.sum((feats[u] - feats[v]) ** 2))
        w = lam * math.exp(-dist / (2.0 * scale)) + 1e-9
        G.add_edge(u, v, capacity=w)
        G.add_edge(v, u, capacity=w)

    _, (source_side, sink_side) = nx.minimum_cut(G, source, sink, capacity="capacity")
    region_fg = np.array([i in source_side for i in range(n)], bool)
    return region_fg[labels], labels


def grabcut(image: np.ndarray, roi_mask: np.ndarray, foreground_mask: Optional[np.ndarray] = None,
            background_mask: Optional[np.ndarray] = None, iterations: int = 5) -> np.ndarray:
    f = to_float01(image)
    if f.ndim == 2:
        rgb = np.repeat((f * 255).astype(np.uint8)[..., None], 3, axis=2)
    else:
        rgb = (f[..., :3] * 255).astype(np.uint8)
    roi = np.asarray(roi_mask, bool)
    if not roi.any():
        raise ValueError("GrabCut requires a non-empty rectangle or polygon ROI.")
    gc = np.full(roi.shape, cv2.GC_BGD, np.uint8)
    gc[roi] = cv2.GC_PR_FGD
    if foreground_mask is not None:
        gc[np.asarray(foreground_mask, bool)] = cv2.GC_FGD
    if background_mask is not None:
        gc[np.asarray(background_mask, bool)] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, gc, None, bgd, fgd, max(1, int(iterations)), cv2.GC_INIT_WITH_MASK)
    return np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD])


def polygon_mask(shape: Tuple[int, int], points: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(out, [pts], 1)
    return out.astype(bool)


def rectangle_mask(shape: Tuple[int, int], p0: Tuple[int, int], p1: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    x0, y0 = p0; x1, y1 = p1
    cv2.rectangle(out, (min(x0,x1), min(y0,y1)), (max(x0,x1), max(y0,y1)), 1, -1)
    return out.astype(bool)


def ellipse_mask(shape: Tuple[int, int], p0: Tuple[int, int], p1: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    x0, y0 = p0; x1, y1 = p1
    cx, cy = (x0+x1)//2, (y0+y1)//2
    ax, ay = max(1, abs(x1-x0)//2), max(1, abs(y1-y0)//2)
    cv2.ellipse(out, (cx,cy), (ax,ay), 0, 0, 360, 1, -1)
    return out.astype(bool)


def masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.array(image, copy=True)
    m = np.asarray(mask, bool)
    if a.ndim == 3:
        a[~m, :] = 0
    else:
        a[~m] = 0
    return a
