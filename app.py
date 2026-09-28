from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from skimage.segmentation import find_boundaries
from streamlit_drawable_canvas import st_canvas

import segmentation_algorithms as alg
from csv_workflow import CSVWorkflow, Case


CANVAS_WIDTH = 920
CANVAS_HEIGHT = 650


@dataclass
class ViewTransform:
    canvas_w: int
    canvas_h: int
    image_w: int
    image_h: int
    scale: float
    x0: float
    y0: float


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Streamlit interface for the Python Image Segmenter.")
    parser.add_argument("--path_to_csv", "--path-to-csv", default=".", help="Directory containing the CSV file.")
    parser.add_argument("--csv_file", "--csv-file", default=None, help="CSV filename inside --path_to_csv.")
    parser.add_argument("csv", nargs="?", default=None, help="Optional full CSV path. This overrides --path_to_csv/--csv_file.")
    args, _ = parser.parse_known_args()
    return args


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv).expanduser().resolve()
    if not args.csv_file:
        raise ValueError("Provide --csv_file together with --path_to_csv, or pass a full CSV path.")
    csv_file = Path(args.csv_file).expanduser()
    if csv_file.is_absolute():
        return csv_file.resolve()
    return (Path(args.path_to_csv).expanduser() / csv_file).resolve()


def _defaults() -> dict:
    return {
        "workflow_path": "",
        "workflow": None,
        "case": None,
        "image": None,
        "mask": None,
        "preview_mask": None,
        "preview_desc": "",
        "history": [],
        "history_index": -1,
        "history_revision": 0,
        "graph_fg": None,
        "graph_bg": None,
        "graph_labels": None,
        "grab_fg": None,
        "grab_bg": None,
        "grab_roi": None,
        "active_panel": "segmentation",
        "manual_tool": "Freehand",
        "graph_mark_mode": "Foreground",
        "grab_mark_mode": "Foreground",
        "grab_phase": "ROI",
        "grab_roi_style": "Rectangle",
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
        "show_binary": False,
        "show_subtraction": False,
        "opacity": 0.45,
        "use_texture": False,
        "show_superpixels": False,
        "canvas_epoch": 0,
        "last_canvas_signature": "",
        "status": "Ready",
        "case_warning": "",
        "complete": False,
        "exited": False,
        "pending_save": False,
    }


def init_session(csv_path: Path) -> None:
    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value

    csv_str = str(csv_path)
    if st.session_state.workflow_path != csv_str:
        workflow = CSVWorkflow(csv_path)
        for key, value in _defaults().items():
            st.session_state[key] = value
        st.session_state.workflow_path = csv_str
        st.session_state.workflow = workflow
        load_next_case(0)


def reset_aux_masks() -> None:
    image = st.session_state.image
    if image is None:
        return
    shape = image.shape[:2]
    st.session_state.graph_fg = np.zeros(shape, np.uint8)
    st.session_state.graph_bg = np.zeros(shape, np.uint8)
    st.session_state.graph_labels = None
    st.session_state.grab_fg = np.zeros(shape, np.uint8)
    st.session_state.grab_bg = np.zeros(shape, np.uint8)
    st.session_state.grab_roi = None


def load_next_case(start: int) -> None:
    workflow: CSVWorkflow = st.session_state.workflow
    case = workflow.next_unprocessed(start)
    if case is None:
        st.session_state.case = None
        st.session_state.image = None
        st.session_state.mask = None
        st.session_state.complete = True
        st.session_state.status = "All unflagged cases have been processed."
        return

    image, existing, warning = workflow.load_case(case)
    mask = existing.copy() if existing is not None else np.zeros(image.shape[:2], bool)

    st.session_state.case = case
    st.session_state.image = image
    st.session_state.mask = mask
    st.session_state.preview_mask = None
    st.session_state.preview_desc = ""
    st.session_state.history = [("Loaded existing mask" if existing is not None else "Load Image", mask.copy())]
    st.session_state.history_index = 0
    st.session_state.history_revision += 1
    st.session_state.zoom = 1.0
    st.session_state.pan_x = 0.0
    st.session_state.pan_y = 0.0
    st.session_state.canvas_epoch += 1
    st.session_state.last_canvas_signature = ""
    st.session_state.case_warning = warning or ""
    st.session_state.complete = False
    st.session_state.pending_save = False
    reset_aux_masks()
    st.session_state.status = ("Existing mask loaded." if existing is not None else "New empty segmentation.")


def commit_mask(new_mask: np.ndarray, description: str) -> None:
    current = st.session_state.mask
    if current is None:
        return
    m = np.asarray(new_mask, bool)
    if m.shape != current.shape:
        raise ValueError("New mask shape does not match image.")
    idx = st.session_state.history_index
    history = list(st.session_state.history[: idx + 1])
    history.append((description, m.copy()))
    st.session_state.history = history
    st.session_state.history_index = len(history) - 1
    st.session_state.history_revision += 1
    st.session_state.mask = m.copy()
    st.session_state.preview_mask = None
    st.session_state.preview_desc = ""
    st.session_state.canvas_epoch += 1
    st.session_state.last_canvas_signature = ""
    st.session_state.status = description


def set_preview(mask: np.ndarray, description: str) -> None:
    st.session_state.preview_mask = np.asarray(mask, bool)
    st.session_state.preview_desc = description
    st.session_state.status = f"Preview: {description}"


def apply_preview() -> None:
    if st.session_state.preview_mask is not None:
        commit_mask(st.session_state.preview_mask, st.session_state.preview_desc or "Applied preview")


def close_preview() -> None:
    st.session_state.preview_mask = None
    st.session_state.preview_desc = ""
    st.session_state.active_panel = "segmentation"
    st.session_state.status = "Preview closed."


def undo_history() -> None:
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.mask = st.session_state.history[st.session_state.history_index][1].copy()
        st.session_state.preview_mask = None
        st.session_state.preview_desc = ""
        st.session_state.history_revision += 1
        st.session_state.canvas_epoch += 1
        st.session_state.status = "Undo"


def delete_history_selection(idx: int) -> None:
    history = st.session_state.history
    if not history or idx <= 0 or idx >= len(history):
        st.session_state.status = "The first history item is the base mask state and cannot be deleted."
        return
    desc = history[idx][0]
    st.session_state.history = history[:idx]
    st.session_state.history_index = idx - 1
    st.session_state.mask = st.session_state.history[-1][1].copy()
    st.session_state.preview_mask = None
    st.session_state.preview_desc = ""
    st.session_state.history_revision += 1
    st.session_state.canvas_epoch += 1
    st.session_state.last_canvas_signature = ""
    st.session_state.status = f"Deleted history item: {desc}"


def save_current_case() -> None:
    workflow: CSVWorkflow = st.session_state.workflow
    case: Case = st.session_state.case
    image = st.session_state.image
    mask = st.session_state.mask
    if case is None or image is None or mask is None:
        return
    segmented = alg.masked_image(image, mask)
    workflow.save_case(case, mask, segmented)
    row = case.row_index
    st.session_state.status = f"Saved row {row + 1}; Segment=1."
    load_next_case(row + 1)


def _display_rgb(a: np.ndarray) -> np.ndarray:
    f = alg.to_float01(a)
    if f.ndim == 2:
        return np.repeat((f * 255).astype(np.uint8)[..., None], 3, axis=2)
    return (f[..., :3] * 255).astype(np.uint8)


def _compose_image_rgb() -> np.ndarray:
    image = st.session_state.image
    mask = (
        st.session_state.preview_mask
        if st.session_state.preview_mask is not None
        else st.session_state.mask
    )
    if image is None:
        return np.zeros((10, 10, 3), np.uint8)

    if st.session_state.show_binary and mask is not None:
        rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    elif st.session_state.show_subtraction and st.session_state.mask is not None:
        rgb = _display_rgb(image)
        rgb = (rgb * st.session_state.mask.astype(bool)[..., None]).astype(np.uint8)
    else:
        rgb = _display_rgb(image)
        if mask is not None:
            alpha = float(st.session_state.opacity)
            m = mask.astype(bool)
            overlay = rgb.astype(np.float32)
            overlay[m] = overlay[m] * (1.0 - alpha) + np.array([255, 0, 0], np.float32) * alpha
            rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    panel = st.session_state.active_panel
    if st.session_state.graph_fg is not None:
        rgb[st.session_state.graph_fg.astype(bool)] = [0, 255, 0]
        rgb[st.session_state.graph_bg.astype(bool)] = [255, 0, 0]

    if panel == "grabcut" and st.session_state.grab_fg is not None:
        rgb[st.session_state.grab_fg.astype(bool)] = [0, 255, 0]
        rgb[st.session_state.grab_bg.astype(bool)] = [255, 0, 0]
        if st.session_state.grab_roi is not None:
            b = find_boundaries(st.session_state.grab_roi, mode="outer")
            rgb[b] = [255, 255, 0]

    if st.session_state.show_superpixels and st.session_state.graph_labels is not None:
        b = find_boundaries(st.session_state.graph_labels)
        rgb[b] = [255, 255, 0]
    return rgb


def render_view(rgb: np.ndarray) -> tuple[Image.Image, ViewTransform]:
    h, w = rgb.shape[:2]
    base_fit = min(CANVAS_WIDTH / max(w, 1), CANVAS_HEIGHT / max(h, 1))
    scale = max(0.02, base_fit * float(st.session_state.zoom))
    dw = max(1, int(round(w * scale)))
    dh = max(1, int(round(h * scale)))

    extra_x = max(0.0, (dw - CANVAS_WIDTH) / 2.0)
    extra_y = max(0.0, (dh - CANVAS_HEIGHT) / 2.0)
    x0 = (CANVAS_WIDTH - dw) / 2.0 + float(st.session_state.pan_x) * extra_x
    y0 = (CANVAS_HEIGHT - dh) / 2.0 + float(st.session_state.pan_y) * extra_y

    resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8)

    sx0 = max(0, int(round(-x0)))
    sy0 = max(0, int(round(-y0)))
    dx0 = max(0, int(round(x0)))
    dy0 = max(0, int(round(y0)))
    copy_w = min(dw - sx0, CANVAS_WIDTH - dx0)
    copy_h = min(dh - sy0, CANVAS_HEIGHT - dy0)
    if copy_w > 0 and copy_h > 0:
        canvas[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = resized[
            sy0 : sy0 + copy_h, sx0 : sx0 + copy_w
        ]

    transform = ViewTransform(
        canvas_w=CANVAS_WIDTH,
        canvas_h=CANVAS_HEIGHT,
        image_w=w,
        image_h=h,
        scale=scale,
        x0=x0,
        y0=y0,
    )
    return Image.fromarray(canvas), transform


def view_mask_to_image(mask_view: np.ndarray, tr: ViewTransform) -> np.ndarray:
    src = (np.asarray(mask_view, bool).astype(np.uint8) * 255)
    M = np.array(
        [[1.0 / tr.scale, 0.0, -tr.x0 / tr.scale], [0.0, 1.0 / tr.scale, -tr.y0 / tr.scale]],
        dtype=np.float32,
    )
    out = cv2.warpAffine(
        src,
        M,
        (tr.image_w, tr.image_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out > 0


def canvas_point_to_image(x: float, y: float, tr: ViewTransform) -> tuple[int, int]:
    ix = int(round((float(x) - tr.x0) / tr.scale))
    iy = int(round((float(y) - tr.y0) / tr.scale))
    ix = int(np.clip(ix, 0, tr.image_w - 1))
    iy = int(np.clip(iy, 0, tr.image_h - 1))
    return ix, iy


def _canvas_call(**kwargs):
    params = inspect.signature(st_canvas).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return st_canvas(**filtered)


def _result_signature(result) -> str:
    data = getattr(result, "json_data", None)
    if not data:
        return ""
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def canvas_alpha_mask(result, background: Image.Image) -> Optional[np.ndarray]:
    arr = getattr(result, "image_data", None)
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[0] != CANVAS_HEIGHT or a.shape[1] != CANVAS_WIDTH:
        return None

    if a.shape[2] >= 4:
        alpha = a[..., 3]
        # Drawable Canvas normally returns only the drawing layer, so alpha is sparse.
        if np.mean(alpha > 0) < 0.95:
            return alpha > 5

    # Compatibility fallback for versions that include the background in image_data.
    rgb = a[..., :3].astype(np.int16)
    bg = np.asarray(background.convert("RGB"), dtype=np.int16)
    diff = np.max(np.abs(rgb - bg), axis=2)
    return diff > 12


def _fabric_path_points(obj: dict) -> list[tuple[float, float]]:
    path = obj.get("path") or []
    raw: list[tuple[float, float]] = []
    for cmd in path:
        if not isinstance(cmd, (list, tuple)) or len(cmd) < 3:
            continue
        op = str(cmd[0]).upper()
        nums = [float(v) for v in cmd[1:] if isinstance(v, (int, float))]
        if op in {"M", "L", "T"} and len(nums) >= 2:
            raw.append((nums[-2], nums[-1]))
        elif op in {"Q", "S", "C"} and len(nums) >= 2:
            raw.append((nums[-2], nums[-1]))
    if len(raw) < 3:
        return raw

    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0))
    height = float(obj.get("height", 0.0))
    sx = float(obj.get("scaleX", 1.0))
    sy = float(obj.get("scaleY", 1.0))
    angle = np.deg2rad(float(obj.get("angle", 0.0)))
    po = obj.get("pathOffset") or {}
    pox = float(po.get("x", 0.0))
    poy = float(po.get("y", 0.0))

    cx = left + width * sx / 2.0
    cy = top + height * sy / 2.0
    ca, sa = float(np.cos(angle)), float(np.sin(angle))
    pts = []
    for px, py in raw:
        lx = (px - pox) * sx
        ly = (py - poy) * sy
        x = cx + ca * lx - sa * ly
        y = cy + sa * lx + ca * ly
        pts.append((x, y))

    in_bounds = sum(0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT for x, y in pts)
    raw_in_bounds = sum(0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT for x, y in raw)
    return pts if in_bounds >= raw_in_bounds else raw


def freehand_fill_mask(result) -> Optional[np.ndarray]:
    data = getattr(result, "json_data", None) or {}
    objects = data.get("objects") or []
    out = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), np.uint8)
    found = False
    for obj in objects:
        if str(obj.get("type", "")).lower() != "path":
            continue
        pts = _fabric_path_points(obj)
        if len(pts) >= 3:
            arr = np.round(np.asarray(pts)).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(out, [arr], 1)
            found = True
    return out.astype(bool) if found else None


def ellipse_from_rect_json(result) -> Optional[np.ndarray]:
    data = getattr(result, "json_data", None) or {}
    objects = data.get("objects") or []
    out = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), np.uint8)
    found = False
    for obj in objects:
        if str(obj.get("type", "")).lower() != "rect":
            continue
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
        height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
        x0, y0 = int(round(left)), int(round(top))
        x1, y1 = int(round(left + width)), int(round(top + height))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ax, ay = max(1, abs(x1 - x0) // 2), max(1, abs(y1 - y0) // 2)
        cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
        found = True
    return out.astype(bool) if found else None


def _preview_apply_close(prefix: str) -> None:
    a, b, c = st.columns(3)
    with a:
        if st.button("Apply", key=f"{prefix}_apply", use_container_width=True):
            apply_preview()
            st.rerun()
    with b:
        if st.button("Close", key=f"{prefix}_close", use_container_width=True):
            close_preview()
            st.rerun()
    with c:
        if st.session_state.preview_mask is not None:
            st.caption(st.session_state.preview_desc)


def requested_superpixels() -> int:
    image = st.session_state.image
    if image is None:
        return 100
    h, w = image.shape[:2]
    val = float(st.session_state.get("superpixel_density", 50))
    return int(round(((h * w / 100.0) * (val / 100.0)) + 100))


def run_operation(fn, success_status: Optional[str] = None):
    try:
        with st.spinner("Working..."):
            value = fn()
        if success_status:
            st.session_state.status = success_status
        return value
    except Exception as exc:
        st.session_state.status = f"Error: {exc}"
        st.error(str(exc))
        return None


def toolbar_group(title: str):
    st.markdown(f'<div class="tool-title">{title}</div>', unsafe_allow_html=True)


def render_toolbar() -> None:
    st.markdown('<div class="ribbon-label">SEGMENTATION</div>', unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns([1.25, 0.9, 2.0], gap="small")
    with r1a:
        with st.container(border=True):
            toolbar_group("LOAD")
            a, b = st.columns(2)
            if a.button("Load Mask", use_container_width=True):
                st.session_state.active_panel = "load_mask"
            if b.button("New Segmentation", use_container_width=True):
                commit_mask(np.zeros(st.session_state.image.shape[:2], bool), "New Segmentation")
                st.rerun()
    with r1b:
        with st.container(border=True):
            toolbar_group("TEXTURE")
            st.session_state.use_texture = st.checkbox(
                "Include Texture Features", value=st.session_state.use_texture
            )
    with r1c:
        with st.container(border=True):
            toolbar_group("CREATE MASK")
            cols = st.columns(4)
            if cols[0].button("Threshold", use_container_width=True):
                st.session_state.active_panel = "threshold"
            if cols[1].button("Graph Cut", use_container_width=True):
                st.session_state.active_panel = "graphcut"
            if cols[2].button("K-means", use_container_width=True):
                m = run_operation(
                    lambda: alg.kmeans_mask(st.session_state.image, st.session_state.use_texture)
                )
                if m is not None:
                    commit_mask(m, "K-means Clustering")
                    st.rerun()
            if cols[3].button("Find Circles", use_container_width=True):
                st.session_state.active_panel = "circles"

    r2a, r2b = st.columns([1.3, 1.15], gap="small")
    with r2a:
        with st.container(border=True):
            toolbar_group("ADD TO MASK")
            cols = st.columns(6)
            if cols[0].button("GrabCut", use_container_width=True):
                st.session_state.active_panel = "grabcut"
            if cols[1].button("Flood Fill", use_container_width=True):
                st.session_state.active_panel = "flood"
            for i, name in enumerate(["Freehand", "Rectangle", "Ellipse", "Polygon"], start=2):
                if cols[i].button(name, use_container_width=True):
                    st.session_state.active_panel = "manual"
                    st.session_state.manual_tool = name
                    st.session_state.canvas_epoch += 1
                    st.session_state.last_canvas_signature = ""
    with r2b:
        with st.container(border=True):
            toolbar_group("REFINE MASK")
            cols = st.columns(5)
            if cols[0].button("Morphology", use_container_width=True):
                st.session_state.active_panel = "morphology"
            if cols[1].button("Active Contours", use_container_width=True):
                st.session_state.active_panel = "active"
            if cols[2].button("Clear Border", use_container_width=True):
                commit_mask(alg.clear_border_mask(st.session_state.mask), "Clear Border")
                st.rerun()
            if cols[3].button("Fill Holes", use_container_width=True):
                commit_mask(alg.fill_holes(st.session_state.mask), "Fill Holes")
                st.rerun()
            if cols[4].button("Invert Mask", use_container_width=True):
                commit_mask(alg.invert_mask(st.session_state.mask), "Invert Mask")
                st.rerun()

    r3a, r3b = st.columns([1.65, 0.55], gap="small")
    with r3a:
        with st.container(border=True):
            toolbar_group("ZOOM / VIEW")
            cols = st.columns([0.6, 0.6, 0.6, 1.0, 1.2, 1.8])
            if cols[0].button("Zoom +", use_container_width=True):
                st.session_state.zoom = min(5.0, st.session_state.zoom * 1.25)
                st.rerun()
            if cols[1].button("Zoom -", use_container_width=True):
                st.session_state.zoom = max(0.25, st.session_state.zoom / 1.25)
                st.rerun()
            if cols[2].button("Pan", use_container_width=True):
                st.session_state.active_panel = "pan"
            def _binary_changed():
                if st.session_state.show_binary:
                    st.session_state.show_subtraction = False

            def _subtraction_changed():
                if st.session_state.show_subtraction:
                    st.session_state.show_binary = False

            cols[3].checkbox(
                "Show Binary", key="show_binary", on_change=_binary_changed
            )
            cols[4].checkbox(
                "Show Subtraction", key="show_subtraction", on_change=_subtraction_changed
            )
            st.session_state.opacity = cols[5].slider(
                "Mask Opacity", 0.0, 1.0, float(st.session_state.opacity), 0.05
            )
    with r3b:
        with st.container(border=True):
            toolbar_group("EXPORT")
            a, b = st.columns(2)
            if a.button("Save & Next [1]", use_container_width=True, type="primary"):
                if st.session_state.preview_mask is not None:
                    st.session_state.pending_save = True
                else:
                    save_current_case()
                    st.rerun()
            if b.button("Exit [0]", use_container_width=True):
                st.session_state.exited = True
                st.rerun()


def render_active_controls() -> None:
    panel = st.session_state.active_panel
    image = st.session_state.image
    mask = st.session_state.mask

    if panel == "load_mask":
        with st.container(border=True):
            st.markdown("**LOAD MASK**")
            server_path = st.text_input("Mask path on server", key="server_mask_path")
            upload = st.file_uploader(
                "Or upload a mask from this browser",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                key="mask_upload",
            )
            a, b = st.columns(2)
            if a.button("Load selected mask", type="primary", use_container_width=True):
                try:
                    if upload is not None:
                        raw = np.array(Image.open(BytesIO(upload.getvalue())).convert("L"))
                    elif server_path.strip():
                        raw = np.array(Image.open(Path(server_path).expanduser()).convert("L"))
                    else:
                        raise ValueError("Choose an uploaded mask or provide a server mask path.")
                    m = raw.astype(np.float32) / 255.0 > 0.5
                    if m.shape != image.shape[:2]:
                        raise ValueError("Mask dimensions must match the current image.")
                    commit_mask(m, "Load Mask")
                    st.session_state.active_panel = "segmentation"
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if b.button("Close", use_container_width=True):
                st.session_state.active_panel = "segmentation"
                st.rerun()

    elif panel == "threshold":
        with st.container(border=True):
            st.markdown("**THRESHOLD**")
            c1, c2, c3, c4 = st.columns(4)
            method = c1.selectbox("Method", ["Global", "Manual", "Adaptive"], key="threshold_method")
            threshold = c2.number_input("Threshold", 0.0, 1.0, 0.5, 0.01, key="threshold_value")
            sensitivity = c3.number_input("Sensitivity", 0.0, 1.0, 0.5, 0.01, key="adapt_sensitivity")
            polarity = c4.selectbox("Foreground Polarity", ["bright", "dark"], key="foreground_polarity")
            a, b, c = st.columns(3)
            if a.button("Preview", key="threshold_preview", use_container_width=True):
                def op():
                    if method == "Global":
                        return alg.global_threshold(image)
                    if method == "Manual":
                        return alg.manual_threshold(image, threshold)
                    return alg.adaptive_threshold(image, sensitivity, polarity)
                m = run_operation(op)
                if m is not None:
                    set_preview(m, f"Threshold ({method})")
                    st.rerun()
            if b.button("Apply", key="threshold_apply", use_container_width=True):
                apply_preview(); st.rerun()
            if c.button("Close", key="threshold_close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "flood":
        with st.container(border=True):
            st.markdown("**FLOOD FILL** — click one seed point on the image")
            c1, c2, c3 = st.columns(3)
            c1.selectbox("Distance Metric", ["Euclidean", "Geodesic"], key="flood_metric")
            c2.number_input("Tolerance", 0.0001, 1.0, 0.05, 0.01, key="flood_tol")
            if c3.button("Apply preview", use_container_width=True):
                apply_preview(); st.rerun()
            if st.button("Close Flood Fill", use_container_width=False):
                close_preview(); st.rerun()

    elif panel == "morphology":
        with st.container(border=True):
            st.markdown("**MORPHOLOGY**")
            c = st.columns(6)
            op = c[0].selectbox("Operation", ["dilate", "erode", "open", "close"], key="morph_op")
            shape = c[1].selectbox("Shape", ["disk", "diamond", "line", "octagon", "square", "rectangle"], key="morph_shape")
            radius = c[2].number_input("Radius", 0, 100, 3, key="morph_radius")
            length = c[3].number_input("Length", 1, 200, 3, key="morph_length")
            angle = c[4].number_input("Degrees", -360.0, 360.0, 0.0, key="morph_angle")
            width = c[5].number_input("Width", 1, 200, 3, key="morph_width")
            a, b, cc = st.columns(3)
            if a.button("Preview", key="morph_preview", use_container_width=True):
                def mop():
                    se = alg.structure_element(shape, radius, length, angle, width)
                    return alg.morphology_operation(mask, op, se)
                m = run_operation(mop)
                if m is not None:
                    set_preview(m, f"Morphology: {op} ({shape})")
                    st.rerun()
            if b.button("Apply", key="morph_apply", use_container_width=True):
                apply_preview(); st.rerun()
            if cc.button("Close", key="morph_close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "active":
        with st.container(border=True):
            st.markdown("**ACTIVE CONTOURS**")
            c1, c2 = st.columns(2)
            method = c1.selectbox("Method", ["Chan-Vese", "edge"], key="active_method")
            iterations = c2.number_input("Iterations", 1, 5000, 100, key="active_iterations")
            a, b, c = st.columns(3)
            if a.button("Preview", key="active_preview", use_container_width=True):
                m = run_operation(lambda: alg.active_contour(image, mask, iterations, method, st.session_state.use_texture))
                if m is not None:
                    set_preview(m, f"Active Contours ({method})")
                    st.rerun()
            if b.button("Apply", key="active_apply", use_container_width=True):
                apply_preview(); st.rerun()
            if c.button("Close", key="active_close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "circles":
        with st.container(border=True):
            st.markdown("**FIND CIRCLES**")
            c = st.columns(4)
            min_d = c[0].number_input("Min Diameter", 2, 5000, 50, key="circle_min")
            max_d = c[1].number_input("Max Diameter", 2, 5000, 150, key="circle_max")
            polarity = c[2].selectbox("Object Polarity", ["bright", "dark"], key="circle_polarity")
            sens = c[3].number_input("Sensitivity", 0.01, 0.99, 0.85, 0.01, key="circle_sens")
            a, b, cc = st.columns(3)
            if a.button("Preview", key="circles_preview", use_container_width=True):
                def cop():
                    minr = max(1, int(min_d) // 2)
                    maxr = max(minr, int(max_d) // 2)
                    centers, radii = alg.find_circles(image, minr, maxr, polarity, sens)
                    return alg.circles_to_mask(image.shape[:2], centers, radii), len(radii)
                out = run_operation(cop)
                if out is not None:
                    m, n = out
                    set_preview(m, f"Find Circles ({n} circles)")
                    st.rerun()
            if b.button("Apply", key="circles_apply", use_container_width=True):
                apply_preview(); st.rerun()
            if cc.button("Close", key="circles_close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "graphcut":
        with st.container(border=True):
            st.markdown("**GRAPH CUT / LAZY SNAPPING**")
            c = st.columns([1.1, 0.8, 1.3, 0.8, 0.8])
            st.session_state.graph_mark_mode = c[0].selectbox(
                "Draw", ["Foreground", "Background", "Erase"],
                index=["Foreground", "Background", "Erase"].index(st.session_state.graph_mark_mode),
                key="graph_mark_mode_widget",
            )
            c[1].number_input("Marker Size", 1, 51, 9, key="marker_size")
            c[2].slider("Superpixel Density", 0, 100, 50, key="superpixel_density")
            st.session_state.show_superpixels = c[3].checkbox(
                "Show Boundaries", value=st.session_state.show_superpixels, key="show_superpixels_widget"
            )
            if c[4].button("Preview", use_container_width=True):
                out = run_operation(lambda: alg.lazy_snapping(
                    image,
                    st.session_state.graph_fg,
                    st.session_state.graph_bg,
                    requested_superpixels(),
                    st.session_state.use_texture,
                ))
                if out is not None:
                    m, labels = out
                    st.session_state.graph_labels = labels
                    set_preview(m, "Graph Cut / Lazy Snapping")
                    st.rerun()
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Clear Foreground", use_container_width=True):
                st.session_state.graph_fg[:] = 0; st.session_state.canvas_epoch += 1; st.rerun()
            if c2.button("Clear Background", use_container_width=True):
                st.session_state.graph_bg[:] = 0; st.session_state.canvas_epoch += 1; st.rerun()
            if c3.button("Apply", use_container_width=True):
                apply_preview(); st.rerun()
            if c4.button("Close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "grabcut":
        with st.container(border=True):
            st.markdown("**GRABCUT**")
            c = st.columns([0.9, 1.0, 1.1, 0.8])
            phase = c[0].selectbox("Mode", ["ROI", "Scribbles"], key="grab_phase_widget")
            st.session_state.grab_phase = phase
            if phase == "ROI":
                st.session_state.grab_roi_style = c[1].selectbox(
                    "ROI", ["Rectangle", "Polygon"], key="grab_roi_style_widget"
                )
                c[2].caption("Draw ROI on image")
            else:
                st.session_state.grab_mark_mode = c[1].selectbox(
                    "Draw", ["Foreground", "Background", "Erase"], key="grab_mark_mode_widget"
                )
                c[2].caption("Foreground / background scribbles")
            if c[3].button("Preview", use_container_width=True):
                if st.session_state.grab_roi is None:
                    st.error("Draw a GrabCut rectangle or polygon ROI first.")
                else:
                    m = run_operation(lambda: alg.grabcut(
                        image,
                        st.session_state.grab_roi,
                        st.session_state.grab_fg,
                        st.session_state.grab_bg,
                    ))
                    if m is not None:
                        set_preview(mask | m, "GrabCut (add to mask)")
                        st.rerun()
            c1, c2, c3 = st.columns(3)
            if c1.button("Clear marks/ROI", use_container_width=True):
                st.session_state.grab_fg[:] = 0
                st.session_state.grab_bg[:] = 0
                st.session_state.grab_roi = None
                st.session_state.canvas_epoch += 1
                st.rerun()
            if c2.button("Apply", use_container_width=True):
                apply_preview(); st.rerun()
            if c3.button("Close", use_container_width=True):
                close_preview(); st.rerun()

    elif panel == "manual":
        tool = st.session_state.manual_tool
        with st.container(border=True):
            st.markdown(f"**{tool.upper()}**")
            if tool == "Freehand":
                st.caption("Draw a closed freehand region. Releasing the mouse adds the filled region to the mask.")
            elif tool == "Ellipse":
                st.caption("Drag the ellipse bounding box. The committed mask is an ellipse inside that box.")
            elif tool == "Polygon":
                st.caption("Click vertices and close the polygon using the canvas polygon control.")
            else:
                st.caption("Draw the region on the image; it is added to the current mask.")

    elif panel == "pan":
        with st.container(border=True):
            st.markdown("**PAN**")
            c1, c2 = st.columns(2)
            st.session_state.pan_x = c1.slider("Horizontal", -1.0, 1.0, float(st.session_state.pan_x), 0.05)
            st.session_state.pan_y = c2.slider("Vertical", -1.0, 1.0, float(st.session_state.pan_y), 0.05)
            st.caption("Pan affects the view when the image is zoomed beyond the viewport.")


def canvas_mode_for_panel() -> tuple[bool, str, str, str, int]:
    panel = st.session_state.active_panel
    if panel == "manual":
        tool = st.session_state.manual_tool
        mode = {"Freehand": "freedraw", "Rectangle": "rect", "Ellipse": "rect", "Polygon": "polygon"}[tool]
        return True, mode, "#FFFF00", "rgba(255,255,0,0.30)", 3
    if panel == "flood":
        return True, "point", "#FFFF00", "rgba(255,255,0,0.65)", 2
    if panel == "graphcut":
        mode_name = st.session_state.graph_mark_mode
        color = {"Foreground": "#00FF00", "Background": "#FF0000", "Erase": "#FFFF00"}[mode_name]
        width = max(1, int(st.session_state.get("marker_size", 9)))
        return True, "freedraw", color, "rgba(0,0,0,0)", width
    if panel == "grabcut":
        if st.session_state.grab_phase == "ROI":
            mode = "rect" if st.session_state.grab_roi_style == "Rectangle" else "polygon"
            return True, mode, "#FFFF00", "rgba(255,255,0,0.25)", 3
        mode_name = st.session_state.grab_mark_mode
        color = {"Foreground": "#00FF00", "Background": "#FF0000", "Erase": "#FFFF00"}[mode_name]
        width = max(1, int(st.session_state.get("marker_size", 9)))
        return True, "freedraw", color, "rgba(0,0,0,0)", width
    return False, "freedraw", "#FFFF00", "rgba(255,255,0,0.30)", 3


def handle_canvas_result(result, background: Image.Image, tr: ViewTransform) -> None:
    if result is None:
        return
    sig = _result_signature(result)
    if not sig or sig == st.session_state.last_canvas_signature:
        return
    data = getattr(result, "json_data", None) or {}
    if not data.get("objects"):
        st.session_state.last_canvas_signature = sig
        return

    panel = st.session_state.active_panel
    view_mask = canvas_alpha_mask(result, background)
    if view_mask is None:
        return

    if panel == "manual":
        tool = st.session_state.manual_tool
        if tool == "Freehand":
            filled = freehand_fill_mask(result)
            if filled is not None:
                view_mask = filled
            else:
                contours, _ = cv2.findContours(
                    view_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                tmp = np.zeros_like(view_mask, np.uint8)
                if contours:
                    cv2.drawContours(tmp, contours, -1, 1, -1)
                    view_mask = tmp.astype(bool)
        elif tool == "Ellipse":
            ellipse_view = ellipse_from_rect_json(result)
            if ellipse_view is not None:
                view_mask = ellipse_view
        addition = view_mask_to_image(view_mask, tr)
        st.session_state.last_canvas_signature = sig
        if addition.any():
            commit_mask(st.session_state.mask | addition, tool)
            st.rerun()

    elif panel == "flood":
        ys, xs = np.nonzero(view_mask)
        if xs.size:
            seed = canvas_point_to_image(float(xs.mean()), float(ys.mean()), tr)
            metric = st.session_state.get("flood_metric", "Euclidean")
            tol = float(st.session_state.get("flood_tol", 0.05))
            fn = alg.flood_fill_euclidean if metric == "Euclidean" else alg.flood_fill_geodesic
            add = run_operation(lambda: fn(st.session_state.image, seed, tol, st.session_state.use_texture))
            st.session_state.last_canvas_signature = sig
            if add is not None:
                set_preview(st.session_state.mask | add, f"Flood Fill ({metric})")
                st.session_state.canvas_epoch += 1
                st.rerun()

    elif panel == "graphcut":
        draw = view_mask_to_image(view_mask, tr).astype(np.uint8)
        mode = st.session_state.graph_mark_mode
        if mode == "Foreground":
            st.session_state.graph_fg[draw.astype(bool)] = 1
            st.session_state.graph_bg[draw.astype(bool)] = 0
        elif mode == "Background":
            st.session_state.graph_bg[draw.astype(bool)] = 1
            st.session_state.graph_fg[draw.astype(bool)] = 0
        else:
            st.session_state.graph_fg[draw.astype(bool)] = 0
            st.session_state.graph_bg[draw.astype(bool)] = 0
        st.session_state.last_canvas_signature = sig
        st.session_state.canvas_epoch += 1
        st.rerun()

    elif panel == "grabcut":
        draw = view_mask_to_image(view_mask, tr)
        if st.session_state.grab_phase == "ROI":
            st.session_state.grab_roi = draw
        else:
            mode = st.session_state.grab_mark_mode
            d = draw.astype(bool)
            if mode == "Foreground":
                st.session_state.grab_fg[d] = 1
                st.session_state.grab_bg[d] = 0
            elif mode == "Background":
                st.session_state.grab_bg[d] = 1
                st.session_state.grab_fg[d] = 0
            else:
                st.session_state.grab_fg[d] = 0
                st.session_state.grab_bg[d] = 0
        st.session_state.last_canvas_signature = sig
        st.session_state.canvas_epoch += 1
        st.rerun()


def render_left_panel() -> None:
    workflow: CSVWorkflow = st.session_state.workflow
    case: Case = st.session_state.case
    st.markdown("### Data Browser")
    if case is not None:
        st.markdown(f"**Row {case.row_index + 1}/{len(workflow.df)}**")
        st.caption("Image")
        st.code(str(case.image_path), language=None)
        st.caption("Mask")
        st.code(str(case.mask_save_path), language=None)
    st.divider()
    st.markdown("### History")

    history = st.session_state.history
    if history:
        options = list(range(len(history)))
        idx = st.radio(
            "History states",
            options,
            index=int(st.session_state.history_index),
            format_func=lambda i: f"{i + 1}. {history[i][0]}",
            key=f"history_{st.session_state.history_revision}",
            label_visibility="collapsed",
        )
        if idx != st.session_state.history_index:
            st.session_state.history_index = int(idx)
            st.session_state.mask = history[idx][1].copy()
            st.session_state.preview_mask = None
            st.session_state.preview_desc = ""

        a, b = st.columns(2)
        if a.button("Undo", use_container_width=True):
            undo_history(); st.rerun()
        if b.button("Delete", use_container_width=True):
            delete_history_selection(int(idx)); st.rerun()


def render_center() -> None:
    if st.session_state.case_warning:
        st.warning(st.session_state.case_warning)
        st.session_state.case_warning = ""

    rgb = _compose_image_rgb()
    background, tr = render_view(rgb)
    enabled, drawing_mode, stroke_color, fill_color, stroke_width = canvas_mode_for_panel()

    if enabled:
        result = _canvas_call(
            fill_color=fill_color,
            stroke_width=max(1, int(round(stroke_width * tr.scale))) if drawing_mode == "freedraw" else stroke_width,
            stroke_color=stroke_color,
            background_image=background,
            update_streamlit=True,
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode=drawing_mode,
            point_display_radius=6,
            return_image_data=True,
            background_image_fit="stretch",
            max_display_height=CANVAS_HEIGHT,
            key=(
                f"segment_canvas_{st.session_state.case.row_index}_"
                f"{st.session_state.active_panel}_{st.session_state.canvas_epoch}_"
                f"{st.session_state.manual_tool}_{st.session_state.graph_mark_mode}_"
                f"{st.session_state.grab_phase}_{st.session_state.grab_mark_mode}_"
                f"{st.session_state.grab_roi_style}"
            ),
        )
        handle_canvas_result(result, background, tr)
    else:
        st.image(background, width=CANVAS_WIDTH)

    s1, s2 = st.columns([3, 1])
    s1.caption(st.session_state.status)
    case: Case = st.session_state.case
    workflow: CSVWorkflow = st.session_state.workflow
    if case is not None:
        s2.caption(f"Row {case.row_index + 1} of {len(workflow.df)}")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 100%; padding-top: .55rem; padding-bottom: .5rem; padding-left: .8rem; padding-right: .8rem;}
        .ribbon-label {font-size: 0.84rem; font-weight: 700; margin: 0 0 .15rem .15rem;}
        .tool-title {font-size: .78rem; font-weight: 700; letter-spacing: .02em; margin-bottom: .25rem;}
        div[data-testid="stVerticalBlockBorderWrapper"] {background: #f7f7f7;}
        div[data-testid="stButton"] button {min-height: 2.15rem; padding: .25rem .45rem;}
        div[data-testid="stCode"] {font-size: .75rem;}
        [data-testid="stSidebar"] {min-width: 250px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Python Image Segmenter — Headless", layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    try:
        args = parse_cli_args()
        csv_path = resolve_csv_path(args)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        init_session(csv_path)
    except Exception as exc:
        st.error(str(exc))
        st.code('streamlit run app.py -- --path_to_csv "/data/Arka/python_imagesegmenter/" --csv_file "sample_cases.csv"', language="bash")
        st.stop()

    if st.session_state.exited:
        st.info("Current Streamlit segmentation session was closed without saving the current row. The CSV row remains unflagged.")
        st.stop()

    if st.session_state.complete:
        st.success("All unflagged cases have been processed.")
        st.stop()

    st.markdown("#### Simple2DSeg  — Python Image Segmenter")
    render_toolbar()

    if st.session_state.pending_save:
        st.warning("A preview is active. Apply the preview before saving this row?")
        a, b = st.columns([1, 1])
        if a.button("Apply preview & Save", type="primary", use_container_width=True):
            apply_preview()
            st.session_state.pending_save = False
            save_current_case()
            st.rerun()
        if b.button("Cancel save", use_container_width=True):
            st.session_state.pending_save = False
            st.rerun()

    render_active_controls()

    left, center = st.columns([0.21, 0.79], gap="small")
    with left:
        render_left_panel()
    with center:
        render_center()


if __name__ == "__main__":
    main()
