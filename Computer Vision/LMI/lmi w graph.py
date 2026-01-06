# hsv_roi_grid_dualcolor_lmi_best_with_plot.py
# -------------------------------------------------------------------
# • File dialog to pick video → draw polygon ROI (click to add, drag to adjust, 'c' to start)
# • Fixed HSV thresholds (edit below)
# • 5×8 grid inside ROI; per-tile % overlays for both colors
# • Weighted Lacey MI (known tracer fraction); shows live on video
# • Resizable windows with scale-to-screen; ROI clicks mapped back to original pixels
# • Logs time vs LMI; plots at end; saves CSV + PNG
# -------------------------------------------------------------------

from __future__ import annotations
import os, time, math, csv
from typing import Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= USER SETTINGS ========================= #
# ---- Segmentation (OpenCV HSV ranges: H ∈ [0,179], S,V ∈ [0,255]) ----
# Set so C1 and C2 isolate your two colors. Choose which color is the TRACER below.
COLOR1_HSV_LO = (0,   0,  80)    # white (example)
COLOR1_HSV_HI = (179, 255, 255)
COLOR2_HSV_LO = (0,   0,    0)    # black (example)
COLOR2_HSV_HI = (179, 255, 80)

TRACER_IS = "C2"                  # "C1" or "C2" (which mask is the tracer for LMI)
KNOWN_TRACER_FRACTION = 0.125      # p (e.g., 20% tracer in the batch)

# ---- Grid & overlays ----
GRID_W, GRID_H = 5, 8             # columns × rows
GRID_COLOR     = (0, 255, 255)
GRID_THICKNESS = 2
MIN_TILE_COVER_FRAC = 0.15        # skip tiles with little ROI coverage

# Overlay tinting (semi-transparent)
TINT1_BGR  = (0, 255, 0)          # C1 tint (green)
TINT2_BGR  = (255, 0, 255)        # C2 tint (magenta)
TINT_ALPHA = 0.05                 # 0..1

# ---- Playback pacing ----
SPEED_MULTIPLIER = 1.0            # 1.0 = real-time, change here (not on the fly)
DROP_FRAMES_TO_CATCH_UP = True
FALLBACK_FPS = 30.0

# ---- Display fit & clarity ----
FIT_TO_SCREEN = False              # scale display copies to fit screen (processing stays native)
DEFAULT_MAX_W, DEFAULT_MAX_H = 1600, 900
SHOW_LMI = True

# ---- Optional: mask denoising (OFF by default; set to >0 to enable) ----
DENOISE = {
    "median_ksize": 0,   # 0/1 = off; try 3 for light salt/pepper
    "open_iters":   0,   # small speck removal (erode→dilate)
    "close_iters":  0,   # fill tiny holes     (dilate→erode)
    "min_blob_area": 0   # remove tiny blobs (< area); try 50–200 if needed
}
# ================================================================ #

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FG = (255, 255, 255)
LABEL_BG = (0, 0, 0)

# ---------------- Display helpers (scale-to-fit while keeping aspect) ---------------- #
def get_screen_size():
    """Return (screen_width, screen_height) in pixels; fall back if unavailable."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try: user32.SetProcessDPIAware()
        except Exception: pass
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return DEFAULT_MAX_W, DEFAULT_MAX_H

def scale_to_fit(img, max_w, max_h):
    """Downscale img to fit within (max_w,max_h). Returns (image_scaled, scale_factor)."""
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)  # only downscale
    if s < 1.0:
        new_size = (max(1, int(w * s)), max(1, int(h * s)))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), s
    return img, 1.0

def put_label(img, x, y, text, fg=LABEL_FG, bg=LABEL_BG, scale=0.7, thick=2, pad=3):
    x = int(x); y = int(y)
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thick)
    x0 = max(0, x - pad); y0 = max(0, y - th - pad)
    x1 = min(img.shape[1]-1, x + tw + pad); y1 = min(img.shape[0]-1, y + baseline + pad)
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness=cv2.FILLED)
    cv2.putText(img, text, (x, y), FONT, scale, fg, thick, cv2.LINE_AA)

# ---------------- File picker ---------------- #
def pick_video_path() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")]
        )
        root.update(); root.destroy()
        if not path or not os.path.exists(path): raise SystemExit("No valid video selected.")
        return path
    except Exception as e:
        raise SystemExit(f"File dialog error: {e}\nTip: install tkinter (e.g., conda install tk).")

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened(): cap = cv2.VideoCapture(path)
    return cap

# ---------------- ROI polygon (scale-aware mouse → original coords) ---------------- #
class PolygonROI:
    def __init__(self, win_name: str, pick_radius: int = 10):
        self.win_name = win_name
        self.points: List[Tuple[int,int]] = []   # stored in ORIGINAL image pixels
        self.closed = False
        self.drag_idx = -1
        self.pick_radius = pick_radius
        self.scale = 1.0                         # display → original scale

    def install(self):
        cv2.setMouseCallback(self.win_name, self._mouse)

    def set_scale(self, s: float):
        self.scale = max(1e-6, float(s))

    def _to_orig(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    def _nearest_idx(self, xo, yo):
        if not self.points: return -1
        pts = np.array(self.points, dtype=np.int32)
        d2 = (pts[:,0] - xo)**2 + (pts[:,1] - yo)**2
        j = int(np.argmin(d2))
        return j if d2[j] <= (self.pick_radius**2) else -1

    def _mouse(self, event, x, y, flags, param):
        xo, yo = self._to_orig(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            j = self._nearest_idx(xo, yo)
            if j >= 0: self.drag_idx = j
            elif not self.closed: self.points.append((xo, yo))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx >= 0:
            self.points[self.drag_idx] = (xo, yo)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1

    def reset(self):
        self.points.clear(); self.closed = False; self.drag_idx = -1

    def mask(self, shape):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if len(self.points) >= 3:
            cv2.fillPoly(mask, [np.array(self.points, dtype=np.int32)], 255)
        return mask

    def draw_overlay(self, img):
        out = img.copy()
        if len(self.points) >= 1:
            for i, p in enumerate(self.points):
                cv2.circle(out, p, 3, (0, 255, 0), -1, cv2.LINE_AA)
                if i > 0: cv2.line(out, self.points[i-1], p, (0, 255, 0), 2, cv2.LINE_AA)
        if self.closed and len(self.points) >= 3:
            cv2.polylines(out, [np.array(self.points, dtype=np.int32)], True, (0, 200, 255), 2, cv2.LINE_AA)
        return out

# ---------------- Grid helpers ---------------- #
def draw_grid_in_roi_bbox(img, polygon_pts, grid_w, grid_h, color=(120,120,120), thickness=1):
    if not polygon_pts or len(polygon_pts) < 3: return img
    H, W = img.shape[:2]
    poly = np.array(polygon_pts, dtype=np.int32)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [poly], 255)
    x, y, w, h = cv2.boundingRect(poly)
    grid_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, grid_w):
        xi = int(round(x + i * (w / grid_w)))
        cv2.line(grid_mask, (xi, y), (xi, y + h), 255, thickness)
    for j in range(1, grid_h):
        yj = int(round(y + j * (h / grid_h)))
        cv2.line(grid_mask, (x, yj), (x + w, yj), 255, thickness)
    grid_mask = cv2.bitwise_and(grid_mask, grid_mask, mask=roi_mask)
    out = img.copy(); out[grid_mask > 0] = color
    return out

# ---------------- Segmentation & optional denoise ---------------- #
def apply_hsv_masks(frame_bgr: np.ndarray, roi_mask: np.ndarray,
                    lo1: Tuple[int,int,int], hi1: Tuple[int,int,int],
                    lo2: Tuple[int,int,int], hi2: Tuple[int,int,int]):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw1 = cv2.inRange(hsv, np.array(lo1, dtype=np.uint8), np.array(hi1, dtype=np.uint8))
    raw2 = cv2.inRange(hsv, np.array(lo2, dtype=np.uint8), np.array(hi2, dtype=np.uint8))
    c1 = cv2.bitwise_and(raw1, raw1, mask=roi_mask)
    c2 = cv2.bitwise_and(raw2, raw2, mask=roi_mask)
    return c1, c2

def clean_mask(mask: np.ndarray) -> np.ndarray:
    m = mask
    if DENOISE["median_ksize"] and DENOISE["median_ksize"] > 1:
        m = cv2.medianBlur(m, DENOISE["median_ksize"])
    if DENOISE["open_iters"] and DENOISE["open_iters"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=DENOISE["open_iters"])
    if DENOISE["close_iters"] and DENOISE["close_iters"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=DENOISE["close_iters"])
    if DENOISE["min_blob_area"] and DENOISE["min_blob_area"] > 0:
        bin8 = (m > 0).astype(np.uint8)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, nlab):
            if stats[i, cv2.CC_STAT_AREA] >= DENOISE["min_blob_area"]:
                keep[labels == i] = 255
        m = keep
    return m

# ---------------- Tile analysis & Lacey MI ---------------- #
def analyze_tiles_tracer(sub_tracer: np.ndarray, valid_sub: np.ndarray,
                         grid_wh: Tuple[int, int], min_cover_frac: float = 0.0):
    """
    Compute per-tile tracer fraction and weighted stats.
    Returns:
      tiles: list of {rect, n, p_tracer}
      p_meas: measured tracer fraction over ROI (sanity check)
      p_w, s2_w: weighted mean and population weighted variance across tiles
      m_bar: average valid pixels per used tile
      used: number of tiles used
    """
    H, W = sub_tracer.shape
    gw, gh = grid_wh
    cell_w = max(1, W // gw)
    cell_h = max(1, H // gh)

    tiles: List[Dict] = []
    weights, p_list = [], []

    total_valid = int(valid_sub.sum())
    if total_valid == 0:
        return tiles, math.nan, math.nan, math.nan, 0.0, 0

    for j in range(gh):
        for i in range(gw):
            x0 = i * cell_w; y0 = j * cell_h
            x1 = W if i == gw - 1 else (i + 1) * cell_w
            y1 = H if j == gh - 1 else (j + 1) * cell_h

            vmask = valid_sub[y0:y1, x0:x1]
            n = int(vmask.sum())
            if n == 0 or (n / float((y1 - y0) * (x1 - x0)) < min_cover_frac):
                continue

            p_t = float(sub_tracer[y0:y1, x0:x1][vmask].mean()) / 255.0
            tiles.append({"rect": (x0, y0, x1, y1), "n": n, "p_tracer": p_t})
            weights.append(n); p_list.append(p_t)

    used = len(tiles)
    if used < 2:
        return tiles, float(sub_tracer[valid_sub].mean()) / 255.0, math.nan, math.nan, 0.0, used

    w = np.array(weights, dtype=np.float64)
    p = np.array(p_list, dtype=np.float64)
    wsum = float(w.sum())
    p_w  = float((w * p).sum() / wsum)
    s2_w = float((w * (p - p_w) ** 2).sum() / wsum)  # population variance
    m_bar = float(wsum / used)
    p_meas = float(sub_tracer[valid_sub].mean()) / 255.0

    return tiles, p_meas, p_w, s2_w, m_bar, used

def lacey_index_weighted_known_p(s2_w: float, p_known: float, m_bar: float) -> float:
    """
    L = (s0^2 - s_w^2) / (s0^2 - s_r^2),
    s0^2 = p (1-p),  s_r^2 = s0^2 / m_bar
    """
    if not np.isfinite(s2_w) or not np.isfinite(p_known) or m_bar <= 1:
        return float("nan")
    s0_sq = p_known * (1.0 - p_known)
    sr_sq = s0_sq / m_bar
    denom = (s0_sq - sr_sq)
    if denom <= 0:
        return float("nan")
    L = (s0_sq - s2_w) / denom
    return float(np.clip(L, 0.0, 1.0))

def format_ts(t: float) -> str:
    if not np.isfinite(t): return "n/a"
    m, s = divmod(float(t), 60.0)
    return f"{int(m):02d}:{s:05.2f}"

# ------------------------------- Main ------------------------------- #
def main():
    path = pick_video_path()
    cap = open_video(path)
    if not cap.isOpened(): raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS

    ok, first = cap.read()
    if not ok: raise SystemExit("Could not read first frame.")

    # ROI window (resizable; scale-to-fit display; map clicks back to original)
    draw_win = "ROI (draw polygon, drag points, 'c' to start)"
    cv2.namedWindow(draw_win, cv2.WINDOW_NORMAL)
    roi = PolygonROI(draw_win); roi.install()

    help_lines = [
        "Left-click: add points | drag vertices to adjust",
        "z: undo | r: reset | c: start | q/ESC: quit",
        f"Grid: {GRID_W}x{GRID_H} | Speed: {SPEED_MULTIPLIER:.2f}×",
    ]

    roi_disp_scale = 1.0
    while True:
        vis = roi.draw_overlay(first)              # original-size overlay
        ytxt = 24
        for line in help_lines:
            put_label(vis, 10, ytxt, line, scale=0.6, thick=2); ytxt += 26

        if FIT_TO_SCREEN:
            sw, sh = get_screen_size()
            vis_disp, s = scale_to_fit(vis, sw - 40, sh - 80)
            roi_disp_scale = s
            roi.set_scale(roi_disp_scale)          # map clicks → original
            cv2.imshow(draw_win, vis_disp)
            cv2.moveWindow(draw_win, 20, 20)
        else:
            roi_disp_scale = 1.0
            roi.set_scale(roi_disp_scale)
            cv2.imshow(draw_win, vis)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')): cap.release(); cv2.destroyAllWindows(); return
        elif k == ord('z') and roi.points: roi.points.pop()
        elif k == ord('r'): roi.reset()
        elif k == ord('c') and len(roi.points) >= 3:
            roi.closed = True; break

    roi_mask = roi.mask(first.shape)
    poly = np.array(roi.points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(poly)
    valid_mask_full = roi_mask.astype(bool)

    # Playback window
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    play_win = "Playback (LMI)"
    cv2.namedWindow(play_win, cv2.WINDOW_NORMAL)

    start_wall = time.perf_counter()
    frame_idx = 0
    tracer_is_c1 = (TRACER_IS.upper() == "C1")

    # Time-series log
    times_s: List[float] = []
    lmi_vals: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        # Segmentation
        c1, c2 = apply_hsv_masks(frame, roi_mask,
                                 COLOR1_HSV_LO, COLOR1_HSV_HI,
                                 COLOR2_HSV_LO, COLOR2_HSV_HI)
        # Optional denoise
        if any([DENOISE["median_ksize"] > 1, DENOISE["open_iters"] > 0,
                DENOISE["close_iters"] > 0, DENOISE["min_blob_area"] > 0]):
            c1 = clean_mask(c1)
            c2 = clean_mask(c2)

        tracer = c1 if tracer_is_c1 else c2

        # Left: tinted original; outside ROI dimmed
        left = frame.copy()
        outside = cv2.bitwise_not(roi_mask)
        left[outside > 0] = (left[outside > 0] * 0.25).astype(np.uint8)

        if TINT_ALPHA > 0:
            t1 = np.zeros_like(left); t1[c1 > 0] = TINT1_BGR
            t2 = np.zeros_like(left); t2[c2 > 0] = TINT2_BGR
            left = cv2.addWeighted(left, 1 - TINT_ALPHA, t1, TINT_ALPHA, 0)
            left = cv2.addWeighted(left, 1 - TINT_ALPHA, t2, TINT_ALPHA, 0)

        if len(roi.points) >= 3:
            cv2.polylines(left, [poly], True, (0, 200, 255), 2, cv2.LINE_AA)
        left = draw_grid_in_roi_bbox(left, roi.points, GRID_W, GRID_H, GRID_COLOR, GRID_THICKNESS)

        # Right: mask channels for debugging
        right = np.zeros_like(frame)
        right[:, :, 1] = c1
        right[:, :, 2] = c2
        right = draw_grid_in_roi_bbox(right, roi.points, GRID_W, GRID_H, GRID_COLOR, GRID_THICKNESS)

        # Tile analysis & LMI
        sub_tr  = tracer[y:y+h, x:x+w]
        valid_sub = valid_mask_full[y:y+h, x:x+w]
        tiles, p_meas, p_w, s2_w, m_bar, used = analyze_tiles_tracer(
            sub_tr, valid_sub, (GRID_W, GRID_H), MIN_TILE_COVER_FRAC
        )
        L = lacey_index_weighted_known_p(s2_w, KNOWN_TRACER_FRACTION, m_bar)

        # Per-tile label overlays (for verification)
        """ for t in tiles:
            x0, y0, x1, y1 = t["rect"]
            gx0, gy0 = x + x0 + 6, y + y0 + 6
            base = min(max(12, x1 - x0), max(12, y1 - y0))
            fscale = max(0.5, min(1.2, base / 120.0))
            thick  = max(1, int(round(1.5 * fscale)))
            vmask = valid_sub[y0:y1, x0:x1]
            if vmask.sum() > 0:
                t_c1 = float(c1[y+y0:y+y1, x+x0:x+x1][vmask].mean()) / 255.0
                t_c2 = float(c2[y+y0:y+y1, x+x0:x+x1][vmask].mean()) / 255.0
            else:
                t_c1 = t_c2 = 0.0
            put_label(left, gx0, gy0 + int(18 * fscale), f"G {int(round(100*t_c1)):3d}%",
                      fg=(20,255,20), bg=LABEL_BG, scale=fscale, thick=thick)
            put_label(left, gx0, gy0 + int(38 * fscale), f"R {int(round(100*t_c2)):3d}%",
                      fg=(255,20,255), bg=LABEL_BG, scale=fscale, thick=thick)
 """
        # Header overlays
        if SHOW_LMI:
            which = "C1" if tracer_is_c1 else "C2"
            put_label(left, 18, 36,
                      f"Lacey MI (weighted, tracer={which}, p_known={KNOWN_TRACER_FRACTION:.2f}): "
                      f"{(L if np.isfinite(L) else float('nan')):.3f}",
                      scale=0.7, thick=2)
            put_label(left, 18, 66, f"Measured tracer p̂={p_meas:.3f} (sanity; not used)",
                      scale=0.7, thick=2)

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_elapsed = pos_ms/1000.0 if pos_ms and pos_ms > 0 else (frame_idx/(fps if fps>0 else FALLBACK_FPS))
        put_label(left, 18, 96, f"t = {int(t_elapsed//60):02d}:{(t_elapsed%60):05.2f}",
                  scale=0.7, thick=2)

        # Log time series
        times_s.append(float(t_elapsed))
        lmi_vals.append(float(L) if np.isfinite(L) else np.nan)

        stacked = stacked = np.vstack([left, right])

        # Fit-to-screen display
        if FIT_TO_SCREEN:
            sw, sh = get_screen_size()
            DISPLAY_SCALE=0.65
            disp = cv2.resize(stacked,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE,interpolation=cv2.INTER_AREA)
            cv2.imshow(play_win, disp); cv2.moveWindow(play_win, 40, 40)
        else:
            cv2.imshow(play_win, stacked)

        # Real-time pacing
        target = frame_idx / (fps * SPEED_MULTIPLIER)
        now = time.perf_counter() - start_wall
        wait_ms = int(max(1, (target - now) * 1000))
        if DROP_FRAMES_TO_CATCH_UP and wait_ms < 1:
            while True:
                now = time.perf_counter() - start_wall
                ahead = now - (frame_idx / (fps * SPEED_MULTIPLIER))
                if ahead < 0: break
                if not cap.grab(): break
                frame_idx += 1
            wait_ms = 1

        if cv2.waitKey(wait_ms) & 0xFF in (27, ord('q')):
            break

    cap.release(); cv2.destroyAllWindows()

    # ---------------- Plot + Save time series ---------------- #
    t_arr = np.array(times_s, dtype=float)
    L_arr = np.array(lmi_vals, dtype=float)

    # Save CSV
    csv_path = "lmi_time_series.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["time_s", "lmi"])
        for t, l in zip(t_arr, L_arr): writer.writerow([t, l])

    # Plot
    plt.figure(figsize=(8, 4.2))
    plt.plot(t_arr, L_arr, linewidth=2)
    plt.ylim(0, 1.05); plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)"); plt.ylabel("Lacey Mixing Index")
    plt.title("Lacey Mixing Index vs Time")
    # Optional target line (uncomment if you want a threshold)
    # target = 0.97
    # plt.axhline(target, linestyle="--"); plt.text(t_arr.max()*0.02, target+0.02, f"Target {target:.2f}")
    png_path = "lmi_time_series.png"
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.show()

    print(f"\nSaved:\n  CSV -> {os.path.abspath(csv_path)}\n  PNG -> {os.path.abspath(png_path)}")

if __name__ == "__main__":
    main()
