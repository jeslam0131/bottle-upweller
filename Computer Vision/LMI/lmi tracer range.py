# hsv_roi_grid_dualcolor_lmi_entropy_gated_with_plot.py
# -------------------------------------------------------------------
# • File dialog → draw polygon ROI (click to add, drag to adjust, 'c' to start)
# • Fixed HSV thresholds (edit below)
# • 5×8 grid inside ROI; per-tile % overlays for both colors
# • Two classic indices:
#     1) Lacey (variance about KNOWN tracer fraction p)
#     2) Shannon Entropy mixing index (histogram of tile fractions)
# • Gating (coverage + spatial spread + hysteresis) to avoid early false "mixed"
# • Resizable windows with scale-to-screen; ROI clicks mapped to original pixels
# • Logs time vs indices; always plots at end; saves CSV + PNG safely
# -------------------------------------------------------------------

from __future__ import annotations
import os, time, math, csv
from typing import Tuple, List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= USER SETTINGS ========================= #
# ---- Segmentation (OpenCV HSV: H ∈ [0,179], S,V ∈ [0,255]) ----
# Tune these so C1 and C2 isolate your two colors. Choose TRACER below.
COLOR1_HSV_LO = (0,   0, 70)     # example: white-ish
COLOR1_HSV_HI = (179, 255, 255)
COLOR2_HSV_LO = (0,   0,   0)     # example: black-ish
COLOR2_HSV_HI = (179, 255,70)

TRACER_IS = "C2"                   # "C1" or "C2" (which mask is the tracer)
KNOWN_TRACER_FRACTION = 0.25    # p (e.g., 20% tracer in the batch)

# ---- Grid & overlays ----
GRID_W, GRID_H = 5, 8              # columns × rows
GRID_COLOR     = (0, 255, 255)
GRID_THICKNESS = 2
MIN_TILE_COVER_FRAC = 0.15      # skip tiles with little ROI coverage

# Overlay tinting (semi-transparent)
TINT1_BGR  = (0, 255, 0)           # C1 tint (green)
TINT2_BGR  = (255, 0, 255)         # C2 tint (magenta)
TINT_ALPHA = 0.05                  # 0..1

# ---- Playback pacing ----
SPEED_MULTIPLIER = 1.0             # 1.0 = real-time, change here (not on the fly)
DROP_FRAMES_TO_CATCH_UP = True
FALLBACK_FPS = 30.0

# ---- Display fit & clarity ----
FIT_TO_SCREEN = True               # scale display copies to fit screen (processing stays native)
DEFAULT_MAX_W, DEFAULT_MAX_H = 1600, 900
SHOW_OVERLAYS = True

# ---- Shannon Entropy bins ----
ENTROPY_BINS = 20                  # histogram bins over [0,1]

# ---- Gating to avoid early false "mixed" (coverage / spread / hysteresis) ----
COVERAGE_GATE = 0.3               # require p_hat_roi / KNOWN_TRACER_FRACTION >= 0.70
TILE_MIN_FRAC = 0.02               # a tile "has tracer" if p_i >= 0.02
SPREAD_TILES_MIN = 8               # require ≥ this many tiles with tracer
HYST_FRAMES = 10                   # gate must hold for N consecutive frames
# ================================================================ #

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FG = (255, 255, 255)
LABEL_BG = (0, 0, 0)

# ---------- overwrite-safe saving ----------
def safe_path(stem_no_ext: str, ext: str) -> str:
    path = f"{stem_no_ext}{ext}"
    if not os.path.exists(path):
        return path
    k = 1
    while True:
        cand = f"{stem_no_ext}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def derive_save_stem(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.dirname(video_path)
    return os.path.join(d if d else ".", f"{base}_mix")

# ---------- display helpers ----------
def get_screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try: user32.SetProcessDPIAware()
        except Exception: pass
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return DEFAULT_MAX_W, DEFAULT_MAX_H

def scale_to_fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)  # only downscale
    if s < 1.0:
        return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s
    return img, 1.0

def put_label(img, x, y, text, fg=LABEL_FG, bg=LABEL_BG, scale=0.7, thick=2, pad=3):
    x = int(x); y = int(y)
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thick)
    x0 = max(0, x - pad); y0 = max(0, y - th - pad)
    x1 = min(img.shape[1]-1, x + tw + pad); y1 = min(img.shape[0]-1, y + baseline + pad)
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness=cv2.FILLED)
    cv2.putText(img, text, (x, y), FONT, scale, fg, thick, cv2.LINE_AA)

# ---------- file picker ----------
def pick_video_path() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files","*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files","*.*")]
        )
        root.update(); root.destroy()
        if not p or not os.path.exists(p): raise SystemExit("No valid video selected.")
        return p
    except Exception as e:
        raise SystemExit(f"File dialog error: {e}")

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened(): cap = cv2.VideoCapture(path)
    return cap

# ---------- ROI (scale-aware) ----------
class PolygonROI:
    def __init__(self, win_name: str, pick_radius: int = 10):
        self.win_name = win_name
        self.points: List[Tuple[int,int]] = []
        self.closed = False
        self.drag_idx = -1
        self.pick_radius = pick_radius
        self.scale = 1.0

    def install(self): cv2.setMouseCallback(self.win_name, self._mouse)
    def set_scale(self, s: float): self.scale = max(1e-6, float(s))
    def _to_orig(self, x, y): return int(round(x / self.scale)), int(round(y / self.scale))

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
        m = np.zeros(shape[:2], dtype=np.uint8)
        if len(self.points) >= 3:
            cv2.fillPoly(m, [np.array(self.points, dtype=np.int32)], 255)
        return m

    def draw_overlay(self, img):
        out = img.copy()
        if self.points:
            for i, p in enumerate(self.points):
                cv2.circle(out, p, 3, (0,255,0), -1, cv2.LINE_AA)
                if i>0: cv2.line(out, self.points[i-1], p, (0,255,0), 2, cv2.LINE_AA)
        if self.closed and len(self.points) >= 3:
            cv2.polylines(out, [np.array(self.points, dtype=np.int32)], True, (0,200,255), 2, cv2.LINE_AA)
        return out

# ---------- grid ----------
def draw_grid_in_roi_bbox(img, polygon_pts, gw, gh, color=(120,120,120), thickness=1):
    if not polygon_pts or len(polygon_pts) < 3: return img
    H, W = img.shape[:2]
    poly = np.array(polygon_pts, dtype=np.int32)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [poly], 255)
    x, y, w, h = cv2.boundingRect(poly)
    grid_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, gw):
        xi = int(round(x + i * (w / gw)))
        cv2.line(grid_mask, (xi, y), (xi, y + h), 255, thickness)
    for j in range(1, gh):
        yj = int(round(y + j * (h / gh)))
        cv2.line(grid_mask, (x, yj), (x + w, yj), 255, thickness)
    grid_mask = cv2.bitwise_and(grid_mask, grid_mask, mask=roi_mask)
    out = img.copy(); out[grid_mask > 0] = color
    return out

# ---------- segmentation ----------
def apply_hsv_masks(bgr: np.ndarray, roi_mask: np.ndarray,
                    lo1, hi1, lo2, hi2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw1 = cv2.inRange(hsv, np.array(lo1, np.uint8), np.array(hi1, np.uint8))
    raw2 = cv2.inRange(hsv, np.array(lo2, np.uint8), np.array(hi2, np.uint8))
    c1 = cv2.bitwise_and(raw1, raw1, mask=roi_mask)
    c2 = cv2.bitwise_and(raw2, raw2, mask=roi_mask)
    return c1, c2

# ---------- tiles; Lacey (centered at KNOWN p) ----------
def analyze_tiles_center_p(sub_tracer, valid_sub, grid_wh, min_cover_frac, p_known):
    H, W = sub_tracer.shape
    gw, gh = grid_wh
    cell_w = max(1, W // gw)
    cell_h = max(1, H // gh)

    tiles = []
    weights, p_list = [], []

    total_valid = int(valid_sub.sum())
    if total_valid == 0:
        return tiles, math.nan, math.nan, 0.0, 0

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
        return tiles, float(sub_tracer[valid_sub].mean()) / 255.0, math.nan, 0.0, used

    w = np.array(weights, np.float64)
    p = np.array(p_list, np.float64)
    wsum = float(w.sum())
    s2_w = float((w * (p - p_known) ** 2).sum() / wsum)   # variance about KNOWN p
    m_bar = float(wsum / used)
    p_meas = float(sub_tracer[valid_sub].mean()) / 255.0
    return tiles, p_meas, s2_w, m_bar, used

def lacey_index_known_p(s2_w: float, p_known: float, m_bar: float) -> float:
    if not np.isfinite(s2_w) or not np.isfinite(p_known) or m_bar <= 1:
        return float("nan")
    s0_sq = p_known * (1 - p_known)
    sr_sq = s0_sq / m_bar
    denom = s0_sq - sr_sq
    if denom <= 0: return float("nan")
    return float(np.clip((s0_sq - s2_w) / denom, 0.0, 1.0))

# ---------- Shannon entropy mixing index ----------
def entropy_mixing_index(p_list, w_list, nbins=20):
    """
    Weighted Shannon entropy of the distribution of tile tracer fractions p_i ∈ [0,1],
    normalized by H_max = ln(nbins), → range [0,1].
      • All tiles identical (all ~0 or all ~1) → H=0 → M_entropy=0 (unmixed)
      • Distribution spread across bins (centered near p) → higher H → toward 1
    """
    p = np.asarray(p_list, float)
    w = np.asarray(w_list, float)
    if p.size == 0 or w.sum() <= 0:
        return float("nan")
    hist, _ = np.histogram(p, bins=nbins, range=(0.0, 1.0), weights=w)
    q = hist.astype(np.float64)
    Z = q.sum()
    if Z <= 0:
        return float("nan")
    q /= Z
    qnz = q[q > 0]
    H = -np.sum(qnz * np.log(qnz))
    Hmax = math.log(nbins)
    return float(np.clip(H / Hmax, 0.0, 1.0)) if Hmax > 0 else float("nan")

def format_ts(t: float) -> str:
    if not np.isfinite(t): return "n/a"
    m, s = divmod(float(t), 60.0)
    return f"{int(m):02d}:{s:05.2f}"

# ------------------------------- main ------------------------------- #
def main():
    path = pick_video_path()
    cap = open_video(path)
    if not cap.isOpened(): raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS

    ok, first = cap.read()
    if not ok: raise SystemExit("Could not read first frame.")

    # ROI pick
    draw_win = "ROI (draw polygon, drag points, 'c' to start)"
    cv2.namedWindow(draw_win, cv2.WINDOW_NORMAL)
    roi = PolygonROI(draw_win); roi.install()

    help_lines = [
        "Left-click: add points | drag vertices to adjust",
        "z: undo | r: reset | c: start | q/ESC: quit",
        f"Grid: {GRID_W}x{GRID_H} | Speed: {SPEED_MULTIPLIER:.2f}×",
    ]

    # ROI drawing loop (scale-to-fit)
    while True:
        vis = roi.draw_overlay(first)
        ytxt = 24
        for line in help_lines:
            put_label(vis, 10, ytxt, line, scale=0.6, thick=2); ytxt += 26
        if FIT_TO_SCREEN:
            sw, sh = get_screen_size()
            vis_disp, s = scale_to_fit(vis, sw-40, sh-80)
            roi.set_scale(s); cv2.imshow(draw_win, vis_disp); cv2.moveWindow(draw_win, 20, 20)
        else:
            roi.set_scale(1.0); cv2.imshow(draw_win, vis)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')): cap.release(); cv2.destroyAllWindows(); return
        elif k == ord('z') and roi.points: roi.points.pop()
        elif k == ord('r'): roi.reset()
        elif k == ord('c') and len(roi.points) >= 3:
            roi.closed = True; break

    roi_mask = roi.mask(first.shape)
    poly = np.array(roi.points, np.int32)
    x, y, w, h = cv2.boundingRect(poly)
    valid_mask_full = roi_mask.astype(bool)

    # Playback
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    play_win = "Playback (Lacey + Entropy, gated)"
    cv2.namedWindow(play_win, cv2.WINDOW_NORMAL)

    start_wall = time.perf_counter()
    frame_idx = 0
    tracer_is_c1 = (TRACER_IS.upper() == "C1")

    # Logs
    times_s: List[float] = []
    lacey_vals: List[float] = []
    entropy_vals: List[float] = []
    coverage_list: List[float] = []
    pmeas_list: List[float] = []
    tiles_tracer_list: List[int] = []
    gate_open_list: List[int] = []

    gate_ok_frames = 0  # hysteresis counter

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        c1, c2 = apply_hsv_masks(frame, roi_mask,
                                 COLOR1_HSV_LO, COLOR1_HSV_HI,
                                 COLOR2_HSV_LO, COLOR2_HSV_HI)
        tracer = c1 if tracer_is_c1 else c2

        # Panels
        left = frame.copy()
        outside = cv2.bitwise_not(roi_mask)
        left[outside > 0] = (left[outside > 0] * 0.25).astype(np.uint8)
        if TINT_ALPHA > 0:
            t1 = np.zeros_like(left); t1[c1 > 0] = TINT1_BGR
            t2 = np.zeros_like(left); t2[c2 > 0] = TINT2_BGR
            left = cv2.addWeighted(left, 1 - TINT_ALPHA, t1, TINT_ALPHA, 0)
            left = cv2.addWeighted(left, 1 - TINT_ALPHA, t2, TINT_ALPHA, 0)
        if len(roi.points) >= 3:
            cv2.polylines(left, [poly], True, (0,200,255), 2, cv2.LINE_AA)
        left = draw_grid_in_roi_bbox(left, roi.points, GRID_W, GRID_H, GRID_COLOR, GRID_THICKNESS)

        right = np.zeros_like(frame)
        right[:,:,1] = c1; right[:,:,2] = c2
        right = draw_grid_in_roi_bbox(right, roi.points, GRID_W, GRID_H, GRID_COLOR, GRID_THICKNESS)

        # Tile analysis about known p
        sub_tr  = tracer[y:y+h, x:x+w]
        valid_sub = valid_mask_full[y:y+h, x:x+w]
        tiles, p_meas, s2_w, m_bar, used = analyze_tiles_center_p(
            sub_tr, valid_sub, (GRID_W, GRID_H), MIN_TILE_COVER_FRAC, KNOWN_TRACER_FRACTION
        )
        w_list = [t["n"] for t in tiles]
        p_list = [t["p_tracer"] for t in tiles]

        # Coverage & spread gates
        EPS = 1e-9
        coverage = float(np.clip(p_meas / max(KNOWN_TRACER_FRACTION, EPS), 0.0, 1.0))
        tiles_with_tracer = sum(1 for p in p_list if p >= TILE_MIN_FRAC)
        meets_gate = (coverage >= COVERAGE_GATE) and (tiles_with_tracer >= SPREAD_TILES_MIN)

        gate_ok_frames = gate_ok_frames + 1 if meets_gate else 0
        gate_open = (gate_ok_frames >= HYST_FRAMES)

        # Classic indices (raw)
        Lacey_raw = lacey_index_known_p(s2_w, KNOWN_TRACER_FRACTION, m_bar)
        Entropy_raw = entropy_mixing_index(p_list, w_list, nbins=ENTROPY_BINS)

        # Gate what we DISPLAY / LOG
        if not gate_open:
            Lacey_to_log   = float("nan")
            Entropy_to_log = float("nan")
            lmi_text = "— (gate)"; ent_text = "— (gate)"
        else:
            Lacey_to_log   = Lacey_raw
            Entropy_to_log = Entropy_raw
            lmi_text = f"{(Lacey_to_log if np.isfinite(Lacey_to_log) else float('nan')):.3f}"
            ent_text = f"{(Entropy_to_log if np.isfinite(Entropy_to_log) else float('nan')):.3f}"

        # Per-tile % overlays (verification)
        for t in tiles:
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
            put_label(left, gx0, gy0 + int(18*fscale), f"G {int(round(100*t_c1)):3d}%",
                      fg=(20,255,20), scale=fscale, thick=thick)
            put_label(left, gx0, gy0 + int(38*fscale), f"R {int(round(100*t_c2)):3d}%",
                      fg=(255,20,255), scale=fscale, thick=thick)

        # Header overlays
        if SHOW_OVERLAYS:
            put_label(left, 18, 36,
                      f"Lacey (p={KNOWN_TRACER_FRACTION:.2f}): {lmi_text}",
                      scale=0.7, thick=2)
            put_label(left, 18, 66,
                      f"Entropy (bins={ENTROPY_BINS}): {ent_text}",
                      scale=0.7, thick=2)
            put_label(left, 18, 96,
                      f"p̂(ROI)={p_meas:.3f}  coverage={coverage:.2f}  tiles≥{TILE_MIN_FRAC:.2f}: {tiles_with_tracer}  gate={'ON' if gate_open else 'OFF'}",
                      scale=0.7, thick=2)

        # Time overlay
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_elapsed = pos_ms/1000.0 if pos_ms and pos_ms > 0 else (frame_idx/(fps if fps>0 else FALLBACK_FPS))
        put_label(left, 18, 126, f"t = {int(t_elapsed//60):02d}:{(t_elapsed%60):05.2f}",
                  scale=0.7, thick=2)

        # Log
        times_s.append(float(t_elapsed))
        lacey_vals.append(float(Lacey_to_log) if np.isfinite(Lacey_to_log) else np.nan)
        entropy_vals.append(float(Entropy_to_log) if np.isfinite(Entropy_to_log) else np.nan)
        coverage_list.append(float(coverage))
        pmeas_list.append(float(p_meas))
        tiles_tracer_list.append(int(tiles_with_tracer))
        gate_open_list.append(int(gate_open))

        # Show
        stacked = np.hstack([left, right])
        if FIT_TO_SCREEN:
            sw, sh = get_screen_size()
            disp, _ = scale_to_fit(stacked, sw - 40, sh - 80)
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
    L_arr = np.array(lacey_vals, dtype=float)
    E_arr = np.array(entropy_vals, dtype=float)

    stem = derive_save_stem(path)

    # CSV
    csv_path = safe_path(stem, ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "lacey_known_p_gated", "entropy_gated",
                    "coverage", "p_hat_roi", "tiles_with_tracer", "gate_open"])
        for i in range(len(t_arr)):
            w.writerow([t_arr[i], L_arr[i], E_arr[i],
                        coverage_list[i], pmeas_list[i], tiles_tracer_list[i], gate_open_list[i]])

    # Plot (always)
    plt.close('all')
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_arr, L_arr, label="Lacey (known p, gated)", linewidth=2)
    # plt.plot(t_arr, E_arr, label=f"Entropy (bins={ENTROPY_BINS}, gated)", linewidth=2)
    plt.ylim(0, 1.05); plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)"); plt.ylabel("Mixing Index (0–1)")
    plt.title("Mixing Indices vs Time (gated)")
    plt.legend()
    png_path = safe_path(stem, ".png")
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.show(block=True)

    print(f"\nSaved files:\n  CSV -> {os.path.abspath(csv_path)}\n  PNG -> {os.path.abspath(png_path)}")

if __name__ == "__main__":
    main()
