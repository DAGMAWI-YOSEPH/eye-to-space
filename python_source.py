import argparse
import math

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
CELL_W, CELL_H = 8, 16
COLS = W // CELL_W
ROWS = H // CELL_H
FPS = 30
DURATION = 15.0

RAMP = " .·:;=+*o#%@█"
GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?|/\\+*#%@&$!()<>=~^"

T_SCLERA_IN = (0.0, 1.5)
T_IRIS_IN = (1.5, 3.2)
T_DISSOLVE = (8.5, 12.0)
T_STARS_IN = (3.5, 6.0)
T_HEART_IN = (11.5, 13.8)


def smooth(a, b, t):
    x = np.clip((t - a) / (b - a), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cell_grids():
    px = (np.arange(COLS) + 0.5) * CELL_W - W / 2.0
    py = -((np.arange(ROWS) + 0.5) * CELL_H - H / 2.0)
    gx, gy = np.meshgrid(px, py)
    return gx, gy


GX, GY = cell_grids()
R_NORM = np.sqrt(GX**2 + GY**2) / (H * 0.30)
ANG = np.arctan2(GY, GX)

rng_stable = np.random.default_rng(7)
THRESH = rng_stable.random((ROWS, COLS))
VESSEL = rng_stable.random((ROWS, COLS))
STAR_N = 110
STAR_X = rng_stable.uniform(-W * 0.48, W * 0.48, STAR_N)
STAR_Y = rng_stable.uniform(-H * 0.48, H * 0.48, STAR_N)
STAR_PH = rng_stable.uniform(0, math.tau, STAR_N)
STAR_SP = rng_stable.uniform(2.0, 5.0, STAR_N)
STAR_BIG = rng_stable.random(STAR_N) < 0.35


def heart_field(gx, gy, t):
    s = H * 0.30
    x, y = gx / s, gy / s
    eq = (x * x + y * y - 1.0) ** 3 - x * x * y**3
    band = 0.60 + 0.40 * (0.5 + 0.5 * np.sin((gy / CELL_H) * 1.05 + 0.4))
    edge = np.clip(-eq * 6.0, 0.0, 1.0)
    return np.clip(edge * 1.4, 0, 1) * band


def eye_layers(t):
    r = R_NORM
    pupil = np.clip((0.55 - r) / 0.05, 0, 1)
    fibers = 0.5 + 0.5 * np.sin(ANG * 42.0 + 2.4 * np.sin(ANG * 11.0) + r * 7.0)
    mid = np.clip((r - 0.55) / 0.08, 0, 1) * np.clip((1.00 - r) / 0.10, 0, 1)
    limbus = np.exp(-(((r - 0.94) / 0.05) ** 2))
    hot = np.exp(-(((r - 1.03) / 0.06) ** 2))

    iris_lum = mid * (0.45 + 0.55 * fibers) * (1.0 - 0.85 * limbus) + 0.95 * hot * mid
    iris_rgb = np.stack(
        [
            np.clip(iris_lum * 1.10, 0, 1),
            np.clip(iris_lum * 0.55 - limbus * 0.35, 0, 1),
            np.clip(iris_lum * 0.16 - limbus * 0.1, 0, 1),
        ],
        axis=-1,
    )

    fig = np.exp(-(((GX + 4) / (W * 0.022)) ** 2 + ((GY + H * 0.03) / (H * 0.075)) ** 2))
    fig2 = np.exp(-(((GX - 12) / (W * 0.016)) ** 2 + ((GY + H * 0.07) / (H * 0.05)) ** 2))
    glint = np.exp(
        -(((GX + W * 0.055) / (W * 0.018)) ** 2 + ((GY - H * 0.10) / (H * 0.032)) ** 2)
    )

    pupil_lum = 0.9 * fig * pupil + 0.55 * fig2 * pupil + 1.5 * glint * pupil
    char_lum = pupil * pupil_lum + mid * np.clip(0.30 + 0.70 * iris_lum, 0, 1)
    return pupil, mid, iris_rgb, pupil_lum, char_lum, glint


def sclera_layer(t, frame_i):
    rng = np.random.default_rng(1000 + (frame_i // 2))
    speck = rng.random((ROWS, COLS))
    vessel = np.clip(
        np.sin(VESSEL * 53.0 + R_NORM * 7.0 + ANG * 2.0) * 0.5 + 0.5 - 0.88, 0, 1
    ) * 5.0
    lum = 0.48 + 0.34 * speck
    lum *= 1.0 - 0.15 * np.clip(R_NORM - 2.2, 0, 1)
    rgb = np.stack([np.ones_like(lum), np.ones_like(lum), np.ones_like(lum)], axis=-1)
    rgb = rgb * lum[..., None]
    vcol = np.stack(
        [np.ones_like(vessel), 0.62 + 0.3 * (1 - vessel), 0.58 + 0.3 * (1 - vessel)],
        axis=-1,
    )
    rgb = rgb * (1 - vessel[..., None] * 0.55) + vcol * (vessel * 0.55)[..., None]
    return np.clip(rgb, 0, 1), lum


def render_cells(t, frame_i):
    rgb = np.zeros((ROWS, COLS, 3), dtype=np.float64)
    lum = np.zeros((ROWS, COLS))

    a_sclera = smooth(*T_SCLERA_IN, t) * (1.0 - smooth(*T_DISSOLVE, t))
    if a_sclera > 0.001:
        wipe_r = 3.7 * smooth(*T_SCLERA_IN, t)
        wipe = np.clip((wipe_r - R_NORM) * 3.0, 0, 1)
        keep = np.clip((THRESH < a_sclera * wipe * 0.88).astype(float), 0, 1)
        s_rgb, s_lum = sclera_layer(t, frame_i)
        m = (wipe * keep)[..., None]
        rgb += s_rgb * m
        lum = np.maximum(lum, s_lum * wipe * keep)

    a_iris = smooth(*T_IRIS_IN, t) * (1.0 - smooth(*T_DISSOLVE, t))
    if a_iris > 0.001:
        pupil, mid, iris_rgb, pupil_lum, char_lum, _ = eye_layers(t)
        keep = np.clip((THRESH + 0.2 < a_iris + 0.35).astype(float), 0, 1)
        m = np.clip(a_iris * keep, 0, 1)
        core = np.clip(pupil + mid, 0, 1) * m
        base = rgb * (1 - core[..., None])
        add = (iris_rgb + pupil_lum[..., None] * np.array([0.9, 0.8, 0.7])) * core[..., None]
        rgb = np.clip(base + add, 0, 1)
        lum = np.maximum(lum, core * char_lum)

    a_heart = smooth(*T_HEART_IN, t)
    if a_heart > 0.001:
        bob = math.sin((t - T_HEART_IN[0]) * 2.1) * H * 0.02
        hfield = heart_field(GX, GY - bob, t)
        keep = np.clip((THRESH + 0.12) < a_heart + 0.4, 0, 1)
        m = np.clip(hfield * a_heart * keep, 0, 1)
        flick = 0.75 + 0.25 * math.sin(t * 7.0)
        h_rgb = np.stack(
            [np.ones_like(m) * 1.0, 0.28 + 0.2 * (1 - m), 0.55 + 0.25 * (1 - m)], axis=-1
        )
        rgb = np.maximum(rgb, h_rgb * m[..., None] * flick)
        lum = np.maximum(lum, m * 0.9 * flick)

    a_stars = smooth(*T_STARS_IN, t)
    if a_stars > 0.001:
        star_m = np.zeros((ROWS, COLS))
        for i in range(STAR_N):
            tw = 0.5 + 0.5 * math.sin(t * STAR_SP[i] + STAR_PH[i])
            cx, cy = STAR_X[i], STAR_Y[i]
            col = int((cx + W / 2) / CELL_W)
            row = int((H / 2 - cy) / CELL_H)
            if not (0 <= row < ROWS and 0 <= col < COLS):
                continue
            b = a_stars * (0.45 + 0.55 * tw)
            star_m[row, col] = max(star_m[row, col], b)
            if STAR_BIG[i] and b > 0.45:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < ROWS and 0 <= cc < COLS:
                        star_m[rr, cc] = max(star_m[rr, cc], b * 0.55)
        bg_g = a_sclera
        white_star = star_m * (1.0 - bg_g)
        dark_star = star_m * bg_g
        lum = lum + white_star * 0.95
        lum = lum * (1.0 - dark_star * 0.9)
        rgb = np.maximum(rgb, white_star[..., None] * np.array([0.95, 0.96, 1.0]))
        rgb = rgb * (1.0 - dark_star * 0.9)[..., None]

    return np.clip(rgb, 0, 1), np.clip(lum, 0, None)


def pick_chars(lum, frame_i):
    rng = np.random.default_rng(frame_i + 4242)
    n_lev = len(RAMP)
    idx = np.clip(((lum**0.8) * n_lev).astype(int), 0, n_lev - 1)
    chars = np.array(list(RAMP))[idx]
    swap = (lum > 0.18) & (rng.random((ROWS, COLS)) < 0.30)
    g = np.array(list(GLYPHS))
    chars = np.where(swap, g[rng.integers(0, len(g), (ROWS, COLS))], chars)
    chars = np.where(lum <= 0.04, " ", chars)
    return chars


def draw_frame(rgb, chars, font):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    for r in range(ROWS):
        y = r * CELL_H
        for c in range(COLS):
            ch = chars[r, c]
            if ch == " ":
                continue
            col = rgb[r, c]
            dr.text(
                (c * CELL_W, y),
                ch,
                font=font,
                fill=(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)),
            )
    return img


def render_frame(t, frame_i, font_holder):
    if font_holder[0] is None:
        font_holder[0] = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    rgb, lum = render_cells(t, frame_i)
    chars = pick_chars(lum, frame_i)
    return draw_frame(rgb, chars, font_holder[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eye_to_space.mp4")
    ap.add_argument("--preview", nargs="*", type=float, default=None)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    font_holder = [None]

    if args.preview is not None:
        times = args.preview or [0.8, 5.0, 10.5, 14.0]
        for t in times:
            fi = int(t * args.fps)
            img = render_frame(t, fi, font_holder)
            img.save(f"preview_{t:04.1f}s.png")
            print(f"preview_{t:04.1f}s.png")
        return

    n = int(args.duration * args.fps)
    container = av.open(args.out, mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for i in range(n):
        t = i / args.fps
        img = render_frame(t, i, font_holder)
        frame = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
        if i % 60 == 0:
            print(f"{i}/{n} ({t:.1f}s)", flush=True)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
