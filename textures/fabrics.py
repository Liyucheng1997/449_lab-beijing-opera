"""戏服刺绣贴图：蟒袍（团龙、如意云、海水江崖）、女蟒（团花）、大靠（鱼鳞甲、兽面、靠旗）、猴衣、虎皮裙。

躯干展开：u∈[0, TORSO_U] 绕身一周（u 中点为正前方），v 为从下摆到领口的弧长比例。
图案在“物理毫米”局部坐标里绘制，再按各高度的周长换算到 UV，保证穿上身后不变形。
"""
import math
import sys
import numpy as np
from scipy import ndimage

from paint import Canvas, catmull, ribbon, rgb
from costume_params import profile_arclen, TORSO_U

rs = np.random.RandomState(5)
GOLD = "#d3a640"
GOLD_D = "#7b5519"


class Garment:
    """把局部毫米坐标映射到躯干 UV。"""

    def __init__(self, canvas, prof_name, sleeve_len=600.0, sleeve_circ=560.0):
        self.c = canvas
        tab = profile_arclen(prof_name)
        self.zs = np.array([t[1] * 1000 for t in tab])  # 弧长 mm
        self.ps = np.array([t[2] * 1000 for t in tab])  # 周长 mm
        self.S = self.zs[-1]
        self.sleeve_len = sleeve_len
        self.sleeve_circ = sleeve_circ
        self.batches = {}

    def P(self, s):
        return np.interp(s, self.zs, self.ps)

    def torso(self, theta_c, s_c):
        """返回把局部 (x_mm, y_mm) 映射到 UV 的函数；theta_c 为绕身角度（π 为正前），s_c 为离下摆弧长 mm。"""
        def f(pts):
            pts = np.asarray(pts, np.float64)
            s = s_c + pts[:, 1]
            u = TORSO_U * (theta_c / (2 * math.pi)) + pts[:, 0] / self.P(s) * TORSO_U
            v = s / self.S
            return np.stack([u, v], 1)
        f.wrap = (0.0, TORSO_U)
        return f

    def sleeve(self, a_c, s_c):
        """袖子：a_c∈[0,1] 绕袖一周（0.5 为袖子外侧/前方），s_c 为离袖口 mm。"""
        def f(pts):
            pts = np.asarray(pts, np.float64)
            u = TORSO_U + (1 - TORSO_U) * (a_c + pts[:, 0] / self.sleeve_circ)
            v = (s_c + pts[:, 1]) / self.sleeve_len
            return np.stack([u, v], 1)
        f.wrap = (TORSO_U, 1.0)
        return f

    # --- 批量绘制（同材质的多边形一次栅格化，速度快）
    ph = 0

    def phase(self):
        self.ph = getattr(self, "ph", 0) + 1

    def add(self, polys_mm, place, key):
        lst = self.batches.setdefault((getattr(self, "ph", 0), key), [])
        wrap = getattr(place, "wrap", None)
        for p in polys_mm:
            q = place(p)
            lst.append(q)
            if wrap:
                lo, hi = wrap
                span = hi - lo
                if q[:, 0].min() < lo:
                    lst.append(q + np.array([span, 0]))
                if q[:, 0].max() > hi:
                    lst.append(q - np.array([span, 0]))

    def flush(self, order=None):
        for (ph, key), polys in list(self.batches.items()):
            if not polys:
                continue
            color, rough, metal, thick, stitch = key
            m = self.c.mask_poly(polys, 0.0)
            self.c.apply(m, color, rough=rough, metal=metal, thick=thick)
            if stitch:
                self.c.height += m * stitch_pattern(self.c, stitch) * 0.5
        self.batches.clear()


def stitch_pattern(c, angle):
    if not hasattr(c, "_st"):
        c._st = {}
    if angle not in c._st:
        yy, xx = np.mgrid[0:c.H, 0:c.W].astype(np.float32)
        a = math.radians(angle)
        c._st[angle] = 0.5 + 0.5 * np.sin((xx * math.cos(a) + yy * math.sin(a)) * 1.9)
    return c._st[angle]


def K(color, rough=0.42, metal=0.0, thick=0.8, stitch=0):
    return (color, rough, metal, thick, stitch)


GOLDK = K(GOLD, 0.32, 0.85, 1.0, 35)
GOLDLINE = K(GOLD_D, 0.4, 0.6, 1.2, 0)


def circle(c, r, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)


def ring(c, r, w, n=60):
    t = np.linspace(0, 2 * np.pi, n + 1)
    outer = np.stack([c[0] + (r + w / 2) * np.cos(t), c[1] + (r + w / 2) * np.sin(t)], 1)
    inner = np.stack([c[0] + (r - w / 2) * np.cos(t[::-1]), c[1] + (r - w / 2) * np.sin(t[::-1])], 1)
    return np.concatenate([outer, inner])


def spiral(c, r0, r1, turns, w, n=60, start=0.0, sign=1):
    t = np.linspace(0, 1, n)
    a = start + sign * t * turns * 2 * np.pi
    r = r0 + (r1 - r0) * t
    pts = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
    return ribbon(pts[::3], w, w * 0.5, n=6)


# ------------------------------------------------------------------ 图案构件
def cloud(g, place, size=60.0, colors=("#1f5f9e", "#4f8fcf", "#bfe0f5"), flip=False):
    """如意云：三瓣云头 + 卷尾，退晕三层 + 金线。"""
    s = size / 60.0
    lobes = [((0, 0), 16), ((-18, 4), 12), ((18, 4), 12), ((-6, 15), 10), ((8, 14), 9)]
    tail = [(16, -6), (34, -12), (52, -8), (64, 2)]
    if flip:
        lobes = [((-x, y), r) for (x, y), r in lobes]
        tail = [(-x, y) for x, y in tail]
    lobes = [((x * s, y * s), r * s) for (x, y), r in lobes]
    tail = [(x * s, y * s) for x, y in tail]
    out = [circle(c, r + 2.2 * s) for c, r in lobes] + [ribbon(tail, 13 * s, 3 * s)]
    g.add(out, place, GOLDLINE)
    for k, (sc, col) in enumerate(zip((1.0, 0.72, 0.42), colors)):
        polys = [circle(c, r * sc) for c, r in lobes] + [ribbon(tail, 10 * s * sc, 1.5 * s)]
        g.add(polys, place, K(col, 0.4, 0.0, 0.6 + 0.2 * k, 0))
    for c, r in lobes[:3]:
        g.add([spiral(c, r * 0.15, r * 0.75, 1.1, 1.6 * s, sign=-1 if c[0] < 0 else 1)], place, GOLDK)


def dragon(g, place, R=150.0):
    """团龙：正面龙头居中，龙身盘绕一周，四爪，火珠。"""
    path = np.array([(18, 50), (62, 88), (112, 48), (116, -22), (80, -88), (8, -112), (-62, -96), (-112, -40), (-114, 30), (-82, 86), (-38, 112), (-4, 120)], float) * (R / 150)
    spine = catmull(path, closed=False, n=10)
    t = np.linspace(0, 1, len(spine))
    width = (36 - 28 * t ** 1.3) * (R / 150)
    body = ribbon(path, 0, wfun=lambda tt: (36 - 28 * tt ** 1.3) * (R / 150), n=10)
    # 背鳍（红）
    d = np.gradient(spine, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    fins = []
    for i in range(2, len(spine) - 6, 3):
        p = spine[i] + nrm[i] * width[i] * 0.45
        tip = spine[i] + nrm[i] * width[i] * 0.95 - d[i] * 6
        fins.append(np.array([p - d[i] * 5, tip, p + d[i] * 5]))
    g.add(fins, place, K("#c23a26", 0.4, 0.0, 0.8, 0))
    # 四爪
    legs = []
    claws = []
    for tt, side in ((0.12, -1), (0.33, -1), (0.58, -1), (0.8, -1)):
        i = int(tt * (len(spine) - 1))
        base = spine[i] - nrm[i] * width[i] * 0.3
        out = base - nrm[i] * 42 * (R / 150) + d[i] * 10
        knee = base - nrm[i] * 22 * (R / 150) - d[i] * 12
        legs.append(ribbon([base, knee, out], 14 * (R / 150), 8 * (R / 150), n=8))
        for a in (-0.7, 0.0, 0.7):
            dirv = -nrm[i] * math.cos(a) + d[i] * math.sin(a)
            perp = np.array([-dirv[1], dirv[0]])
            claws.append(ribbon([out, out + dirv * 9 + perp * 2, out + dirv * 13 - perp * 3], 3.2, 0.6, n=6))
        g.add([ribbon([knee, knee + d[i] * 16 - nrm[i] * 4, knee + d[i] * 22 + nrm[i] * 2], 5, 0.5, n=6)], place, K("#c23a26", 0.4))
    g.add([body] + legs, place, GOLDLINE)
    inner = ribbon(path, 0, wfun=lambda tt: (36 - 28 * tt ** 1.3) * (R / 150) - 3.5, n=10)
    g.add([inner] + [ribbon([l[0], l[len(l) // 2]], 1, 1) for l in []], place, GOLDK)
    g.add(legs, place, GOLDK)
    g.add(claws, place, K("#f1ece0", 0.35, 0.2, 1.0, 0))
    # 鳞片：沿身体的弧线
    scales = []
    for i in range(1, len(spine) - 2):
        w = width[i] * 0.42
        for k in (-0.5, 0.0, 0.5):
            c0 = spine[i] + nrm[i] * w * k
            arc = [c0 - nrm[i] * w * 0.35 - d[i] * 1.5, c0 + d[i] * 2.5, c0 + nrm[i] * w * 0.35 - d[i] * 1.5]
            scales.append(ribbon(arc, 1.0, 1.0, n=4))
    g.add(scales, place, GOLDLINE)
    # 龙头（正面）
    S = R / 150
    mane = []
    for k in range(13):
        a = math.pi * (0.05 + 0.9 * k / 12)
        b = np.array((math.cos(a), math.sin(a)))
        p0 = np.array((0, 26)) * S + b * 28 * S
        mane.append(ribbon([p0, p0 + b * 18 * S + np.array((b[1], -b[0])) * 5 * S, p0 + b * 30 * S], 7 * S, 0.8, n=6))
    g.add(mane, place, K("#c23a26", 0.42))
    horns = []
    for s in (1, -1):
        horns.append(ribbon([(12 * s * S, 44 * S), (24 * s * S, 64 * S), (34 * s * S, 84 * S), (46 * s * S, 92 * S)], 6 * S, 2 * S, n=8))
        horns.append(ribbon([(26 * s * S, 66 * S), (38 * s * S, 70 * S), (44 * s * S, 78 * S)], 4 * S, 1.5 * S, n=6))
    g.add(horns, place, GOLDK)
    face = [(-30, 40), (-34, 22), (-30, 0), (-24, -14), (0, -22), (24, -14), (30, 0), (34, 22), (30, 40), (0, 50)]
    face = catmull(np.array(face) * S)
    g.add([face], place, GOLDLINE)
    g.add([catmull(np.array([(-27, 38), (-31, 22), (-27, 2), (-21, -10), (0, -18), (21, -10), (27, 2), (31, 22), (27, 38), (0, 46)]) * S)], place, GOLDK)
    brows = [ribbon([(6 * s * S, 34 * S), (16 * s * S, 40 * S), (28 * s * S, 38 * S), (34 * s * S, 46 * S)], 5 * S, 1.5 * S, n=6) for s in (1, -1)]
    g.add(brows, place, K("#c23a26", 0.4, 0, 1.1))
    g.add([circle((14 * s * S, 26 * S), 7 * S) for s in (1, -1)], place, K("#f4efe4", 0.35, 0, 1.2))
    g.add([circle((14 * s * S, 26 * S), 3.2 * S) for s in (1, -1)], place, K("#111111", 0.3, 0, 1.2))
    g.add([catmull(np.array([(-16, -4), (0, -10), (16, -4), (10, -16), (0, -19), (-10, -16)]) * S)], place, K("#a01f22", 0.4))
    g.add([circle((8 * s * S, 8 * S), 4 * S) for s in (1, -1)], place, K(GOLD_D, 0.4, 0.6, 1.3))
    whisk = []
    for s in (1, -1):
        whisk.append(ribbon([(22 * s * S, 4 * S), (48 * s * S, 0), (70 * s * S, -14 * S), (78 * s * S, -4 * S), (72 * s * S, 4 * S)], 3 * S, 1 * S, n=10))
        whisk.append(ribbon([(18 * s * S, -10 * S), (36 * s * S, -30 * S), (58 * s * S, -36 * S), (64 * s * S, -28 * S)], 2.6 * S, 0.8 * S, n=10))
    g.add(whisk, place, GOLDK)
    # 火珠
    pc = np.array((0, -58)) * S
    flames = [ribbon([pc + np.array((math.cos(a), math.sin(a))) * 14 * S, pc + np.array((math.cos(a + 0.3), math.sin(a + 0.3))) * 26 * S], 6 * S, 0.5, n=4) for a in np.linspace(0, 2 * np.pi, 9, endpoint=False)]
    g.add(flames, place, K("#e0702a", 0.4))
    g.add([circle(pc, 13 * S)], place, K("#c5262a", 0.35, 0, 1.2))
    g.add([spiral(pc, 1, 10 * S, 1.2, 2.0 * S)], place, GOLDK)


def peony(g, place, R=60.0, colors=("#b3264a", "#e0607c", "#f7c6cf")):
    """团花牡丹：叶 + 三层花瓣 + 花心。"""
    leaves = []
    for k in range(6):
        a = k / 6 * 2 * np.pi + 0.3
        b = np.array((math.cos(a), math.sin(a)))
        p = [b * R * 0.4, b * R * 0.8 + np.array((-b[1], b[0])) * R * 0.15, b * R * 1.05]
        leaves.append(ribbon(p, R * 0.28, R * 0.02, n=8))
    g.add(leaves, place, K("#2f6b3a", 0.45))
    g.add([ribbon([l[0], l[len(l) // 4]], 1, 1) for l in []], place, GOLDLINE)
    for layer, (sc, col) in enumerate(zip((0.85, 0.62, 0.38), colors)):
        petals = []
        n = 7 - layer
        for k in range(n):
            a = k / n * 2 * np.pi + layer * 0.4
            c0 = np.array((math.cos(a), math.sin(a))) * R * sc * 0.45
            t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            r = R * sc * 0.42 * (1 + 0.12 * np.cos(3 * t))
            pts = np.stack([np.cos(t) * r, np.sin(t) * r * 0.8], 1)
            ca, sa = math.cos(a), math.sin(a)
            pts = pts @ np.array([[ca, sa], [-sa, ca]]) + c0
            petals.append(pts)
        g.add([circle((0, 0), R * sc * 0.95)], place, GOLDLINE) if layer == 0 else None
        g.add(petals, place, K(col, 0.4, 0, 0.8 + 0.2 * layer))
    g.add([circle((0, 0), R * 0.12)], place, GOLDK)


def wave_band(c, g, s0, s1, palette, stripe=14.0, slant=0.7):
    """立水：从下摆起的斜向彩条，左右对称，条内退晕。"""
    H, W = c.H, c.W
    v = 1 - (np.arange(H) + 0.5) / H
    u = (np.arange(W) + 0.5) / W
    s = v * g.S
    rows = np.where((s >= s0) & (s <= s1))[0]
    cols = np.where(u <= TORSO_U)[0]
    if len(rows) == 0:
        return
    S_ = s[rows][:, None]
    P = g.P(S_)
    x = (u[cols][None, :] / TORSO_U) * P  # mm 绕身
    # 到最近的前/后中线距离（左右对称）
    xf = np.abs(((x - 0.5 * P) + 0.5 * P) % P - 0.5 * P)
    ph = (xf * 0.6 + S_ * slant + 8 * np.sin(S_ / 25.0)) / stripe
    idx = np.floor(ph).astype(int) % len(palette)
    frac = ph - np.floor(ph)
    pal = np.array([rgb(p) for p in palette])
    col = pal[idx] * (0.78 + 0.32 * np.sin(frac * np.pi))[..., None]
    gold = np.clip(1 - np.minimum(frac, 1 - frac) / 0.06, 0, 1)
    sub = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    c.color[sub] = col * (1 - gold[..., None]) + rgb(GOLD) * gold[..., None]
    c.metal[sub] = 0.85 * gold
    c.rough[sub] = 0.42 - 0.1 * gold
    c.height[sub] += 0.8 + 0.4 * gold


def flat_waves(c, g, s0, s1, colors=("#1f4f8f", "#5b8fd0", "#e8f0f7")):
    """平水：鳞状波浪。"""
    H, W = c.H, c.W
    v = 1 - (np.arange(H) + 0.5) / H
    u = (np.arange(W) + 0.5) / W
    s = v * g.S
    rows = np.where((s >= s0) & (s <= s1))[0]
    cols = np.where(u <= TORSO_U)[0]
    S_ = s[rows][:, None]
    P = g.P(S_)
    x = (u[cols][None, :] / TORSO_U) * P
    cell = 26.0
    r_i = np.floor((S_ - s0) / (cell * 0.6))
    xx = x + (r_i % 2) * cell / 2
    lx = (xx % cell) - cell / 2
    ly = (S_ - s0) - r_i * cell * 0.6
    d = np.sqrt(lx ** 2 + ly ** 2) / (cell / 2)
    band = (np.floor(d * 3).astype(int)) % 3
    pal = np.array([rgb(cc) for cc in colors])
    col = pal[np.clip(band, 0, 2)]
    line = np.clip(1 - np.abs(d * 3 - np.round(d * 3)) / 0.12, 0, 1)
    sub = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    c.color[sub] = col * (1 - 0.5 * line[..., None]) + rgb(GOLD) * 0.5 * line[..., None]
    c.metal[sub] = 0.4 * line
    c.height[sub] += 0.7 + 0.5 * line


def mountain(g, place, w=150.0, h=170.0, colors=("#1d5a3a", "#3f8f5c", "#a7d6a8", "#f2ead2")):
    """江崖：三峰山石，退晕层。"""
    peaks = np.array([(-w / 2, 0), (-w * 0.42, h * 0.4), (-w * 0.3, h * 0.55), (-w * 0.2, h * 0.5), (-w * 0.12, h * 0.85), (0, h), (w * 0.12, h * 0.85),
                      (w * 0.2, h * 0.5), (w * 0.3, h * 0.55), (w * 0.42, h * 0.4), (w / 2, 0)])
    g.add([np.concatenate([peaks, [(w * 0.5, -6), (-w * 0.5, -6)]])], place, GOLDLINE)
    for k, col in enumerate(colors):
        sc = 1 - 0.2 * k
        p = peaks * np.array([sc * 0.96, sc * 0.96]) + np.array([0, -2])
        g.add([np.concatenate([p, [(w * 0.5 * sc, -4), (-w * 0.5 * sc, -4)]])], place, K(col, 0.42, 0, 0.8 + 0.2 * k))
    sprays = []
    for sx in (-1, 1):
        for k in range(3):
            c0 = (sx * (w * 0.55 + k * 18), 10 + k * 14)
            sprays.append(spiral(c0, 1, 9, 1.0, 2.2, sign=sx))
    g.add(sprays, place, K("#f4f1ea", 0.4, 0, 1.0))


def scatter_clouds(g, s_lo, s_hi, n, avoid, colors, size=(45, 70), sleeve=False, seed=1):
    r = np.random.RandomState(seed)
    placed = []
    tries = 0
    while len(placed) < n and tries < 4000:
        tries += 1
        th = r.uniform(0, 2 * math.pi)
        s = r.uniform(s_lo, s_hi)
        sz = r.uniform(*size)
        P = g.P(s) if not sleeve else g.sleeve_circ
        x = th / (2 * math.pi) * P
        ok = True
        for (ax, as_, ar) in avoid + placed:
            dx = min(abs(x - ax), P - abs(x - ax))
            if math.hypot(dx, s - as_) < ar + sz * 0.9:
                ok = False
                break
        if ok:
            placed.append((x, s, sz))
            place = g.sleeve(th / (2 * math.pi), s) if sleeve else g.torso(th, s)
            cloud(g, place, sz, colors, flip=r.rand() < 0.5)
    return placed


def border(c, g, s0, s1, color, gold=True):
    """领口/下摆镶边。"""
    H, W = c.H, c.W
    v = 1 - (np.arange(H) + 0.5) / H
    s = v * g.S
    rows = np.where((s >= s0) & (s <= s1))[0]
    W_t = int(W * TORSO_U)
    sub = (slice(rows[0], rows[-1] + 1), slice(0, W_t))
    c.color[sub] = rgb(color)
    c.rough[sub] = 0.36
    c.metal[sub] = 0
    if gold:
        for edge in (rows[0], rows[-1]):
            e = slice(max(edge - 2, 0), edge + 3)
            c.color[e, :W_t] = rgb(GOLD)
            c.metal[e, :W_t] = 0.85


def damask(c, base, amount=0.06, region=(0, 1)):
    """暗纹：极淡的团花底纹。"""
    yy, xx = np.mgrid[0:c.H, 0:c.W].astype(np.float32)
    p = np.sin(xx / 9.0) * np.sin(yy / 9.0) + 0.6 * np.sin((xx + yy) / 13.0)
    m = (p > 0.9).astype(np.float32)
    m = ndimage.gaussian_filter(m, 1.0)
    c.color *= (1 + amount * m[..., None])


def sleeve_cuff(c, g, colors, width=90.0):
    """袖口镶边 + 立水条。"""
    H, W = c.H, c.W
    v = 1 - (np.arange(H) + 0.5) / H
    s = v * g.sleeve_len
    rows = np.where(s <= width)[0]
    W0 = int(W * TORSO_U)
    pal = np.array([rgb(p) for p in colors])
    u = (np.arange(W0, W) + 0.5) / W
    x = (u - TORSO_U) / (1 - TORSO_U) * g.sleeve_circ
    S_ = s[rows][:, None]
    ph = (x[None, :] * 0.5 + S_ * 0.8) / 12.0
    idx = np.floor(ph).astype(int) % len(pal)
    frac = ph - np.floor(ph)
    col = pal[idx] * (0.8 + 0.3 * np.sin(frac * np.pi))[..., None]
    gold = np.clip(1 - np.minimum(frac, 1 - frac) / 0.07, 0, 1)
    sub = (slice(rows[0], rows[-1] + 1), slice(W0, W))
    c.color[sub] = col * (1 - gold[..., None]) + rgb(GOLD) * gold[..., None]
    c.metal[sub] = 0.85 * gold
    c.height[sub] += 0.8
    # 金边
    e = rows[0]
    c.color[e - 3:e + 3, W0:] = rgb(GOLD)
    c.metal[e - 3:e + 3, W0:] = 0.9


# ------------------------------------------------------------------ 成品
LISHUI = ["#173f7a", "#2f6db5", "#8fc0e8", "#f2efe6", "#2f7a4a", "#8cc79a", "#f2efe6", "#b8343a", "#efa7a7", "#f2efe6", "#c89a2e", "#f3dc94", "#f2efe6"]


def mang(name, base, cloud_cols, dragon_on=True, prof="mang", peony_cols=None):
    c = Canvas(size=2048, mapping="uv", w_mm=1.0, h_mm=1.0, base=base, rough=0.4)
    c.x0 = c.z0 = 0.0
    g = Garment(c, prof)
    damask(c, base)
    S = g.S
    hem = 0 if prof == "mang" else 0
    wave_band(c, g, 0, 175 if prof == "mang" else 130, LISHUI)
    flat_waves(c, g, 175 if prof == "mang" else 130, 255 if prof == "mang" else 190)
    order = []
    avoid = []
    top_wave = 255 if prof == "mang" else 190
    for th in (math.pi, 0.0, math.pi / 2, 3 * math.pi / 2):
        P = g.P(150)
        g.phase()
        mountain(g, g.torso(th, 120 if prof == "mang" else 90), w=170 if th in (math.pi, 0.0) else 120, h=190 if th in (math.pi, 0.0) else 140)
        avoid.append((th / (2 * math.pi) * g.P(top_wave + 40), top_wave + 40, 90))
    chest = S - 300
    if dragon_on:
        for th in (math.pi, 0.0):
            g.phase()
            dragon(g, g.torso(th, chest), 150)
            avoid.append((th / (2 * math.pi) * g.P(chest), chest, 170))
        for th in (math.pi * 0.62, math.pi * 1.38):  # 下身两侧行龙（小团龙）
            g.phase()
            dragon(g, g.torso(th, (top_wave + chest) / 2 - 60), 95)
            avoid.append((th / (2 * math.pi) * g.P((top_wave + chest) / 2 - 60), (top_wave + chest) / 2 - 60, 110))
    if peony_cols:
        for th, s in ((math.pi, chest), (0.0, chest), (math.pi * 0.6, (top_wave + chest) / 2), (math.pi * 1.4, (top_wave + chest) / 2), (math.pi, (top_wave + chest) / 2 - 40)):
            g.phase()
            peony(g, g.torso(th, s), 95 if s == chest else 70, peony_cols)
            avoid.append((th / (2 * math.pi) * g.P(s), s, 105))
    g.phase()
    scatter_clouds(g, top_wave + 60, S - 80, 26, avoid, cloud_cols, seed=3)
    # 袖子：云纹
    scatter_clouds(g, 120, g.sleeve_len - 60, 9, [], cloud_cols, size=(40, 55), sleeve=True, seed=9)
    keys = [GOLDLINE] + [k for k in g.batches if k not in (GOLDLINE, GOLDK)] + [GOLDK]
    # 维持插入顺序：按批次插入先后绘制（退晕层在 GOLDLINE 之后、GOLDK 之前）
    ordered = [GOLDLINE] + [k for k in list(g.batches.keys()) if k not in (GOLDLINE, GOLDK)] + [GOLDK]
    g.flush(ordered)
    # 第二批：白爪、眼睛等写在 GOLDK 之后的元素已含在 ordered 中
    sleeve_cuff(c, g, LISHUI[:7])
    border(c, g, S - 26, S, "#141414")
    return c.save(name, pores=0.0, nstrength=2.4)


def scales_texture(c, region, scale_mm=(26, 20), base="#151414", rim=GOLD, inner="#2a2622", per_mm=None):
    """鱼鳞甲（逐像素）：金边黑鳞。region: (u0,u1,v0,v1)，per_mm: (宽 mm, 高 mm)。"""
    u0, u1, v0, v1 = region
    W, H = c.W, c.H
    cols = np.arange(int(u0 * W), int(u1 * W))
    rows = np.arange(int((1 - v1) * H), int((1 - v0) * H))
    uw, vh = per_mm
    x = (cols[None, :] - cols[0]) / len(cols) * uw
    y = (rows[-1] - rows[:, None]) / len(rows) * vh
    cw, ch = scale_mm
    r_i = np.floor(y / ch)
    xx = x + (r_i % 2) * cw / 2
    lx = (xx % cw) - cw / 2
    ly = y - r_i * ch
    d = np.sqrt((lx / (cw / 2)) ** 2 + ((ly - ch) / (ch * 1.25)) ** 2)
    rimm = np.clip(1 - np.abs(d - 0.92) / 0.09, 0, 1)
    shade = np.clip(1 - d, 0, 1)
    col = rgb(inner) * (0.6 + 0.8 * shade[..., None]) * (1 - rimm[..., None]) + rgb(rim) * rimm[..., None]
    sub = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    c.color[sub] = col
    c.metal[sub] = 0.9 * rimm + 0.2 * (1 - rimm)
    c.rough[sub] = 0.3 + 0.15 * (1 - rimm)
    c.height[sub] += 1.2 * shade + 0.8 * rimm


def kao():
    """大靠：靠身与袖子的鱼鳞纹 + 镶边。"""
    c = Canvas(size=2048, mapping="uv", w_mm=1.0, h_mm=1.0, base="#141313", rough=0.38)
    c.x0 = c.z0 = 0.0
    g = Garment(c, "kao", sleeve_len=560)
    scales_texture(c, (0, TORSO_U, 0, 1), scale_mm=(44, 32), per_mm=(g.P(g.S * 0.6), g.S))
    scales_texture(c, (TORSO_U, 1, 0, 1), scale_mm=(30, 22), per_mm=(g.sleeve_circ, g.sleeve_len))
    border(c, g, g.S - 30, g.S, "#1e1e1e")
    border(c, g, 0, 36, "#7e1416")
    sleeve_cuff(c, g, ["#7e1416", "#c8a040", "#1f4f8f"], width=60)
    c.save("kao", pores=0.0, nstrength=2.2)

    # 靠腿 / 靠肚 / 护肩用的方形贴图：鳞片 + 彩边
    p = Canvas(size=1024, mapping="uv", w_mm=1.0, h_mm=1.0, base="#141313", rough=0.38)
    p.x0 = p.z0 = 0.0
    scales_texture(p, (0, 1, 0, 1), scale_mm=(24, 18), per_mm=(300, 300))
    X, Z = p.grid()
    edge = np.minimum(np.minimum(X, 1 - X), np.minimum(Z, 1 - Z))
    for w, col in ((0.075, "#c8a040"), (0.06, "#7e1416"), (0.035, "#1f4f8f"), (0.015, "#c8a040")):
        m = (edge < w).astype(np.float32)
        p.apply(ndimage.gaussian_filter(m, 0.7), col, rough=0.35, metal=0.85 if col == "#c8a040" else 0.0, thick=0.6, noise=0)
    p.save("kao_panel", pores=0.0, nstrength=2.0)

    # 靠肚：兽面
    b = Canvas(size=1024, mapping="uv", w_mm=300.0, h_mm=300.0, base="#7e1416", rough=0.38)
    b.x0, b.z0 = -150.0, -150.0
    gb = Garment.__new__(Garment)
    gb.c, gb.batches = b, {}
    ident = lambda pts: (np.asarray(pts) + 150.0) / 300.0
    b.x0 = b.z0 = 0.0
    b.sx = b.sz = 1.0
    gb.add([ring((0, 0), 132, 14, 90)], ident, GOLDK)
    for s in (1, -1):
        gb.add([catmull(np.array([(20, 30), (60, 62), (100, 50), (110, 20), (70, 30), (40, 14)]) * np.array([s, 1]))], ident, GOLDK)  # 眉
        gb.add([circle((45 * s, 5), 24)], ident, K("#f2ede2", 0.3, 0, 1.2))
        gb.add([circle((45 * s, 5), 11)], ident, K("#111111", 0.3, 0, 1.4))
        gb.add([spiral((90 * s, -50), 2, 22, 1.2, 5, sign=s)], ident, GOLDK)
    gb.add([catmull(np.array([(-18, 20), (18, 20), (26, -24), (0, -34), (-26, -24)]))], ident, GOLDK)  # 鼻
    gb.add([catmull(np.array([(-70, -48), (0, -62), (70, -48), (50, -96), (0, -108), (-50, -96)]))], ident, K("#1a1212", 0.3))
    gb.phase()
    teeth = [np.array([(x - 8, -62), (x + 8, -62), (x, -80)]) for x in range(-48, 56, 16)]
    gb.add(teeth, ident, K("#f2ede2", 0.3, 0, 1.2))
    gb.flush([GOLDK, K("#f2ede2", 0.3, 0, 1.2), K("#111111", 0.3, 0, 1.4), K("#1a1212", 0.3)] + list(gb.batches.keys()))
    b.save("kao_belly", pores=0.0, nstrength=2.2)

    # 靠旗：黑底金龙 + 火焰边
    f = Canvas(size=(512, 1024), mapping="uv", w_mm=1.0, h_mm=1.0, base="#121212", rough=0.4)
    f.x0 = f.z0 = 0.0
    gf = Garment.__new__(Garment)
    gf.c, gf.batches = f, {}
    pl = lambda pts: np.asarray(pts) / np.array([250.0, 700.0]) + np.array([0.0, 0.0])
    # 三角旗：u=0 为旗杆一侧，v=1 为顶端；斜边从 (0,0) 到 (1,1)
    flames = []
    for k in range(16):
        t0, t1 = k / 16, (k + 1) / 16
        a = np.array((250 * t0, 700 * t0))
        bb = np.array((250 * t1, 700 * t1))
        mid = (a + bb) / 2 + np.array((24, -10))
        flames.append(np.array([a, mid, bb, (bb + a) / 2 - np.array((14, -4))]))
    gf.add(flames, pl, K("#d8452a", 0.4, 0, 1.0))
    gf.add([ribbon([(10, 30), (232, 690)], 18, 18)], pl, GOLDK)
    gf.add([ribbon([(4, 20), (4, 690)], 10, 10), ribbon([(0, 690), (250, 690)], 12, 12)], pl, GOLDK)
    gf.flush([K("#d8452a", 0.4, 0, 1.0), GOLDK])
    gd = Garment.__new__(Garment)
    gd.c, gd.batches = f, {}
    dragon(gd, lambda pts: (np.asarray(pts) * 0.42 + np.array([75, 470])) / np.array([250.0, 700.0]), 150)
    gd.flush()
    f.save("kao_flag", pores=0.0, nstrength=1.6)


def houyi():
    """猴衣：杏黄底 + 黑色短毛纹；虎皮裙。"""
    c = Canvas(size=1024, mapping="uv", w_mm=1.0, h_mm=1.0, base="#e2a925", rough=0.45)
    c.x0 = c.z0 = 0.0
    yy, xx = np.mgrid[0:c.H, 0:c.W].astype(np.float32)
    r = np.random.RandomState(4)
    m = np.zeros((c.H, c.W), np.float32)
    for k in range(1400):
        x0, y0 = r.uniform(0, c.W), r.uniform(0, c.H)
        L = r.uniform(8, 14)
        a = r.uniform(-0.4, 0.4) + math.pi / 2
        for t in np.linspace(0, 1, 12):
            px, py = int(x0 + math.cos(a) * L * t) % c.W, int(y0 + math.sin(a) * L * t) % c.H
            m[py, px] = 1
    m = ndimage.gaussian_filter(ndimage.maximum_filter(m, 2), 0.5)
    c.apply(np.clip(m * 1.5, 0, 1), "#3a2210", rough=0.5, thick=0.6, noise=0)
    c.save("houyi", pores=0.2, nstrength=1.5)

    t = Canvas(size=1024, mapping="uv", w_mm=1.0, h_mm=1.0, base="#e07a1f", rough=0.6)
    t.x0 = t.z0 = 0.0
    X, Z = t.grid()
    n = ndimage.gaussian_filter(np.random.RandomState(8).randn(t.H, t.W), 18)
    n /= np.abs(n).max()
    ph = X * 8 + n * 0.45 + np.sin(Z * 9 + X * 20) * 0.08
    taper = 0.55 + 0.35 * np.sin(Z * 14 + X * 31)
    stripe = np.clip((np.cos(ph * 2 * np.pi) - taper) * 6, 0, 1) * np.clip(Z * 1.4, 0, 1)
    t.apply(stripe, "#151010", rough=0.6, thick=0.5, noise=0)
    belly = np.clip((0.12 - Z) * 14, 0, 1)
    t.apply(belly, "#f2e8d6", rough=0.7, thick=0.8, noise=0)
    t.save("tiger", pores=0.6, nstrength=2.0)


def yunjian():
    """云肩：径向展开（u 绕颈一周，v 从颈到外缘），多层如意头色带。"""
    c = Canvas(size=1024, mapping="uv", w_mm=1.0, h_mm=1.0, base="#1f4f8f", rough=0.38)
    c.x0 = c.z0 = 0.0
    X, Z = c.grid()
    rad = 1 - Z  # 0 颈 -> 1 外缘
    a = X * 2 * np.pi
    lobe = 0.5 + 0.5 * np.cos(a * 8)
    bands = [("#c8a040", 0.12), ("#b3264a", 0.32), ("#f0b8c4", 0.45), ("#2f6b3a", 0.6), ("#8cc79a", 0.72), ("#1f4f8f", 0.86), ("#c8a040", 0.95)]
    for col, r0 in bands:
        edge = r0 + 0.05 * lobe
        m = np.clip((rad - edge) * 120, 0, 1)
        c.apply(m, col, rough=0.36, metal=0.85 if col == "#c8a040" else 0.0, thick=0.4, noise=0)
    gold_lines = np.zeros_like(rad)
    for col, r0 in bands:
        edge = r0 + 0.05 * lobe
        gold_lines = np.maximum(gold_lines, np.clip(1 - np.abs(rad - edge) * 220, 0, 1))
    c.apply(gold_lines, GOLD, rough=0.3, metal=0.9, thick=1.0, noise=0)
    c.save("yunjian", pores=0.0, nstrength=2.0)


def skirt():
    c = Canvas(size=1024, mapping="uv", w_mm=1.0, h_mm=1.0, base="#efe8dc", rough=0.45)
    c.x0 = c.z0 = 0.0
    X, Z = c.grid()
    m = np.clip((0.16 - Z) * 60, 0, 1)
    c.apply(m, "#b3264a", rough=0.4, thick=0.5, noise=0)
    for z0 in (0.17, 0.2):
        c.apply(np.clip(1 - np.abs(Z - z0) * 300, 0, 1), GOLD, rough=0.3, metal=0.9, thick=0.8, noise=0)
    # 马面上的折枝花
    for k in range(24):
        x = (k + 0.5) / 24
        c.ellipse((x, 0.08), 0.012, 0.03, "#f0b8c4", rough=0.4)
    c.save("skirt", pores=0.0, nstrength=1.5)


def silk():
    c = Canvas(size=512, mapping="uv", w_mm=1.0, h_mm=1.0, base="#f4f2ec", rough=0.32)
    c.x0 = c.z0 = 0.0
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    c.height += 0.15 * np.sin(xx * 1.7) + 0.15 * np.sin(yy * 1.7)
    c.save("silk", pores=0.0, nstrength=1.0)


JOBS = {
    "mang_green": lambda: mang("mang_green", "#1d5432", ("#1f5f9e", "#4f8fcf", "#bfe0f5")),
    "mang_red": lambda: mang("mang_red", "#93191c", ("#1d4f8a", "#5b8fd0", "#cfe6f7")),
    "mang_yellow": lambda: mang("mang_yellow", "#d6a320", ("#1f5f9e", "#6aa0d8", "#d4e9f7"), dragon_on=False, prof="nvmang", peony_cols=("#b3264a", "#e0607c", "#f7c6cf")),
    "kao": kao,
    "houyi": houyi,
    "yunjian": yunjian,
    "skirt": skirt,
    "silk": silk,
}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(JOBS)):
        print(n, JOBS[n]())
