"""脸谱/织物贴图绘制工具（系统 Python：Pillow + numpy + scipy）。

画布坐标：毫米，脸部正面投影 (x 向右, z 向上)。每个图层写入颜色、厚度和粗糙度，最终输出：
  *_base.jpg  颜色（sRGB）
  *_orm.png   R=AO(1)，G=粗糙度，B=金属度（glTF metallicRoughness 约定）
  *_nrm.png   切线空间法线（由颜料厚度 + 细噪声计算）
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blender"))
from face_params import TEX_X0, TEX_Z0, TEX_SIZE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "tex")
os.makedirs(OUT, exist_ok=True)


def rgb(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    c = np.asarray(c, np.float32)
    return c / 255.0 if c.max() > 1.0 else c


def catmull(pts, closed=True, n=24):
    pts = [np.asarray(p, np.float64) for p in pts]
    if len(pts) < 3:
        return np.array(pts)
    P = pts + pts[:3] if closed else [pts[0]] + pts + [pts[-1]]
    out = []
    rng = range(len(pts)) if closed else range(len(pts) - 1)
    for i in rng:
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(pts[-1])
    return np.array(out)


def ribbon(pts, w0, w1=None, n=24, wfun=None):
    """沿开放曲线生成变宽笔画多边形（宽度单位与坐标一致）。"""
    c = catmull(pts, closed=False, n=n)
    w1 = w0 if w1 is None else w1
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(c))
    w = wfun(t) if wfun else (w0 + (w1 - w0) * t)
    left = c + nrm * (w[:, None] / 2)
    right = c - nrm * (w[:, None] / 2)
    return np.concatenate([left, right[::-1]])


class Canvas:
    """以毫米为单位的绘制画布。mapping='face' 时使用脸谱正面投影；否则 (0..w_mm, 0..h_mm)。"""

    def __init__(self, size=2048, ss=2, mapping="face", w_mm=None, h_mm=None, base="#808080", rough=0.45, metal=0.0, height=(None)):
        self.W = size if isinstance(size, int) else size[0]
        self.H = size if isinstance(size, int) else size[1]
        self.ss = ss
        self.mapping = mapping
        if mapping == "face":
            self.x0, self.z0 = TEX_X0 * 1000, TEX_Z0 * 1000
            self.sx = self.sz = TEX_SIZE * 1000
        else:
            self.x0, self.z0 = 0.0, 0.0
            self.sx, self.sz = w_mm, h_mm
        self.color = np.ones((self.H, self.W, 3), np.float32) * rgb(base)
        self.rough = np.full((self.H, self.W), rough, np.float32)
        self.metal = np.full((self.H, self.W), metal, np.float32)
        self.height = np.zeros((self.H, self.W), np.float32)

    # 坐标换算
    def px(self, pts):
        pts = np.asarray(pts, np.float64)
        u = (pts[:, 0] - self.x0) / self.sx * self.W * self.ss
        v = (1.0 - (pts[:, 1] - self.z0) / self.sz) * self.H * self.ss
        return np.stack([u, v], 1)

    def mm_per_px(self):
        return self.sx / self.W

    def grid(self):
        u = (np.arange(self.W) + 0.5) / self.W
        v = 1.0 - (np.arange(self.H) + 0.5) / self.H
        X = self.x0 + u * self.sx
        Z = self.z0 + v * self.sz
        return np.meshgrid(X, Z)

    def mask_poly(self, polys, blur=0.0):
        im = Image.new("L", (self.W * self.ss, self.H * self.ss), 0)
        dr = ImageDraw.Draw(im)
        for p in polys:
            q = [tuple(x) for x in self.px(p)]
            if len(q) >= 3:
                dr.polygon(q, fill=255)
        im = im.resize((self.W, self.H), Image.LANCZOS)
        m = np.asarray(im, np.float32) / 255.0
        if blur > 0:
            m = ndimage.gaussian_filter(m, blur / self.mm_per_px())
        return m

    def mask_draw(self, fn, blur=0.0):
        im = Image.new("L", (self.W * self.ss, self.H * self.ss), 0)
        dr = ImageDraw.Draw(im)
        fn(dr, self)
        im = im.resize((self.W, self.H), Image.LANCZOS)
        m = np.asarray(im, np.float32) / 255.0
        if blur > 0:
            m = ndimage.gaussian_filter(m, blur / self.mm_per_px())
        return m

    def apply(self, m, color, rough=None, metal=None, thick=0.6, alpha=1.0, noise=0.03):
        a = np.clip(m * alpha, 0, 1)[..., None]
        col = rgb(color)
        if noise:
            col = col[None, None, :] * (1.0 + noise * self.lownoise[..., None])
        self.color = self.color * (1 - a) + col * a
        if rough is not None:
            self.rough = self.rough * (1 - a[..., 0]) + rough * a[..., 0]
        if metal is not None:
            self.metal = self.metal * (1 - a[..., 0]) + metal * a[..., 0]
        self.height += thick * a[..., 0]

    @property
    def lownoise(self):
        if not hasattr(self, "_ln"):
            rs = np.random.RandomState(7)
            n = rs.randn(self.H // 16 + 2, self.W // 16 + 2).astype(np.float32)
            n = ndimage.zoom(n, 16, order=3)[: self.H, : self.W]
            self._ln = n / (np.abs(n).max() + 1e-6)
        return self._ln

    # 便捷图元
    def fill(self, pts, color, mirror=False, blur=0.25, closed_smooth=True, **kw):
        polys = [catmull(pts) if closed_smooth else np.asarray(pts, np.float64)]
        if mirror:
            polys.append(polys[0] * np.array([-1, 1]))
        self.apply(self.mask_poly(polys, blur), color, **kw)

    def stroke(self, pts, w0, w1=None, color="#000", mirror=False, blur=0.2, wfun=None, **kw):
        polys = [ribbon(pts, w0, w1, wfun=wfun)]
        if mirror:
            polys.append(polys[0] * np.array([-1, 1]))
        self.apply(self.mask_poly(polys, blur), color, **kw)

    def ellipse(self, c, rx, rz, color, mirror=False, blur=0.25, rot=0.0, **kw):
        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ca, sa = math.cos(rot), math.sin(rot)
        ex, ez = rx * np.cos(t), rz * np.sin(t)
        pts = np.stack([c[0] + ex * ca - ez * sa, c[1] + ex * sa + ez * ca], 1)
        polys = [pts]
        if mirror:
            polys.append(pts * np.array([-1, 1]))
        self.apply(self.mask_poly(polys, blur), color, **kw)

    def radial(self, c, rx, rz, color, alpha=1.0, power=1.5, mirror=False, **kw):
        X, Z = self.grid()
        def g(cx):
            r = np.sqrt(((X - cx) / rx) ** 2 + ((Z - c[1]) / rz) ** 2)
            return np.clip(1 - r, 0, 1) ** power
        m = g(c[0])
        if mirror:
            m = np.maximum(m, g(-c[0]))
        self.apply(m, color, alpha=alpha, thick=0.0, **kw)

    def vignette(self, color, r0, r1, center=(0, 0), alpha=1.0):
        X, Z = self.grid()
        r = np.sqrt((X - center[0]) ** 2 + ((Z - center[1]) * 0.8) ** 2)
        m = np.clip((r - r0) / (r1 - r0), 0, 1) ** 1.4
        self.apply(m, color, alpha=alpha, thick=0.0, noise=0.0)

    # 输出
    def save(self, name, pores=0.35, nstrength=1.6, quality=92):
        rs = np.random.RandomState(11)
        fine = ndimage.gaussian_filter(rs.randn(self.H, self.W).astype(np.float32), 1.2)
        fine /= np.abs(fine).max() + 1e-6
        col = self.color * (1.0 + 0.025 * fine[..., None])
        col = np.clip(col, 0, 1)
        Image.fromarray((col * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f"{name}_base.jpg"), quality=quality)
        rough = np.clip(self.rough + 0.04 * fine, 0.02, 1)
        orm = np.stack([np.ones_like(rough), rough, np.clip(self.metal, 0, 1)], -1)
        Image.fromarray((orm * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f"{name}_orm.png"))
        h = ndimage.gaussian_filter(self.height, 1.6) + pores * fine * 0.25
        gy, gx = np.gradient(h)
        nx, ny = -gx * nstrength, gy * nstrength
        nz = np.ones_like(nx)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz)
        nrm = np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5
        Image.fromarray((nrm * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f"{name}_nrm.png"))
        prev = Image.fromarray((col * 255 + 0.5).astype(np.uint8)).resize((512, int(512 * self.H / self.W)))
        prev.save(os.path.join(OUT, f"{name}_preview.png"))
        return os.path.join(OUT, f"{name}_base.jpg")
