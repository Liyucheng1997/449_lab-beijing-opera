"""眼球、髯口发丝、翎子、绒球等小贴图。"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from paint import OUT

rs = np.random.RandomState(3)


def save(arr, name):
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name))


def eye():
    N = 512
    y, x = np.mgrid[0:N, 0:N]
    u = (x + 0.5) / N * 2 - 1
    v = 1 - (y + 0.5) / N * 2
    r = np.sqrt(u * u + v * v)
    a = np.arctan2(v, u)
    col = np.ones((N, N, 3), np.float32) * np.array([0.93, 0.9, 0.88])
    veins = ndimage.gaussian_filter(rs.rand(N, N), 2) > 0.62
    col[veins & (r > 0.6)] = col[veins & (r > 0.6)] * np.array([1.0, 0.78, 0.76])
    R_IRIS, R_PUP = 0.48, 0.19
    fib = 0.5 + 0.5 * np.sin(a * 90 + 4 * ndimage.gaussian_filter(rs.randn(N, N), 3))
    iris = np.array([0.24, 0.14, 0.07])[None, None] * (0.7 + 0.5 * fib[..., None]) * (0.6 + 0.6 * (r[..., None] / R_IRIS))
    m = np.clip((R_IRIS - r) / 0.012, 0, 1)[..., None]
    col = col * (1 - m) + iris * m
    limb = np.clip(1 - np.abs(r - R_IRIS + 0.02) / 0.035, 0, 1)[..., None]
    col = col * (1 - 0.7 * limb)
    pm = np.clip((R_PUP - r) / 0.01, 0, 1)[..., None]
    col = col * (1 - pm) + 0.015 * pm
    save(col, "eye_base.jpg")


def hair(name="hair", W=256, H=1024, n=120, taper=True, curl=0.0):
    a = np.zeros((H, W), np.float32)
    val = np.zeros((H, W), np.float32)
    ys = np.arange(H)
    for _ in range(n):
        x0 = rs.uniform(0, W)
        length = rs.uniform(0.6, 1.0) * H
        width = rs.uniform(0.8, 2.0)
        bright = rs.uniform(0.55, 1.0)
        wob = rs.uniform(2, 6) + curl * 30
        ph = rs.uniform(0, 6.28)
        xs = x0 + wob * np.sin(ys / H * rs.uniform(2, 5) + ph)
        for yy in range(0, int(length)):
            xx = xs[yy]
            lo, hi = int(xx - width - 1), int(xx + width + 2)
            for k in range(max(lo, 0), min(hi, W)):
                d = abs(k + 0.5 - xx)
                aa = max(0.0, 1 - d / width)
                if taper:
                    aa *= min(1.0, (length - yy) / (0.25 * H))
                if aa > a[yy, k]:
                    a[yy, k] = aa
                    val[yy, k] = bright
    a = np.clip(a * 1.4, 0, 1)
    rgb = np.stack([val, val, val], -1) * 0.85 + 0.15
    out = np.concatenate([rgb, a[..., None]], -1)
    Image.fromarray((out * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT, f"{name}.png"))


def feather():
    """雉鸡翎：白底褐色横斑，中轴，边缘透明，末端收尖。"""
    W, H = 128, 2048
    y, x = np.mgrid[0:H, 0:W]
    u = (x + 0.5) / W * 2 - 1
    t = (y + 0.5) / H  # 0 根部 -> 1 尖
    half = 0.95 * (1 - t ** 3) * np.clip(t * 25, 0, 1)
    alpha = np.clip((half - np.abs(u)) * 40, 0, 1)
    jag = ndimage.gaussian_filter(rs.rand(H, W), (0.5, 3)) * 0.12
    alpha *= np.clip((half - np.abs(u) - jag) * 30, 0, 1)
    band = 0.5 + 0.5 * np.sin(t * 160 + np.abs(u) * 2.2)
    base = np.array([0.82, 0.74, 0.58])
    dark = np.array([0.25, 0.14, 0.07])
    col = base[None, None] * (1 - band[..., None] * 0.85) + dark[None, None] * band[..., None] * 0.85
    shaft = np.clip(1 - np.abs(u) / 0.05, 0, 1)[..., None]
    col = col * (1 - shaft) + np.array([0.92, 0.88, 0.78]) * shaft
    barbs = 0.85 + 0.15 * np.sin((y * 0.9 + np.abs(x - W / 2) * 3.0))
    col *= barbs[..., None]
    out = np.concatenate([col, alpha[..., None]], -1)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT, "feather.png"))


def fur():
    """绒球 / 狐尾的绒毛法线噪声。"""
    N = 512
    n = ndimage.gaussian_filter(rs.randn(N, N), 0.8)
    gy, gx = np.gradient(n)
    k = 6.0
    nx, ny, nz = -gx * k, gy * k, np.ones_like(gx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    save(np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5, "fur_nrm.png")


if __name__ == "__main__":
    eye()
    hair()
    hair("hair_curly", n=150, curl=0.25)
    feather()
    fur()
    print("ok")
