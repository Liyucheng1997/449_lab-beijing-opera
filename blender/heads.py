"""角色头像：脸谱头部 + 眼球 + 水纱 + 盔头 + 髯口。"""
import math
import numpy as np
import bpy

import geo
from geo import MB, HeadShell, interp_deg
import mats
import headgen
from face_params import HEADS

R = np.random.RandomState(42)
DEG = math.pi / 180


def empty(name, parent=None):
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = parent
    return e


def ring_from_shell(sh, theta_fn, offset, n=120, phi_range=(-math.pi, math.pi)):
    closed = phi_range == (-math.pi, math.pi)
    pts = []
    for ph in np.linspace(phi_range[0], phi_range[1], n, endpoint=not closed):
        p, nr = sh.hit(theta_fn(ph), ph)
        pts.append(np.array(p) + np.array(nr) * offset)
    pts = np.array(pts)
    if closed:
        pts = np.concatenate([pts, pts[:1]])
    return pts


def surf(sh, theta, phi, offset):
    p, n = sh.hit(theta, phi)
    return np.array(p) + np.array(n) * offset, np.array(n)


def pompom(mb, c, r):
    v, f, uv = geo.uv_sphere(c, r, 14, 9)
    v = v + (v - np.asarray(c)) * R.uniform(-0.05, 0.05, (len(v), 1))
    mb.add(v, f, uv * 3)


def pearls_along(mb, pts, r, step=None):
    pts = np.asarray(pts)
    if step:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        targets = np.arange(0, s[-1], step)
        pts = np.stack([np.interp(targets, s, pts[:, i]) for i in range(3)], 1)
    for p in pts:
        mb.add(*geo.uv_sphere(p, r, 8, 6))


def plate(poly2d, depth, center, normal, up=(0, 0, 1), tilt=0.0):
    """把二维轮廓放到空间：轮廓 x→水平切向，y→向上；normal 为朝外方向。"""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    upv = np.asarray(up, float)
    xax = np.cross(upv, n)
    xax /= np.linalg.norm(xax)
    yax = np.cross(n, xax)
    if tilt:
        # 绕 xax 向后仰
        c, s = math.cos(tilt), math.sin(tilt)
        yax, n = yax * c - n * s, n * c + yax * s
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = xax, yax, n, center
    return M


def outline_tube(mb, poly2d, M, r, z=0.0):
    pts = np.c_[np.asarray(poly2d), np.full(len(poly2d), z)]
    pts = np.concatenate([pts, pts[:1]])
    pts = geo.transform(pts, M)
    mb.add(*geo.tube(pts, r, 6, caps=False))


def shape(kind, w, h, n=48):
    """常用二维轮廓（底边中点为原点）。"""
    t = np.linspace(0, 1, n)
    if kind == "shield":  # 盾形 / 火焰顶
        xs = np.concatenate([np.linspace(-w / 2, -w / 2 * 0.9, 6), -w / 2 * 0.9 * np.cos(np.linspace(0, np.pi / 2, 20)) * (1 - 0.1 * np.linspace(0, 1, 20))])
        left = [(-w / 2, 0), (-w / 2 * 0.95, h * 0.45), (-w * 0.38, h * 0.75), (-w * 0.2, h * 0.9), (0, h)]
        pts = left + [(-x, y) for x, y in reversed(left[:-1])]
        from_pts = np.array(pts)
        return geo.smooth_path(np.c_[from_pts, np.zeros(len(from_pts))], 60)[:, :2]
    if kind == "arch":  # 尖拱
        left = [(-w / 2, 0), (-w / 2 * 0.9, h * 0.5), (-w * 0.25, h * 0.88), (0, h)]
        pts = left + [(-x, y) for x, y in reversed(left[:-1])]
        p = np.array(pts)
        return geo.smooth_path(np.c_[p, np.zeros(len(p))], 50)[:, :2]
    if kind == "ellipse":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(a) * w / 2, np.sin(a) * h / 2 + h / 2], 1)
    if kind == "flower":
        a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        r = w / 2 * (0.78 + 0.22 * np.cos(5 * a))
        return np.stack([np.cos(a) * r, np.sin(a) * r], 1)
    if kind == "flame":  # 侧翼火焰形
        p = np.array([(0, 0), (w * 0.5, h * 0.15), (w * 0.8, h * 0.45), (w * 0.6, h * 0.55), (w, h), (w * 0.35, h * 0.7), (w * 0.1, h * 0.45), (-w * 0.05, h * 0.2)])
        return geo.smooth_path(np.c_[p, np.zeros(len(p))], 60)[:, :2]
    if kind == "lotus":
        left = [(-w / 2, 0), (-w * 0.48, h * 0.35), (-w * 0.32, h * 0.62), (-w * 0.18, h * 0.55), (-w * 0.1, h * 0.85), (0, h)]
        pts = left + [(-x, y) for x, y in reversed(left[:-1])]
        p = np.array(pts)
        return geo.smooth_path(np.c_[p, np.zeros(len(p))], 60)[:, :2]
    raise KeyError(kind)


# ------------------------------------------------------------------ 基础头部
_base_cache = {}


def base_head(kind):
    """生成（并缓存网格数据）基础头部。"""
    if kind not in _base_cache:
        ob = headgen.build_head(kind)
        _base_cache[kind] = ob.data
        return ob
    me = _base_cache[kind].copy()
    ob = bpy.data.objects.new(f"Head_{kind}", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def eyes(kind, parent):
    p = HEADS[kind]
    mb = MB()
    r = p["eye_rad"]
    for s in (1, -1):
        c = np.array((p["eye_c"][0] * s, p["eye_c"][1], p["eye_c"][2]))
        v, f, uv = geo.uv_sphere((0, 0, 0), r, 28, 18)
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)  # 极点朝 -Y（正前）
        uv = np.stack([0.5 + v[:, 0] / (2 * r), 0.5 + v[:, 2] / (2 * r)], 1)
        mb.add(v + c, f, uv)
    return mb.obj("Eyes", mats.get("eye"), parent=parent)


WANGZI_THETA = interp_deg([(0, 50), (30, 53), (70, 64), (95, 70), (120, 108), (150, 128), (180, 134)])


def wangzi(sh, parent):
    v, f, uv = sh.shell(WANGZI_THETA, 0.0013, nphi=120, nth=36)
    return MB().add(v, f, uv * 6).obj("Wangzi", mats.get("wangzi"), parent=parent)


# ------------------------------------------------------------------ 髯口
def wire_y(x):
    return -0.1115 + 12.0 * x * x


def beard(style, color, parent, kind="male"):
    mb = MB()
    z0 = -0.0565
    anchors = []

    def hang(a, L, spread, fwd=0.0, out=0.0, n=14, wild=0.0):
        a = np.asarray(a, float)
        sway = R.uniform(-0.012, 0.012)
        p = [a,
             a + (out * 0.25, -0.008 - fwd * 0.3, -0.03),
             a + (spread * 0.3 + out * 0.6, -0.014 - fwd * 0.7, -0.09),
             a + (spread * 0.7 + out + sway, -0.02 - fwd, -0.09 - L * 0.45),
             a + (spread + out * 1.1 + sway * 2, -0.028 - fwd * 1.2 + R.uniform(-0.01, 0.01), -L)]
        if wild:
            p = [q + R.uniform(-wild, wild, 3) * (i / 4) for i, q in enumerate(p)]
            p[0] = a
        return geo.smooth_path(p, n)

    def add_card(path, w):
        side = np.array((1.0, R.uniform(-0.3, 0.3), 0))
        mb.add(*geo.card(path, np.linspace(w, w * 0.45, len(path)), side, twist=R.uniform(-0.6, 0.6)))

    if style == "man":  # 满髯：整片垂帘
        for layer in range(4):
            for x in np.linspace(-0.038, 0.038, 26):
                x += R.uniform(-0.002, 0.002)
                a = (x, wire_y(x) - 0.002 * layer, z0 - 0.002 * layer)
                L = R.uniform(0.40, 0.46)
                add_card(hang(a, L, x * 0.9, fwd=0.007 * layer), R.uniform(0.009, 0.013))
    elif style in ("san", "wu"):  # 三髯 / 五髯：分绺
        groups = [(-0.024, 0.0), (0.024, 0.0), (0.0, -1.0)]
        if style == "wu":
            groups += [(-0.046, 0.5), (0.046, 0.5)]
        for gx, kindg in groups:
            L = 0.47 if style == "wu" else 0.42
            if kindg == -1.0:  # 中绺挂在下唇下
                base = np.array((0.0, -0.103, -0.094))
                L -= 0.04
            elif kindg == 0.5:  # 两腮
                base = np.array((gx, wire_y(gx) + 0.004, z0 - 0.006))
                L -= 0.06
            else:
                base = np.array((gx, wire_y(gx), z0))
            for k in range(34):
                a = base + (R.uniform(-0.009, 0.009), R.uniform(-0.004, 0.003), R.uniform(-0.003, 0.003))
                tip_x = gx * 0.6 + R.uniform(-0.009, 0.009)
                path = hang(a, L * R.uniform(0.85, 1.0), tip_x - a[0], fwd=R.uniform(0.0, 0.016), wild=0.004)
                add_card(path, R.uniform(0.006, 0.009))
    elif style == "zha":  # 扎髯：嘴部开口，四散蓬松
        for s in (1, -1):
            for k in range(64):
                x = s * R.uniform(0.012, 0.048)
                a = (x, wire_y(x), z0 + R.uniform(-0.004, 0.004))
                L = R.uniform(0.26, 0.34)
                add_card(hang(a, L, s * R.uniform(0.03, 0.07), fwd=R.uniform(0.01, 0.03), out=s * R.uniform(0.01, 0.04), wild=0.02), R.uniform(0.014, 0.02))
        for k in range(50):
            x = R.uniform(-0.03, 0.03)
            a = (x, -0.1 + 10 * x * x, -0.097)
            add_card(hang(a, R.uniform(0.24, 0.3), x * 1.5, fwd=0.02, wild=0.02), R.uniform(0.014, 0.02))
    hair = mb.obj("Beard", mats.get(f"hair_{color}"), smooth=True, parent=parent)
    # 髯口挂架：过耳的细钢丝
    w = MB()
    for s in (1, -1):
        xs = np.linspace(0, 0.045, 10) * s
        front = [(x, wire_y(x) - 0.001, z0) for x in xs]
        back = [(0.062 * s, -0.045, -0.045), (0.078 * s, -0.01, -0.02), (0.084 * s, 0.01, 0.012), (0.08 * s, 0.022, 0.03), (0.072 * s, 0.03, 0.02)]
        w.add(*geo.tube(geo.smooth_path(front + back, 50), 0.0008, 6))
    wire = w.obj("BeardWire", mats.get("wire"), parent=hair)
    return hair


# ------------------------------------------------------------------ 盔头
BRIM = interp_deg([(0, 57), (40, 59), (80, 64), (110, 72), (180, 80)])


def crown_shell(sh, theta_fn, off, mat, parent, name, nphi=110, rim=True, rim_mat="gold", rim_r=0.0035):
    v, f, uv = sh.shell(theta_fn, lambda t, ph: off + 0.002 * t, nphi=nphi, nth=34)
    ob = MB().add(v, f, uv * 4).obj(name, mats.get(mat), parent=parent)
    if rim:
        g = MB()
        g.add(*geo.tube(ring_from_shell(sh, theta_fn, off + 0.002, 140), rim_r, 8, caps=False))
        g.obj(name + "_Rim", mats.get(rim_mat), parent=parent)
    return ob


def tassel(mb_hair, mb_gold, top, L=0.26, n=26, spread=0.012, knot=0.009):
    top = np.asarray(top, float)
    mb_gold.add(*geo.uv_sphere(top, knot, 12, 8))
    for k in range(n):
        a = top + (R.uniform(-spread, spread), R.uniform(-spread, spread), -knot * 0.6)
        path = geo.smooth_path([a, a + (R.uniform(-0.01, 0.01), R.uniform(-0.01, 0.01), -L * 0.5), a + (R.uniform(-0.02, 0.02), R.uniform(-0.02, 0.02), -L)], 10)
        mb_hair.add(*geo.card(path, 0.009, (1, R.uniform(-1, 1), 0)))


def front_point(sh, theta, off):
    return surf(sh, theta, 0.0, off)


def hg_fuzi(sh, parent, color="satin_green"):
    """夫子盔（关羽）：绿缎盔身、金边、红绒球、后兜、黄穗。"""
    root = empty("Headgear", parent)
    crown_shell(sh, BRIM, 0.006, color, root, "Kui")
    band = interp_deg([(0, 52), (60, 55), (120, 62), (180, 70)])
    g, red, pomg, tas = MB(), MB(), MB(), MB()
    g.add(*geo.tube(ring_from_shell(sh, band, 0.009, 140), 0.0025, 8, caps=False))
    # 前额立板
    c, n = front_point(sh, 52 * DEG, 0.009)
    M = plate(None, 0, c + (0, 0, -0.005), (0, -1, 0.0), tilt=12 * DEG)
    poly = shape("shield", 0.12, 0.12)
    body = MB().add(*geo.extrude2d(poly, 0.004, M))
    body.obj("Kui_Front", mats.get(color), parent=root)
    outline_tube(g, poly, M, 0.0025, 0.0025)
    inner = shape("shield", 0.08, 0.085)
    Mi = M.copy()
    Mi[:3, 3] = M[:3, 3] + M[:3, 1] * 0.012
    outline_tube(g, inner, Mi, 0.0014, 0.0035)
    disc_c = geo.transform([(0, 0.06, 0.004)], M)[0]
    g.add(*geo.uv_sphere(disc_c, 0.014, 20, 12, scale=(1, 0.35, 1)))
    red.add(*geo.uv_sphere(disc_c + M[:3, 2] * 0.004, 0.006, 12, 8))
    pompom(pomg, geo.transform([(0, 0.13, 0.0)], M)[0], 0.02)
    for k, ph in enumerate(np.linspace(-60, 60, 7)):
        p, nn = surf(sh, 57 * DEG, ph * DEG, 0.012)
        pompom(pomg, p + nn * 0.004, 0.0085)
    # 后兜
    cb, nb = surf(sh, 40 * DEG, math.pi, 0.008)
    Mb = plate(None, 0, cb + (0, 0, -0.01), (0, 1, 0.2), tilt=-5 * DEG)
    pb = shape("arch", 0.15, 0.17)
    MB().add(*geo.extrude2d(pb, 0.004, Mb)).obj("Kui_Back", mats.get(color), parent=root)
    outline_tube(g, pb, Mb, 0.0028, 0.0025)
    # 两侧耳子 + 黄穗
    for s in (1, -1):
        p, nn = surf(sh, 64 * DEG, s * 80 * DEG, 0.012)
        Mf = plate(None, 0, p, nn, tilt=0)
        fl = shape("flame", 0.05, 0.07) * np.array([s, 1])
        MB().add(*geo.extrude2d(fl, 0.003, Mf)).obj(f"Kui_Ear{s}", mats.get("gold"), parent=root)
        tassel(tas, g, p + nn * 0.01 + (0, -0.01, -0.01), L=0.3)
    g.obj("Kui_Gold", mats.get("gold"), parent=root)
    red.obj("Kui_Ruby", mats.get("ruby"), parent=root)
    pomg.obj("Kui_Pom", mats.get("pom_red"), parent=root)
    tas.obj("Kui_Tassel", mats.get("hair_yellow"), parent=root)
    return root


def hg_xiangdiao(sh, parent, wing="oval", gold_scale=1.0):
    """相貂（乌纱）：前低后高，两侧展翅。"""
    root = empty("Headgear", parent)
    brim = interp_deg([(0, 56), (60, 60), (110, 72), (180, 82)])
    crown_shell(sh, brim, 0.007, "satin_black", root, "Diao")
    g, jade = MB(), MB()
    # 后山（高起的后冠）
    prof = np.array([(-0.07, 0.0), (-0.072, 0.07), (-0.06, 0.1), (-0.03, 0.115), (0.0, 0.118), (0.03, 0.115), (0.06, 0.1), (0.072, 0.07), (0.07, 0.0)])
    prof = geo.smooth_path(np.c_[prof, np.zeros(len(prof))], 50)[:, :2]
    Mb = plate(None, 0, (0, 0.05, 0.065), (0, -1, 0))
    v, f, uv = geo.extrude2d(prof, 0.075, Mb)
    MB().add(v, f, uv).obj("Diao_Back", mats.get("satin_black"), parent=root)
    outline_tube(g, prof, Mb, 0.0022, 0.0375)
    outline_tube(g, prof, Mb, 0.0022, -0.0375)
    # 展翅
    for s in (1, -1):
        rod = [(s * 0.072, 0.07, 0.15), (s * 0.12, 0.072, 0.152), (s * 0.15, 0.072, 0.153)]
        MB().add(*geo.tube(geo.smooth_path(rod, 10), 0.003, 8)).obj(f"Diao_Rod{s}", mats.get("satin_black"), parent=root)
        if wing == "oval":
            wp = shape("ellipse", 0.17, 0.065)
        else:  # 方翅
            wp = np.array([(-0.085, 0.0), (0.085, 0.0), (0.09, 0.03), (0.085, 0.06), (-0.085, 0.06), (-0.09, 0.03)])
            wp = geo.smooth_path(np.c_[np.concatenate([wp, wp[:1]]), np.zeros(len(wp) + 1)], 60)[:-1, :2]
        Mw = plate(None, 0, (s * 0.24, 0.072, 0.12), (0, -1, 0), tilt=0)
        Mw = Mw @ geo.mat_from(rot=(0, 0, s * 6 * DEG))
        MB().add(*geo.extrude2d(wp, 0.002, Mw)).obj(f"Diao_Wing{s}", mats.get("satin_black"), parent=root)
        outline_tube(g, wp, Mw, 0.0018 * gold_scale, 0.0)
    # 帽正：金花托玉
    c, n = front_point(sh, 54 * DEG, 0.01)
    g.add(*geo.extrude2d(shape("flower", 0.024, 0.024), 0.003, plate(None, 0, c, n)))
    jade.add(*geo.uv_sphere(c + n * 0.003, 0.0065, 14, 10, scale=(1, 1, 0.6)))
    g.add(*geo.tube(ring_from_shell(sh, brim, 0.009, 140), 0.0022, 8, caps=False))
    g.obj("Diao_Gold", mats.get("gold"), parent=root)
    jade.obj("Diao_Jade", mats.get("jade"), parent=root)
    return root


def hg_zhajin(sh, parent, color="satin_black", pom="pom_red"):
    """扎巾：黑缎包头，前竖额子，红绒球，后垂飘带。"""
    root = empty("Headgear", parent)
    brim = interp_deg([(0, 58), (60, 62), (110, 76), (180, 88)])
    crown_shell(sh, brim, 0.008, color, root, "Zhajin")
    g, pomg, tas = MB(), MB(), MB()
    c, n = front_point(sh, 55 * DEG, 0.011)
    M = plate(None, 0, c + (0, 0, -0.006), (0, -1, 0), tilt=22 * DEG)
    poly = shape("arch", 0.13, 0.11)
    MB().add(*geo.extrude2d(poly, 0.004, M)).obj("Zhajin_Front", mats.get(color), parent=root)
    outline_tube(g, poly, M, 0.0026, 0.0026)
    for k in range(5):
        pt = geo.transform([(0, 0.02 + k * 0.018, 0.004)], M)[0]
        g.add(*geo.uv_sphere(pt, 0.0045, 10, 8, scale=(1, 1, 0.6)))
    pompom(pomg, geo.transform([(0, 0.125, 0.0)], M)[0], 0.022)
    for s in (1, -1):
        p, nn = surf(sh, 60 * DEG, s * 70 * DEG, 0.014)
        pompom(pomg, p + nn * 0.006, 0.013)
        tassel(tas, g, p + nn * 0.01 + (0, 0, -0.012), L=0.24)
    # 后垂飘带
    rib = MB()
    for s in (1, -1):
        top = np.array((s * 0.025, 0.105, 0.0))
        path = geo.smooth_path([top, top + (s * 0.01, 0.025, -0.08), top + (s * 0.02, 0.04, -0.2), top + (s * 0.03, 0.045, -0.36)], 20)
        rib.add(*geo.card(path, np.linspace(0.035, 0.05, len(path)), (1, 0, 0), twist=s * 0.4))
    rib.obj("Zhajin_Ribbon", mats.get(color), parent=root)
    g.obj("Zhajin_Gold", mats.get("gold"), parent=root)
    pomg.obj("Zhajin_Pom", mats.get(pom), parent=root)
    tas.obj("Zhajin_Tassel", mats.get("hair_redtassel"), parent=root)
    return root


def hg_bawang(sh, parent):
    """霸王盔：黑金盔身，顶插金枪缨，两侧金翅。"""
    root = empty("Headgear", parent)
    brim = interp_deg([(0, 58), (60, 63), (110, 82), (180, 92)])
    crown_shell(sh, brim, 0.009, "satin_black", root, "Bawang")
    g, pomg, red, tas = MB(), MB(), MB(), MB()
    for th in (40, 25):
        g.add(*geo.tube(ring_from_shell(sh, lambda ph, th=th: th * DEG + (brim(ph) - 58 * DEG) * th / 58, 0.012, 140), 0.0025, 8, caps=False))
    c, n = front_point(sh, 54 * DEG, 0.012)
    M = plate(None, 0, c + (0, 0, -0.01), (0, -1, 0), tilt=15 * DEG)
    poly = shape("lotus", 0.13, 0.1)
    g.add(*geo.extrude2d(poly, 0.004, M))
    red.add(*geo.uv_sphere(geo.transform([(0, 0.04, 0.004)], M)[0], 0.01, 14, 10, scale=(1, 1, 0.6)))
    for s in (1, -1):
        red.add(*geo.uv_sphere(geo.transform([(s * 0.035, 0.03, 0.004)], M)[0], 0.006, 12, 8, scale=(1, 1, 0.6)))
    # 顶枪 + 缨
    top, tn = surf(sh, 0.0, 0.0, 0.01)
    stem = [top, top + (0, 0, 0.04), top + (0, 0.005, 0.11)]
    sp = geo.smooth_path(stem, 12)
    g.add(*geo.tube(sp, np.linspace(0.008, 0.003, len(sp)), 10))
    for h in (0.02, 0.05, 0.08):
        g.add(*geo.uv_sphere(top + (0, 0.0015 * h * 30, h), 0.011 - h * 0.05, 14, 8, scale=(1, 1, 0.55)))
    tip = top + (0, 0.005, 0.115)
    pompom(pomg, tip, 0.022)
    for k in range(40):
        a = tip + R.uniform(-0.01, 0.01, 3)
        d = np.array((R.uniform(-1, 1), R.uniform(-0.2, 1), 0))
        d /= np.linalg.norm(d)
        path = geo.smooth_path([a, a + d * 0.05 + (0, 0, 0.0), a + d * 0.08 + (0, 0.02, -0.08), a + d * 0.1 + (0, 0.04, -0.2)], 12)
        tas.add(*geo.card(path, 0.012, (d[1], -d[0], 0) if abs(d[0]) > 0.1 else (1, 0, 0)))
    # 两侧金翅
    for s in (1, -1):
        p, nn = surf(sh, 58 * DEG, s * 85 * DEG, 0.012)
        Mf = plate(None, 0, p, nn)
        fl = shape("flame", 0.07, 0.11) * np.array([-s, 1])
        g.add(*geo.extrude2d(fl, 0.003, Mf))
        for ph in (55, 30):
            q, qn = surf(sh, 60 * DEG, s * ph * DEG, 0.014)
            pompom(pomg, q + qn * 0.004, 0.01)
    g.obj("Bawang_Gold", mats.get("gold"), parent=root)
    red.obj("Bawang_Ruby", mats.get("ruby"), parent=root)
    pomg.obj("Bawang_Pom", mats.get("pom_red"), parent=root)
    tas.obj("Bawang_Ying", mats.get("hair_redtassel"), parent=root)
    return root


def hg_zijinguan(sh, parent):
    """紫金冠 + 翎子 + 狐尾（孙悟空）。"""
    root = empty("Headgear", parent)
    band_lo = interp_deg([(0, 60), (90, 68), (180, 80)])
    band_hi = interp_deg([(0, 48), (90, 56), (180, 66)])
    # 只取冠圈：用 band_hi..band_lo 的条带
    rings = []
    for i in range(6):
        t = i / 5
        rings.append(ring_from_shell(sh, lambda ph, t=t: band_hi(ph) + (band_lo(ph) - band_hi(ph)) * t, 0.008 + 0.001 * math.sin(t * math.pi), 110)[:-1])
    g = MB().add(*geo.loft(np.array(rings), True))
    MB().add(*sh.shell(lambda ph: 38 * DEG, 0.012, nphi=60, nth=12)).obj("Guan_Cap", mats.get("satin_purple"), parent=root)
    red, pomg, fur = MB(), MB(), MB()
    for ph in np.linspace(-90, 90, 9):
        p, n = surf(sh, (band_hi(ph * DEG) + band_lo(ph * DEG)) / 2, ph * DEG, 0.011)
        red.add(*geo.uv_sphere(p, 0.004, 10, 8, scale=(1, 1, 1)))
    c, n = front_point(sh, 50 * DEG, 0.012)
    M = plate(None, 0, c + (0, 0, -0.004), (0, -1, 0), tilt=10 * DEG)
    lot = shape("lotus", 0.1, 0.095)
    g.add(*geo.extrude2d(lot, 0.003, M))
    outline_tube(g, lot * 0.8 + np.array([0, 0.01]), M, 0.0015, 0.002)
    red.add(*geo.uv_sphere(geo.transform([(0, 0.035, 0.003)], M)[0], 0.009, 14, 10, scale=(1, 1, 0.6)))
    pompom(pomg, geo.transform([(0, 0.1, 0)], M)[0], 0.015)
    for s in (1, -1):
        Ms = M @ geo.mat_from(loc=(s * 0.06, -0.005, -0.004), rot=(0, 0, -s * 25 * DEG), scale=(0.6, 0.6, 1))
        g.add(*geo.extrude2d(lot, 0.003, Ms))
        pompom(pomg, geo.transform([(0, 0.1, 0)], Ms)[0], 0.01)
    # 翎子：一对长雉尾，向后上方弯出
    fe = MB()
    for s in (1, -1):
        base, _ = surf(sh, 35 * DEG, s * 25 * DEG, 0.012)
        ctrl = [base, base + (s * 0.03, 0.0, 0.18), base + (s * 0.1, 0.08, 0.55), base + (s * 0.2, 0.3, 0.9), base + (s * 0.3, 0.62, 1.02), base + (s * 0.38, 0.9, 0.95)]
        path = geo.smooth_path(ctrl, 80)
        fe.add(*geo.card(path, np.linspace(0.032, 0.026, len(path)), (1, 0.15 * s, 0), twist=s * 0.5))
        g.add(*geo.tube(path[:6], 0.004, 8))
    fe.obj("Lingzi", mats.get("feather"), parent=root)
    # 狐尾
    for s in (1, -1):
        top, _ = surf(sh, 70 * DEG, s * 95 * DEG, 0.01)
        path = geo.smooth_path([top, top + (s * 0.015, -0.03, -0.08), top + (s * 0.02, -0.07, -0.25), top + (s * 0.01, -0.09, -0.48)], 40)
        rad = 0.022 * np.sin(np.linspace(0.25, 1.0, len(path)) * np.pi) ** 0.6 + 0.004
        fur.add(*geo.tube(path, rad, 16))
    fur.obj("Fur_Hushen", mats.get("pom_white"), parent=root)
    g.obj("Guan_Gold", mats.get("gold"), parent=root)
    red.obj("Guan_Ruby", mats.get("ruby"), parent=root)
    pomg.obj("Guan_Pom", mats.get("pom_red"), parent=root)
    return root


def phoenix_poly(scale=1.0):
    p = np.array([(0, 0), (0.012, 0.01), (0.03, 0.006), (0.05, 0.02), (0.035, 0.03), (0.02, 0.026), (0.012, 0.04), (0.006, 0.055),
                  (0.014, 0.075), (0.004, 0.09), (-0.006, 0.07), (-0.004, 0.05), (-0.014, 0.034), (-0.03, 0.028), (-0.05, 0.02), (-0.03, 0.006), (-0.012, 0.01)])
    p = geo.smooth_path(np.c_[np.concatenate([p, p[:1]]), np.zeros(len(p) + 1)], 120)[:-1, :2]
    return p * scale


def hg_fengguan(sh, parent):
    """凤冠：金点翠，多层花片、正中凤凰、珠帘、挑子与红绒球；后有发髻。"""
    root = empty("Headgear", parent)
    band_lo = interp_deg([(0, 58), (90, 68), (180, 86)])
    band_hi = interp_deg([(0, 50), (90, 58), (180, 74)])
    rings = []
    for i in range(6):
        t = i / 5
        rings.append(ring_from_shell(sh, lambda ph, t=t: band_hi(ph) + (band_lo(ph) - band_hi(ph)) * t, 0.007, 110)[:-1])
    g, cui, pearl, pomg, ruby, tas = MB(), MB(), MB(), MB(), MB(), MB()
    g.add(*geo.loft(np.array(rings), True))
    pearls_along(pearl, ring_from_shell(sh, band_lo, 0.01, 200), 0.0028, step=0.0065)
    # 三层花片
    for tier, (th, off, n, size) in enumerate([(46, 0.016, 13, 0.026), (36, 0.03, 11, 0.03), (25, 0.045, 7, 0.032)]):
        for ph in np.linspace(-105, 105, n):
            p, nn = surf(sh, th * DEG, ph * DEG, off)
            M = plate(None, 0, p, nn, up=(0, 0.3, 1))
            fl = shape("flower", size, size)
            cui.add(*geo.extrude2d(fl, 0.003, M))
            outline_tube(g, fl, M, 0.0012, 0.0016)
            pearl.add(*geo.uv_sphere(p + nn * 0.004, size * 0.16, 10, 8))
            # 支杆
            q, qn = surf(sh, th * DEG, ph * DEG, 0.004)
            g.add(*geo.tube([q, p - nn * 0.002], 0.0011, 5))
    # 正中凤凰
    c, n = surf(sh, 30 * DEG, 0.0, 0.05)
    M = plate(None, 0, c, (0, -1, 0.15))
    pp = phoenix_poly(1.1)
    g.add(*geo.extrude2d(pp, 0.004, M))
    cui.add(*geo.extrude2d(pp * 0.8 + np.array([0, 0.008]), 0.005, M))
    ruby.add(*geo.uv_sphere(geo.transform([(0, 0.03, 0.004)], M)[0], 0.006, 12, 8))
    for k in range(5):
        pt = geo.transform([(0, -0.004 - k * 0.0065, 0)], M)[0]
        pearl.add(*geo.uv_sphere(pt, 0.0028, 8, 6))
    # 红绒球
    for ph in (-60, -30, 0, 30, 60):
        p, nn = surf(sh, 20 * DEG, ph * DEG, 0.055)
        pompom(pomg, p, 0.011)
    # 珠帘（额前短串）
    for ph in np.linspace(-70, 70, 15):
        p, nn = surf(sh, band_lo(ph * DEG) + 1 * DEG, ph * DEG, 0.012)
        pts = [p + (0, -0.002, -0.0055 * k) for k in range(5)]
        pearls_along(pearl, pts, 0.0024)
    # 两侧挑子：长珠串 + 红穗
    for s in (1, -1):
        p, nn = surf(sh, 62 * DEG, s * 92 * DEG, 0.03)
        path = geo.smooth_path([p, p + (s * 0.01, -0.005, -0.12), p + (s * 0.012, -0.012, -0.24)], 40)
        pearls_along(pearl, path, 0.0032, step=0.0075)
        tassel(tas, g, path[-1] + (0, 0, -0.006), L=0.1, n=18, spread=0.006, knot=0.007)
    # 发髻与线尾子
    hair = MB()
    hair.add(*geo.uv_sphere((0, 0.098, 0.02), 1.0, 24, 16, scale=(0.055, 0.038, 0.045)))
    path = geo.smooth_path([(0, 0.11, -0.02), (0, 0.13, -0.12), (0, 0.135, -0.3), (0, 0.14, -0.55)], 30)
    widths = np.linspace(0.06, 0.03, len(path))
    hair.add(*geo.tube(path, widths * 0.5, 16, scale2=0.35))
    hair.obj("Hair_Bun", mats.get("bun"), parent=root)
    g.obj("Feng_Gold", mats.get("gold"), parent=root)
    cui.obj("Feng_Cui", mats.get("cui"), parent=root)
    pearl.obj("Feng_Pearl", mats.get("pearl"), parent=root)
    pomg.obj("Feng_Pom", mats.get("pom_red"), parent=root)
    ruby.obj("Feng_Ruby", mats.get("ruby"), parent=root)
    tas.obj("Feng_Tassel", mats.get("hair_redtassel"), parent=root)
    return root


# ------------------------------------------------------------------ 角色配置
CHARS = {
    "guanyu": dict(kind="male", beard=("wu", "black"), hg=lambda sh, p: hg_fuzi(sh, p)),
    "baozheng": dict(kind="male", beard=("man", "black"), hg=lambda sh, p: hg_xiangdiao(sh, p, wing="square")),
    "caocao": dict(kind="male", beard=("san", "black"), hg=lambda sh, p: hg_xiangdiao(sh, p, wing="oval")),
    "zhangfei": dict(kind="male", beard=("zha", "black"), hg=lambda sh, p: hg_zhajin(sh, p)),
    "douerdun": dict(kind="male", beard=("zha", "red"), hg=lambda sh, p: hg_zhajin(sh, p, color="satin_blue", pom="pom_red")),
    "sunwukong": dict(kind="male", beard=None, hg=lambda sh, p: hg_zijinguan(sh, p)),
    "xiangyu": dict(kind="male", beard=("man", "black"), hg=lambda sh, p: hg_bawang(sh, p)),
    "yangguifei": dict(kind="female", beard=None, hg=lambda sh, p: hg_fengguan(sh, p)),
}


def build(cid):
    cfg = CHARS[cid]
    root = empty(f"Head_{cid}")
    head = base_head(cfg["kind"])
    head.name = "Face"
    head.parent = root
    head.data.materials.clear()
    head.data.materials.append(mats.get(f"face_{cid}"))
    eyes(cfg["kind"], root)
    sh = HeadShell(head)
    wangzi(sh, root)
    if cfg["beard"]:
        beard(cfg["beard"][0], cfg["beard"][1], root, cfg["kind"])
    cfg["hg"](sh, root)
    return root


def all_objects(root):
    out = [root]
    for c in root.children_recursive:
        out.append(c)
    return out
