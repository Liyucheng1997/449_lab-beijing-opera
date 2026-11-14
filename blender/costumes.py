"""戏服全身模型：蟒、女蟒、大靠、猴衣。头部通过 HeadSocket 空物体在网页端挂载。"""
import math
import numpy as np
import bpy

import geo
from geo import MB
import mats
import sdf as S
import headgen
from blib import pbr, TEX
from costume_params import PROFILES, HEAD_Z, ring_point, profile_arclen, TORSO_U
import os

R = np.random.RandomState(17)
P_RING = 128


def T(n):
    return os.path.join(TEX, n)


_mat = {}


def fabric(name, double=True, sheen=0.22, nstr=0.8):
    if name not in _mat:
        _mat[name] = pbr(f"Cloth_{name}", base_tex=T(f"{name}_base.jpg"), rough_tex=T(f"{name}_orm.png"),
                         normal_tex=T(f"{name}_nrm.png"), normal_strength=nstr, sheen=sheen, double=double)
    return _mat[name]


def skin():
    if "skin" not in _mat:
        _mat["skin"] = pbr("Skin_hand", color=(0.78, 0.58, 0.47, 1), rough=0.45, sss=0.25)
    return _mat["skin"]


def satin(color, name):
    if name not in _mat:
        _mat[name] = pbr(name, color=color, rough=0.4, sheen=0.4, double=True)
    return _mat[name]


def solid(ob, t=0.005):
    m = ob.modifiers.new("solid", "SOLIDIFY")
    m.thickness = t
    m.offset = -1
    m.use_even_offset = True
    return ob


def grid_mesh(rings, uvs):
    """rings (R, P+1, 3) 已含缝列；uvs (R, P+1, 2)。"""
    Rn, C, _ = rings.shape
    faces = []
    for i in range(Rn - 1):
        for j in range(C - 1):
            a = i * C + j
            faces.append((a, a + C, a + C + 1, a + 1))
    return rings.reshape(-1, 3), faces, uvs.reshape(-1, 2)


# ------------------------------------------------------------------ 躯干
def torso(prof_name, fold_amp=lambda z: 0.003, nz=90, z_cut=None, hem_wave=0.008, flip=False):
    prof = np.array(PROFILES[prof_name], float)
    rows = geo.smooth_path(prof, nz)
    if z_cut is not None:
        rows = rows[rows[:, 0] >= z_cut]
    tab = profile_arclen(prof_name)
    tz = np.array([t[0] for t in tab])
    ts = np.array([t[1] for t in tab])
    Stot = ts[-1]
    th = np.linspace(0, 2 * math.pi, P_RING + 1)
    phases = R.uniform(0, 6.28, 6)
    rings, uvs = [], []
    for k, (z, a, b, yc) in enumerate(rows):
        pts = []
        A = fold_amp(z)
        for t in th:
            x, y = ring_point(a, b, yc, t)
            dx, dy = x, y - yc
            ln = math.hypot(dx, dy) + 1e-9
            f = (math.sin(9 * t + phases[0] + z * 3) + 0.6 * math.sin(14 * t + phases[1] - z * 5) + 0.35 * math.sin(23 * t + phases[2] + z * 9))
            d = A * f
            zz = z
            if k == len(rows) - 1:
                zz = z + hem_wave * math.sin(7 * t + phases[3]) * (A / 0.02 if A < 0.02 else 1)
            pts.append((x + dx / ln * d, y + dy / ln * d, zz))
        rings.append(pts)
        v = np.interp(z, tz[::-1] if tz[0] > tz[-1] else tz, (ts[::-1] if tz[0] > tz[-1] else ts)) / Stot
        uvs.append([(TORSO_U * t / (2 * math.pi), v) for t in th])
    return grid_mesh(np.array(rings), np.array(uvs))


def mang_folds(z):
    if z > 1.05:
        return 0.0025
    return 0.0025 + 0.022 * min(1.0, (1.05 - z) / 0.9) ** 1.4


# ------------------------------------------------------------------ 袖子与水袖
def arm_path(side, kind="mang"):
    if kind == "mang":
        pts = [(side * 0.165, 0.018, 1.435), (side * 0.27, 0.0, 1.27), (side * 0.345, -0.045, 1.10), (side * 0.385, -0.085, 0.96)]
    elif kind == "female":
        pts = [(side * 0.145, 0.018, 1.44), (side * 0.235, 0.0, 1.28), (side * 0.30, -0.045, 1.12), (side * 0.335, -0.085, 0.98)]
    else:  # 窄袖：手臂更贴身
        pts = [(side * 0.165, 0.018, 1.435), (side * 0.235, 0.01, 1.26), (side * 0.275, -0.01, 1.08), (side * 0.29, -0.03, 0.93)]
    return geo.smooth_path(pts, 30)


def sleeve(side, kind, r0, r1, sag, sleeve_len_tex=0.6, mat=None, name="Sleeve"):
    path = arm_path(side, "mang" if kind == "mang" else kind)
    T_, N_, B_ = geo.frames(path)
    up = np.array((0, 0, 1.0))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    total = L[-1]
    a = np.linspace(0, 2 * math.pi, 49)
    rings, uvs = [], []
    for i, p in enumerate(path):
        t = L[i] / total
        r = r0 + (r1 - r0) * t ** 1.3
        top = up - T_[i] * (up @ T_[i])
        top /= np.linalg.norm(top)
        side_v = np.cross(T_[i], top)
        ring = []
        for ang in a:
            o = math.cos(ang) * top + math.sin(ang) * side_v
            hang = ((1 - math.cos(ang)) / 2) ** 2 * sag * t
            fold = 0.004 * t * math.sin(ang * 7 + t * 5)
            ring.append(p + o * (r + fold) - up * hang)
        rings.append(ring)
        uvs.append([(TORSO_U + (1 - TORSO_U) * (ang / (2 * math.pi)), (total - L[i]) / sleeve_len_tex) for ang in a])
    return np.array(rings), np.array(uvs), path


def shuixiu(cuff_ring, side, length=0.6, width=0.15):
    """水袖：从袖口自然垂落的白绸。"""
    cuff = np.asarray(cuff_ring)
    c0 = cuff[:-1].mean(0)
    n = 26
    rings, uvs = [], []
    C = len(cuff)
    ang = np.linspace(0, 2 * math.pi, C)
    ph = R.uniform(0, 6.28, 3)
    for k in range(n):
        t = k / (n - 1)
        drop = length * t
        tgt = []
        for j, a in enumerate(ang):
            w = width * (1 + 0.25 * t)
            x = math.sin(a) * w * 0.5
            y = math.cos(a) * 0.035 * (1 + 0.6 * t) + 0.012 * math.sin(a * 5 + ph[0] + t * 3) * t
            x += 0.01 * math.sin(a * 3 + ph[1]) * t
            tgt.append(c0 + np.array((x, y - 0.01 * t, -0.03 - drop)))
        tgt = np.array(tgt)
        s = min(1.0, t * 3.5)
        s = s * s * (3 - 2 * s)
        rings.append(cuff * (1 - s) + tgt * s)
        uvs.append([(j / (C - 1), 1 - t) for j in range(C)])
    return grid_mesh(np.array(rings), np.array(uvs))


# ------------------------------------------------------------------ 配件
def collar(mat, z=1.508, a=0.078, b=0.074, yc=0.022, r=0.012):
    t = np.linspace(0, 2 * math.pi, 64)
    path = np.stack([np.sin(t) * a, yc + np.cos(t) * b, np.full_like(t, z)], 1)
    return MB().add(*geo.tube(path, r, 12, caps=False))


def yudai(z=1.02, a=0.258, b=0.205, drop=0.07, h=0.045, t=0.014):
    """玉带：前低后高的环形硬带，红漆 + 金边 + 玉片。"""
    th = np.linspace(0, 2 * math.pi, 97)
    core, gold, jade = MB(), MB(), MB()
    pts = []
    for ang in th:
        x = -math.sin(ang) * a
        y = math.cos(ang) * b
        zz = z + drop * 0.5 * math.cos(ang)
        pts.append((x, y, zz))
    pts = np.array(pts)
    rings = []
    for i, p in enumerate(pts):
        rad = np.array((p[0], p[1], 0))
        rad /= np.linalg.norm(rad)
        up = np.array((0, 0, 1.0))
        sq = [p + rad * t / 2 + up * h / 2, p - rad * t / 2 + up * h / 2, p - rad * t / 2 - up * h / 2, p + rad * t / 2 - up * h / 2]
        rings.append(sq)
    rings = np.array(rings)
    v, f, uv = geo.loft(rings.transpose(0, 1, 2), closed=True)
    core.add(v, f, uv)
    gold.add(*geo.tube(rings[:, 0] + np.array((0, 0, 0.002)), 0.0026, 6, caps=False))
    gold.add(*geo.tube(rings[:, 3] - np.array((0, 0, 0.002)), 0.0026, 6, caps=False))
    for ang in np.linspace(math.pi * 0.45, math.pi * 1.55, 9):
        x, y = -math.sin(ang) * (a + t * 0.6), math.cos(ang) * (b + t * 0.6)
        zz = z + drop * 0.5 * math.cos(ang)
        out = np.array((x, y, 0)) / math.hypot(x, y)
        vv, ff, uu = geo.uv_sphere((0, 0, 0), 1.0, 16, 10)
        vv = vv * np.array((0.026, 0.026, 0.017))
        # 让玉片朝外
        ang2 = math.atan2(out[1], out[0])
        ca, sa = math.cos(ang2), math.sin(ang2)
        rot = np.array([[0.3 * ca, -sa, 0], [0.3 * sa, ca, 0], [0, 0, 1]])
        jade.add(vv @ rot.T + np.array((x, y, zz)), ff, uu)
    return core, gold, jade


def boots(sole_h=0.085, white=True):
    sole, up = MB(), MB()
    for s in (1, -1):
        c = np.array((s * 0.1, -0.03, 0))
        def rr(w, l, z, sh=0.0):
            t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
            x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.6 * w / 2
            y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.6 * l / 2
            return np.stack([c[0] + x, c[1] + y + sh, np.full_like(t, z)], 1)
        rings = [rr(0.09, 0.25, 0.0), rr(0.105, 0.285, 0.004), rr(0.11, 0.29, sole_h * 0.5), rr(0.108, 0.285, sole_h - 0.004), rr(0.1, 0.27, sole_h)]
        sole.add(*geo.loft(np.array(rings), True, True, True))
        urings = [rr(0.1, 0.265, sole_h), rr(0.098, 0.25, sole_h + 0.04), rr(0.09, 0.2, sole_h + 0.08, 0.025), rr(0.085, 0.11, sole_h + 0.14, 0.05), rr(0.088, 0.1, 0.42, 0.05)]
        up.add(*geo.loft(np.array(urings), True, False, True))
    return sole, up


def hand_mesh(side):
    """握拳的手（SDF → 网格），腕部在原点，向 -Z 伸展。"""
    h = 0.0012
    xs = np.arange(-0.06, 0.06, h, dtype=np.float32)
    ys = np.arange(-0.06, 0.05, h, dtype=np.float32)
    zs = np.arange(-0.13, 0.03, h, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d = S.ellipsoid(X, Y, Z, (0, 0, -0.05), (0.04, 0.021, 0.048))
    d = S.smin(d, S.capsule(X, Y, Z, (0, 0, 0.02), (0, 0, -0.02), 0.026), 0.02)
    for i in range(4):
        x = -0.026 + i * 0.0175
        L = 1.0 - 0.08 * abs(i - 1.3)
        d = S.smin(d, S.capsule(X, Y, Z, (x, -0.008, -0.088 * L), (x, -0.03, -0.072 * L), 0.0095), 0.006)
        d = S.smin(d, S.capsule(X, Y, Z, (x, -0.03, -0.072 * L), (x, -0.026, -0.05), 0.0088), 0.006)
    tx = 0.036 * side
    d = S.smin(d, S.capsule(X, Y, Z, (tx, 0.0, -0.028), (tx * 0.6, -0.03, -0.055), 0.0105), 0.008)
    v, q = S.surface_nets(d, (xs[0], ys[0], zs[0]), h)
    return v, q


def hands(paths, mat):
    out = MB()
    for side, path in paths:
        v, q = hand_mesh(side)
        # 对准小臂方向
        T_ = path[-1] - path[-3]
        T_ /= np.linalg.norm(T_)
        down = np.array((0, 0, -1.0))
        axis = np.cross(down, T_)
        ang = math.acos(np.clip(down @ T_, -1, 1))
        if np.linalg.norm(axis) > 1e-6:
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
        else:
            Rm = np.eye(3)
        v = v @ Rm.T + path[-1] - T_ * 0.02
        out.add(v, [tuple(f) for f in q], np.zeros((len(v), 2)))
    ob = out.obj("Hands", mat)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    m = ob.modifiers.new("sm", "LAPLACIANSMOOTH")
    m.lambda_factor = 0.6
    m.iterations = 4
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = 0.35
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def socket(root):
    e = bpy.data.objects.new("HeadSocket", None)
    bpy.context.scene.collection.objects.link(e)
    e.location = (0, 0, HEAD_Z)
    e.parent = root
    return e


def empty(name):
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    return e


def parent_all(root, objs):
    for o in objs:
        if o.parent is None:
            o.parent = root


def recalc(ob):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


# ------------------------------------------------------------------ 成套戏服
def mang_set(cid, tex, female=False):
    root = empty(f"Costume_{cid}")
    objs = []
    prof = "nvmang" if female else "mang"
    folds = (lambda z: 0.0025 + 0.012 * min(1.0, max(0.0, 1.05 - z) / 0.45) ** 1.4) if female else mang_folds
    v, f, uv = torso(prof, folds)
    body = recalc(solid(MB().add(v, f, uv).obj("Robe", fabric(tex)), 0.005))
    objs.append(body)
    kind = "female" if female else "mang"
    paths = []
    for s in (1, -1):
        rings, uvs, path = sleeve(s, kind, 0.07 if not female else 0.06, 0.115 if not female else 0.1, 0.05)
        sv, sf, su = grid_mesh(rings, uvs)
        objs.append(recalc(solid(MB().add(sv, sf, su).obj(f"Sleeve{s}", fabric(tex)), 0.004)))
        xv, xf, xu = shuixiu(rings[-1], s, length=0.62 if not female else 0.58, width=0.16 if not female else 0.14)
        objs.append(recalc(MB().add(xv, xf, xu * np.array((1, 2))).obj(f"Shuixiu{s}", fabric("silk", sheen=0.8, nstr=0.3))))
        paths.append((s, path))
    objs.append((collar(None, a=0.068, b=0.064) if female else collar(None)).obj("Collar", satin((0.92, 0.9, 0.86, 1), "Satin_white")))
    a_belt = 0.258 if not female else 0.225
    core, gold, jade = yudai(z=1.02 if not female else 1.0, a=a_belt, b=a_belt * 0.8, h=0.045 if not female else 0.032)
    objs.append(core.obj("Yudai", satin((0.32, 0.03, 0.03, 1), "Lacquer_red")))
    objs.append(gold.obj("Yudai_Gold", mats.get("gold")))
    objs.append(jade.obj("Yudai_Jade", mats.get("jade")))
    if not female:
        # 两侧硬摆
        for s in (1, -1):
            pts = np.array([(s * 0.212, -0.06, 1.26), (s * 0.212, 0.06, 1.26), (s * 0.29, 0.085, 0.5), (s * 0.29, -0.085, 0.5)])
            mb = MB().add(pts, [(0, 1, 2, 3)], np.array([(0.18, 0.75), (0.2, 0.75), (0.2, 0.3), (0.18, 0.3)]) * np.array((1, 1)))
            ob = solid(mb.obj(f"Bai{s}", fabric(tex)), 0.006)
            objs.append(ob)
        sole, up = boots()
        objs.append(sole.obj("Boot_Sole", satin((0.93, 0.92, 0.88, 1), "Sole_white")))
        objs.append(up.obj("Boot", mats.get("satin_black")))
    else:
        # 女蟒：下身长裙 + 云肩 + 流苏
        sk = PROFILES["nvmang"]
        rings, uvs = [], []
        th = np.linspace(0, 2 * math.pi, 241)
        zs = np.linspace(1.0, 0.012, 50)
        for z in zs:
            t = (1.0 - z) / 0.99
            a = 0.19 + 0.09 * t ** 1.2
            b = 0.14 + 0.06 * t ** 1.2
            ring = []
            for ang in th:
                x, y = ring_point(a, b, 0.008, ang, 2.2)
                pleat = 0.0035 * (0.4 + t) * math.sin(ang * 60)
                rr = math.hypot(x, y - 0.008) + 1e-9
                ring.append((x + x / rr * pleat, y + (y - 0.008) / rr * pleat, z))
            rings.append(ring)
            uvs.append([(ang / (2 * math.pi) * 3, 1 - t) for ang in th])
        v, f, uv = grid_mesh(np.array(rings), np.array(uvs))
        objs.append(recalc(MB().add(v, f, uv).obj("Skirt", fabric("skirt"))))
        # 云肩：从领口覆盖到肩外，八个如意头
        rings, uvs = [], []
        th = np.linspace(0, 2 * math.pi, 161)
        for k in range(14):
            t = k / 13
            ring = []
            for ang in th:
                lobe = 0.5 + 0.5 * math.cos(ang * 8)
                rr = 0.075 + t * (0.17 + 0.035 * lobe)
                x = -math.sin(ang) * rr * 1.25
                y = 0.02 + math.cos(ang) * rr * 0.9
                z = 1.512 - 0.13 * t ** 1.6 - 0.02 * t * lobe
                ring.append((x, y, z))
            rings.append(ring)
            uvs.append([(ang / (2 * math.pi), 1 - t) for ang in th])
        rings = np.array(rings)
        # 避免穿进肩部：外推到躯干之外
        v, f, uv = grid_mesh(rings, np.array(uvs))
        objs.append(recalc(solid(MB().add(v, f, uv).obj("Yunjian", fabric("yunjian")), 0.003)))
        tas = MB()
        edge = rings[-1]
        for i in range(0, len(edge) - 1, 2):
            p = edge[i] + np.array((0, 0, -0.004))
            out = np.array((p[0], p[1] - 0.02, 0))
            out /= np.linalg.norm(out) + 1e-9
            path = geo.smooth_path([p, p + out * 0.01 + (0, 0, -0.05), p + out * 0.015 + (0, 0, -0.11)], 8)
            tas.add(*geo.card(path, 0.012, (-out[1], out[0], 0)))
        objs.append(tas.obj("Yunjian_Tassel", mats.get("hair_pink")))
    sock = socket(root)
    parent_all(root, objs)
    return root


def kao_set(cid):
    root = empty(f"Costume_{cid}")
    objs = []
    v, f, uv = torso("kao", lambda z: 0.0015)
    objs.append(recalc(solid(MB().add(v, f, uv).obj("Kao_Body", fabric("kao")), 0.006)))
    # 衬裙（黑缎，到脚面）
    rings, uvs = [], []
    th = np.linspace(0, 2 * math.pi, 129)
    for z in np.linspace(0.99, 0.16, 30):
        t = (0.99 - z) / 0.83
        a, b = 0.2 + 0.1 * t, 0.15 + 0.07 * t
        rings.append([(x + 0.012 * t * math.sin(9 * ang) * -math.sin(ang), y + 0.012 * t * math.sin(9 * ang) * math.cos(ang), z) for ang in th for x, y in [ring_point(a, b, 0.01, ang)]])
        uvs.append([(ang / (2 * math.pi), 1 - t) for ang in th])
    v, f, uv = grid_mesh(np.array(rings), np.array(uvs))
    objs.append(recalc(MB().add(v, f, uv).obj("Chenqun", mats.get("satin_black"))))
    # 靠腿：前两片 + 侧两片 + 后一片，略向外张
    panels = MB()
    for c_ang, w, L in ((math.pi - 0.28, 0.2, 0.52), (math.pi + 0.28, 0.2, 0.52), (math.pi * 0.5, 0.2, 0.46), (math.pi * 1.5, 0.2, 0.46), (0.0, 0.3, 0.5)):
        rows, uvr = [], []
        for k in range(12):
            t = k / 11
            z = 0.97 - L * t
            a, b = 0.215 + 0.12 * t, 0.165 + 0.08 * t
            row, ur = [], []
            for j in range(9):
                s = (j / 8 - 0.5)
                ang = c_ang + s * w / ((a + b) / 2)
                x, y = ring_point(a, b, 0.0, ang)
                row.append((x, y, z - (0.04 * (1 - abs(s) * 2) * t if c_ang != 0.0 else 0)))
                ur.append((j / 8, 1 - t))
            rows.append(row)
            uvr.append(ur)
        v, f, uv = grid_mesh(np.array(rows), np.array(uvr))
        panels.add(v, f, uv)
    objs.append(solid(panels.obj("Kao_Legs", fabric("kao_panel")), 0.006))
    # 靠肚：腹前凸起的弧形护片
    rows, uvr = [], []
    for k in range(14):
        t = k / 13
        z = 1.2 - 0.27 * t
        row, ur = [], []
        for j in range(17):
            s = j / 16 - 0.5
            bulge = math.cos(s * math.pi) * math.sin(t * math.pi) * 0.045
            row.append((s * 0.32, -0.155 - bulge - 0.015 * math.sin(t * math.pi), z))
            ur.append((j / 16, 1 - t))
        rows.append(row)
        uvr.append(ur)
    v, f, uv = grid_mesh(np.array(rows), np.array(uvr))
    objs.append(solid(MB().add(v, f, uv).obj("Kao_Belly", fabric("kao_belly")), 0.005))
    # 披膀（护肩）
    pb = MB()
    for s in (1, -1):
        vv, ff, uu = geo.uv_sphere((0, 0, 0), 1.0, 24, 12)
        keep = vv[:, 2] > -0.25
        vv = vv * np.array((0.085, 0.1, 0.07)) + np.array((s * 0.205, 0.01, 1.4))
        pb.add(vv, ff, uu)
    objs.append(pb.obj("Kao_Shoulder", fabric("kao_panel")))
    # 袖：窄袖 + 手
    paths = []
    for s in (1, -1):
        rings, uvs, path = sleeve(s, "narrow", 0.062, 0.05, 0.0, 0.56)
        sv, sf, su = grid_mesh(rings, uvs)
        objs.append(recalc(MB().add(sv, sf, su).obj(f"Sleeve{s}", fabric("kao"))))
        cuff = MB().add(*geo.tube(rings[-1][:-1].tolist() + [rings[-1][0].tolist()], 0.008, 8, caps=False))
        objs.append(cuff.obj(f"Cuff{s}", mats.get("gold")))
        paths.append((s, path))
    objs.append(hands(paths, skin()))
    objs.append(collar(None).obj("Collar", satin((0.92, 0.9, 0.86, 1), "Satin_white")))
    # 腰间金带
    core, gold, jade = yudai(z=1.0, a=0.215, b=0.17, drop=0.02, h=0.03, t=0.01)
    objs.append(core.obj("Belt", mats.get("satin_black")))
    objs.append(gold.obj("Belt_Gold", mats.get("gold")))
    objs.append(jade.obj("Belt_Jade", mats.get("ruby")))
    # 靠旗：背后四面三角旗
    flags, poles, tips, poms = MB(), MB(), MB(), MB()
    hold = np.array((0, 0.17, 1.30))
    poles.add(*geo.uv_sphere(hold + (0, 0.02, 0.0), 1.0, 16, 10, scale=(0.05, 0.035, 0.09)))
    for ang_deg in (-34, -12, 12, 34):
        ang = math.radians(ang_deg)
        d = np.array((math.sin(ang), math.sin(math.radians(12)), math.cos(ang)))
        d /= np.linalg.norm(d)
        base = hold + np.array((math.sin(ang) * 0.03, 0.03, 0.07))
        Lp = 1.05
        poles.add(*geo.tube([base, base + d * Lp], 0.0065, 8))
        tip = base + d * Lp
        tips.add(*geo.tube([tip, tip + d * 0.06], [0.012, 0.0005], 8))
        pompom = geo.uv_sphere(tip - d * 0.01, 0.022, 14, 9)
        poms.add(*pompom)
        sg = 1.0 if ang_deg > 0 else -1.0
        out = sg * np.array((math.cos(ang), 0.0, -math.sin(ang)))
        A = base + d * 0.28
        Bp = base + d * (Lp - 0.04)
        n_u, n_v = 8, 14
        verts, uvs_, faces = [], [], []
        for iv in range(n_v):
            tv = iv / (n_v - 1)
            pole_pt = A + (Bp - A) * tv
            for iu in range(n_u):
                tu = iu / (n_u - 1) * tv  # 三角形：顶端最宽
                wave = 0.012 * math.sin(tu * 9 + tv * 4) * tu
                verts.append(pole_pt + out * 0.3 * tu + np.array((0, wave + 0.03 * tu, 0)))
                uvs_.append((tu, tv))
        for iv in range(n_v - 1):
            for iu in range(n_u - 1):
                a_ = iv * n_u + iu
                faces.append((a_, a_ + 1, a_ + n_u + 1, a_ + n_u))
        flags.add(np.array(verts), faces, np.array(uvs_))
    objs.append(flags.obj("Kaoqi", fabric("kao_flag", sheen=0.5)))
    objs.append(poles.obj("Kaoqi_Pole", mats.get("satin_black")))
    objs.append(tips.obj("Kaoqi_Tip", mats.get("gold")))
    objs.append(poms.obj("Kaoqi_Pom", mats.get("pom_red")))
    sole, up = boots()
    objs.append(sole.obj("Boot_Sole", satin((0.93, 0.92, 0.88, 1), "Sole_white")))
    objs.append(up.obj("Boot", mats.get("satin_black")))
    socket(root)
    parent_all(root, objs)
    return root


def houyi_set(cid):
    root = empty(f"Costume_{cid}")
    objs = []
    v, f, uv = torso("houyi", lambda z: 0.0015)
    objs.append(recalc(solid(MB().add(v, f, uv * np.array((4, 3))).obj("Houyi", fabric("houyi")), 0.004)))
    paths = []
    for s in (1, -1):
        rings, uvs, path = sleeve(s, "narrow", 0.058, 0.046, 0.0)
        sv, sf, su = grid_mesh(rings, uvs)
        objs.append(recalc(MB().add(sv, sf, su * np.array((4, 2))).obj(f"Sleeve{s}", fabric("houyi"))))
        cuff = MB().add(*geo.tube(rings[-1][:-1].tolist() + [rings[-1][0].tolist()], 0.007, 8, caps=False))
        objs.append(cuff.obj(f"Cuff{s}", mats.get("satin_red")))
        paths.append((s, path))
    objs.append(hands(paths, pbr("Fur_hand", color=(0.85, 0.6, 0.18, 1), rough=0.6, sheen=0.6)))
    # 裤 + 绑腿
    legs, wraps = MB(), MB()
    for s in (1, -1):
        path = geo.smooth_path([(s * 0.09, 0.0, 1.0), (s * 0.1, -0.01, 0.6), (s * 0.1, 0.0, 0.2), (s * 0.1, 0.0, 0.1)], 30)
        rad = np.interp(path[:, 2], [0.1, 0.25, 0.55, 0.8, 1.0], [0.045, 0.05, 0.068, 0.085, 0.1])
        v, f, uv = geo.tube(path, rad, 24)
        legs.add(v, f, uv * np.array((3, 4)))
        wp = path[(path[:, 2] < 0.45) & (path[:, 2] > 0.14)]
        wr = np.interp(wp[:, 2], [0.1, 0.25, 0.45], [0.05, 0.056, 0.07])
        wr = wr + 0.003 * np.abs(np.sin(wp[:, 2] * 120))
        wraps.add(*geo.tube(wp, wr, 24))
    objs.append(legs.obj("Trousers", fabric("houyi")))
    objs.append(wraps.obj("Bangtui", mats.get("satin_black")))
    # 虎皮裙（锯齿下摆）
    rings, uvs = [], []
    th = np.linspace(0, 2 * math.pi, 161)
    for k in range(16):
        t = k / 15
        ring = []
        for ang in th:
            a, b = 0.19 + 0.07 * t, 0.145 + 0.05 * t
            x, y = ring_point(a, b, 0.004, ang)
            zig = 0.05 * abs(((ang * 10 / math.pi) % 2) - 1) * t
            ring.append((x, y, 1.0 - 0.36 * t + zig))
        rings.append(ring)
        uvs.append([(ang / (2 * math.pi) * 2, 1 - t) for ang in th])
    v, f, uv = grid_mesh(np.array(rings), np.array(uvs))
    objs.append(recalc(solid(MB().add(v, f, uv).obj("Tiger_Skirt", fabric("tiger", sheen=0.8)), 0.005)))
    # 鸾带：红腰带 + 前垂两条带穗
    t = np.linspace(0, 2 * math.pi, 80)
    belt = np.stack([-np.sin(t) * 0.2, 0.004 + np.cos(t) * 0.152, np.full_like(t, 0.99)], 1)
    bm, tas, gld = MB(), MB(), MB()
    bm.add(*geo.tube(belt, 0.016, 10, caps=False, scale2=0.5))
    for s in (1, -1):
        top = np.array((s * 0.05, -0.155, 0.985))
        path = geo.smooth_path([top, top + (s * 0.01, -0.03, -0.12), top + (s * 0.015, -0.03, -0.33)], 20)
        bm.add(*geo.card(path, 0.035, (1, 0, 0)))
        tassel_top = path[-1]
        gld.add(*geo.uv_sphere(tassel_top, 0.01, 12, 8))
        for k in range(16):
            a = tassel_top + R.uniform(-0.006, 0.006, 3)
            tas.add(*geo.card(geo.smooth_path([a, a + (R.uniform(-0.01, 0.01), 0, -0.08), a + (R.uniform(-0.02, 0.02), 0, -0.15)], 8), 0.009, (1, R.uniform(-1, 1), 0)))
    objs.append(bm.obj("Luandai", mats.get("satin_red")))
    objs.append(tas.obj("Luandai_Tassel", mats.get("hair_redtassel")))
    objs.append(gld.obj("Luandai_Gold", mats.get("gold")))
    # 薄底靴
    boot = MB()
    for s in (1, -1):
        c = np.array((s * 0.1, -0.03, 0))
        def rr(w, l, z, sh=0.0):
            tt = np.linspace(0, 2 * math.pi, 40, endpoint=False)
            return np.stack([c[0] + np.cos(tt) * w / 2, c[1] + np.sin(tt) * l / 2 + sh, np.full_like(tt, z)], 1)
        rings = [rr(0.085, 0.24, 0.0), rr(0.095, 0.26, 0.02), rr(0.095, 0.25, 0.05), rr(0.085, 0.16, 0.1, 0.04), rr(0.09, 0.11, 0.2, 0.03), rr(0.1, 0.115, 0.3, 0.03)]
        boot.add(*geo.loft(np.array(rings), True, True, True))
    objs.append(boot.obj("Boot", mats.get("satin_black")))
    # 金箍棒：右手（-X 一侧）竖握
    hand_pos = paths[1][1][-1] - (paths[1][1][-1] - paths[1][1][-3]) / np.linalg.norm(paths[1][1][-1] - paths[1][1][-3]) * 0.06
    staff = MB()
    base = np.array((hand_pos[0], hand_pos[1] - 0.012, 0.0))
    staff.add(*geo.tube([base + (0, 0, 0.12), base + (0, 0, 1.78)], 0.0135, 16))
    gold = MB()
    for z0, z1 in ((0.0, 0.14), (1.76, 1.9)):
        gold.add(*geo.tube([base + (0, 0, z0), base + (0, 0, z1)], 0.016, 16))
    objs.append(staff.obj("Jingubang", mats.get("satin_red")))
    objs.append(gold.obj("Jingubang_Gold", mats.get("gold")))
    objs.append(collar(None, r=0.009).obj("Collar", mats.get("satin_red")))
    socket(root)
    parent_all(root, objs)
    return root


COSTUMES = {
    "guanyu": dict(head="guanyu", build=lambda: mang_set("guanyu", "mang_green")),
    "caocao": dict(head="caocao", build=lambda: mang_set("caocao", "mang_red")),
    "xiangyu": dict(head="xiangyu", build=lambda: kao_set("xiangyu")),
    "yangguifei": dict(head="yangguifei", build=lambda: mang_set("yangguifei", "mang_yellow", female=True)),
    "sunwukong": dict(head="sunwukong", build=lambda: houyi_set("sunwukong")),
}
