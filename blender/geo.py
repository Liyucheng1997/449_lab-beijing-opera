"""几何构件：网格累加器、放样、管道、卡片、贴合头部的壳体等。"""
import math
import numpy as np
import bpy
import mathutils
from mathutils.bvhtree import BVHTree


class MB:
    """网格累加器：把许多小部件合成一个对象（减少 glTF 节点数）。"""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []
        self.n = 0

    def add(self, verts, faces, uvs=None):
        verts = np.asarray(verts, np.float64).reshape(-1, 3)
        if uvs is None:
            uvs = np.zeros((len(verts), 2))
        self.v.append(verts)
        self.uv.append(np.asarray(uvs, np.float64).reshape(-1, 2))
        self.f.extend([tuple(int(i) + self.n for i in face) for face in faces])
        self.n += len(verts)
        return self

    def extend(self, other):
        for v, uv in zip(other.v, other.uv):
            pass
        verts = np.concatenate(other.v) if other.v else np.zeros((0, 3))
        uvs = np.concatenate(other.uv) if other.uv else np.zeros((0, 2))
        self.add(verts, other.f, uvs)
        return self

    def obj(self, name, mat=None, smooth=True, parent=None, coll=None):
        verts = np.concatenate(self.v)
        uvs = np.concatenate(self.uv)
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(x) for x in verts], [], self.f)
        me.update()
        uvl = me.uv_layers.new(name="UVMap")
        vi = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", vi)
        uvl.data.foreach_set("uv", uvs[vi].astype(np.float32).ravel())
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(name, me)
        (coll or bpy.context.scene.collection).objects.link(ob)
        if mat is not None:
            for m in (mat if isinstance(mat, (list, tuple)) else [mat]):
                me.materials.append(m)
        if parent is not None:
            ob.parent = parent
        return ob


def grid_faces(R, P, closed):
    faces = []
    cols = P + 1 if closed else P
    for i in range(R - 1):
        for j in range(cols - 1):
            a = i * cols + j
            faces.append((a, a + 1, a + cols + 1, a + cols))
    return faces


def loft(rings, closed=True, cap0=False, cap1=False, v_range=(0, 1), u_scale=1.0):
    """rings: (R, P, 3)。返回 verts, faces, uvs。闭合时复制首列形成 UV 缝。"""
    rings = np.asarray(rings, np.float64)
    R, P, _ = rings.shape
    if closed:
        rings = np.concatenate([rings, rings[:, :1]], 1)
    cols = rings.shape[1]
    verts = rings.reshape(-1, 3)
    u = np.tile(np.linspace(0, 1, cols) * u_scale, R)
    v = np.repeat(np.linspace(v_range[0], v_range[1], R), cols)
    faces = grid_faces(R, P, closed)
    uvs = np.stack([u, v], 1)
    extra_v, extra_uv = [], []
    n = len(verts)
    if cap0:
        c = rings[0, :P].mean(0)
        extra_v.append(c)
        extra_uv.append((0.5, v_range[0]))
        faces += [(n, (j + 1) % P, j) for j in range(P)]
        n += 1
    if cap1:
        c = rings[-1, :P].mean(0)
        extra_v.append(c)
        extra_uv.append((0.5, v_range[1]))
        base = (R - 1) * cols
        faces += [(n, base + j, base + (j + 1) % P) for j in range(P)]
        n += 1
    if extra_v:
        verts = np.concatenate([verts, np.array(extra_v)])
        uvs = np.concatenate([uvs, np.array(extra_uv)])
    return verts, faces, uvs


def frames(path, up=(0, 0, 1)):
    path = np.asarray(path, np.float64)
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    up = np.asarray(up, np.float64)
    N = np.cross(T[0], up)
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], (1, 0, 0))
    N /= np.linalg.norm(N)
    Ns, Bs = [], []
    for t in T:
        N = N - t * (N @ t)
        N /= np.linalg.norm(N) + 1e-12
        B = np.cross(t, N)
        Ns.append(N.copy())
        Bs.append(B)
    return T, np.array(Ns), np.array(Bs)


def tube(path, radius, seg=10, caps=True, scale2=None):
    path = np.asarray(path, np.float64)
    r = np.broadcast_to(np.asarray(radius, np.float64), (len(path),))
    _, N, B = frames(path)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    s2 = 1.0 if scale2 is None else scale2
    rings = path[:, None, :] + r[:, None, None] * (np.cos(a)[None, :, None] * N[:, None, :] + s2 * np.sin(a)[None, :, None] * B[:, None, :])
    return loft(rings, True, caps, caps)


def smooth_path(pts, n=40):
    """Catmull-Rom 平滑三维控制点。"""
    pts = [np.asarray(p, np.float64) for p in pts]
    P = [pts[0]] + pts + [pts[-1]]
    out = []
    segs = len(pts) - 1
    per = max(2, n // segs)
    for i in range(segs):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.array(out)


def uv_sphere(c, r, nu=16, nv=10, scale=(1, 1, 1)):
    c = np.asarray(c, np.float64)
    th = np.linspace(0, np.pi, nv + 1)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    rings = []
    for t in th:
        ring = np.stack([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.full_like(ph, np.cos(t))], 1)
        rings.append(c + ring * r * np.asarray(scale))
    return loft(np.array(rings), True)


def card(path, width, side, twist=0.0):
    """沿路径的发丝卡片。side：宽度方向（会正交化）。"""
    path = np.asarray(path, np.float64)
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    n = len(path)
    w = np.broadcast_to(np.asarray(width, np.float64), (n,))
    verts, uvs = [], []
    side = np.asarray(side, np.float64)
    for i in range(n):
        s = side - T[i] * (side @ T[i])
        s /= np.linalg.norm(s) + 1e-12
        if twist:
            ang = twist * i / (n - 1)
            s = s * math.cos(ang) + np.cross(T[i], s) * math.sin(ang)
        verts += [path[i] - s * w[i] / 2, path[i] + s * w[i] / 2]
        v = 1 - i / (n - 1)
        uvs += [(0, v), (1, v)]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(n - 1)]
    return np.array(verts), faces, np.array(uvs)


def extrude2d(poly, depth, M=None, bevel=0.0):
    """二维轮廓（N,2）沿 z 挤出，M 为 4x4 变换。"""
    poly = [mathutils.Vector((p[0], p[1], 0)) for p in poly]
    tris = mathutils.geometry.tessellate_polygon([poly])
    N = len(poly)
    v = [(p.x, p.y, depth / 2) for p in poly] + [(p.x, p.y, -depth / 2) for p in poly]
    f = [tuple(t[::-1]) for t in tris] + [tuple(i + N for i in t) for t in tris]
    for i in range(N):
        j = (i + 1) % N
        f.append((i, j, j + N, i + N))
    v = np.array(v)
    xs, ys = v[:, 0], v[:, 1]
    uv = np.stack([(xs - xs.min()) / (np.ptp(xs) + 1e-9), (ys - ys.min()) / (np.ptp(ys) + 1e-9)], 1)
    if M is not None:
        M = np.asarray(M)
        v = (np.c_[v, np.ones(len(v))] @ M.T)[:, :3]
    return v, f, uv


def mat_from(loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    return np.array(mathutils.Matrix.LocRotScale(mathutils.Vector(loc), mathutils.Euler(rot), mathutils.Vector(scale)))


def transform(verts, M):
    verts = np.asarray(verts, np.float64)
    return (np.c_[verts, np.ones(len(verts))] @ np.asarray(M).T)[:, :3]


class HeadShell:
    """在真实头部表面上取样，生成贴合的帽子/水纱壳体。"""

    def __init__(self, head_ob, center=(0, 0.012, 0.0)):
        dg = bpy.context.evaluated_depsgraph_get()
        self.bvh = BVHTree.FromObject(head_ob, dg)
        self.c = mathutils.Vector(center)

    def hit(self, theta, phi):
        d = mathutils.Vector((math.sin(theta) * math.sin(phi), -math.sin(theta) * math.cos(phi), math.cos(theta)))
        far = self.c + d * 0.4
        loc, nrm, _, _ = self.bvh.ray_cast(far, -d, 0.5)
        if loc is None:
            return self.c + d * 0.09, d
        return loc, nrm

    def shell(self, theta_max, offset, nphi=96, nth=40, theta_min=0.0, phi_range=(-math.pi, math.pi), flare=None):
        """theta_max(phi)->弧度。offset 可为常数或函数 offset(t, phi)。"""
        rings = []
        phis = np.linspace(phi_range[0], phi_range[1], nphi, endpoint=(phi_range != (-math.pi, math.pi)))
        for i in range(nth):
            t = i / (nth - 1)
            row = []
            for ph in phis:
                th = theta_min + (theta_max(ph) - theta_min) * t
                th = max(th, 1e-3)
                p, n = self.hit(th, ph)
                off = offset(t, ph) if callable(offset) else offset
                row.append(np.array(p) + np.array(n) * off)
            rings.append(row)
        closed = phi_range == (-math.pi, math.pi)
        return loft(np.array(rings), closed=closed)


def interp_deg(table):
    xs = np.radians([a for a, _ in table])
    ys = np.radians([b for _, b in table])
    return lambda ph: float(np.interp(abs(ph), xs, ys))
