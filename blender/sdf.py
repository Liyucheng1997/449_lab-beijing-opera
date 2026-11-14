"""numpy 版 SDF 工具与 Surface Nets 网格化（在 Blender 自带 Python 中运行）。"""
import numpy as np


def ellipsoid(X, Y, Z, c, r):
    px, py, pz = (X - c[0]) / r[0], (Y - c[1]) / r[1], (Z - c[2]) / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    qx, qy, qz = px / r[0], py / r[1], pz / r[2]
    k1 = np.sqrt(qx * qx + qy * qy + qz * qz) + 1e-9
    return k0 * (k0 - 1.0) / k1


def sphere(X, Y, Z, c, r):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def capsule(X, Y, Z, a, b, r):
    return round_cone(X, Y, Z, a, b, r, r)


def round_cone(X, Y, Z, a, b, r1, r2):
    """线性插值半径的胶囊（近似圆锥），足够用于建模。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ba = b - a
    L2 = float(ba @ ba)
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / L2, 0.0, 1.0)
    dx, dy, dz = px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h
    return np.sqrt(dx * dx + dy * dy + dz * dz) - (r1 + (r2 - r1) * h)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def ssub(a, b, k):
    """从 a 中平滑减去 b。"""
    return smax(a, -b, k)


def surface_nets(F, origin, h):
    """F: (nx, ny, nz) 标量场，负值为内部。返回 (verts[N,3], quads[M,4])。"""
    nx, ny, nz = F.shape
    inside = F < 0
    # 8 个角点
    corners = [(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    cs = [F[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for (i, j, k) in corners]
    ins = [c < 0 for c in cs]
    any_in = np.zeros(cs[0].shape, bool)
    all_in = np.ones(cs[0].shape, bool)
    for m in ins:
        any_in |= m
        all_in &= m
    active = any_in & ~all_in
    idx = np.argwhere(active)
    cell_id = -np.ones(active.shape, np.int64)
    cell_id[active] = np.arange(len(idx))

    # 12 条边
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            da = np.subtract(corners[b], corners[a])
            if np.abs(da).sum() == 1:
                edges.append((a, b))
    acc = np.zeros((len(idx), 3), np.float64)
    cnt = np.zeros(len(idx), np.float64)
    sel = tuple(idx.T)
    for a, b in edges:
        fa = cs[a][sel].astype(np.float64)
        fb = cs[b][sel].astype(np.float64)
        cross = (fa < 0) != (fb < 0)
        t = np.where(cross, fa / np.where(cross, fa - fb, 1.0), 0.0)
        pa = np.array(corners[a], np.float64)
        pb = np.array(corners[b], np.float64)
        pos = pa[None, :] + (pb - pa)[None, :] * t[:, None]
        acc[cross] += pos[cross]
        cnt[cross] += 1
    verts = idx + acc / np.maximum(cnt, 1)[:, None]
    verts = np.asarray(origin) + verts * h

    quads = []
    # x 方向边：点 (i,j,k)-(i+1,j,k)，周围单元 (i, j-1..j, k-1..k)
    for axis in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        e_in0 = inside[tuple(sl0)]
        e_in1 = inside[tuple(sl1)]
        crossing = e_in0 != e_in1
        # 另外两个轴需要 >=1 且 <= n-2（单元索引范围）
        o1, o2 = [a for a in range(3) if a != axis]
        pts = np.argwhere(crossing)
        n = np.array(active.shape)
        ok = (pts[:, o1] >= 1) & (pts[:, o2] >= 1) & (pts[:, o1] <= n[o1] - 1) & (pts[:, o2] <= n[o2] - 1) & (pts[:, axis] <= n[axis] - 1)
        pts = pts[ok]
        flip = e_in0[tuple(pts.T)]  # 起点在内部
        def cid(d1, d2):
            q = pts.copy()
            q[:, o1] -= d1
            q[:, o2] -= d2
            return cell_id[tuple(q.T)]
        c00, c10, c11, c01 = cid(1, 1), cid(0, 1), cid(0, 0), cid(1, 0)
        f = np.stack([c00, c10, c11, c01], 1)
        f[flip] = f[flip][:, ::-1]
        if axis == 1:
            f = f[:, ::-1]
        quads.append(f)
    quads = np.concatenate(quads)
    quads = quads[(quads >= 0).all(1)]
    return verts.astype(np.float32), quads
