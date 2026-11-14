"""程序化写实头部：SDF 组合五官 → Surface Nets → Blender 网格。"""
import math
import numpy as np
import bpy
import bmesh

import sdf as S
from face_params import HEADS, NECK_CUT, TEX_X0, TEX_Z0, TEX_SIZE


def mirror_x(c):
    return (-c[0], c[1], c[2])


def ear_sdf(X, Y, Z, p, side):
    cx, cy, cz = p["ear_c"]
    cx *= side
    # 耳廓外翻约 18°：绕 z 轴旋转
    a = math.radians(18) * side
    dx, dy = X - cx, Y - cy
    lx = dx * math.cos(a) + dy * math.sin(a)
    ly = -dx * math.sin(a) + dy * math.cos(a)
    lz = Z - cz
    rx, ry, rz = p["ear_r"]
    plate = S.ellipsoid(lx, ly, lz, (0, 0, 0), (rx, ry, rz))
    rim = S.ellipsoid(lx, ly, lz, (0, 0.002, 0.004), (rx * 1.25, ry * 0.95, rz * 0.95))
    concha = S.ellipsoid(lx, ly, lz, (side * 0.0045, -0.003, -0.006), (0.0042, ry * 0.45, rz * 0.38))
    fossa = S.ellipsoid(lx, ly, lz, (side * 0.0045, 0.001, 0.012), (0.0028, ry * 0.5, rz * 0.3))
    e = S.smin(plate, rim, 0.002)
    e = S.ssub(e, concha, 0.0015)
    e = S.ssub(e, fossa, 0.0012)
    return e


def head_sdf(X, Y, Z, p):
    d = S.ellipsoid(X, Y, Z, p["cranium_c"], p["cranium_r"])
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["forehead_c"], p["forehead_r"]), 0.02)
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["jaw_c"], p["jaw_r"]), 0.03)
    for s in (1, -1):
        a = (p["mand_a"][0] * s, *p["mand_a"][1:])
        b = (p["mand_b"][0] * s, *p["mand_b"][1:])
        d = S.smin(d, S.capsule(X, Y, Z, a, b, p["mand_r"]), 0.02)
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["chin_c"], p["chin_r"]), 0.015)
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["muzzle_c"], p["muzzle_r"]), 0.02)
    for s in (1, -1):
        c = (p["cheek_c"][0] * s, *p["cheek_c"][1:])
        d = S.smin(d, S.ellipsoid(X, Y, Z, c, p["cheek_r"]), 0.02)
        c = (p["zyg_c"][0] * s, *p["zyg_c"][1:])
        d = S.smin(d, S.ellipsoid(X, Y, Z, c, p["zyg_r"]), 0.015)
        b = (p["brow_b"][0] * s, *p["brow_b"][1:])
        d = S.smin(d, S.capsule(X, Y, Z, p["brow_a"], b, p["brow_rad"]), 0.014)
    # 太阳穴微凹
    for s in (1, -1):
        d = S.ssub(d, S.ellipsoid(X, Y, Z, (0.078 * s, -0.045, 0.022), (0.012, 0.03, 0.025)), 0.012)
    # 眼窝
    for s in (1, -1):
        c = (p["socket_c"][0] * s, *p["socket_c"][1:])
        d = S.ssub(d, S.ellipsoid(X, Y, Z, c, p["socket_r"]), 0.010)
    # 眼睑（略大于眼球的球壳，再挖出杏仁形眼裂）
    for s in (1, -1):
        ce = (p["eye_c"][0] * s, *p["eye_c"][1:])
        d = S.smin(d, S.sphere(X, Y, Z, ce, p["eye_rad"] + 0.0011), 0.004)
        dx = (X - ce[0]) * s  # 正值朝外眼角
        dz = Z - ce[2]
        w = p["eye_w"]
        tt = np.clip(dx / w, -1.0, 1.0)
        bell = 1.0 - tt * tt
        skew = 1.0 + 0.25 * tt  # 上睑外侧略高
        zu = p["eye_up"] * bell * skew + p["eye_tilt"] * tt
        zl = -p["eye_lo"] * bell * (1.0 - 0.15 * tt) + p["eye_tilt"] * tt
        d2 = np.maximum(np.maximum(dz - zu, zl - dz), np.abs(dx) - w)
        d_open = np.maximum(d2, Y - (ce[1] + 0.003))
        d_open = np.maximum(d_open, S.sphere(X, Y, Z, ce, 0.022))
        d = S.smax(d, -d_open, 0.0012)
    # 鼻
    d = S.smin(d, S.round_cone(X, Y, Z, p["nose_a"], p["nose_b"], p["nose_r1"], p["nose_r2"]), 0.008)
    d = S.smin(d, S.sphere(X, Y, Z, p["nose_tip_c"], p["nose_tip_r"]), 0.006)
    base_c = (0.0, p["nose_tip_c"][1] + 0.017, p["nose_tip_c"][2] - 0.005)
    d = S.smin(d, S.ellipsoid(X, Y, Z, base_c, (0.018, 0.012, 0.008)), 0.006)
    for s in (1, -1):
        c = (p["ala_c"][0] * s, *p["ala_c"][1:])
        d = S.smin(d, S.ellipsoid(X, Y, Z, c, p["ala_r"]), 0.005)
    for s in (1, -1):
        c = (p["nostril_c"][0] * s, *p["nostril_c"][1:])
        d = S.ssub(d, S.ellipsoid(X, Y, Z, c, p["nostril_r"]), 0.0018)
    # 人中两侧
    ul = p["upper_lip_c"]
    for s in (1, -1):
        d = S.smin(d, S.capsule(X, Y, Z, (0.0035 * s, ul[1] - 0.004, ul[2] + 0.003),
                               (0.0045 * s, ul[1] + 0.001, p["nose_tip_c"][2] - 0.011), 0.0022), 0.003)
    # 唇
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["upper_lip_c"], p["upper_lip_r"]), 0.004)
    d = S.smin(d, S.ellipsoid(X, Y, Z, p["lower_lip_c"], p["lower_lip_r"]), 0.004)
    d = S.ssub(d, S.ellipsoid(X, Y, Z, (0.0, -0.104, p["mouth_z"]), (p["mouth_w"], 0.016, 0.0010)), 0.0013)
    lc = p["lower_lip_c"]
    d = S.ssub(d, S.ellipsoid(X, Y, Z, (0.0, lc[1] - 0.0025, lc[2] - 0.0125), (0.015, 0.006, 0.0038)), 0.006)
    for s in (1, -1):  # 嘴角小凹
        d = S.ssub(d, S.sphere(X, Y, Z, (p["mouth_w"] * 0.98 * s, ul[1] + 0.006, p["mouth_z"]), 0.0028), 0.004)
    # 耳
    for s in (1, -1):
        d = S.smin(d, ear_sdf(X, Y, Z, p, s), 0.005)
    # 颈
    d = S.smin(d, S.round_cone(X, Y, Z, p["neck_a"], p["neck_b"], p["neck_r1"], p["neck_r2"]), 0.03)
    if p["adam"]:
        d = S.smin(d, S.ellipsoid(X, Y, Z, (0, -0.031, -0.14), (0.010, 0.012, 0.016)), 0.014)
    d = np.maximum(d, NECK_CUT - Z)
    return d


def eval_grid(p, h=0.0009):
    x0, x1 = -0.105, 0.105
    y0, y1 = -0.135, 0.125
    z0, z1 = NECK_CUT - 0.006, 0.13
    xs = np.arange(x0, x1 + h, h, dtype=np.float32)
    ys = np.arange(y0, y1 + h, h, dtype=np.float32)
    zs = np.arange(z0, z1 + h, h, dtype=np.float32)
    F = np.empty((len(xs), len(ys), len(zs)), np.float32)
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    step = 24
    for k0 in range(0, len(zs), step):
        zz = zs[k0:k0 + step]
        X = np.repeat(X2[:, :, None], len(zz), 2)
        Y = np.repeat(Y2[:, :, None], len(zz), 2)
        Z = np.broadcast_to(zz[None, None, :], X.shape).astype(np.float32)
        F[:, :, k0:k0 + step] = head_sdf(X, Y, Z, p)
    return F, (x0, y0, z0), h


def mesh_from_arrays(name, verts, quads):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build_head(kind="male", target_faces=70000):
    p = HEADS[kind]
    F, origin, h = eval_grid(p)
    verts, quads = S.surface_nets(F, origin, h)
    ob = mesh_from_arrays(f"Head_{kind}", verts, quads)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    # 平滑表面网格的阶梯
    m = ob.modifiers.new("smooth", "LAPLACIANSMOOTH")
    m.lambda_factor = 0.6
    m.iterations = 3
    m.use_volume_preserve = True
    bpy.ops.object.modifier_apply(modifier=m.name)
    ratio = min(1.0, target_faces / max(1, len(ob.data.polygons) * 2))
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.shade_smooth()
    front_uv(ob)
    return ob


def front_uv(ob):
    me = ob.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers.active.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    u = (co[vi, 0] - TEX_X0) / TEX_SIZE
    v = (co[vi, 2] - TEX_Z0) / TEX_SIZE
    uv.foreach_set("uv", np.stack([u, v], 1).ravel())
