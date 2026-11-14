"""批量构建戏服：导出 site/models/costume_<id>.glb（不含头），并装上对应头部渲染全身缩略图。
用法：blender -b --factory-startup -P build_costumes.py -- [id ...] [--preview] [--noexport]
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import mathutils
import numpy as np
import blib
import heads
import costumes
from costume_params import HEAD_Z

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
flags = {a for a in argv if a.startswith("--")}
ids = [a for a in argv if not a.startswith("--")] or list(costumes.COSTUMES)

blib.reset()
blib.render_setup((900, 1200), 160 if "--preview" not in flags else 64, transparent=True)
area, _ = blib.studio(strength=1.0)
area("key", (-2.2, -2.6, 3.0), 2.0, 900)
area("fill", (2.6, -1.8, 1.2), 3.0, 260)
area("rim", (1.2, 2.6, 2.6), 1.5, 700)
area("top", (0, -0.5, 4.0), 2.0, 200)
floor = bpy.data.objects.new("Floor", bpy.data.meshes.new("Floor"))
floor.data.from_pydata([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], [], [(0, 1, 2, 3)])
bpy.context.scene.collection.objects.link(floor)
floor.is_shadow_catcher = True


def fit_camera(objs, direction, lens=50, margin=1.08, aspect=0.75):
    pts = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            pts.append(o.matrix_world @ mathutils.Vector(c))
    pts = np.array([tuple(p) for p in pts])
    center = (pts.min(0) + pts.max(0)) / 2
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    up = np.array((0, 0, 1.0))
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, d)
    rel = pts - center
    ex = np.abs(rel @ right).max()
    ey = np.abs(rel @ up2).max()
    depth = (rel @ -d).max()
    sensor = 36.0
    fov_x = 2 * math.atan(sensor / 2 / lens)
    fov_y = 2 * math.atan(math.tan(fov_x / 2) / aspect) if aspect < 1 else fov_x
    if aspect < 1:  # 竖幅：传感器适配宽度 -> 以宽度为准
        fov_x_eff = fov_x * 1.0
        fov_x = 2 * math.atan(math.tan(fov_x / 2) * aspect)
        fov_y = 2 * math.atan(sensor / 2 / lens)
    dist = max(ex / math.tan(fov_x / 2), ey / math.tan(fov_y / 2)) * margin + depth
    cam = blib.camera(tuple(center - d * dist), tuple(center), lens=lens, name="cam")
    cam.data.sensor_fit = "VERTICAL" if aspect < 1 else "AUTO"
    cam.data.sensor_height = 36.0
    return cam


for cid in ids:
    t = time.time()
    cfg = costumes.COSTUMES[cid]
    root = cfg["build"]()
    objs = heads.all_objects(root)
    print(f"[{cid}] costume built {len(objs)} objects in {time.time() - t:.1f}s")
    if "--noexport" not in flags:
        blib.export_glb(os.path.join(blib.OUT_MODELS, f"costume_{cid}.glb"), objs)
    head = heads.build(cfg["head"])
    head.location = (0, 0, HEAD_Z)
    hobjs = heads.all_objects(head)
    bpy.context.view_layer.update()
    views = [("thumb", (0.42, 1.0, -0.12))]
    if "--preview" in flags:
        views += [("back", (-0.5, -1.0, -0.1)), ("side", (1.0, 0.0, -0.05))]
    for name, dvec in views:
        fit_camera(objs + hobjs, dvec, lens=50)
        out = os.path.join(blib.OUT_THUMBS, f"costume_{cid}.png") if (name == "thumb" and "--preview" not in flags) else os.path.join(blib.PREVIEW, f"costume_{cid}_{name}.png")
        blib.render(out)
    for o in reversed(objs + hobjs):
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"[{cid}] done {time.time() - t:.1f}s")
