"""批量构建脸谱头像：导出 site/models/head_<id>.glb，渲染 site/thumbs/head_<id>.png。
用法：blender -b --factory-startup -P build_heads.py -- [id ...] [--preview] [--noexport]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import blib
import mats
import heads

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
flags = {a for a in argv if a.startswith("--")}
ids = [a for a in argv if not a.startswith("--")] or list(heads.CHARS)

blib.reset()
blib.render_setup((800, 800), 128 if "--preview" not in flags else 48, transparent=True)


def lights():
    area, ls = blib.studio(strength=1.0)
    area("key", (-0.55, -0.7, 0.45), 0.6, 70)
    area("fill", (0.7, -0.5, 0.05), 1.0, 22)
    area("rim", (0.35, 0.65, 0.4), 0.5, 80)
    area("top", (0, -0.1, 0.9), 0.6, 25)
    return ls


ls = lights()
for cid in ids:
    t = time.time()
    root = heads.build(cid)
    objs = heads.all_objects(root)
    print(f"[{cid}] built {len(objs)} objects in {time.time() - t:.1f}s")
    if "--noexport" not in flags:
        blib.export_glb(os.path.join(blib.OUT_MODELS, f"head_{cid}.glb"), objs)
    views = [("thumb", (-0.42, -0.95, 0.12), 60)] if "--preview" not in flags else [("front", (0, -1.1, 0.03), 60), ("thumb", (-0.42, -0.95, 0.12), 60), ("side", (-1.1, 0.0, 0.05), 60)]
    for name, loc, lens in views:
        tall = 0.06 if cid != "sunwukong" else 0.06
        blib.camera(loc, (0, 0, 0.0), lens=lens, name="cam_" + name)
        out = os.path.join(blib.OUT_THUMBS, f"head_{cid}.png") if name == "thumb" and "--preview" not in flags else os.path.join(blib.PREVIEW, f"{cid}_{name}.png")
        blib.render(out)
    for o in reversed(objs):
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"[{cid}] done {time.time() - t:.1f}s")
