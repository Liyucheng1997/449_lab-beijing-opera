"""把 Blender 渲染出的 PNG 缩略图转成 WebP（网页用）。"""
import glob
import os
from PIL import Image

d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site", "thumbs")
for f in glob.glob(os.path.join(d, "*.png")):
    im = Image.open(f)
    im.thumbnail((600, 800))
    im.save(f[:-4] + ".webp", quality=86, method=6)
    os.remove(f)
    print("webp", os.path.basename(f))
