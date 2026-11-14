"""共享材质库（名称前缀供网页端识别：Hair_/Feather/Fur_ 需要透明裁切）。"""
import os
import bpy
from blib import pbr, TEX

_cache = {}


def T(name):
    return os.path.join(TEX, name)


def get(key):
    if key in _cache:
        return _cache[key]
    m = make(key)
    _cache[key] = m
    return m


def reset():
    _cache.clear()


def make(key):
    if key.startswith("face_"):
        n = key[5:]
        return pbr(f"Face_{n}", base_tex=T(f"face_{n}_base.jpg"), rough_tex=T(f"face_{n}_orm.png"),
                   normal_tex=T(f"face_{n}_nrm.png"), normal_strength=0.6, sss=0.08, extend=True)
    if key.startswith("hair_"):
        col = {"black": (0.012, 0.011, 0.011, 1), "red": (0.55, 0.03, 0.025, 1), "white": (0.9, 0.88, 0.84, 1),
               "gold": (0.9, 0.62, 0.12, 1), "yellow": (0.95, 0.75, 0.15, 1), "redtassel": (0.7, 0.04, 0.05, 1),
               "pink": (0.9, 0.35, 0.45, 1), "green": (0.1, 0.45, 0.25, 1)}[key[5:]]
        tex = "hair_curly.png" if key == "hair_red" else "hair.png"
        return pbr(f"Hair_{key[5:]}", color=col, rough=0.45, base_tex=T(tex), alpha_tex="same", double=True, sheen=0.4)
    if key == "feather":
        return pbr("Feather_pheasant", rough=0.5, base_tex=T("feather.png"), alpha_tex="same", double=True, sheen=0.3)
    table = {
        "gold": dict(color=(1.0, 0.72, 0.32, 1), rough=0.26, metal=1.0),
        "gold_dark": dict(color=(0.85, 0.55, 0.2, 1), rough=0.35, metal=1.0),
        "silver": dict(color=(0.9, 0.9, 0.92, 1), rough=0.25, metal=1.0),
        "pearl": dict(color=(0.93, 0.9, 0.84, 1), rough=0.18, coat=1.0),
        "jade": dict(color=(0.45, 0.72, 0.5, 1), rough=0.12, coat=1.0),
        "ruby": dict(color=(0.6, 0.02, 0.05, 1), rough=0.08, coat=1.0),
        "cui": dict(color=(0.04, 0.32, 0.62, 1), rough=0.25, metal=0.3, coat=0.6),  # 点翠
        "satin_black": dict(color=(0.012, 0.012, 0.014, 1), rough=0.38, sheen=0.2, double=True),
        "satin_green": dict(color=(0.02, 0.16, 0.06, 1), rough=0.4, sheen=0.3, double=True),
        "satin_blue": dict(color=(0.03, 0.07, 0.28, 1), rough=0.4, sheen=0.25, double=True),
        "satin_red": dict(color=(0.5, 0.03, 0.03, 1), rough=0.4, sheen=0.25, double=True),
        "satin_purple": dict(color=(0.25, 0.06, 0.3, 1), rough=0.4, sheen=0.25, double=True),
        "wangzi": dict(color=(0.012, 0.012, 0.014, 1), rough=0.5, sheen=0.5),  # 水纱 / 网子
        "bun": dict(color=(0.01, 0.01, 0.012, 1), rough=0.3, coat=0.3),
        "wire": dict(color=(0.05, 0.05, 0.05, 1), rough=0.4, metal=0.6),
        "eye": dict(color=(1, 1, 1, 1), rough=0.05, coat=1.0, base_tex=T("eye_base.jpg")),
    }
    if key in table:
        return pbr(key.capitalize(), **table[key])
    if key.startswith("pom_"):
        col = {"red": (0.72, 0.03, 0.04, 1), "pink": (0.92, 0.4, 0.5, 1), "green": (0.06, 0.4, 0.2, 1),
               "blue": (0.06, 0.2, 0.6, 1), "yellow": (0.95, 0.7, 0.1, 1), "white": (0.92, 0.9, 0.86, 1)}[key[4:]]
        return pbr(f"Fur_{key[4:]}", color=col, rough=0.85, sheen=0.5, normal_tex=T("fur_nrm.png"), normal_strength=1.0)
    raise KeyError(key)
