"""八张京剧脸谱 / 旦角妆面的贴图绘制。坐标单位：毫米（正面投影，x 向观众右侧，z 向上）。"""
import numpy as np
from paint import Canvas, catmull
from face_params import landmarks

L_M = {k: (v * 1000 if isinstance(v, float) else tuple(np.array(v) * 1000)) for k, v in landmarks("male").items()}
L_F = {k: (v * 1000 if isinstance(v, float) else tuple(np.array(v) * 1000)) for k, v in landmarks("female").items()}

BLACK = "#121111"
WHITE = "#f2efe8"


def almond(L, side=1, grow=0.0, n=40):
    """与几何眼裂一致的杏仁形轮廓（单侧），grow 为外扩毫米。"""
    cx, cz = L["eye"]
    w, up, lo, tilt = L["eye_w"], L["eye_up"], L["eye_lo"], L["eye_tilt"]
    t = np.linspace(-1, 1, n)
    bell = 1 - t * t
    zu = up * bell * (1 + 0.25 * t) + tilt * t
    zl = -lo * bell * (1 - 0.15 * t) + tilt * t
    xs = cx + t * (w + grow)
    top = np.stack([xs, cz + zu + grow * np.sqrt(np.clip(bell, 0, 1) + 0.15)], 1)
    bot = np.stack([xs, cz + zl - grow * np.sqrt(np.clip(bell, 0, 1) + 0.15)], 1)
    pts = np.concatenate([top, bot[::-1]])
    pts[:, 0] = pts[:, 0] * side if side == 1 else -pts[:, 0]
    return pts


def eye_liner(c, L, width=1.4, color=BLACK, wing=None, lower=True):
    """沿眼裂描黑线，可带上挑的眼尾。"""
    for s in (1, -1):
        pts = almond(L, 1, grow=width * 0.5)
        inner = almond(L, 1, grow=-0.2)
        poly = [pts, inner]
        m = c.mask_poly([pts * np.array([s, 1])], 0.15) - c.mask_poly([inner * np.array([s, 1])], 0.15) * (0 if lower else 0)
        c.apply(np.clip(m, 0, 1), color, rough=0.28, thick=0.5)
        if wing:
            c.stroke([(x * s, z) for x, z in wing], width, 0.4, color=color, rough=0.28)


def nostrils(c, L, color=BLACK, r=(4.2, 3.0)):
    nx, nz = L["nostril"]
    c.ellipse((nx, nz), r[0], r[1], color, mirror=True, rough=0.3)


def lips(c, L, color, w_scale=1.0, rough=0.3):
    mz, mw = L["mouth_z"], L["mouth_w"] * w_scale
    ul, ll = L["upper_lip_z"], L["lower_lip_z"]
    up = [(-mw, mz), (-mw * 0.5, ul - 1.0), (-mw * 0.15, ul + 3.2), (0, ul + 2.2), (mw * 0.15, ul + 3.2), (mw * 0.5, ul - 1.0), (mw, mz),
          (mw * 0.6, ll - 3.5), (0, ll - 5.5), (-mw * 0.6, ll - 3.5)]
    c.fill(up, color, rough=rough, thick=0.4)


def neck_fade(c, color, alpha=0.5):
    X, Z = c.grid()
    m = np.clip((-(Z) - 128) / 25, 0, 1)
    c.apply(m, color, alpha=alpha, thick=0, noise=0)


# ---------------------------------------------------------------- 关羽：红整脸
def guanyu():
    L = L_M
    c = Canvas(base="#a3161b", rough=0.36)
    c.vignette("#6e0c10", 40, 95, center=(0, 5), alpha=0.85)  # 揉脸：四周深
    c.radial((0, 40), 30, 26, "#c0262a", alpha=0.35)
    # 卧蚕眉：粗重上扬，压在眉骨之上
    brow = [(8, 25), (16, 31), (28, 37), (42, 45), (55, 54), (64, 62), (67, 63), (66, 57), (57, 47), (44, 37), (30, 29), (17, 23), (9, 21)]
    c.fill(brow, BLACK, mirror=True, rough=0.3, thick=1.0)
    for k in range(9):  # 眉毛的丝缕
        t = k / 8
        x0 = 12 + 50 * t
        c.stroke([(x0, 24 + 30 * t), (x0 + 6, 30 + 33 * t)], 0.7, 0.2, color="#2a0b0b", mirror=True, alpha=0.6)
    # 丹凤眼：细长，眼尾高挑入鬓
    eye_liner(c, L, 1.8, wing=[(45, 3), (54, 9), (63, 17), (68, 22)])
    c.stroke([(18, 6.5), (30, 9.5), (44, 9), (54, 13)], 1.0, 0.6, color=BLACK, mirror=True, rough=0.3)
    # 印堂纹
    c.stroke([(2.5, 28), (3.2, 38), (4.5, 50)], 1.1, 0.5, color=BLACK, mirror=True)
    # 鼻窝与嘴
    nostrils(c, L)
    c.stroke([(14, -40), (21, -48), (20, -56)], 1.6, 0.6, color=BLACK, mirror=True, alpha=0.8)
    lips(c, L, "#2a0a0a")
    # 破脸：一颗痣
    c.ellipse((-38, -36), 2.6, 2.4, BLACK, rough=0.3)
    neck_fade(c, "#8f1418", 0.4)
    return c.save("face_guanyu")


# ---------------------------------------------------------------- 包拯：黑整脸
def baozheng():
    L = L_M
    c = Canvas(base="#141314", rough=0.24)
    c.radial((0, 0), 70, 80, "#2a2627", alpha=0.35)
    # 额上白月牙（角朝上）
    X, Z = c.grid()
    outer = np.clip(1 - (np.sqrt(X ** 2 + (Z - 60) ** 2) - 11.5) / 0.4, 0, 1)
    inner = np.clip((np.sqrt(X ** 2 + (Z - 64.5) ** 2) - 10.0) / 0.4, 0, 1)
    c.apply(np.clip(outer, 0, 1) * np.clip(inner, 0, 1), WHITE, rough=0.4, thick=1.0)
    # 白色皱眉（紧锁上挑）
    c.stroke([(4, 22), (10, 30), (20, 38), (36, 42), (52, 41), (63, 35)], 2.6, 1.0, color=WHITE, mirror=True, rough=0.4)
    c.stroke([(4, 22), (8, 34), (9, 45)], 1.4, 0.6, color=WHITE, mirror=True, rough=0.4)
    c.stroke([(14, 27), (28, 32), (44, 33), (56, 29)], 1.2, 0.6, color=WHITE, mirror=True, rough=0.4)
    # 白眼圈
    for s in (1, -1):
        ring = c.mask_poly([almond(L, 1, grow=3.6) * np.array([s, 1])], 0.2) - c.mask_poly([almond(L, 1, grow=1.6) * np.array([s, 1])], 0.2)
        c.apply(np.clip(ring, 0, 1), WHITE, rough=0.4, thick=0.8)
    c.stroke([(48, 4), (58, 9), (66, 13)], 1.6, 0.4, color=WHITE, mirror=True)
    # 鼻窝白线、嘴边白纹
    c.stroke([(9, -30), (19, -38), (24, -48), (19, -57)], 1.8, 0.8, color=WHITE, mirror=True, rough=0.4)
    c.stroke([(22, -62), (30, -72), (28, -86)], 1.6, 0.6, color=WHITE, mirror=True, rough=0.4)
    lips(c, L, "#3b0d10")
    return c.save("face_baozheng")


# ---------------------------------------------------------------- 曹操：水白脸
def caocao():
    L = L_M
    c = Canvas(base="#efe9df", rough=0.42)
    c.vignette("#d8cfc2", 45, 95, alpha=0.6)
    c.radial((34, 6), 22, 14, "#e2a7a0", alpha=0.45, mirror=True)  # 眼窝淡红晕
    # 细长立眉
    c.stroke([(7, 21), (20, 28), (34, 36), (50, 47), (63, 58)], 2.6, 0.5, color=BLACK, mirror=True, rough=0.3)
    # 三角眼 + 眼尾长纹
    eye_liner(c, L, 1.4, wing=[(45, 2), (55, 9), (65, 20)])
    c.stroke([(44, -3), (52, 2), (60, 10)], 0.9, 0.3, color=BLACK, mirror=True)
    c.stroke([(17, 7.5), (30, 11), (44, 10), (50, 12)], 0.8, 0.4, color=BLACK, mirror=True)
    # 额纹、印堂纹
    for k, (dx, h) in enumerate([(3, 0), (6, 1)]):
        c.stroke([(dx, 24), (dx + 1, 36), (dx + 3, 48)], 0.9, 0.3, color=BLACK, mirror=True)
    for z in (52, 59, 66):
        c.stroke([(-14, z - 3), (0, z), (14, z - 3)], 0.7, 0.7, color="#3a3433", alpha=0.7)
    # 鱼尾纹、法令纹
    c.stroke([(52, -6), (60, -10), (66, -12)], 0.7, 0.2, color=BLACK, mirror=True)
    c.stroke([(52, -11), (58, -17), (62, -22)], 0.7, 0.2, color=BLACK, mirror=True)
    c.stroke([(21, -44), (27, -56), (30, -70), (29, -82)], 1.0, 0.3, color=BLACK, mirror=True)
    nostrils(c, L)
    lips(c, L, "#8e1d22", rough=0.28)
    return c.save("face_caocao")


# ---------------------------------------------------------------- 张飞：黑十字门蝴蝶脸
def zhangfei():
    L = L_M
    c = Canvas(base="#ebe6dd", rough=0.4)
    c.vignette("#cfc7ba", 50, 100, alpha=0.6)
    c.radial((40, -36), 22, 20, "#e59a8f", alpha=0.55, mirror=True)  # 笑脸腮红
    # 黑“门”：额中竖条贯到鼻梁
    c.fill([(-11, 100), (11, 100), (9, 70), (6, 45), (4.5, 20), (5, 0), (-5, 0), (-4.5, 20), (-6, 45), (-9, 70)], BLACK, rough=0.28, thick=1.0)
    # 蝴蝶翅：大眉
    wing = [(6, 26), (10, 40), (20, 58), (36, 74), (54, 80), (68, 70), (70, 52), (62, 38), (48, 31), (30, 27), (14, 22)]
    c.fill(wing, BLACK, mirror=True, rough=0.28, thick=1.0)
    c.stroke([(20, 40), (32, 56), (48, 66), (60, 62), (58, 52), (50, 50)], 1.4, 1.0, color=WHITE, mirror=True, rough=0.4)
    c.ellipse((52, 56), 2.6, 2.6, WHITE, mirror=True)
    # 白眼窝 + 黑色笑眼
    c.fill([(13, 16), (28, 22), (46, 20), (62, 10), (64, -4), (52, -14), (34, -14), (18, -8), (12, 2)], WHITE, mirror=True, rough=0.4)
    for s in (1, -1):
        m = c.mask_poly([almond(L, 1, grow=2.8) * np.array([s, 1])], 0.2)
        c.apply(m, BLACK, rough=0.28, thick=0.7)
    c.stroke([(14, 12), (28, 16), (44, 15), (58, 6), (64, -2)], 2.0, 1.4, color=BLACK, mirror=True)
    c.stroke([(46, -4), (56, -10), (60, -16)], 1.6, 0.6, color=BLACK, mirror=True)
    # 鼻头与大笑口
    c.fill([(-6, -2), (6, -2), (9, -24), (20, -36), (22, -48), (12, -56), (0, -54), (-12, -56), (-22, -48), (-20, -36), (-9, -24)], BLACK, rough=0.28, thick=1.0)
    mouth = [(0, -58), (16, -58), (32, -52), (46, -42), (50, -44), (44, -58), (30, -72), (14, -82), (0, -84)]
    c.fill(mouth + [(-x, z) for x, z in reversed(mouth[1:-1])], BLACK, rough=0.28, thick=1.0)
    return c.save("face_zhangfei")


# ---------------------------------------------------------------- 窦尔敦：蓝花三块瓦脸
def douerdun():
    L = L_M
    c = Canvas(base="#1f4fa8", rough=0.34)
    c.vignette("#123276", 45, 95, alpha=0.8)
    c.radial((0, 50), 28, 22, "#3b6fcc", alpha=0.45)
    # 黑眼窝大块 + 白边
    patch = [(9, 22), (22, 34), (40, 40), (58, 34), (68, 18), (66, -2), (56, -18), (38, -24), (22, -18), (12, -6), (8, 8)]
    c.fill(patch, WHITE, mirror=True, rough=0.4)
    shrink = [(11.5, 21), (23, 31.5), (40, 37), (56.5, 32), (65.5, 17), (63.5, -1.5), (54.5, -15.5), (38, -21.5), (23, -16), (14, -5), (10.8, 8)]
    c.fill(shrink, BLACK, mirror=True, rough=0.28, thick=1.0)
    # 白色眉纹与白眼圈
    c.stroke([(14, 21), (26, 29), (42, 33), (56, 28), (62, 20), (58, 15), (54, 19)], 1.8, 1.2, color=WHITE, mirror=True, rough=0.4)
    for s in (1, -1):
        ring = c.mask_poly([almond(L, 1, grow=3.0) * np.array([s, 1])], 0.2) - c.mask_poly([almond(L, 1, grow=1.3) * np.array([s, 1])], 0.2)
        c.apply(np.clip(ring, 0, 1), WHITE, rough=0.4, thick=0.6)
    c.radial((40, -28), 14, 9, "#c23a3a", alpha=0.7, mirror=True)
    # 额上“双钩”
    for s in (1, -1):
        hook = [(3, 30), (4.5, 50), (5.5, 68), (8, 80), (14, 86), (20, 84), (21, 77), (16, 73), (13, 76), (15, 79), (12, 80), (9.5, 74), (8.5, 64), (8, 50), (7, 30)]
        pts = [(x * s, z) for x, z in hook]
        c.fill(pts, BLACK, blur=0.3, rough=0.28, thick=0.6)
        inner = [(x * s * 0.97 + 0.3 * s, z * 0.995) for x, z in hook]
        c.stroke([(5 * s, 32), (6.2 * s, 52), (7 * s, 68), (9.5 * s, 78), (14 * s, 83), (18 * s, 81)], 2.2, 1.0, color="#e2b23a", rough=0.35, metal=0.2)
    c.fill([(-3, 30), (3, 30), (2, 46), (0, 52), (-2, 46)], "#c62a2a", rough=0.35)
    # 鼻窝、嘴
    c.fill([(-5, -10), (5, -10), (8, -30), (18, -38), (22, -48), (14, -56), (0, -52), (-14, -56), (-22, -48), (-18, -38), (-8, -30)], BLACK, rough=0.28)
    c.stroke([(19, -38), (24, -48), (16, -57)], 1.2, 0.8, color=WHITE, mirror=True)
    mouth = [(0, -60), (18, -60), (30, -56), (34, -62), (26, -74), (12, -80), (0, -81)]
    c.fill(mouth + [(-x, z) for x, z in reversed(mouth[1:-1])], BLACK, rough=0.28)
    return c.save("face_douerdun")


# ---------------------------------------------------------------- 孙悟空：猴脸
def sunwukong():
    L = L_M
    GOLD = "#d39a2a"
    c = Canvas(base=GOLD, rough=0.36, metal=0.35)
    c.vignette("#8a5a12", 50, 95, alpha=0.7)
    # 额上毛纹（拱形排列）
    for row, (r, n) in enumerate([(58, 11), (66, 13), (74, 15)]):
        for k in range(n):
            a = np.pi * (0.18 + 0.64 * k / (n - 1))
            x, z = r * np.cos(a), r * np.sin(a) - 8
            dx, dz = np.cos(a) * 6, np.sin(a) * 6
            c.stroke([(x, z), (x + dx * 0.6 + 1.2, z + dz * 0.6), (x + dx + 2, z + dz)], 1.5, 0.3, color="#1a1210", rough=0.3, metal=0.0)
    # 倒栗形红脸
    peach = [(0, 52), (14, 54), (30, 48), (46, 36), (56, 18), (57, -4), (50, -26), (38, -48), (22, -70), (0, -92)]
    c.fill(peach + [(-x, z) for x, z in reversed(peach[1:-1])], "#b81e22", rough=0.32, metal=0.0, thick=1.0)
    c.stroke(peach + [(-x, z) for x, z in reversed(peach[1:])], 1.8, 1.8, color=BLACK, rough=0.3)
    # 金眼窝 + 黑圈
    for s in (1, -1):
        c.ellipse((33 * s, 3), 19, 14, "#1a1210", rot=0.08 * s, rough=0.3, metal=0.0)
        c.ellipse((33 * s, 3), 17.2, 12.2, "#e8b64a", rot=0.08 * s, rough=0.3, metal=0.55)
    eye_liner(c, L, 2.0)
    # 卷眉
    c.stroke([(10, 22), (20, 28), (36, 30), (50, 26), (56, 20), (52, 15), (47, 18)], 2.2, 1.4, color=BLACK, mirror=True)
    # 白嘴套与鼻、口
    c.ellipse((0, -56), 27, 27, "#f2ece0", rough=0.4, metal=0.0, thick=0.8)
    c.ellipse((0, -56), 27, 27, BLACK, alpha=0.0)
    c.ellipse((6.5, -46), 3.6, 2.4, BLACK, mirror=True, rot=0.5, rough=0.3)
    c.stroke([(0, -40), (0, -60)], 1.0, 1.0, color=BLACK)
    c.stroke([(-24, -58), (-12, -66), (0, -62), (12, -66), (24, -58)], 1.6, 1.6, color=BLACK)
    c.stroke([(-14, -72), (0, -78), (14, -72)], 1.2, 1.2, color=BLACK)
    # 两颊毛
    for k in range(14):
        z = 30 - k * 8
        x = 60 - max(0, -z) * 0.12
        c.stroke([(x, z), (x + 5, z - 3), (x + 9, z - 7)], 1.4, 0.3, color="#1a1210", mirror=True)
    return c.save("face_sunwukong")


# ---------------------------------------------------------------- 项羽：霸王哭脸
def xiangyu():
    L = L_M
    c = Canvas(base="#ece7de", rough=0.4)
    c.vignette("#cfc5b6", 50, 100, alpha=0.6)
    # 下垂的哭眼大块
    patch = [(10, 24), (24, 30), (42, 30), (58, 22), (66, 6), (64, -16), (56, -36), (50, -44), (46, -32), (40, -20), (26, -12), (13, -4), (9, 10)]
    c.fill(patch, BLACK, mirror=True, rough=0.28, thick=1.0)
    for s in (1, -1):
        ring = c.mask_poly([almond(L, 1, grow=2.8) * np.array([s, 1])], 0.2) - c.mask_poly([almond(L, 1, grow=1.2) * np.array([s, 1])], 0.2)
        c.apply(np.clip(ring, 0, 1), WHITE, rough=0.4, thick=0.5)
    # 下垂粗眉
    c.stroke([(5, 34), (16, 42), (32, 46), (50, 42), (62, 32), (68, 16)], 5.0, 2.0, color=BLACK, mirror=True, rough=0.28)
    c.stroke([(52, 40), (60, 34), (60, 28), (55, 30)], 1.4, 1.4, color=WHITE, mirror=True)
    # 额中“寿”字纹
    c.fill([(0, 38), (6, 50), (5, 64), (0, 86), (-5, 64), (-6, 50)], BLACK, rough=0.28)
    for z, w in ((48, 13), (58, 11), (68, 9), (77, 7)):
        c.stroke([(-w, z), (w, z)], 2.0, 2.0, color=BLACK)
    c.ellipse((0, 58), 2.4, 3.2, "#b81e22")
    # 鼻、嘴（下撇）
    c.fill([(-4, -6), (4, -6), (8, -28), (20, -40), (22, -50), (12, -56), (0, -52), (-12, -56), (-22, -50), (-20, -40), (-8, -28)], BLACK, rough=0.28)
    mouth = [(0, -60), (16, -61), (28, -66), (36, -80), (30, -82), (20, -76), (8, -78), (0, -80)]
    c.fill(mouth + [(-x, z) for x, z in reversed(mouth[1:-1])], BLACK, rough=0.28)
    return c.save("face_xiangyu")


# ---------------------------------------------------------------- 杨贵妃：旦角俊扮
def yangguifei():
    L = L_F
    c = Canvas(base="#f3e1d6", rough=0.42)
    c.vignette("#e9cbbd", 45, 90, alpha=0.5)
    # 胭脂：眼皮最浓，向两颊晕开，鼻梁留白
    c.radial((32, 2), 30, 40, "#d9566a", alpha=0.85, power=1.1, mirror=True)
    c.radial((33, 10), 18, 12, "#c23048", alpha=0.55, power=1.2, mirror=True)
    X, Z = c.grid()
    bridge = np.clip(1 - np.abs(X) / 8, 0, 1) * np.clip((Z + 40) / 30, 0, 1) * np.clip((40 - Z) / 15, 0, 1)
    c.apply(bridge, "#f3e1d6", alpha=0.7, thick=0, noise=0)
    # 柳叶吊眉
    c.stroke([(11, 21), (22, 27), (36, 31), (52, 33), (64, 30)], 2.4, 0.6, color="#161012", mirror=True, rough=0.3)
    # 吊眼线
    eye_liner(c, L, 1.3, wing=[(46, 4), (55, 9), (64, 15)])
    c.stroke([(17, 7.5), (30, 10.5), (44, 10.5), (52, 12)], 1.6, 0.6, color="#161012", mirror=True)
    # 樱桃小口
    lips(c, L, "#c0182e", w_scale=0.55, rough=0.22)
    # 片子：额前小弯 + 两颊大片（黑亮）
    for k in range(-5, 6):
        x = k * 9.5
        z = 74 - 0.012 * x * x
        c.ellipse((x, z), 6.0, 7.5, "#0d0b0c", rough=0.15, thick=1.2)
    for s in (1, -1):
        dapian = [(48 * s, 78), (60 * s, 70), (66 * s, 40), (64 * s, 0), (60 * s, -40), (57 * s, -58), (53 * s, -44), (54 * s, -6), (52 * s, 30), (44 * s, 60)]
        c.fill(dapian, "#0d0b0c", rough=0.15, thick=1.2)
    X, Z = c.grid()
    c.apply(np.clip((Z - 80) / 3, 0, 1), "#0d0b0c", rough=0.15, thick=1.0)
    return c.save("face_yangguifei")


ALL = dict(guanyu=guanyu, baozheng=baozheng, caocao=caocao, zhangfei=zhangfei,
           douerdun=douerdun, sunwukong=sunwukong, xiangyu=xiangyu, yangguifei=yangguifei)

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(ALL[n]())
