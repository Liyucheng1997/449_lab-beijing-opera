"""头部几何参数与五官定位（Blender 与贴图脚本共用）。

坐标系：Blender 世界坐标，单位米。脸朝 -Y，Z 向上，X 为角色左手方向（观众视角的右侧为 -X）。
头部原点在两耳连线中点、约眼睛高度；脖子底部切口在 z = NECK_CUT。
脸谱贴图采用正面正交投影：u = (x - TEX_X0) / TEX_SIZE，v = (z - TEX_Z0) / TEX_SIZE。
"""

TEX_X0 = -0.13
TEX_Z0 = -0.15
TEX_SIZE = 0.26
NECK_CUT = -0.215

# 男性（净、生）基准参数
MALE = dict(
    scale=1.0,
    cranium_c=(0.0, 0.012, 0.025), cranium_r=(0.076, 0.098, 0.093),
    forehead_c=(0.0, -0.045, 0.038), forehead_r=(0.060, 0.050, 0.055),
    jaw_c=(0.0, -0.022, -0.056), jaw_r=(0.061, 0.068, 0.058),
    mand_a=(0.056, 0.002, -0.042), mand_b=(0.033, -0.058, -0.097), mand_r=0.013,
    chin_c=(0.0, -0.073, -0.104), chin_r=(0.024, 0.019, 0.019),
    muzzle_c=(0.0, -0.070, -0.060), muzzle_r=(0.037, 0.030, 0.034),
    cheek_c=(0.043, -0.060, -0.030), cheek_r=(0.028, 0.027, 0.032),
    zyg_c=(0.050, -0.060, -0.008), zyg_r=(0.020, 0.018, 0.014),
    brow_a=(0.0, -0.088, 0.024), brow_b=(0.050, -0.071, 0.018), brow_rad=0.011,
    socket_c=(0.032, -0.089, 0.004), socket_r=(0.020, 0.016, 0.013),
    eye_c=(0.032, -0.0735, 0.0), eye_rad=0.0122,
    eye_w=0.0145, eye_up=0.0056, eye_lo=0.0042, eye_tilt=0.0012,
    nose_a=(0.0, -0.091, 0.012), nose_b=(0.0, -0.112, -0.036), nose_r1=0.0075, nose_r2=0.0095,
    nose_tip_c=(0.0, -0.111, -0.041), nose_tip_r=0.0105,
    ala_c=(0.0145, -0.099, -0.045), ala_r=(0.0075, 0.009, 0.0075),
    nostril_c=(0.0075, -0.1035, -0.0515), nostril_r=(0.0035, 0.0055, 0.0025),
    upper_lip_c=(0.0, -0.0985, -0.0645), upper_lip_r=(0.023, 0.009, 0.0058),
    lower_lip_c=(0.0, -0.0955, -0.0755), lower_lip_r=(0.020, 0.0095, 0.0065),
    mouth_z=-0.0702, mouth_w=0.0245,
    ear_c=(0.077, 0.012, -0.012), ear_r=(0.0055, 0.019, 0.031),
    neck_a=(0.0, 0.018, -0.07), neck_b=(0.0, 0.025, -0.23), neck_r1=0.057, neck_r2=0.064,
    adam=True,
)

FEMALE = dict(MALE)
FEMALE.update(
    cranium_r=(0.071, 0.095, 0.092),
    forehead_c=(0.0, -0.044, 0.036), forehead_r=(0.058, 0.050, 0.055),
    jaw_c=(0.0, -0.025, -0.058), jaw_r=(0.052, 0.064, 0.055),
    mand_a=(0.047, 0.0, -0.045), mand_b=(0.024, -0.060, -0.096), mand_r=0.010,
    chin_c=(0.0, -0.072, -0.100), chin_r=(0.019, 0.017, 0.017),
    muzzle_c=(0.0, -0.069, -0.058), muzzle_r=(0.033, 0.029, 0.031),
    cheek_c=(0.038, -0.063, -0.026), cheek_r=(0.026, 0.026, 0.030),
    zyg_c=(0.046, -0.061, -0.006), zyg_r=(0.018, 0.016, 0.013),
    brow_a=(0.0, -0.086, 0.022), brow_b=(0.048, -0.070, 0.017), brow_rad=0.0085,
    socket_r=(0.019, 0.015, 0.013),
    eye_c=(0.031, -0.0735, 0.0), eye_rad=0.0122,
    eye_w=0.0150, eye_up=0.0062, eye_lo=0.0045, eye_tilt=0.0018,
    nose_a=(0.0, -0.089, 0.010), nose_b=(0.0, -0.106, -0.034), nose_r1=0.0062, nose_r2=0.0080,
    nose_tip_c=(0.0, -0.1055, -0.039), nose_tip_r=0.0088,
    ala_c=(0.0125, -0.095, -0.043), ala_r=(0.0065, 0.008, 0.0065),
    nostril_c=(0.0065, -0.0985, -0.0485), nostril_r=(0.0030, 0.0050, 0.0022),
    upper_lip_c=(0.0, -0.0975, -0.0615), upper_lip_r=(0.019, 0.0095, 0.0062),
    lower_lip_c=(0.0, -0.0945, -0.0725), lower_lip_r=(0.0175, 0.0105, 0.0072),
    mouth_z=-0.0672, mouth_w=0.0205,
    ear_c=(0.074, 0.012, -0.012), ear_r=(0.005, 0.017, 0.028),
    neck_a=(0.0, 0.018, -0.07), neck_b=(0.0, 0.022, -0.23), neck_r1=0.045, neck_r2=0.050,
    adam=False,
)

HEADS = {"male": MALE, "female": FEMALE}


def landmarks(kind="male"):
    """贴图绘制所需的五官正面坐标 (x, z)，单位米。"""
    p = HEADS[kind]
    ex, _, ez = p["eye_c"]
    return dict(
        eye=(ex, ez), eye_w=p["eye_w"], eye_up=p["eye_up"], eye_lo=p["eye_lo"], eye_tilt=p["eye_tilt"],
        brow_z=p["brow_a"][2], nose_tip=(0.0, p["nose_tip_c"][2]), nostril=(p["nostril_c"][0], p["nostril_c"][2]),
        ala_x=p["ala_c"][0] + p["ala_r"][0], mouth_z=p["mouth_z"], mouth_w=p["mouth_w"],
        upper_lip_z=p["upper_lip_c"][2], lower_lip_z=p["lower_lip_c"][2],
        chin_z=p["chin_c"][2] - p["chin_r"][2], face_half=p["cranium_r"][0],
        top_z=p["cranium_c"][2] + p["cranium_r"][2],
    )
