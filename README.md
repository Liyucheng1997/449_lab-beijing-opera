# 京剧 · 脸谱与戏服 3D

用 Blender（Python 脚本）程序化建模京剧脸谱、盔头、髯口与戏服，在网页里用 Three.js 实时展示：可旋转、缩放、切换灯光、摘下盔头/髯口，还能给戏服“换脸”。

## 内容

- **脸谱（8）**：关羽红整脸、包拯黑整脸、曹操水白脸、张飞十字门蝴蝶脸、窦尔敦蓝花三块瓦、孙悟空猴脸、项羽寿字眉哭脸、杨贵妃旦角俊扮
- **戏服（5）**：关羽绿蟒、曹操红蟒、项羽黑靠（带靠旗）、杨贵妃黄女蟒（云肩、长裙）、孙悟空猴衣（虎皮裙、金箍棒）
- 盔头：夫子盔、相貂（方翅/圆翅）、扎巾、霸王盔、紫金冠 + 翎子 + 狐尾、凤冠（点翠、珠帘、挑子）
- 髯口：五髯、三髯、满髯、扎髯（黑 / 红）

## 运行网站

```bash
python -m http.server 8449 --directory site
```

然后打开 http://localhost:8449 。链接可直达：`#mask/baozheng`、`#costume/xiangyu`、`#costume/guanyu/zhangfei`（给绿蟒换上张飞的脸）。

## 重建模型

需要 Blender 5.0 与 Python（Pillow、numpy、scipy）。

```bash
powershell -ExecutionPolicy Bypass -File build_all.ps1
```

## 目录

| 路径 | 作用 |
| --- | --- |
| `blender/face_params.py` | 头部比例与五官定位（男 / 女），贴图与建模共用 |
| `blender/headgen.py`、`sdf.py` | SDF 组合五官 → Surface Nets 网格 → 平滑、减面、正面投影 UV |
| `blender/heads.py` | 眼球、水纱、髯口发丝卡片、各式盔头 |
| `blender/costumes.py` | 蟒 / 女蟒 / 大靠 / 猴衣：放样躯干（带垂褶）、袖子与水袖、玉带、靠旗、厚底靴、手 |
| `textures/masks.py` | 八张脸谱贴图（颜色、粗糙度/金属度、法线） |
| `textures/fabrics.py` | 刺绣：团龙、如意云、海水江崖、团花牡丹、鱼鳞甲、兽面、靠旗 |
| `site/` | 网页（Three.js r170，GLB 使用 Draco 压缩） |

纹样为风格化再现，细节以舞台实物为准。
