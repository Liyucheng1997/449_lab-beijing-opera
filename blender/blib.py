"""Blender 通用工具：清场、材质、渲染、导出。"""
import math
import os
import bpy
import mathutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "build", "tex")
OUT_MODELS = os.path.join(ROOT, "site", "models")
OUT_THUMBS = os.path.join(ROOT, "site", "thumbs")
PREVIEW = os.path.join(ROOT, "build", "preview")


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for t in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = t
            prefs.get_devices()
            ok = False
            for d in prefs.devices:
                d.use = d.type != "CPU"
                ok |= d.use
            if ok:
                return t
        except Exception:
            pass
    return "CPU"


def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def img(path, colorspace="sRGB"):
    name = os.path.basename(path)
    im = bpy.data.images.get(name)
    if im is None:
        im = bpy.data.images.load(path)
    im.colorspace_settings.name = colorspace
    return im


def pbr(name, color=(0.8, 0.8, 0.8, 1), rough=0.5, metal=0.0, base_tex=None, rough_tex=None,
        normal_tex=None, normal_strength=1.0, alpha_tex=None, sheen=0.0, sss=0.0, coat=0.0,
        emission=None, double=False, alpha_clip=False, extend=False):
    """生成 glTF 兼容的 Principled BSDF 材质（导出器能识别的节点结构）。"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
        bsdf.inputs["Sheen Roughness"].default_value = 0.4
        bsdf.inputs["Sheen Tint"].default_value = tuple(min(1.0, c * 1.6 + 0.05) for c in color[:3]) + (1,)
    if sss:
        bsdf.inputs["Subsurface Weight"].default_value = sss
        bsdf.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
    if coat:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.15
    x = -500
    if base_tex:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(base_tex)
        t.location = (x, 300)
        if extend:
            t.extension = "EXTEND"
        if color != (0.8, 0.8, 0.8, 1) and color is not None and tuple(color[:3]) != (1, 1, 1):
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.inputs[7].default_value = color
            nt.links.new(t.outputs["Color"], mix.inputs[6])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha_tex == "same":
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    if rough_tex:
        # glTF：粗糙度 = G 通道，金属度 = B 通道
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(rough_tex, "Non-Color")
        t.location = (x, 0)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (x + 250, 0)
        nt.links.new(t.outputs["Color"], sep.inputs[0])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if normal_tex:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(normal_tex, "Non-Color")
        t.location = (x, -300)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nm.location = (x + 250, -300)
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if emission:
        bsdf.inputs["Emission Color"].default_value = emission
        bsdf.inputs["Emission Strength"].default_value = 1.0
    if alpha_tex:
        m.blend_method = "CLIP" if alpha_clip else "HASHED" if hasattr(m, "blend_method") else None
        try:
            m.surface_render_method = "DITHERED"
        except Exception:
            pass
    m.use_backface_culling = not double
    return m


def look_at(ob, target):
    d = mathutils.Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def studio(scene=None, strength=1.0, floor=False, bg=(0.02, 0.02, 0.022)):
    """三点布光 + 柔和环境，用于缩略图与检查渲染。"""
    scene = scene or bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    bgn.inputs[0].default_value = (*bg, 1)
    bgn.inputs[1].default_value = 1.0
    lights = []

    def area(name, loc, size, energy, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size = size
        ld.energy = energy * strength
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        link(lo)
        lights.append(lo)
        return lo

    return area, lights


def camera(loc, target, lens=85, name="Cam"):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    link(co)
    look_at(co, target)
    bpy.context.scene.camera = co
    return co


def render_setup(res=(900, 900), samples=96, transparent=False):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    dev = enable_gpu()
    sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = transparent
    for vt in ("Khronos PBR Neutral", "Standard"):
        try:
            sc.view_settings.view_transform = vt
            print("view transform:", vt)
            break
        except TypeError:
            pass
    sc.render.image_settings.file_format = "PNG"
    return dev


def render(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def export_glb(path, objects):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_image_format="JPEG",
        export_jpeg_quality=88,
        export_materials="EXPORT",
    )
