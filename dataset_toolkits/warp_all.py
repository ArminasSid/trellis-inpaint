import argparse
import random
import sys
from pathlib import Path

import bmesh
import bpy
import mathutils

random.seed()

REMOVE_RATIO = 0.40


def parse_args():
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]

    parser = argparse.ArgumentParser(
        description="Apply bounding-box mesh damage to GLB objects."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help=f"Base dataset directory",
    )
    return parser.parse_args(script_args)


def compute_global_bbox(meshes):
    corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(corner))

    min_corner = mathutils.Vector((
        min(v.x for v in corners),
        min(v.y for v in corners),
        min(v.z for v in corners),
    ))

    max_corner = mathutils.Vector((
        max(v.x for v in corners),
        max(v.y for v in corners),
        max(v.z for v in corners),
    ))

    return min_corner, max_corner


def apply_damage(input_path, output_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_path))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        print(f"No mesh objects found in {input_path}")
        return

    # Apply transforms
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        obj.select_set(False)

    # Compute global bounding volume
    min_corner, max_corner = compute_global_bbox(meshes)
    size = max_corner - min_corner
    center = (min_corner + max_corner) / 2

    # Compute cube size
    volume_scale = REMOVE_RATIO ** (1 / 3)
    cube_size = size * volume_scale

    offset = mathutils.Vector((
        random.uniform(-size.x / 4, size.x / 4),
        random.uniform(-size.y / 4, size.y / 4),
        random.uniform(-size.z / 4, size.z / 4),
    ))

    cube_center = center + offset
    min_region = cube_center - cube_size / 2
    max_region = cube_center + cube_size / 2

    # Remove geometry inside cube
    for obj in meshes:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        verts_to_remove = set()

        for v in bm.verts:
            world_co = obj.matrix_world @ v.co
            if (
                min_region.x <= world_co.x <= max_region.x
                and min_region.y <= world_co.y <= max_region.y
                and min_region.z <= world_co.z <= max_region.z
            ):
                verts_to_remove.add(v)

        if verts_to_remove:
            faces_to_remove = set()
            for v in verts_to_remove:
                faces_to_remove.update(v.link_faces)

            bmesh.ops.delete(bm, geom=list(faces_to_remove), context="FACES")

        bm.to_mesh(mesh)
        bm.free()

    print(f"Trying to make dir {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_apply=True,
    )

    print(f"Processed: {input_path} -> {output_path}")


def main():
    args = parse_args()
    input_root = args.output_dir / "raw" / "objects"
    output_root = args.output_dir / "raw" / "objects_dmg"

    glb_files = list(input_root.rglob("*.glb"))
    glb_files.sort()

    for input_path in glb_files:
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        apply_damage(input_path, output_path)


if __name__ == "__main__":
    main()