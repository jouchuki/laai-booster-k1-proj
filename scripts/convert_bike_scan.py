"""Convert the uploaded textured bike scan into an articulated MuJoCo asset.

Linux and macOS, from the repository root (headless):
    uv run python scripts/convert_bike_scan.py

The source files remain unchanged. Alignment and spatial cuts are specific to
this inspected scan; they are approximations of mechanical part boundaries.
"""

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/bike_scan/source/real_bike.obj"
SOURCE_HASH = "27e141881121c985eb9fae675a54c84a53008e722f8bbbe0de93ca475cc2d741"
TEXTURE = "textures/base_color.png"


def numbers(values):
    return " ".join(f"{float(value):.9g}" for value in np.atleast_1d(values))


def read_obj(source):
    """Read the inspected export, retaining per-corner texture coordinates."""
    vertices, texcoords, faces, uv_faces = [], [], [], []
    for line in source.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "vt":
            texcoords.append([float(x) for x in fields[1:3]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Expected the inspected triangulated OBJ")
            corners = [field.split("/") for field in fields[1:]]
            faces.append([int(corner[0]) - 1 for corner in corners])
            uv_faces.append([int(corner[1]) - 1 for corner in corners])
    return tuple(np.asarray(array) for array in (vertices, texcoords, faces, uv_faces))


def align(vertices, wheelbase, radius):
    """Fit the two tyre annuli, then level the axle centres and lateral axis."""
    centres, normals = [], []
    # Manually identified regions in this source's oblique coordinate system.
    for guess in ([0.223, 0.538], [-0.013, 0.105]):
        distance = np.linalg.norm(vertices[:, :2] - guess, axis=1)
        annulus = vertices[(distance > 0.085) & (distance < 0.116)]
        centre = annulus.mean(0)
        _, _, axes = np.linalg.svd(annulus - centre, full_matrices=False)
        normal = axes[-1] * np.sign(axes[-1, 2])
        projected = (annulus - centre) @ axes[:2].T
        system = np.column_stack((2 * projected, np.ones(len(projected))))
        solution = np.linalg.lstsq(system, (projected**2).sum(1), rcond=None)[0]
        centres.append(centre + solution[:2] @ axes[:2])
        normals.append(normal)
    rear, front = centres
    forward = (front - rear) / np.linalg.norm(front - rear)
    lateral = np.mean(normals, axis=0)
    lateral -= forward * (forward @ lateral)
    lateral /= np.linalg.norm(lateral)
    rotation = np.array([forward, lateral, np.cross(forward, lateral)])
    scale = wheelbase / np.linalg.norm(front - rear)
    points = (vertices - rear) @ rotation.T * scale + [0, 0, radius]
    return points, rotation, scale, centres


def assign_faces(points, faces, wheelbase, radius, scale):
    """Cut fused scan triangles by inspected wheel and headset regions."""
    centres = points[faces].mean(1)
    # The fitted annuli and the following cuts use the default 0.565 m scale.
    reference = centres / scale
    reference[:, 2] += 0.125 - radius / scale
    labels = np.full(len(faces), "frame", dtype="U11")
    # Keep the complete outer head tube on the frame. Cut only at its lower
    # bearing and above the upper clamp, rather than slicing along its axis.
    fork = ((reference[:, 2] < 0.31) & (reference[:, 0] > 0.47)) | (
        (reference[:, 2] > 0.45) & (reference[:, 0] > 0.30)
    )
    labels[fork] = "fork"
    for name, x in (("rear_wheel", 0), ("front_wheel", wheelbase / scale)):
        radial = np.linalg.norm(reference[:, [0, 2]] - [x, 0.125], axis=1)
        labels[(radial < 0.13) & (np.abs(reference[:, 1]) < 0.037)] = name
    return labels


def write_visual(path, points, texcoords, faces, uv_faces):
    used, indices = np.unique(faces, return_inverse=True)
    used_uv, uv_indices = np.unique(uv_faces, return_inverse=True)
    with path.open("w") as stream:
        stream.write("# Derived from the project owner's uploaded bike scan.\n")
        np.savetxt(stream, points[used], fmt="v %.9g %.9g %.9g")
        np.savetxt(stream, texcoords[used_uv], fmt="vt %.9g %.9g")
        for face, uv in zip(indices.reshape(-1, 3) + 1, uv_indices.reshape(-1, 3) + 1):
            stream.write("f " + " ".join(f"{v}/{t}" for v, t in zip(face, uv)) + "\n")


def write_tyre(path, radius, half_width):
    """Convex contact hull with a rounded tyre section, as on the green bike."""
    around, section = np.meshgrid(
        np.linspace(0, 2 * np.pi, 96, endpoint=False),
        np.linspace(0, 2 * np.pi, 16, endpoint=False),
        indexing="ij",
    )
    radial = radius - half_width + half_width * np.cos(section)
    vertices = np.stack(
        (radial * np.cos(around), half_width * np.sin(section), radial * np.sin(around)), -1
    ).reshape(-1, 3)
    with path.open("w") as stream:
        np.savetxt(stream, vertices, fmt="v %.9g %.9g %.9g")
        for i in range(96):
            for j in range(16):
                a, b = i * 16 + j, ((i + 1) % 96) * 16 + j
                c, d = ((i + 1) % 96) * 16 + (j + 1) % 16, i * 16 + (j + 1) % 16
                stream.write(f"f {a + 1} {c + 1} {b + 1}\nf {a + 1} {d + 1} {c + 1}\n")


def convert(source, output, wheelbase=0.565, mass=1.5):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError("Source differs from the inspected bike scan; inspect its geometry first")
    texture = source.parent / TEXTURE
    if not texture.is_file():
        raise ValueError(f"Missing source colour texture: {texture}")
    vertices, texcoords, faces, uv_faces = read_obj(source)
    dimension_scale = wheelbase / 0.565
    radius, half_width = 0.125 * dimension_scale, 0.028 * dimension_scale
    points, rotation, scale, centres = align(vertices, wheelbase, radius)
    labels = assign_faces(points, faces, wheelbase, radius, dimension_scale)
    lower = np.array([0.485, 0, 0.31]) * dimension_scale
    upper = np.array([0.445, 0, 0.43]) * dimension_scale
    axis = (upper - lower) / np.linalg.norm(upper - lower)
    grips = []
    for side in (1, -1):
        region = points[
            (points[:, 2] > 0.55 * dimension_scale) & (side * points[:, 1] > 0.13 * dimension_scale)
        ]
        grips.append((region.min(0) + region.max(0)) / 2)
    seat_points = points[
        (points[:, 0] < 0.21 * dimension_scale) & (points[:, 2] > 0.45 * dimension_scale)
    ]
    seat_lo, seat_hi = seat_points.min(0), seat_points.max(0)
    seat_centre = (seat_lo + seat_hi) / 2
    seat_centre[1] = 0
    origins = {
        "frame": np.zeros(3),
        "fork": lower,
        "rear_wheel": np.array([0, 0, radius]),
        "front_wheel": np.array([wheelbase, 0, radius]),
    }
    output.mkdir(parents=True, exist_ok=True)
    meshes = output / "meshes"
    meshes.mkdir(exist_ok=True)
    root = ET.Element("mujoco", model="real_bike")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir="meshes")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="0.8 0.005 0.0001")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="scan_color",
        type="2d",
        file=Path(os.path.relpath(texture, output)).as_posix(),
    )
    ET.SubElement(
        asset,
        "material",
        name="scan_material",
        texture="scan_color",
        specular="0.1",
        shininess="0.1",
    )
    world = ET.SubElement(root, "worldbody")
    frame = ET.SubElement(world, "body", name="frame")
    ET.SubElement(frame, "freejoint", name="root")
    fork = ET.SubElement(frame, "body", name="fork", pos=numbers(lower))
    ET.SubElement(
        fork,
        "joint",
        name="steer",
        type="hinge",
        axis=numbers(axis),
        range=numbers([-np.pi / 4, np.pi / 4]),
        damping="0.05",
        armature="0.001",
    )
    bodies = {"frame": frame, "fork": fork}
    for name in ("rear_wheel", "front_wheel"):
        parent = frame if name == "rear_wheel" else fork
        offset = origins[name] - (np.zeros(3) if name == "rear_wheel" else lower)
        bodies[name] = ET.SubElement(parent, "body", name=name, pos=numbers(offset))
        ET.SubElement(bodies[name], "joint", name=name, type="hinge", axis="0 1 0", damping="0.002")
    for name, body in bodies.items():
        mask = labels == name
        if not mask.any():
            raise ValueError(f"Spatial cut produced no triangles for {name}")
        visual = f"{name}_visual"
        write_visual(
            meshes / f"{visual}.obj", points - origins[name], texcoords, faces[mask], uv_faces[mask]
        )
        ET.SubElement(asset, "mesh", name=visual, file=f"{visual}.obj")
        ET.SubElement(
            body,
            "geom",
            name=visual,
            type="mesh",
            mesh=visual,
            material="scan_material",
            mass="0",
            contype="0",
            conaffinity="0",
            group="2",
        )
    write_tyre(meshes / "tyre_collision.obj", radius, half_width)
    ET.SubElement(asset, "mesh", name="tyre_collision", file="tyre_collision.obj")
    for name in ("rear_wheel", "front_wheel"):
        wheel_mass = mass * 0.6 / 3.7
        major = radius - half_width
        axial = wheel_mass * (major**2 + 0.75 * half_width**2)
        transverse = wheel_mass * (0.5 * major**2 + 0.625 * half_width**2)
        ET.SubElement(
            bodies[name],
            "inertial",
            pos="0 0 0",
            mass=numbers(wheel_mass),
            diaginertia=numbers([transverse, axial, transverse]),
        )
        ET.SubElement(
            bodies[name],
            "geom",
            name=name,
            type="mesh",
            mesh="tyre_collision",
            mass="0",
            group="3",
            rgba="0.8 0.2 0.1 0.4",
            friction="1 0.005 0.0001",
        )

    def capsule(body, name, start, end, width, geom_mass):
        origin = origins[body]
        ET.SubElement(
            bodies[body],
            "geom",
            name=name,
            type="capsule",
            size=numbers(width),
            fromto=numbers([*(start - origin), *(end - origin)]),
            mass=numbers(geom_mass),
            group="3",
            rgba="0.2 0.4 0.8 0.4",
        )

    frame_mass, fork_mass = mass * 1.8 / 3.7, mass * 0.7 / 3.7
    seat_base = np.array([0.16, 0, 0.16]) * dimension_scale
    bend = np.array([0.36, 0, 0.15]) * dimension_scale
    for side in (-1, 1):
        offset = np.array([0, side * (half_width + 0.02 * dimension_scale), 0])
        capsule(
            "frame",
            f"rear_stay_{side}",
            origins["rear_wheel"] + offset,
            seat_base + offset,
            0.012 * dimension_scale,
            frame_mass * 0.15,
        )
        capsule(
            "fork",
            f"fork_leg_{side}",
            lower + offset,
            origins["front_wheel"] + offset,
            0.012 * dimension_scale,
            fork_mass * 0.2,
        )
    capsule("frame", "main_tube_rear", seat_base, bend, 0.023 * dimension_scale, frame_mass * 0.175)
    capsule("frame", "main_tube_front", bend, lower, 0.023 * dimension_scale, frame_mass * 0.175)
    capsule(
        "frame", "seat_post", seat_base, seat_centre, 0.012 * dimension_scale, frame_mass * 0.15
    )
    capsule("frame", "head_tube", lower, upper, 0.025 * dimension_scale, 0)
    ET.SubElement(
        frame,
        "geom",
        name="saddle",
        type="box",
        pos=numbers(seat_centre),
        size=numbers((seat_hi - seat_lo) / 2),
        mass=numbers(frame_mass * 0.2),
        group="3",
    )
    left, right = grips
    capsule("fork", "stem", lower, (left + right) / 2, 0.012 * dimension_scale, fork_mass * 0.2)
    capsule("fork", "handlebar", right, left, 0.016 * dimension_scale, fork_mass * 0.4)
    seat = seat_centre.copy()
    seat[2] = seat_hi[2]
    for name, point, body in (
        ("seat", seat, "frame"),
        ("left_grip", left, "fork"),
        ("right_grip", right, "fork"),
    ):
        ET.SubElement(
            bodies[body],
            "site",
            name=name,
            pos=numbers(point - origins[body]),
            size="0.005",
            rgba="1 0.5 0 1",
            group="4",
        )
    ET.indent(root)
    ET.ElementTree(root).write(output / "bike.xml", encoding="unicode", xml_declaration=True)
    report = {
        "source": Path(os.path.relpath(source, output)).as_posix(),
        "sha256": digest,
        "color_texture_sha256": hashlib.sha256(texture.read_bytes()).hexdigest(),
        "source_axle_centres": [centre.tolist() for centre in centres],
        "source_to_bike_rotation": rotation.tolist(),
        "uniform_scale": scale,
        "wheelbase_m": wheelbase,
        "collision_tyre_radius_m": radius,
        "collision_tyre_width_m": 2 * half_width,
        "saddle_top_m": float(seat[2]),
        "grip_centres_m": [point.tolist() for point in grips],
        "steering_axis": axis.tolist(),
        "estimated_total_mass_kg": mass,
        "source_triangle_count": len(faces),
        "triangles_per_body": {name: int((labels == name).sum()) for name in bodies},
        "physics_assumptions": [
            "Uniform scale uses the approximate measured 0.565 m wheelbase by default.",
            "Scan saddle and handlebar heights are preserved, not fitted to measurements.",
            "Fused scan is cut spatially; seams and misassigned boundary triangles are possible.",
            "Steering axis and collision primitives are manually approximated from the scan.",
            "Default tyre collider radius is the approximate measured 0.125 m; width is inferred.",
            "Default 1.5 kg mass is provisional; component ratios follow the original prototype.",
            "Wheel inertias assume a uniform torus; contact uses a rounded convex hull.",
            "Friction, damping and steering limits are unmeasured; no clips or controller.",
            "Only the colour texture is used; source AO and normal maps remain in the upload.",
        ],
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "models/bike_scan")
    parser.add_argument("--wheelbase", type=float, default=0.565, help="wheel-centre distance (m)")
    parser.add_argument("--mass", type=float, default=1.5, help="provisional total mass (kg)")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.wheelbase, args.mass)):
        parser.error("--wheelbase and --mass must be positive finite numbers")
    convert(args.source.resolve(), args.output.resolve(), args.wheelbase, args.mass)


if __name__ == "__main__":
    main()
