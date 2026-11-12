"""Prepare this project's Sketchfab bike for MuJoCo with an adjustable stem.

Linux and macOS, from the repository root (headless; no mjpython required):
    uv run --with usd-core python scripts/convert_bike_usdz.py

The source USDZ remains unchanged. usd-core is needed only for conversion.
Mesh separation and bearing identification below are specific to this asset,
not a general USD-to-robot converter. See the generated measurements.json.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = "093e92bfed8b3b044b388b08cf42c226960d785a664b8291b3a60219f076096e"
DEFAULT_WHEELBASE = 0.565
DEFAULT_MASS = 1.5  # Provisional point within the owner's unweighed 1–2 kg estimate.
DEFAULT_HANDLEBAR_HEIGHT = 0.45


@dataclass
class Part:
    material: str
    component: int
    points: np.ndarray
    faces: np.ndarray
    color: np.ndarray

    @property
    def centre(self):
        return (self.points.min(0) + self.points.max(0)) / 2


def connected_parts(points, faces):
    """Join coincident seam vertices for connectivity, without changing geometry."""
    parents = list(range(len(points)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[b] = a

    seen = {}
    for index, point in enumerate(np.round(points, 7)):
        key = tuple(point)
        if key in seen:
            join(index, seen[key])
        else:
            seen[key] = index
    for face in faces:
        join(int(face[0]), int(face[1]))
        join(int(face[0]), int(face[2]))
    groups = defaultdict(list)
    for face in faces:
        groups[find(int(face[0]))].append(face)
    result = []
    for group in groups.values():
        vertices, inverse = np.unique(np.asarray(group), return_inverse=True)
        result.append((points[vertices], inverse.reshape(-1, 3)))
    return sorted(result, key=lambda item: -len(item[0]))


def read_parts(source):
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise ValueError(f"Cannot open {source}")
    if UsdGeom.GetStageUpAxis(stage) != "Y":
        raise ValueError("This asset's conversion expects its authored Y-up frame")
    cache = UsdGeom.XformCache()
    parts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = (np.column_stack((raw, np.ones(len(raw)))) @ matrix)[:, :3]
        points *= UsdGeom.GetStageMetersPerUnit(stage)
        points = points[:, [0, 2, 1]] * [1, -1, 1]  # Y-up -> right-handed Z-up.
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        triangles, offset = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            triangles.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
            offset += count
        triangles = np.asarray(triangles, dtype=int)
        if np.linalg.det(matrix[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        # This older export has direct bindings without the API schema applied.
        # Read the relationship without authoring anything into the source stage.
        binding = prim.GetRelationship("material:binding").GetTargets()
        material = UsdShade.Material(stage.GetPrimAtPath(binding[0]))
        shader = material.ComputeSurfaceSource()[0]
        color = np.asarray(shader.GetInput("diffuseColor").Get(), dtype=float)
        # USD Preview Surface colours are linear; retain their appearance in MJCF.
        color = np.where(color <= 0.0031308, 12.92 * color, 1.055 * color ** (1 / 2.4) - 0.055)
        name = prim.GetName().removeprefix("Balance_Bike_Dyno_v22_073_")
        for component, (vertices, faces) in enumerate(connected_parts(points, triangles)):
            parts.append(Part(name, component, vertices, faces, color))
    return parts


def assign_body(part, rear, front, radius):
    """Assign this asset's components to four rigid bodies; inspect after conversion."""
    nearest = (
        "front_wheel"
        if abs(part.centre[0] - front[0]) < abs(part.centre[0] - rear[0])
        else "rear_wheel"
    )
    wheel_centre = front if nearest == "front_wheel" else rear
    radial_extent = np.linalg.norm((part.points - wheel_centre)[:, [0, 2]], axis=1).max()
    if part.material.startswith("Nylon_12"):
        return nearest
    if part.material.startswith("Aluminum") and part.points[:, 2].max() < 1.2:
        return nearest  # Rims and rotating hubs; handlebar is above this band.
    if (
        part.material.startswith("Steel___Satin")
        and part.points[:, 2].max() < 1.1
        and radial_extent > 0.35 * radius
    ):
        return nearest  # Spokes. Axle end fasteners remain on frame/fork.
    if part.material.startswith("Plastic___Translucent"):
        return "frame"
    return "fork" if part.centre[0] > (rear[0] + front[0]) / 2 else "frame"


def write_obj(path, points, faces):
    with path.open("w") as stream:
        stream.write("# Derived from Balance bike by maulanarizal752; CC BY 4.0.\n")
        np.savetxt(stream, points, fmt="v %.9g %.9g %.9g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")


def rounded_tyre(radius, half_width):
    """Ideal circular tyre profile; the convex hull fills the unused hub opening."""
    around, section = np.meshgrid(
        np.linspace(0, 2 * np.pi, 96, endpoint=False),
        np.linspace(0, 2 * np.pi, 16, endpoint=False),
        indexing="ij",
    )
    radial = radius - half_width + half_width * np.cos(section)
    points = np.stack(
        (radial * np.cos(around), half_width * np.sin(section), radial * np.sin(around)), -1
    ).reshape(-1, 3)
    faces = []
    for i in range(96):
        for j in range(16):
            a, b = i * 16 + j, ((i + 1) % 96) * 16 + j
            c, d = ((i + 1) % 96) * 16 + (j + 1) % 16, i * 16 + (j + 1) % 16
            faces.extend(((a, c, b), (a, d, c)))
    return points, np.asarray(faces)


def numbers(values):
    return " ".join(f"{float(value):.9g}" for value in np.atleast_1d(values))


def convert(
    source,
    output,
    wheelbase=DEFAULT_WHEELBASE,
    mass=DEFAULT_MASS,
    handlebar_height=DEFAULT_HANDLEBAR_HEIGHT,
):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError(
            "Source differs from the inspected Sketchfab asset; inspect its parts first"
        )
    parts = read_parts(source)
    tyres = sorted(
        (p for p in parts if p.material.startswith("Nylon_12")), key=lambda p: p.centre[0]
    )
    if len(tyres) != 2:
        raise ValueError("Expected two separate tyre components")
    rear, front = (part.centre for part in tyres)
    delta = front - rear
    angle = math.atan2(delta[2], delta[0])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    scale = wheelbase / np.linalg.norm(delta[[0, 2]])
    raw_radius = max(
        np.linalg.norm((part.points - part.centre)[:, [0, 2]], axis=1).max() for part in tyres
    )
    radius = raw_radius * scale

    def transform(points):
        return (points - rear) @ rotation.T * scale + [0, 0, radius]

    # The two bearing rings identify a steering axis independently of the wheels.
    bearings = {p.component: p for p in parts if p.material.startswith("Steel___Satin")}
    lower = transform(bearings[7].centre)
    upper = transform(bearings[8].centre)
    axis = (upper - lower) / np.linalg.norm(upper - lower)
    grips = sorted(
        (transform(p.centre) for p in parts if p.material.startswith("Opaque_76_76_76")),
        key=lambda p: p[1],
    )
    source_grip_height = float(np.mean(grips, axis=0)[2])
    rise = 0.0 if handlebar_height is None else handlebar_height - source_grip_height
    if rise < -1e-9:
        raise ValueError("--handlebar-height must not lower the source handlebars")
    extension = axis * (max(0.0, rise) / axis[2])

    def extend_stem(points):
        # This asset's upper headset ends 9.8 mm above the upper bearing centre;
        # rigid handlebar/clamp geometry begins at 10.7 mm. Lengthen the short
        # stem band between them, leaving the headset and lower fork untouched.
        axial = (points - upper) @ axis
        weight = np.clip((axial - 0.0099) / 0.0007, 0, 1)
        return points + weight[..., None] * extension

    grips = [point + extension for point in grips]
    origins = {
        "frame": np.zeros(3),
        "fork": lower,
        "rear_wheel": transform(rear),
        "front_wheel": transform(front),
    }
    groups = defaultdict(list)
    materials = {}
    assignments = []
    for part in parts:
        body = assign_body(part, rear, front, raw_radius)
        points = transform(part.points)
        if body == "fork":
            points = extend_stem(points)
        groups[body, part.material].append((points - origins[body], part.faces))
        materials[part.material] = part.color
        assignments.append({"material": part.material, "component": part.component, "body": body})

    output.mkdir(parents=True, exist_ok=True)
    meshes = output / "meshes"
    meshes.mkdir(exist_ok=True)
    root = ET.Element("mujoco", model="sketchfab_balance_bike")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir="meshes")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="0.8 0.005 0.0001")
    asset = ET.SubElement(root, "asset")
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
        parent_origin = np.zeros(3) if name == "rear_wheel" else lower
        body = ET.SubElement(parent, "body", name=name, pos=numbers(origins[name] - parent_origin))
        ET.SubElement(body, "joint", name=name, type="hinge", axis="0 1 0", damping="0.002")
        bodies[name] = body

    for index, ((body, material), group) in enumerate(sorted(groups.items())):
        vertices, triangles, offset = [], [], 0
        for points, faces in group:
            vertices.append(points)
            triangles.append(faces + offset)
            offset += len(points)
        name = f"{body}_visual_{index:02d}"
        write_obj(meshes / f"{name}.obj", np.concatenate(vertices), np.concatenate(triangles))
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.obj")
        ET.SubElement(
            bodies[body],
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba=numbers([*materials[material], 1]),
            mass="0",
            contype="0",
            conaffinity="0",
            group="2",
        )

    tyre_width = float(np.ptp(transform(tyres[0].points)[:, 1]))
    half_width = tyre_width / 2
    write_obj(meshes / "tyre_collision.obj", *rounded_tyre(radius, half_width))
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

    seat_points = transform(
        next(p.points for p in parts if p.material.startswith("Leather___Matte__Black"))
    )
    seat_lo, seat_hi = seat_points.min(0), seat_points.max(0)
    seat_centre = (seat_lo + seat_hi) / 2
    right, left = grips
    seat_base = np.array([seat_centre[0], 0, radius * 1.1])

    def capsule(body_name, name, start, end, size, geom_mass):
        origin = origins[body_name]
        ET.SubElement(
            bodies[body_name],
            "geom",
            name=name,
            type="capsule",
            size=numbers(size),
            fromto=numbers([*(start - origin), *(end - origin)]),
            mass=numbers(geom_mass),
            group="3",
            rgba="0.2 0.4 0.8 0.4",
        )

    frame_mass, fork_mass = mass * 1.8 / 3.7, mass * 0.7 / 3.7
    for side in (-1, 1):
        offset = np.array([0, side * (half_width + 0.012), 0])
        capsule(
            "frame",
            f"rear_stay_{side}",
            origins["rear_wheel"] + offset,
            seat_base + offset,
            0.012,
            frame_mass * 0.15,
        )
        capsule(
            "fork",
            f"fork_leg_{side}",
            lower + offset,
            origins["front_wheel"] + offset,
            0.009,
            fork_mass * 0.2,
        )
    capsule("frame", "main_tube", seat_base, lower, 0.024, frame_mass * 0.35)
    capsule("frame", "seat_post", seat_base, seat_centre, 0.012, frame_mass * 0.15)
    capsule("frame", "head_tube", lower, upper, 0.018, 0)
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
    capsule("fork", "stem", lower, (left + right) / 2, 0.012, fork_mass * 0.2)
    capsule("fork", "handlebar", right, left, 0.012, fork_mass * 0.4)
    seat = np.array([seat_centre[0], 0, seat_hi[2]])
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
        "source": source.name,
        "sha256": digest,
        "wheelbase_reference_m": wheelbase,
        "uniform_scale_after_usd_units": scale,
        "source_pitch_removed_rad": angle,
        "wheel_radius_m": radius,
        "tyre_diameter_m": 2 * radius,
        "tyre_width_m": tyre_width,
        "saddle_top_m": float(seat_hi[2]),
        "source_grip_centre_height_m": source_grip_height,
        "handlebar_target_height_m": handlebar_height,
        "handlebar_extension_vector_m": extension.tolist(),
        "stem_extension_m": float(np.linalg.norm(extension)),
        "grip_centre_height_m": float((left[2] + right[2]) / 2),
        "grip_centre_spacing_m": float(np.linalg.norm(left - right)),
        "steering_axis": axis.tolist(),
        "estimated_total_mass_kg": mass,
        "physics_assumptions": [
            "Mass is a provisional point within the owner's unweighed 1-2 kg estimate.",
            "Component mass fractions retain the placeholder's frame/fork/wheel ratios.",
            "Wheel inertias assume a uniform torus; the rounded convex tyre is an idealization.",
            "Frame/fork collisions are compound primitives approximating the visual geometry.",
            "Steering axis is inferred from mesh bearing rings; limits and damping are unmeasured.",
            "Friction and compliance are untuned; no robot, clips or stabilizer is included.",
        ],
        "part_assignments": assignments,
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("part_assignments", "physics_assumptions")
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "models/bike/source/Balance_bike.usdz"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models/bike")
    parser.add_argument(
        "--wheelbase", type=float, default=DEFAULT_WHEELBASE, help="wheel-centre distance in metres"
    )
    parser.add_argument(
        "--mass", type=float, default=DEFAULT_MASS, help="provisional total mass in kg"
    )
    parser.add_argument(
        "--handlebar-height",
        type=float,
        default=DEFAULT_HANDLEBAR_HEIGHT,
        help="grip-centre height in metres; extend the stem along the steering axis",
    )
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.wheelbase, args.mass)):
        parser.error("--wheelbase and --mass must be positive finite numbers")
    if args.handlebar_height is not None and (
        not math.isfinite(args.handlebar_height) or args.handlebar_height <= 0
    ):
        parser.error("--handlebar-height must be a positive finite number")
    convert(args.source, args.output, args.wheelbase, args.mass, args.handlebar_height)


if __name__ == "__main__":
    main()
