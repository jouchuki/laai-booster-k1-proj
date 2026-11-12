"""Saddle collision that follows the visible saddle, so the K1 visibly sits on it.

The converted bikes give the saddle one flat collision box at the height of its highest
point. A real saddle is dished: on the Puky the visible surface is 16 mm lower in the middle,
where the rider sits, than at its back and front. Sitting on the flat box left the K1's pelvis
hovering that far above the visible saddle while pushing on the invisible box.

`fit_saddle_to_visual` casts rays down onto the bike's visible meshes over the box's
footprint and replaces the box's collision with a row of thin convex slices whose tops follow
the measured surface (dish included). The box keeps its name, mass and inertia; it just no
longer collides (it moves to geom group 5). Bikes whose saddle has no separate visible mesh (the primitive bike) keep
their box unchanged.

Colliding parts under the saddle that would poke through its dip (the Puky's seat post)
are shortened to end below it. It also adds a site `saddle_dip` at the lowest point of the saddle's middle (the dip a rider
sits in), which the seated pose solver aims the K1's pelvis at.
"""

import mujoco
import numpy as np

SADDLE_GEOM = "saddle"
DIP_SITE = "saddle_dip"
DIP_MIDDLE = 0.6  # the dip is searched in the middle 60% of the saddle's length
SURFACE_PREFIX = "saddle_surface"
SLICES = 12  # along the bike; each ~1.8 cm on the Puky, so the curve is followed closely
GRID = (49, 15)  # rays along x and across y over the box footprint
THICKNESS = 0.02  # each slice reaches 2 cm below the surface
MIN_HITS = 40  # fewer surface points than this: no visible saddle mesh, keep the box
RETIRED_GROUP = 5  # geom group for the box once it no longer collides (hidden by default)
POST_CLEARANCE = 0.005  # collision parts under the saddle end this far below its dip


def saddle_geom_ids(model: mujoco.MjModel, prefix: str = "") -> list[int]:
    """Ids of the colliding geoms that make up the saddle (the slices, or the box)."""

    def is_saddle(name: str) -> bool:
        return name == f"{prefix}{SADDLE_GEOM}" or name.startswith(f"{prefix}{SURFACE_PREFIX}_")

    return [
        g
        for g in range(model.ngeom)
        if is_saddle(model.geom(g).name) and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def _visible_saddle_points(bike: mujoco.MjSpec):
    """Surface points of the visible saddle in the saddle body's frame, plus the box's centre
    and its x (along), y (across) and z (up) axes in that frame."""
    model = bike.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = model.geom(SADDLE_GEOM).id
    body = model.geom_bodyid[box]
    centre, rot, half = data.geom_xpos[box], data.geom_xmat[box].reshape(3, 3), model.geom_size[box]
    # Only the non-colliding meshes on the saddle's body count as "visible saddle".
    visual = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and not (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    groups = np.zeros(6, np.uint8)
    for g in visual:
        groups[model.geom_group[g]] = 1
    points = []
    hit = np.zeros(1, np.int32)
    above = half[2] + 0.1
    for x in np.linspace(-half[0], half[0], GRID[0]):
        for y in np.linspace(-half[1], half[1], GRID[1]):
            start = centre + rot @ np.array([x, y, above])
            dist = mujoco.mj_ray(model, data, start, -rot[:, 2], groups, 1, -1, hit)
            if 0 <= dist <= above + half[2] + 0.03 and hit[0] in visual:
                points.append(start - dist * rot[:, 2])
    body_pos, body_rot = data.xpos[body], data.xmat[body].reshape(3, 3)
    local = (np.array(points) - body_pos) @ body_rot if points else np.zeros((0, 3))
    axes = body_rot.T @ rot
    return local, body_rot.T @ (centre - body_pos), axes[:, 0], axes[:, 1], axes[:, 2]


def _add_dip_site(body, pos) -> None:
    body.add_site(name=DIP_SITE, pos=list(pos), size=[0.006, 0, 0], rgba=[0, 1, 0, 1], group=4)


def _dip(points, centre, along, across, up) -> np.ndarray:
    """Lowest point of the saddle's crown line within the middle of its length."""
    s = (points - centre) @ along
    lateral = (points - centre) @ across
    height = (points - centre) @ up
    lo, hi = s.min(), s.max()
    margin = (1 - DIP_MIDDLE) / 2 * (hi - lo)
    best = None
    for x in np.unique(s.round(5)):
        if not lo + margin <= x <= hi - margin:
            continue
        row = (np.abs(s - x) < 1e-4) & (np.abs(lateral) < 0.012)
        if row.any():
            crown = height[row].max()  # surface height on the centre line at this x
            if best is None or crown < best[1]:
                best = (x, crown)
    return centre + best[0] * along + best[1] * up


def _copy_contact(src, dst) -> None:
    for attr in ("friction", "condim", "priority", "solref", "solimp", "margin", "gap"):
        setattr(dst, attr, getattr(src, attr))
    dst.contype, dst.conaffinity = src.contype, src.conaffinity


def _retire(geom) -> None:
    """Keep a geom's mass and inertia but stop it colliding (and hide it from group 3)."""
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = RETIRED_GROUP


def _trim_parts_under_saddle(bike: mujoco.MjSpec, prefix: str) -> list[str]:
    """Shorten colliding capsules (the seat post) whose top pokes up through the saddle.

    With the flat box, the Puky's seat post ended inside the box; with the saddle surface
    following its dip, the post's rounded top stuck 4 mm out of the dip and the K1 sat on
    it. Such a capsule gets a massless collision copy that ends POST_CLEARANCE below the dip;
    the original keeps its mass and inertia but no longer collides.
    """
    model = bike.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = model.geom(f"{prefix}{SADDLE_GEOM}").id
    body = model.geom_bodyid[box]
    centre, rot, half = data.geom_xpos[box], data.geom_xmat[box].reshape(3, 3), model.geom_size[box]
    dip = data.site_xpos[model.site(f"{prefix}{DIP_SITE}").id]
    limit = (dip - centre) @ rot[:, 2] - POST_CLEARANCE  # highest allowed top, box "up" axis
    body_pos, body_rot = data.xpos[body], data.xmat[body].reshape(3, 3)
    trimmed = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if (
            model.geom_bodyid[g] != body
            or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE
            or not (model.geom_contype[g] or model.geom_conaffinity[g])
            or not name
        ):
            continue
        radius, half_len = model.geom_size[g][:2]
        axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = [data.geom_xpos[g] + half_len * axis, data.geom_xpos[g] - half_len * axis]
        rel = [rot.T @ (e - centre) for e in ends]
        top = int(rel[1][2] > rel[0][2])  # index of the upper end
        upper, lower = rel[top], rel[1 - top]
        inside = abs(upper[0]) < half[0] and abs(upper[1]) < half[1]
        if not inside or upper[2] + radius <= limit:
            continue
        # Slide the upper end down the capsule's axis until its top reaches the limit.
        t = (upper[2] + radius - limit) / (upper[2] - lower[2])
        new_upper = ends[top] + t * (ends[1 - top] - ends[top])
        local = [(p - body_pos) @ body_rot for p in (new_upper, ends[1 - top])]
        original = bike.geom(name)
        copy = original.parent.add_geom(
            name=f"{name}_collision",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.concatenate(local).tolist(),
            size=[radius, 0, 0],
            density=0,
            group=original.group,
            rgba=list(original.rgba),
        )
        _copy_contact(original, copy)
        _retire(original)
        trimmed.append(name)
    return trimmed


def fit_saddle_to_visual(bike: mujoco.MjSpec, prefix: str = "") -> int:
    """Give the bike's saddle a collision surface that follows its visible saddle.

    Call on the bike's own spec before attaching it to a scene. Returns the number of
    slices added (0: no visible saddle mesh, the box is kept).
    """
    box = bike.geom(f"{prefix}{SADDLE_GEOM}")
    points, centre, along, across, up = _visible_saddle_points(bike)
    body = box.parent
    if len(points) < MIN_HITS:
        # No visible saddle mesh: the box is the saddle; its dip is the middle of its top.
        _add_dip_site(body, centre + box.size[2] * up)
        return 0
    _add_dip_site(body, _dip(points, centre, along, across, up))
    s = points @ along
    edges = np.linspace(s.min(), s.max(), SLICES + 1)
    step = edges[1] - edges[0]
    added = 0
    for i in range(SLICES):
        # Include points just past both edges so neighbouring slices overlap without gaps.
        top = points[(s >= edges[i] - 0.25 * step) & (s <= edges[i + 1] + 0.25 * step)]
        if len(top) < 6:
            continue
        verts = np.vstack([top, top - THICKNESS * up])
        name = f"{prefix}{SURFACE_PREFIX}_{added}"
        bike.add_mesh(name=name, uservert=verts.ravel().tolist())
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            density=0,
            group=box.group,
            rgba=[0.2, 0.4, 0.8, 0.4],
        )
        _copy_contact(box, geom)
        added += 1
    _retire(box)  # out of the collision group (3), so the viewer shows the slices
    _trim_parts_under_saddle(bike, prefix)
    return added
