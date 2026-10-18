#!/usr/bin/env python3
"""
phy2clip.py - Turn Source .phy collision data into idTech3 clip brushes.

------------------------------------------------------------------------------
WHY THIS WORKS
------------------------------------------------------------------------------
A Source .phy file is NOT a single hull - it is VPhysics' authored CONVEX
DECOMPOSITION of the prop: a set of convex pieces ("ledges"), which is
exactly how Source represents concave collision. That means the hard part
of "complex collision" was already solved by the original artists, and we
just have to read it.

idTech3 brushes are themselves convex polyhedra defined by planes, so each
ledge maps to exactly one brush - no hull algorithm, no approximation, no
convex decomposition of our own. Better still, each ledge's triangles ARE
its hull faces, so the brush planes come straight out of the file; we only
need to de-duplicate triangles that share a plane (e.g. the two triangles
of a box face collapse into one brush side).

Coordinate conversion was derived empirically by comparing a real .phy's
extracted bounds against the same model's .mdl hull_min/hull_max:

    source_units = (phy.x, phy.z, -phy.y) * 39.37

(39.37 = 1/0.0254: VPhysics stores metres, Source uses inches/units. The
axis swap is VPhysics' Y-up convention vs Source's Z-up.)

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 phy2clip.py <vmf> <phy_dir> <output.map fragment> [options]

      --scale S          world scale, must match what you passed vmf2map.py
      --model-scale S    model scale, must match vmf2map.py
      --clip-texture T   shader for the generated brushes
                         (default tools/_clip - playerclip+monsterclip+nodraw)
      --max-planes N     skip any ledge needing more than N planes (default 40)
                         - guards against a dense mesh blowing MAX_MAP_PLANES,
                           which is the exact failure auto-clip produces
      --skip-list FILE   newline-separated model paths to NOT generate clip for
      --skip-ids FILE    vmf2map.py's <map>_dropped_props.json: VMF entity ids of
                         props the prop budget dropped - their collision is left
                         out too, so no invisible prop stays behind

Output is a .map fragment containing only brushes - paste its contents
inside your worldspawn block (before the final closing brace), or use
--merge-into to have it written directly into an existing .map.

    --merge-into <map>   insert the generated brushes into this .map's
                         worldspawn automatically (writes a .bak first)
"""

import argparse
import math
import os
import struct
import sys

PHY_TO_SOURCE_SCALE = 39.37
PLANE_EPSILON = 1e-4


# ------------------------------------------------------------------ .phy ---

def parse_phy(path):
    """Return a list of solids; each solid is a list of ledges; each ledge is
    a list of triangles, each triangle a 3-tuple of (x, y, z) in SOURCE units."""
    with open(path, 'rb') as f:
        d = f.read()
    if len(d) < 16:
        return []
    hsize, _id, solid_count, _checksum = struct.unpack_from('<iiii', d, 0)
    solids = []
    off = hsize
    for _s in range(solid_count):
        if off + 4 > len(d):
            break
        solid_size = struct.unpack_from('<i', d, off)[0]
        if d[off + 4:off + 8] != b'VPHY':
            break
        surf = off + 32  # past compactsurfaceheader_t
        # ivpcompactsurface_t: mass_center[3], rotation_inertia[3],
        # upper_limit_radius, packed flags, offset_ledgetree_root
        mass_center = struct.unpack_from('<3f', d, surf)
        upper_limit_radius = struct.unpack_from('<f', d, surf + 24)[0]
        offset_ledgetree_root = struct.unpack_from('<i', d, surf + 32)[0]
        ledges = []

        # The ledge tree walk can land on bogus offsets in malformed or
        # unusual .phy layouts, producing ledges whose "vertices" are just
        # reinterpreted neighbouring bytes - which show up as enormous
        # clip brushes stretching across the map. The file itself tells us
        # the solid's bounding sphere (upper_limit_radius, in VPhysics
        # units), so anything well outside it cannot be real collision for
        # this prop and is rejected rather than emitted.
        radius_limit = upper_limit_radius * 1.75 if upper_limit_radius > 0 else None

        def read_ledge(lo):
            if lo + 16 > len(d):
                return None
            c_point_offset, _node_offset, _bits, n_tri, _future = \
                struct.unpack_from('<iiIhh', d, lo)
            if n_tri <= 0 or n_tri > 20000:
                return None
            pts_base = lo + c_point_offset
            tris = []
            for t in range(n_tri):
                to = lo + 16 + t * 16
                if to + 16 > len(d):
                    break
                vids = []
                for e in range(3):
                    ev = struct.unpack_from('<I', d, to + 4 + e * 4)[0]
                    vids.append(ev & 0xFFFF)
                tri = []
                ok = True
                for vid in vids:
                    po = pts_base + vid * 16
                    if po + 12 > len(d):
                        ok = False
                        break
                    x, y, z = struct.unpack_from('<3f', d, po)
                    # VPhysics -> Source units/axes (derived empirically, see docstring)
                    tri.append((x * PHY_TO_SOURCE_SCALE,
                                z * PHY_TO_SOURCE_SCALE,
                                -y * PHY_TO_SOURCE_SCALE))
                if ok:
                    if radius_limit is not None:
                        too_far = False
                        for v in tri:
                            # tri is in SOURCE units: (phy_x, phy_z, -phy_y)*S.
                            # Convert back to raw phy axes to compare against
                            # mass_center, which is stored in phy axes.
                            # Getting this mapping wrong silently deletes real
                            # collision: an earlier version used
                            # (v[2]/S - mass_center[1]), i.e. (0.668 - phy_y)
                            # instead of (phy_y - mass_center[1]). Near the
                            # model's base that barely differs, but for
                            # geometry high up it roughly ADDS the two offsets
                            # instead of subtracting, pushing valid vertices
                            # past the radius limit. On a wire-spool prop that
                            # removed every tall ledge and left only two flat
                            # discs.
                            phy_x = v[0] / PHY_TO_SOURCE_SCALE
                            phy_y = -v[2] / PHY_TO_SOURCE_SCALE
                            phy_z = v[1] / PHY_TO_SOURCE_SCALE
                            dx = phy_x - mass_center[0]
                            dy = phy_y - mass_center[1]
                            dz = phy_z - mass_center[2]
                            if (dx * dx + dy * dy + dz * dz) > radius_limit * radius_limit:
                                too_far = True
                                break
                        if too_far:
                            continue
                    tris.append(tri)
            return tris or None

        def walk(node_off, depth=0):
            if depth > 64 or node_off + 8 > len(d):
                return
            offset_right_node, offset_compact_ledge = struct.unpack_from('<ii', d, node_off)
            if offset_right_node == 0:
                lg = read_ledge(node_off + offset_compact_ledge)
                if lg:
                    ledges.append(lg)
            else:
                walk(node_off + 28, depth + 1)                  # left child is inline
                walk(node_off + offset_right_node, depth + 1)

        walk(surf + offset_ledgetree_root)
        if ledges:
            solids.append(ledges)
        off += 4 + solid_size
    return solids


# ------------------------------------------------------------- geometry ---

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize(a):
    l = math.sqrt(dot(a, a))
    return (0.0, 0.0, 0.0) if l < 1e-12 else (a[0] / l, a[1] / l, a[2] / l)


def rotation_matrix(pitch, yaw, roll):
    """Source 'angles' are (pitch, yaw, roll) about (Y, Z, X). Composed as
    Rz(yaw) * Ry(pitch) * Rx(roll), matching Source's own AngleMatrix."""
    p, y, r = map(math.radians, (pitch, yaw, roll))
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp,     cp * sr,                cp * cr),
    )


def transform(v, m, origin, model_scale):
    """v is a collision vertex in SOURCE units, local to the model's origin.

    Scale is applied EXACTLY ONCE here. The prop's world position already has
    --scale baked into it (origin_out below), and the in-engine model is drawn
    at its "modelscale" key value, so the local offset must be multiplied by
    that model scale alone. Multiplying by the world scale again as well is a
    double-application - it produced collision hulls ~1.275x too large on a
    real map, which is exactly the kind of bug that would show up in-game as
    invisible walls standing off the prop."""
    lv = (v[0] * model_scale, v[1] * model_scale, v[2] * model_scale)
    rv = (dot(m[0], lv), dot(m[1], lv), dot(m[2], lv))
    return (origin[0] + rv[0], origin[1] + rv[1], origin[2] + rv[2])


def solve_brush_vertices(planes):
    """Intersect every triple of planes, keeping points that satisfy every
    plane's half-space - the brush's real, solved vertex set. Same role as
    bobToolz's DBrush::BuildPoints()."""
    verts = []
    n = len(planes)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                n1, d1 = planes[i][0], planes[i][1]
                n2, d2 = planes[j][0], planes[j][1]
                n3, d3 = planes[k][0], planes[k][1]
                det = (n1[0] * (n2[1] * n3[2] - n2[2] * n3[1])
                       - n1[1] * (n2[0] * n3[2] - n2[2] * n3[0])
                       + n1[2] * (n2[0] * n3[1] - n2[1] * n3[0]))
                if abs(det) < 1e-9:
                    continue
                x = (d1 * (n2[1] * n3[2] - n2[2] * n3[1]) - n1[1] * (d2 * n3[2] - n2[2] * d3)
                     + n1[2] * (d2 * n3[1] - n2[1] * d3)) / det
                y = (n1[0] * (d2 * n3[2] - n2[2] * d3) - d1 * (n2[0] * n3[2] - n2[2] * n3[0])
                     + n1[2] * (n2[0] * d3 - d2 * n3[0])) / det
                z = (n1[0] * (n2[1] * d3 - d2 * n3[1]) - n1[1] * (n2[0] * d3 - d2 * n3[0])
                     + d1 * (n2[0] * n3[1] - n2[1] * n3[0])) / det
                p = (x, y, z)
                if all(dot(p, pn) <= pd + PLANE_EPSILON * 50 for pn, pd, _t in planes):
                    verts.append(p)
    return verts


def drop_redundant_planes(planes):
    """Port of bobToolz's DBrush::IsRedundant(): a plane that doesn't touch
    at least 3 of the brush's own real solved vertices isn't defining an
    actual face of the final shape - it's a leftover from the ledge's
    original triangulation that the intersection of the OTHER planes has
    already made irrelevant. Read directly from bobToolz's own source
    (DPlane.cpp) rather than guessed: MAX_ROUND_ERROR there is 0.05."""
    if len(planes) < 4:
        return planes
    verts = solve_brush_vertices(planes)
    if not verts:
        return planes
    kept = []
    for pn, pd, t in planes:
        touching = sum(1 for v in verts if abs(dot(v, pn) - pd) < 0.05)
        if touching >= 3:
            kept.append((pn, pd, t))
    return kept if len(kept) >= 4 else planes


def unique_planes(tris):
    """Collapse a ledge's triangles into its distinct face planes. Two
    triangles of the same box face share a plane and must become ONE brush
    side, or the brush ends up with redundant/duplicate sides.

    Tolerance matches bobToolz's own DPlane::operator== exactly (read from
    its source, DPlane.cpp): two planes are equal if |normal_a - normal_b|
    < MAX_ROUND_ERROR (0.05) and |dist_a - dist_b| < 0.05. For unit vectors,
    |a-b| = sqrt(2-2*dot(a,b)), so a 0.05 vector-difference threshold is
    exactly dot(a,b) > 1 - 0.05**2/2 = 0.99875 - not the same as an
    arbitrarily chosen dot threshold, this is the precise equivalent."""
    DUP_DOT_THRESHOLD = 1 - 0.05 ** 2 / 2  # == 0.99875
    DUP_DIST_THRESHOLD = 0.05
    planes = []
    for tri in tris:
        n = normalize(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])))
        if n == (0.0, 0.0, 0.0):
            continue  # degenerate triangle
        dist = dot(n, tri[0])
        for pn, pd, _pts in planes:
            if dot(pn, n) > DUP_DOT_THRESHOLD and abs(pd - dist) < DUP_DIST_THRESHOLD:
                break
        else:
            planes.append((n, dist, tri))
    return planes


def plane_basis(n):
    """Any two orthonormal vectors spanning the plane with normal n."""
    arbitrary = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = normalize(sub(arbitrary, (n[0] * dot(arbitrary, n),
                                   n[1] * dot(arbitrary, n),
                                   n[2] * dot(arbitrary, n))))
    v = cross(n, u)
    return u, v


def synth_plane_points(n, dist, centroid, spread=64.0):
    """Build three WIDELY-SPACED points on the plane instead of reusing the
    collision triangle's own vertices.

    A .phy hull contains plenty of long thin sliver triangles. Their three
    vertices are nearly collinear, so when the editor/compiler recomputes a
    plane normal from them at map-scale coordinates (often in the thousands)
    the cross product underflows and you get 'brush plane with no normal'.
    The plane itself is perfectly well defined - only the choice of points
    describing it was bad. Emitting three far-apart points on that same
    plane, centred near the ledge so coordinates stay small, removes the
    precision problem entirely without changing the geometry at all.

    Winding: the .map brush side's points must reproduce -n, matching the
    reversed order the triangle path used (a .phy triangle normal points OUT
    of the solid; a brush plane faces INTO the brush volume). cross(v, u) is
    -n by construction, so the order (p, p+v, p+u) is correct."""
    p = (centroid[0] - n[0] * (dot(n, centroid) - dist),
         centroid[1] - n[1] * (dot(n, centroid) - dist),
         centroid[2] - n[2] * (dot(n, centroid) - dist))
    u, v = plane_basis(n)
    return (p,
            (p[0] + v[0] * spread, p[1] + v[1] * spread, p[2] + v[2] * spread),
            (p[0] + u[0] * spread, p[1] + u[1] * spread, p[2] + u[2] * spread))


# ------------------------------------------------------------------ vmf ---

_TOKEN_RE = None


def tokenize(text):
    import re
    global _TOKEN_RE
    if _TOKEN_RE is None:
        _TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(//[^\n]*)|([{}])|(\S+)')
    out = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            out.append(m.group(1).replace('\\"', '"'))
        elif m.group(2) is not None:
            continue
        elif m.group(3) is not None:
            out.append(m.group(3))
        else:
            out.append(m.group(4))
    return out


def parse_vdf(tokens):
    from collections import OrderedDict
    pos = [0]

    def block():
        node = OrderedDict()
        while pos[0] < len(tokens):
            tok = tokens[pos[0]]
            if tok == '}':
                pos[0] += 1
                return node
            key = tok
            pos[0] += 1
            if pos[0] >= len(tokens):
                break
            nxt = tokens[pos[0]]
            if nxt == '{':
                pos[0] += 1
                node.setdefault(key, []).append(block())
            else:
                pos[0] += 1
                node.setdefault(key, []).append(nxt)
        return node

    root = block()
    return root


def get1(node, key, default=None):
    v = node.get(key)
    return v[0] if v else default


PROP_CLASSES = {'prop_static', 'prop_dynamic', 'prop_physics',
                'prop_physics_multiplayer', 'prop_dynamic_override',
                'prop_ragdoll'}


# $surfaceprop -> clip shader. Defaults target the "_real*" naming convention
# (clip brushes that are invisible and solid but carry the right
# q3map_material, so footsteps and bullet impacts match the prop instead of
# defaulting to generic). Override wholesale with --clip-map JSON.
DEFAULT_CLIP_MAP = {
    # Targets are SOF2's OWN shipped tools.shader entries, so they are
    # guaranteed to exist. The _shoot_* family is used rather than _clip_*
    # because a Source .phy is the prop's real trace collision - it stops
    # bullets as well as players. SOF2's _clip has "surfaceparm noimpact"
    # and NO shotclip, so shots pass straight through it; only the _shoot_*
    # family carries shotclip. Each _shoot_* variant also sets the matching
    # q3map_material, so impacts and footsteps sound right for the prop.
    'metal': 'tools/_shoot_metal',
    'solidmetal': 'tools/_shoot_metal',
    'iron': 'tools/_shoot_metal',
    'metal_box': 'tools/_shoot_metal',
    'metalpanel': 'tools/_shoot_metal',
    'metalvent': 'tools/_shoot_metal',
    'metalgrate': 'tools/_shoot_metal',
    'chainlink': 'tools/_shoot_metal',
    'grate': 'tools/_shoot_metal',
    'wood': 'tools/_shoot_wood',
    'wood_plank': 'tools/_shoot_wood',
    'wood_panel': 'tools/_shoot_wood',
    'wood_solid': 'tools/_shoot_wood',
    'wood_box': 'tools/_shoot_wood',
    'wood_crate': 'tools/_shoot_wood',
    'wood_furniture': 'tools/_shoot_wood',
    'cardboard': 'tools/_shoot_wood',
    'computer': 'tools/_shoot_computer',
    'canvas': 'tools/_shoot_cloth',
    'carpet': 'tools/_shoot_cloth',
    'upholstery': 'tools/_shoot_cloth',
    'fabric': 'tools/_shoot_cloth',
    'concrete': 'tools/_shoot_stone',
    'concrete_block': 'tools/_shoot_stone',
    'brick': 'tools/_shoot_stone',
    'rock': 'tools/_shoot_stone',
    'boulder': 'tools/_shoot_stone',
    'stone': 'tools/_shoot_stone',
    'tile': 'tools/_shoot_stone',
    'plaster': 'tools/_shoot_stone',
    'sand': 'tools/_shoot_sandbag',
    'dirt': 'tools/_shoot_sandbag',
    'gravel': 'tools/_shoot_sandbag',
    # No shipped _shoot_* variant matches these, so they fall back to plain
    # _shoot (still blocks bullets, just generic impact material).
}


def cstr(data, off):
    end = data.index(b'\x00', off)
    return data[off:end].decode('latin1')


def mdl_hull(mdl_path):
    """Read a Source .mdl's declared hull_min/hull_max (its own bounding box,
    in source units). This is the authoritative per-model bound - far better
    than any shape heuristic, because it comes from the model itself rather
    than a guess about what collision "should" look like. Verified against
    real data: utility_truck.mdl reports hull (-104,-51,-2)..(105,51,69) and
    its .phy collision sits just inside that, as expected."""
    try:
        with open(mdl_path, 'rb') as f:
            d = f.read(200)
    except OSError:
        return None
    if len(d) < 128 or d[0:4] != b'IDST':
        return None
    try:
        hmin = struct.unpack_from('<3f', d, 104)
        hmax = struct.unpack_from('<3f', d, 116)
    except struct.error:
        return None
    if all(abs(v) < 1e-6 for v in hmin + hmax):
        return None
    return hmin, hmax


CONTENTS_SOLID = 0x1
CONTENTS_GRATE = 0x8
# Surfaceprops CS:S treats as see-through mesh. Its bullets go through these
# almost for free (the penetration code's grate/chainlink material), and SOF2
# has no penetration, only shotclip on or off - so the nearest match is "off".
GRATE_SURFACEPROPS = {'chainlink', 'metalgrate', 'grate', 'metal_grate'}


def mdl_contents(mdl_path):
    """studiohdr_t.contents ($contents in the .qc), or None. Offset 332,
    right after mass at 328 - checked on a real .mdl (utility_truck: mass
    3000, contents 0x1)."""
    try:
        with open(mdl_path, 'rb') as f:
            d = f.read(336)
        if len(d) < 336 or d[0:4] != b'IDST':
            return None
        return struct.unpack_from('<i', d, 332)[0]
    except (OSError, struct.error):
        return None


def mdl_surfaceprop(mdl_path):
    """studiohdr_t.surfacepropindex (offset 308, right before keyvalueindex;
    contents at 332 is checked on real files) -> the model's own $surfaceprop
    from its .qc. That is the prop's physics material - what Source itself
    uses for its impacts and footsteps - which a texture's .vmt only guesses
    at (most model .vmts have no $surfaceprop at all)."""
    try:
        with open(mdl_path, 'rb') as f:
            d = f.read()
        if len(d) < 336 or d[0:4] != b'IDST':
            return None
        off = struct.unpack_from('<i', d, 308)[0]
        if not (0 < off < len(d)):
            return None
        name = cstr(d, off).strip().lower()
        return name or None
    except (OSError, struct.error, ValueError):
        return None


def _load_sibling(module_name):
    """vmt2shader.py, wherever this script is run from (command line, the
    GUI's file-path loading, or a PyInstaller bundle)."""
    try:
        return __import__(module_name)
    except ImportError:
        pass
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(os.path.abspath(sys.argv[0])), getattr(sys, '_MEIPASS', '')):
        path = os.path.join(d, module_name + '.py')
        if d and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(module_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def mdl_first_material(mdl_path):
    """Read a Source .mdl's cdmaterials + first texture name, returning the
    engine-relative material path its .vmt would live at."""
    try:
        with open(mdl_path, 'rb') as f:
            d = f.read()
    except OSError:
        return None
    if len(d) < 240 or d[0:4] != b'IDST':
        return None
    try:
        numtextures, textureindex = struct.unpack_from('<ii', d, 204)
        numcdtextures, cdtextureindex = struct.unpack_from('<ii', d, 212)
        if numtextures < 1 or numcdtextures < 1:
            return None
        off = struct.unpack_from('<i', d, cdtextureindex)[0]
        cdmat = cstr(d, off).rstrip('\\/').replace('\\', '/')
        sznameindex = struct.unpack_from('<i', d, textureindex)[0]
        tex = cstr(d, textureindex + sznameindex)
        return "{}/{}".format(cdmat, tex).lower()
    except (struct.error, ValueError, IndexError):
        return None


def vmt_surfaceprop(vmt_path):
    try:
        with open(vmt_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except OSError:
        return None
    import re as _re
    m = _re.search(r'"?\$surfaceprop"?\s+"?([A-Za-z0-9_]+)"?', text, _re.IGNORECASE)
    return m.group(1).lower() if m else None


def build_vmt_index(materials_dir):
    index = {}
    for dirpath, _dirs, files in os.walk(materials_dir):
        for fn in files:
            if fn.lower().endswith('.vmt'):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, materials_dir).replace('\\', '/').lower()
                index[os.path.splitext(rel)[0]] = full
    return index


def collect_props(root):
    props = []
    for ent in root.get('entity', []):
        if not isinstance(ent, dict):
            continue
        cls = (get1(ent, 'classname') or '').lower()
        if cls not in PROP_CLASSES:
            continue
        model = get1(ent, 'model')
        origin = get1(ent, 'origin')
        if not model or not origin:
            continue
        try:
            o = tuple(float(x) for x in origin.split())
        except ValueError:
            continue
        ang = get1(ent, 'angles', '0 0 0')
        try:
            a = tuple(float(x) for x in ang.split())
        except ValueError:
            a = (0.0, 0.0, 0.0)
        ms = get1(ent, 'modelscale', '1')
        try:
            ms = float(ms)
        except ValueError:
            ms = 1.0
        # Source's per-placement collision switch: 0 = not solid, 2 = bbox,
        # 6 = vphysics (the default when absent).
        solid = (get1(ent, 'solid', '6') or '6').strip()
        props.append({'model': model.replace('\\', '/').lower(),
                      'origin': o, 'angles': a, 'modelscale': ms, 'solid': solid,
                      'id': str(get1(ent, 'id') or '').strip()})
    return props


def find_phy(model_path, phy_index):
    base = os.path.splitext(os.path.basename(model_path))[0].lower()
    return phy_index.get(base)


def build_phy_index(phy_dir):
    index = {}
    for dirpath, _dirs, files in os.walk(phy_dir):
        for fn in files:
            if fn.lower().endswith('.phy'):
                key = os.path.splitext(fn)[0].lower()
                index.setdefault(key, os.path.join(dirpath, fn))
    return index


# --------------------------------------------------------------- output ---

def fmt_point(p):
    return "( {:.6g} {:.6g} {:.6g} )".format(p[0], p[1], p[2])


def aabb_planes(world_tris, min_thickness):
    """Build a plain axis-aligned box (6 perpendicular planes) around this
    piece's vertices, expanded so no dimension is thinner than
    min_thickness. Used instead of the exact hull for pieces thin enough to
    need padding at all.

    An earlier version tried to nudge just the two planes bounding the thin
    axis outward, preserving the exact tilted shape. That broke on real
    content: a cardboard box's collision turned out to have TWO pairs of
    near-parallel planes (not just the intended thin axis), and nudging one
    pair shifted the geometry enough to destabilize vertex solving involving
    the OTHER pair, producing a wildly wrong result elsewhere in the same
    piece. A plain AABB has none of this risk - its 6 planes are always
    exactly perpendicular by construction, so there is no near-parallel
    pair for any padding operation to interact badly with. The cost is
    precision (an axis-aligned box instead of the model's exact tilted
    shape), which is an acceptable trade for a piece already this thin -
    it is not carrying meaningfully precise collision to begin with."""
    vs = [v for tri in world_tris for v in tri]
    lo = [min(v[i] for v in vs) for i in range(3)]
    hi = [max(v[i] for v in vs) for i in range(3)]
    for i in range(3):
        if hi[i] - lo[i] < min_thickness:
            pad = (min_thickness - (hi[i] - lo[i])) / 2.0
            lo[i] -= pad
            hi[i] += pad
    planes = []
    for i in range(3):
        n = [0.0, 0.0, 0.0]
        n[i] = 1.0
        planes.append((tuple(n), hi[i], None))
        n2 = [0.0, 0.0, 0.0]
        n2[i] = -1.0
        planes.append((tuple(n2), -lo[i], None))
    return planes


def brush_from_planes(planes, texture, centroid):
    lines = ["{"]
    for n, dist, _tri in planes:
        q0, q1, q2 = synth_plane_points(n, dist, centroid)
        lines.append("{} {} {} {} 0 0 0 0.5 0.5 0 0 0".format(
            fmt_point(q0), fmt_point(q1), fmt_point(q2), texture))
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vmf")
    ap.add_argument("phy_dir")
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--model-scale", type=float, default=1.0)
    ap.add_argument("--clip-texture", default="tools/_shoot",
                     help="fallback clip shader when the prop's material can't be "
                          "resolved or has no mapping (default tools/_clip)")
    ap.add_argument("--materials-dir",
                     help="your .vmt root. When given, each prop's .mdl is read for its "
                          "material, that .vmt's $surfaceprop is looked up, and a "
                          "material-appropriate clip shader is chosen (see --clip-map) so "
                          "footsteps/impacts match the prop instead of being generic.")
    ap.add_argument("--clip-map",
                     help="JSON {\"surfaceprop\": \"tools/_shader\"} overriding the built-in "
                          "surfaceprop -> clip shader table.")
    ap.add_argument("--max-planes", type=int, default=80,
                     help="skip any ledge needing more than N planes (default 80) - guards "
                          "against a dense mesh blowing MAX_MAP_PLANES, which is the exact "
                          "failure auto-clip produces. Set generously: a genuinely round piece "
                          "(a cylinder's side wall) needs far more planes than a boxy one to "
                          "represent its curve at all - confirmed on real content, a wire-spool "
                          "prop's cylindrical body needed 50 planes and was being silently "
                          "discarded (not corrupted, just deleted) by an earlier, too-conservative "
                          "default of 40.")
    ap.add_argument("--hull-margin", type=float, default=1.25,
                     help="reject any collision piece extending beyond the model's own "
                          "hull_min/hull_max (read from its .mdl) times this factor "
                          "(default 1.25). This is the authoritative corruption check - "
                          "set to 0 to disable.")
    ap.add_argument("--min-compile-thickness", type=float, default=2.0,
                     help="pad any collision piece thinner than this (world units, default 2) "
                          "up to this thickness, symmetrically along its thin axis. This is "
                          "NOT about rejecting bad data - the source geometry can be perfectly "
                          "correct and still be too thin for idTech3's own CSG/BSP math to "
                          "handle reliably at compile time (confirmed: a 0.91-unit-thick "
                          "cardboard box's authored collision, verified exact in our own math, "
                          "still came out corrupted after compiling). Set to 0 to disable and "
                          "emit the raw thickness as-is.")
    ap.add_argument("--min-thickness", type=float, default=0.1,
                     help="reject any collision piece whose SMALLEST dimension is under this "
                          "(world units, default 0.1). This exists ONLY to catch genuine "
                          "mathematical degeneracy (a piece with ~zero volume, which has no "
                          "solvable interior for the compiler). It is deliberately set very low: "
                          "Source physics legitimately authors thin real collision (a chainlink "
                          "fence's mesh panel measured 0.9 world units thick in real content and "
                          "needs to stay solid) - a higher threshold here would silently delete "
                          "correct collision, not just corrupt data. If you're chasing walk-"
                          "through geometry, --needle-min-thickness below is almost always the "
                          "one to raise, not this one.")
    ap.add_argument("--needle-min-thickness", type=float, default=0.0,
                     help="OFF by default (0). Originally rejected long+thin pieces as "
                          "'corrupt needles', but real content disproved that assumption: a "
                          "chainlink fence's mesh panel is legitimately ~508 units long and 0.9 "
                          "units thick, and this heuristic deleted it, leaving the fence walk-"
                          "through. The actual corruption source (a sign error in the ledge "
                          "bounding-sphere check) is fixed directly now, and --hull-margin "
                          "below is the principled replacement: it checks each piece against "
                          "that SPECIFIC model's own declared hull instead of guessing from "
                          "shape. Set this > 0 only if you still see spike-shaped geometry after "
                          "confirming --hull-margin isn't the fix for it.")
    ap.add_argument("--needle-min-length", type=float, default=200.0,
                     help="length above which --needle-min-thickness applies, if set > 0.")
    ap.add_argument("--skip-list")
    ap.add_argument("--skip-ids", help="vmf2map.py's <map>_dropped_props.json (VMF entity ids "
                                       "of props it dropped); their collision is left out too.")
    ap.add_argument("--merge-into")
    args = ap.parse_args()

    skip = set()
    if args.skip_list and os.path.isfile(args.skip_list):
        with open(args.skip_list) as f:
            skip = {l.strip().lower() for l in f if l.strip()}

    skip_ids = set()
    if args.skip_ids and os.path.isfile(args.skip_ids):
        import json
        with open(args.skip_ids, encoding='utf-8') as f:
            skip_ids = {str(x).strip() for x in json.load(f)}

    v2s = _load_sibling('vmt2shader')
    if v2s is None:
        print("NOTE: vmt2shader.py not found next to this script - collision falls back to "
              "SOF2's own few tools/_shoot_* shaders.", file=sys.stderr)

    clip_map = dict(DEFAULT_CLIP_MAP)
    if args.clip_map and os.path.isfile(args.clip_map):
        import json
        with open(args.clip_map) as f:
            clip_map.update({k.lower(): v for k, v in json.load(f).items()})

    vmt_index = build_vmt_index(args.materials_dir) if args.materials_dir else {}

    with open(args.vmf, 'r', encoding='utf-8', errors='replace') as f:
        root = parse_vdf(tokenize(f.read()))

    props = collect_props(root)
    phy_index = build_phy_index(args.phy_dir)
    print("{} prop entities, {} .phy files indexed".format(len(props), len(phy_index)),
          file=sys.stderr)

    brushes = []
    stats = {'ok': 0, 'no_phy': 0, 'skipped': 0, 'too_complex': 0,
             'ledges': 0, 'planes': 0}
    missing = {}

    for p in props:
        if p['model'] in skip:
            stats['skipped'] += 1
            continue
        if p.get('id') and p['id'] in skip_ids:
            stats['dropped_prop'] = stats.get('dropped_prop', 0) + 1
            continue
        # "solid" "0" in the .vmf: the mapper made this placement non-solid
        # (wall lamps, wires, cloth, trim - 502 of de_mirage_cs2's 923
        # prop_statics). It still has a .phy, because the MODEL can be solid
        # elsewhere, but here players and bullets go straight through it.
        if p.get('solid') == '0':
            stats['nonsolid'] = stats.get('nonsolid', 0) + 1
            continue
        phy_path = find_phy(p['model'], phy_index)
        if not phy_path:
            stats['no_phy'] += 1
            missing[p['model']] = missing.get(p['model'], 0) + 1
            continue
        try:
            solids = parse_phy(phy_path)
        except Exception:
            stats['no_phy'] += 1
            continue
        if not solids:
            stats['no_phy'] += 1
            continue

        # Pick a clip shader matching the prop's own material, so the
        # generated collision produces the right footstep/impact response.
        clip_tex = args.clip_texture
        prop_sp = None
        mdl_path = os.path.splitext(phy_path)[0] + '.mdl'
        if os.path.isfile(mdl_path):
            prop_sp = mdl_surfaceprop(mdl_path)
            if prop_sp in (None, 'default') and vmt_index:
                mat = mdl_first_material(mdl_path)
                vmt = (vmt_index.get(mat) or vmt_index.get('materials/' + mat)) if mat else None
                sp = vmt_surfaceprop(vmt) if vmt else None
                if sp:
                    prop_sp = sp.lower()
        if prop_sp:
            stats.setdefault('surfaceprops', {})
            stats['surfaceprops'][prop_sp] = stats['surfaceprops'].get(prop_sp, 0) + 1
            if prop_sp in clip_map and args.clip_map:
                clip_tex = clip_map[prop_sp]          # the user's own table wins
            else:
                sof2_mat = v2s.q3map_material_for(prop_sp) if v2s else None
                if sof2_mat:
                    clip_tex = v2s.shoot_clip_name(sof2_mat)
                else:
                    clip_tex = clip_map.get(prop_sp, args.clip_texture)
                    stats.setdefault('unmapped_sp', {})
                    stats['unmapped_sp'][prop_sp] = stats['unmapped_sp'].get(prop_sp, 0) + 1

        mdl_for_hull = os.path.splitext(phy_path)[0] + '.mdl'
        hull = mdl_hull(mdl_for_hull) if os.path.isfile(mdl_for_hull) else None

        # Fences and grates: block players, let bullets through. Source marks
        # these on the model ($contents "grate") or on the material
        # (chainlink/metalgrate surfaceprop). SOF2's tools/_clip is exactly
        # playerclip without shotclip.
        contents = mdl_contents(mdl_for_hull) if os.path.isfile(mdl_for_hull) else None
        if contents == 0:
            stats['nonsolid'] = stats.get('nonsolid', 0) + 1   # $contents "notsolid"
            continue
        is_grate = ((contents is not None and contents & CONTENTS_GRATE
                     and not contents & CONTENTS_SOLID)
                    or prop_sp in GRATE_SURFACEPROPS)
        if is_grate:
            clip_tex = 'tools/_clip'
            stats.setdefault('grate_models', set()).add(p['model'])
        hull_margin = args.hull_margin

        m = rotation_matrix(*p['angles'])
        origin_out = (p['origin'][0] * args.scale,
                      p['origin'][1] * args.scale,
                      p['origin'][2] * args.scale)
        eff_model_scale = p['modelscale'] * args.model_scale

        made_any = False
        for ledges in solids:
            for tris in ledges:
                world_tris = [
                    [transform(v, m, origin_out, eff_model_scale) for v in tri]
                    for tri in tris
                ]
                # Needle rejection: corrupt ledges show up as extremely long,
                # near-zero-thickness slivers (e.g. 500 units long with a
                # 2-unit cross-section). The bounding-sphere test upstream
                # can't catch these because a large prop legitimately has a
                # large bounding sphere - it's the SHAPE that is impossible,
                # not the size. Requiring BOTH "very long" and "very thin"
                # keeps legitimately thin small parts and legitimately long
                # chunky parts.
                # Authoritative check first: does this ledge fit inside the
                # model's OWN declared hull? Corrupt ledges (bad tree offsets
                # reinterpreting neighbouring bytes as vertices) produce
                # geometry far outside the box the model itself declares.
                # Unlike a shape heuristic this can't mistake a legitimately
                # big or oddly-proportioned prop for corruption, because the
                # bound comes from that specific model.
                if hull is not None:
                    hmin, hmax = hull
                    over = False
                    for tri in tris:
                        for v in tri:  # tri is local, in source units
                            for i in range(3):
                                lo = hmin[i] * hull_margin - 4.0
                                hi = hmax[i] * hull_margin + 4.0
                                if lo > hi:
                                    lo, hi = hi, lo
                                if v[i] < lo or v[i] > hi:
                                    over = True
                                    break
                            if over:
                                break
                        if over:
                            break
                    if over:
                        stats['out_of_hull'] = stats.get('out_of_hull', 0) + 1
                        continue

                vs = [v for tri in world_tris for v in tri]
                ext = [max(v[i] for v in vs) - min(v[i] for v in vs) for i in range(3)]

                # Degenerate-thickness check. This is separate from the needle
                # test below and fires at ANY size: a piece 62 x 62 x 0 is just
                # as unusable as one 500 x 3 x 3. Real example: a wire-spool
                # prop whose .phy contains near-flat disc hulls - legal in
                # Source's physics, but a zero-thickness .map brush has no
                # solvable interior and the compiler turns it into stretched
                # garbage across the level.
                if min(ext) < args.min_thickness:
                    stats['flat'] = stats.get('flat', 0) + 1
                    continue

                if (max(ext) > args.needle_min_length
                        and min(ext) < args.needle_min_thickness):
                    stats['needles'] = stats.get('needles', 0) + 1
                    continue

                vs_check = [v for tri in world_tris for v in tri]
                raw_ext = [max(v[i] for v in vs_check) - min(v[i] for v in vs_check) for i in range(3)]
                if args.min_compile_thickness > 0 and min(raw_ext) < args.min_compile_thickness:
                    planes = aabb_planes(world_tris, args.min_compile_thickness)
                    stats['aabb_fallback'] = stats.get('aabb_fallback', 0) + 1
                else:
                    planes = unique_planes(world_tris)
                    if len(planes) < 4:
                        continue  # not a closed volume
                    planes = drop_redundant_planes(planes)
                if len(planes) > args.max_planes:
                    stats['too_complex'] += 1
                    continue
                cx = sum(v[0] for tri in world_tris for v in tri)
                cy = sum(v[1] for tri in world_tris for v in tri)
                cz = sum(v[2] for tri in world_tris for v in tri)
                npts = len(world_tris) * 3
                centroid = (cx / npts, cy / npts, cz / npts)
                brushes.append(brush_from_planes(planes, clip_tex, centroid))
                stats['ledges'] += 1
                stats['planes'] += len(planes)
                made_any = True
        if made_any:
            stats['ok'] += 1

    body = "\n".join(brushes) + ("\n" if brushes else "")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(body)

    print("\nGenerated {} clip brush(es) from {} prop(s)".format(stats['ledges'], stats['ok']),
          file=sys.stderr)
    print("  total planes: {} (auto-clip would have produced one per triangle)".format(
        stats['planes']), file=sys.stderr)
    print("  props with no .phy (no collision authored): {}".format(stats['no_phy']),
          file=sys.stderr)
    print("  ledges skipped as too complex (> --max-planes {}): {}".format(
        args.max_planes, stats['too_complex']), file=sys.stderr)
    if stats.get('aabb_fallback'):
        print("  ledges replaced with a padded bounding box (thinner than "
              "--min-compile-thickness {}): {}".format(
                  args.min_compile_thickness, stats['aabb_fallback']), file=sys.stderr)
    if stats.get('flat'):
        print("  ledges rejected as degenerate (thinner than --min-thickness {}): {}".format(
            args.min_thickness, stats['flat']), file=sys.stderr)
    if stats.get('out_of_hull'):
        print("  ledges rejected as outside the model's own .mdl hull: {}".format(
            stats['out_of_hull']), file=sys.stderr)
    if stats.get('needles'):
        print("  ledges rejected as corrupt needles (long + near-zero thickness): {}".format(
            stats['needles']), file=sys.stderr)
    if stats.get('dropped_prop'):
        print("  props the prop budget dropped (no collision left behind): {}".format(
            stats['dropped_prop']), file=sys.stderr)
    if stats.get('surfaceprops'):
        print("  collision material by $surfaceprop: " + ", ".join(
            "{} x{}".format(k, v) for k, v in sorted(stats['surfaceprops'].items(),
                                                     key=lambda kv: -kv[1])), file=sys.stderr)
    if stats.get('unmapped_sp'):
        print("  $surfaceprop with no SOF2 material (plain {}): {}".format(
            args.clip_texture, ", ".join(sorted(stats['unmapped_sp']))), file=sys.stderr)
    if stats['skipped']:
        print("  props skipped via --skip-list: {}".format(stats['skipped']), file=sys.stderr)
    if stats.get('nonsolid'):
        print("  props non-solid in Source (\"solid\" \"0\" or $contents notsolid), no clip: "
              "{}".format(stats['nonsolid']), file=sys.stderr)
    if stats.get('grate_models'):
        print("  fence/grate models clipped with tools/_clip (players blocked, bullets pass):",
              file=sys.stderr)
        for mdl in sorted(stats['grate_models']):
            print("    " + mdl, file=sys.stderr)

    if args.merge_into:
        with open(args.merge_into, 'r', encoding='utf-8', errors='replace') as f:
            map_text = f.read()
        with open(args.merge_into + '.bak', 'w', encoding='utf-8') as f:
            f.write(map_text)
        # worldspawn is the first entity block; insert before its closing brace
        idx = map_text.find('"classname" "worldspawn"')
        if idx == -1:
            print("ERROR: no worldspawn found in --merge-into target", file=sys.stderr)
            sys.exit(1)
        depth = 0
        end = None
        start = map_text.rfind('{', 0, idx)
        for i in range(start, len(map_text)):
            if map_text[i] == '{':
                depth += 1
            elif map_text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            print("ERROR: couldn't find end of worldspawn", file=sys.stderr)
            sys.exit(1)
        merged = map_text[:end] + body + map_text[end:]
        with open(args.merge_into, 'w', encoding='utf-8') as f:
            f.write(merged)
        print("\nMerged into {} (backup at {}.bak)".format(args.merge_into, args.merge_into),
              file=sys.stderr)
    else:
        print("\nWrote {} - paste its contents inside your worldspawn block,".format(args.output),
              file=sys.stderr)
        print("or re-run with --merge-into <your.map> to do it automatically.", file=sys.stderr)


if __name__ == "__main__":
    main()
