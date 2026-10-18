#!/usr/bin/env python3
"""
vmf2map.py - Convert Valve Map Format (.vmf) files into idTech3 .map files
compatible with Soldier of Fortune II: Double Helix (and other Quake III
Arena derived engines / editors such as GtkRadiant, SOF2Radiant, NetRadiant).

------------------------------------------------------------------------------
WHAT THIS DOES
------------------------------------------------------------------------------
- Parses the VMF KeyValues tree (versioninfo/world/entity/solid/side/...).
- Converts every brush ("solid") into the classic (non-brush-primitives)
  Quake3 .map brush syntax:

      ( x1 y1 z1 ) ( x2 y2 z2 ) ( x3 y3 z3 ) texture sX sY rot scX scY 0 0 0

- Reconstructs texture rotation/scale/shift from Source's uaxis/vaxis by
  solving for the rotation that reproduces Source's texture-axis unit
  vectors within the standard idTech3 base-axis basis for that face's
  dominant normal direction (the same "TextureAxisFromPlane" table id
  Software's tools use). Scale and shift are passed straight through,
  since both formats use the same "dot(P, axis)/scale + shift" texel
  formula.
- Emits worldspawn brushes as worldspawn, and brush/point entities as
  entities, preserving keys/values as-is (extra unknown keys are ignored
  harmlessly by the idTech3 engine).
- func_detail brushes are folded into worldspawn (idTech3 has no direct
  "detail entity" equivalent in the classic .map format; detail status
  is normally controlled by shader/content flags instead).

------------------------------------------------------------------------------
WHAT THIS CANNOT DO PERFECTLY (please read before relying on the output)
------------------------------------------------------------------------------
1. TEXTURES: Source materials (materials/foo/bar.vmt) and SOF2 textures/
   shaders (textures/foo/bar) are different asset sets. This script just
   rewrites the path (lowercased, backslashes normalized) and applies a
   small built-in table for common Source "tool" textures (nodraw, clip,
   trigger, skip, hint, sky -> tools/_caulk, tools/_clip, etc.). Anything
   else will almost certainly not exist in SOF2's tree and needs a real
   texture, supplied via --texture-map (see below). Entries you supply in
   --texture-map should NOT include a "textures/" prefix (e.g.
   "textures/walls/euro_brick1" is wrong, "walls/euro_brick1" is right) -
   .map source files (as read by Radiant/GtkRadiant-family editors)
   reference textures relative to the textures/ folder already; it's
   q3map2 (the compiler) that prepends "textures/" when building the
   compiled .bsp. That's also why the SOF2 .shader scripts vmt2shader.py
   generates DO carry that prefix - .shader names are matched against the
   already-prefixed, compiled form, a different file with a different
   convention from the .map source.

2. ENTITIES: Source and idTech3/SOF2 use different entity classnames and
   keys (Source I/O "connections" blocks have no idTech3 equivalent and
   are dropped with a warning). Point/brush entities are passed through
   as-is so you keep the geometry and keys, but you should expect to
   re-target classnames (e.g. prop_static/prop_dynamic have no brush
   geometry and won't do anything useful in SOF2) via --class-map or
   --skip-classes.

3. SCALE/UNITS: Source and Quake/idTech3 use the same base unit, so no
   scale conversion is applied by default. Use --scale if your particular
   map was built at a different working scale.

4. This produces the classic id .map brush format (3-point planes, no
   brush-primitives texture matrices), which is what SOF2Radiant / older
   GtkRadiant expect for SOF2 authoring.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 vmf2map.py input.vmf output.map
    python3 vmf2map.py input.vmf output.map --texture-map mytextures.json
    python3 vmf2map.py input.vmf output.map --class-map myclasses.json \
                        --skip-classes prop_static,prop_dynamic,env_fog_controller

--texture-map is a JSON object: {"BRICK/BRICKWALL051D": "walls/euro_brick1", ...}
  (keys are matched case-insensitively against the VMF "material" value)

--class-map is a JSON object: {"info_player_start": "info_player_deathmatch"}

A summary of unmapped textures / dropped entity data / warnings is printed
to stderr at the end of the run so you know what needs manual attention.
"""

import argparse
import json
import glob
import math
import re
import os
import shutil
import struct
import sys
from collections import OrderedDict, Counter


# idTech3 shader names are 64 bytes (63 + NUL), and q3map2 -rename appends
# "_BSP" to misc_model surfaces, so names longer than 59 characters are written
# in a deterministic short form. vmt2shader.py renames the matching shader block
# the same way (see its comment for the full reasoning). KEEP IDENTICAL to
# short_qpath in vmt2shader.py.
QPATH_NAME_MAX = 59


def short_qpath(name, limit=QPATH_NAME_MAX):
    if len(name) <= limit:
        return name
    import hashlib
    digest = hashlib.md5(name.lower().encode('utf-8')).hexdigest()[:8]
    return name[:limit - 9].rstrip('/_.') + '_' + digest


def _short_map_texture(tex):
    """A .map texture is the shader name minus q3map2's implicit 'textures/'."""
    if tex.lower().startswith('textures/'):
        return short_qpath(tex)
    full = 'textures/' + tex
    short = short_qpath(full)
    return tex if short == full else short[len('textures/'):]


_FACE_TEX_RE = re.compile(r'^(\s*(?:\(\s*[^()]*\)\s*){3})(\S+)', re.M)
_PATCH_TEX_RE = re.compile(r'(patchDef[23]\s*\n\s*\{\s*\n\s*)(\S+)')


def _md3_counts(path):
    """(vertices, triangles) of an .md3, summed over its surfaces."""
    try:
        with open(path, 'rb') as f:
            d = f.read()
    except OSError:
        return None
    if d[:4] != b'IDP3' or len(d) < 108:
        return None
    nsurf, ofs = struct.unpack_from('<i', d, 84)[0], struct.unpack_from('<i', d, 100)[0]
    verts = tris = 0
    for _ in range(max(0, min(nsurf, 256))):
        if ofs <= 0 or ofs + 108 > len(d) or d[ofs:ofs + 4] != b'IDP3':
            break
        verts += struct.unpack_from('<i', d, ofs + 80)[0]
        tris += struct.unpack_from('<i', d, ofs + 84)[0]
        ofs += struct.unpack_from('<i', d, ofs + 104)[0]
    return verts, tris


# RBSP drawvert: xyz, st, 4 lightmap st, normal, 4 colours = 80 bytes
RBSP_DRAWVERT_BYTES = 80

# q3map2's draw-index ceiling. In ydnar's 2.5.17 - which is what GtkRadiant
# 1.6.7 ships - the draw vertices and indexes live in FIXED C arrays sized
# MAX_MAP_DRAW_VERTS / MAX_MAP_DRAW_INDEXES (0x80000 = 524,288 each), and the
# compile stops at "MAX_MAP_DRAW_INDEXES" in FilterDrawsurfsIntoTree. The
# NetRadiant fork grows those arrays on demand instead, so it has no such
# ceiling. Measured on de_overpass_csgo with GtkRadiant 1.6.7 (2026-10-17):
# 71,175 world meta triangles = 213,525 indexes, and the map's full-quality
# props then overflowed. Three indexes per triangle.
Q3MAP2_MAX_DRAW_INDEXES = 0x80000
# What the world (brushwork, terrain, patches) typically takes of that before
# a single prop is baked - de_overpass_csgo's measured 213,525, rounded up.
Q3MAP2_WORLD_INDEX_ALLOWANCE = 220000


def report_baked_prop_geometry(map_text, md3_dir):
    """Every misc_model is copied into the .bsp once per placement (model_static
    is not: the game loads the .md3 once). Print what that costs, so a .bsp that
    comes out far bigger than the map's brushwork explains itself."""
    per_model = Counter()
    for blk in re.finditer(r'\{([^{}]*)\}', map_text):
        body = blk.group(1)
        cls = re.search(r'"classname"\s+"([^"]+)"', body)
        mdl = re.search(r'"model"\s+"([^"]+\.md3)"', body, re.I)
        if cls and mdl and cls.group(1).lower() == 'misc_model':
            per_model[mdl.group(1).replace('\\', '/')] += 1
    if not per_model:
        return
    rows, missing = [], 0
    for model, n in per_model.items():
        rel = re.sub(r'^/?models/', '', model, flags=re.I)
        c = _md3_counts(os.path.join(md3_dir, *rel.split('/')))
        if c is None:
            missing += n
            continue
        rows.append((c[0] * n, c[1] * n, n, c[0], model))
    if not rows:
        return
    rows.sort(reverse=True)
    tv = sum(r[0] for r in rows)
    tt = sum(r[1] for r in rows)
    print("\nBaked props: {:,} misc_model placement(s) of {} model(s) put {:,} vertices / {:,} "
          "triangles into the .bsp (about {:.0f} MB of draw vertices before lighting; each "
          "placement is its own copy).{} Largest contributors:".format(
              sum(per_model.values()), len(per_model), tv, tt,
              tv * RBSP_DRAWVERT_BYTES / 1e6,
              " {} placement(s) had no .md3 to measure.".format(missing) if missing else ""),
          file=sys.stderr)
    for tot_v, _tot_t, n, v, model in rows[:12]:
        print("  {:>9,} verts  {:>4} x {:>6,}  {}".format(tot_v, n, v, model), file=sys.stderr)
    prop_indexes = tt * 3
    room = Q3MAP2_MAX_DRAW_INDEXES - Q3MAP2_WORLD_INDEX_ALLOWANCE
    if prop_indexes > room:
        print("\n! These props need about {:,} draw indexes ({:,} triangles x 3), and a q3map2 "
              "with the stock FIXED lumps has {:,} in total (MAX_MAP_DRAW_INDEXES) - about "
              "{:,} once the world's own ~{:,} are in. Such a build (GtkRadiant 1.6.7 ships "
              "one) will stop with \"MAX_MAP_DRAW_INDEXES\" in FilterDrawsurfsIntoTree.\n"
              "  Compile this map with NetRadiant's q3map2: it grows those lumps as needed, so "
              "the geometry above goes through unchanged. Only if you must use a fixed-lump "
              "q3map2, convert without \"Props at original quality\" (or with a smaller "
              "--prop-vertex-budget) to bring the triangles down.".format(
                  prop_indexes, tt, Q3MAP2_MAX_DRAW_INDEXES, room,
                  Q3MAP2_WORLD_INDEX_ALLOWANCE),
              file=sys.stderr)


# How much baked prop geometry the .bsp may carry, in md3 vertices.
#
# SOF2 reads the whole .bsp into ONE allocation when a map loads ("Z_Malloc():
# Failed to malloc %d bytes!" in retail SoF2MP.exe), and SoF2MP.exe is NOT
# large-address-aware (its PE header lacks the flag), so that block has to fit
# in the largest free hole of a 2 GB address space - which depends on the
# player's machine. Measured points: de_dust2 at 28 MB loads, de_overpass_csgo
# at 45.7 MB loads, de_mirage_cs2 at 81.5 MB failed with "Failed to malloc
# 81514312 bytes" - the .bsp's size to the byte.
#
# More measured points, all de_overpass_csgo on KENNITH's machine
# (2026-10-17): 59.7 MiB loaded, 64.0 loaded once, 63.65 and 55.7 FAILED.
# The cause was not the .bsp: com_hunkMegs was 1024, which reserves half the
# 2 GB at startup. sof2_memmap.py at the main menu measured the largest free
# hole at 54.0 MiB with hunk 1024 and 393.2 MiB with hunk 512 (same machine,
# NVIDIA's nvgpucomp32.dll alone takes 85 MiB). meminfo on that map: hunk
# peak ~134 MB (low tempHighwater 57.5 + high 76.1), so com_hunkMegs 256 runs
# it with 2x headroom - the compile .bat says so (install_to_sof2.py).
# With a 256-512 hunk the target is nowhere near the limit; 64 MiB stays under
# de_mirage_cs2's 81.5 MB failure, whose hunk setting is unknown.
BSP_SIZE_TARGET = 64 * 1048576
# Everything in a .bsp that is not a baked prop or a decal: brushwork,
# terrain, planes, lightmaps, vis. Measured on de_overpass_csgo's compile of
# 2026-10-17: 52.86 MiB .bsp - 31.04 MiB of props (363,309 vertices) - 2.84
# MiB of decals (33,231) = 18.98 MiB; ~19 MB on de_mirage_cs2. 24 MiB left
# about 5 MiB of the target unused (that compile came out at 52.9 MiB against
# 58) and cost props for nothing; 21 MiB keeps 10% over the measured value.
BSP_NON_PROP_ALLOWANCE = 21 * 1048576
# What one baked md3 vertex costs in the .bsp, MEASURED on de_overpass_csgo by
# matching its .map to the .bsp q3map2 produced: 903 misc_model placements,
# 290,285 md3 vertices -> 286,891 draw vertices (0.988 each: q3map2 keeps them
# essentially 1:1) plus their indexes and surface records = 26.0 MB, i.e. 89.6
# bytes per md3 vertex (the 80-byte RBSP draw vertex is most of it).
BSP_BYTES_PER_BAKED_VERTEX = 89.6
# Decals and overlays are not in BSP_NON_PROP_ALLOWANCE (it was measured when
# they were a few thousand vertices). Since they are split finely for their
# vertex lighting (DECAL_VERTEX_LIGHT_SPACING) they are tens of thousands:
# 60,228 on de_overpass_csgo, 4.6 MiB, which is what pushed that map from
# 59.7 to 64.0 MiB. So their actual count is taken off the prop budget per map
# (decal_vertex_count), at the same cost per vertex.
DECAL_MESH_POINTS = [0]   # overlay mesh points written, set by convert_vmf
PROP_VERTEX_BUDGET = int((BSP_SIZE_TARGET - BSP_NON_PROP_ALLOWANCE) / BSP_BYTES_PER_BAKED_VERTEX)

def decal_vertex_count(map_text):
    """Vertices the decals add to the .bsp: the overlay mesh's points plus
    every patchDef2's control points (a patch is stored as its control grid).
    Displacement terrain is never a patch in --terrain-mode lwo, so in that
    mode the patches are all decals."""
    n = DECAL_MESH_POINTS[0]
    for w, h in re.findall(r'patchDef2\s*\{\s*\S+\s*\(\s*(\d+)\s+(\d+)', map_text):
        n += int(w) * int(h)
    return n


_MISC_MODEL_BLOCK_RE = re.compile(
    r'\{\n"classname" "misc_model"\n((?:"[^"\n]*" "[^"\n]*"\n)*)\}')


def _md3_lod_chain(md3_dir, model):
    """[(path, vertices)] for LOD0 and every <name>_lodN.md3 mdl_to_md3 wrote."""
    rel = re.sub(r'^/?models/', '', model, flags=re.I)
    base = os.path.join(md3_dir, *rel.split('/'))
    c = _md3_counts(base)
    if c is None:
        return []
    chain = [(model, c[0])]
    stem = model[:-4]
    for lod in range(1, 8):
        p = base[:-4] + '_lod{}.md3'.format(lod)
        c = _md3_counts(p)
        if c is None:
            break
        chain.append(('{}_lod{}.md3'.format(stem, lod), c[0]))
    return chain


# idTech3 entity slots (GENTITYNUM_BITS 10) and the ones a map can't have.
MAX_GENTITIES = 1024
SOF2_MAX_CLIENTS = 64
# Kept free for what the game spawns while it runs: dropped weapons,
# grenades, gametype items, corpses and the temporary event entities idTech3
# makes for every sound/effect/impact. A map that leaves none free crashes in
# the middle of a round instead of at load.
RUNTIME_ENTITY_RESERVE = 200
# Classnames that don't hold an entity slot in game: baked or merged by q3map2
# (misc_model, func_group, light, _skybox), or freed by the game at spawn
# (info_null).
_SLOTLESS_CLASSES = {'worldspawn', 'misc_model', 'func_group', 'light', '_skybox', 'info_null'}


def model_static_cap(map_text):
    """How many model_static entities the map can still take."""
    live = sum(1 for c in re.findall(r'"classname" "([^"]+)"', map_text)
               if c not in _SLOTLESS_CLASSES)
    return max(0, MAX_GENTITIES - SOF2_MAX_CLIENTS - RUNTIME_ENTITY_RESERVE - live)


# --- The snapshot limit -------------------------------------------------------
#
# In retail SOF2 a model_static is a NETWORKED SERVER ENTITY, not a client-side
# static model. Verified against the shipped binaries: retail cgamex86.dll has
# no static-model code at all (no "misc_model_static", "model_static" or
# "zoffset" string anywhere in it), while gamex86.dll registers a model_static
# handler that gives the entity a model index (G_ModelIndex -> CS_MODELS) and
# a stationary trajectory - an ordinary entity the server sends in snapshots.
# (OpenSoF2MP draws them client-side in CG_DrawMiscStaticModels; retail does
# not, so nothing learned from that code about culling applies here.)
#
# Every snapshot the client receives is truncated to 256 entities. Retail
# SoF2MP.exe 1.03, CL_GetSnapshot, at 0x4011c5:
#     cmp  edx, 0x100        ; entities in this snapshot vs 256
#     jle  ok
#     call Com_DPrintf       ; "CL_GetSnapshot: truncated %i entities to %i"
#     mov  edx, 0x100        ; keep the first 256, drop the rest
# Com_DPrintf only prints with "developer 1", so this is silent in normal play.
# Snapshots are sorted by entity number (delta compression needs that order),
# so the HIGHEST-numbered entities are the ones dropped.
#
# This was the "props missing / popping in" bug: de_overpass_csgo had 625
# model_statics, and in 28% of its vis clusters more than 256 props alone were
# in the PVS (up to 587). The fountain and the truck were entities #680 and
# #762 - last in line, so first to go.
#
# The entity-slot cap above (model_static_cap) allowed all 625, because 1024
# slots is not the limit that binds. Overpass's open vis lets one cluster see
# nearly every prop, so the networked props have to fit in ONE snapshot
# together, alongside the players and whatever they are doing.
SNAPSHOT_ENTITY_LIMIT = 256
# What a converted map should stay correct for. Over this many players nothing
# breaks: the props networked last (the smallest - see plan_prop_budget) are
# simply the first to flicker.
SNAPSHOT_PLAYERS = 32
# Per player, room for one transient entity in view at a time: a dropped
# weapon, a grenade in flight, a gametype item, or one of the short-lived event
# entities idTech3 spawns for every impact and sound.
SNAPSHOT_TRANSIENTS_PER_PLAYER = 1
# Map entities that are themselves networked and so share the snapshot. func_*
# brush entities are drawn by the client from snapshots; func_group and
# func_detail are not (q3map2 merges them into the world), so the slotless set
# is excluded explicitly.
_NETWORKED_CLASSES = {'model_static', 'target_speaker', 'gametype_item'}


def _is_networked_class(classname):
    if classname in _SLOTLESS_CLASSES or classname in ('func_detail',):
        return False
    return classname.startswith('func_') or classname in _NETWORKED_CLASSES


# A target_speaker with "soundSet" is EF_PERMANENT (1.03 SDK g_target.c
# SP_target_speaker): sent once in the game state, skipped when snapshots are
# built - so it takes an entity slot but no snapshot room.
_SOUNDSET_SPEAKER_RE = re.compile(r'"classname" "target_speaker"[^{}]*?"soundSet"')


def _networked_entity_count(map_text, exclude=()):
    n = sum(1 for c in re.findall(r'"classname" "([^"]+)"', map_text)
            if _is_networked_class(c) and c not in exclude)
    if 'target_speaker' not in exclude:
        n -= len(_SOUNDSET_SPEAKER_RE.findall(map_text))
    return n


def networked_prop_cap(map_text):
    """How many more props the map can network without the snapshot overflowing:
    256, less every player and one transient each, less every entity in the map
    that is already networked. Never more than the entity slots allow either."""
    networked = _networked_entity_count(map_text)
    snap = (SNAPSHOT_ENTITY_LIMIT
            - SNAPSHOT_PLAYERS * (1 + SNAPSHOT_TRANSIENTS_PER_PLAYER)
            - networked)
    return max(0, min(snap, model_static_cap(map_text)))


# --- MD3 simplification ------------------------------------------------------
#
# Used when the baked props still exceed the .bsp budget after Source's own
# LODs and after the snapshot-limited model_static allowance is spent: the
# heaviest remaining models get reduced-detail copies instead of being dropped.
#
# Method: quadric-error edge collapse (Garland & Heckbert 1997), in its
# HALF-EDGE form and applied to POSITIONS rather than raw vertices.
#
# Why positions. An MD3 vertex is a (position, normal, UV) triple, so a hard
# edge, a UV seam or a double-sided card is stored as several vertices at one
# position. Measured on de_overpass_csgo: on the subway car 68% of shared
# positions differ only by normal (hard edges), on the balcony planter 98%
# (double-sided foliage cards), on the giant tree 100% by UV (seams). Treating
# those as untouchable left 69 of 313 models unreducible and cut the map's
# prop vertices by only 9%. Collapsing a whole position group at once fixes
# that without ever opening a crack:
#
#   Position P is merged onto an adjacent position R by mapping EVERY live
#   vertex at P onto the one vertex at R it shares an edge with. If any vertex
#   at P has zero or several such partners, or two vertices at P would land on
#   the same vertex at R, the collapse is refused. That one rule is what keeps
#   creases sharp, seams closed and texture islands separate - and it refuses
#   on its own to move a box corner or a seam junction.
#
# Because it is half-edge (R never moves) no position and no texture
# coordinate is ever invented: every output vertex is an exact copy of an
# input vertex, still on MD3's 1/64 grid.
#
# Boundary lines - open edges, seams, creases, the outline of a foliage card -
# get Garland & Heckbert's perpendicular "boundary plane" quadrics, weighted
# x BOUNDARY_WEIGHT, so a collapse that bends one is expensive while one that
# slides along a straight one is free. A collapse is also refused if it would
# flip or fold any triangle or break the link condition (which would make the
# surface non-manifold).
#
# Reduction is bounded by ERROR, not by a triangle count: collapses happen
# cheapest-first and stop the moment the next one would move the surface more
# than `max_error` (a fraction of the model's own size). A model is only made
# as simple as it can be without changing shape beyond that, and the budget
# loop measures what it actually got rather than assuming a ratio.

import heapq as _heapq

# Each step bounds how far the simplified surface may sit from the original:
# a fraction of the model's bounding-box diagonal, capped at an absolute
# distance in map units. The fraction alone is wrong for tall thin props - 1.5%
# of a 420-unit streetlight is 6 units, more than its pole's radius, and
# measured coverage fell 10% - so the cap keeps thin parts thin. One SOF2 map
# unit is about 2 cm at this pipeline's 1.25 scale.
# The last two are coarse - 3% and 6% of the model's size, at most 5 and 10
# units - and are only used where they cost less than dropping something
# (plan_prop_budget weighs both by surface movement per vertex saved).
MD3_SIMPLIFY_STEPS = ((0.004, 1.0), (0.015, 2.5), (0.03, 5.0), (0.06, 10.0))  # (fraction of diagonal, max units)
PHYSICS_DROP_WEIGHT = 0.5
# Props larger than this (map units across) are only ever simplified, and
# dropped only once nothing smaller is left to cut. The cost measure (size per
# vertex saved) is low for a large prop with many vertices, so on its own it
# dropped de_overpass_csgo's fountain (360 units), its swing seats, 11 big
# bushes and its railings on 2026-10-17, while clutter was still left in.
DROP_PROTECT_SIZE = 96.0
# Thin props - railings, fences, poles - are protected the same way at any
# size: they are part of how a route is read and walked along (KENNITH:
# "don't remove railings, rather screwdrivers/tools"), and their diagonal
# undersells them. Thin = the smallest extent under this fraction of the
# largest (a 64-unit railing section is 64 x 36 x 3), and the largest at
# least DROP_LINEAR_MIN_LENGTH.
DROP_LINEAR_RATIO = 0.12
# Props that ARE the map's architecture - window frames, arches, wall trims,
# doors - are never dropped at all, not even as a last resort: without them a
# wall has a hole in it (de_cbble_go, 2026-10-17: its window frames, arches
# and corner trims went once the budget ran out of everything else). When
# only these are left, the .bsp is allowed to come out over BSP_SIZE_TARGET
# and the report says by how much. Matched against the model path.
ARCHITECTURE_PROP_RE = re.compile(
    r'window|arch|trim|wall|door|frame|column|pillar|beam|stair|step|railing|fence|'
    r'gate|roof|gutter|ledge|curb|molding|moulding|facade|building|balcon|awning|'
    r'shutter|porch|port_|bridge|cliff', re.I)
DROP_LINEAR_MIN_LENGTH = 48.0
MD3_SURF_MAX_VERTS = 999            # engine per-surface caps (SHADER_MAX_VERTEXES 1000,
MD3_SURF_MAX_TRIS = 1999            # SHADER_MAX_INDEXES 6000) - as mdl_to_md3.py splits
MD3_SIMPLIFY_MIN_GAIN = 0.10          # a step must remove >= 10% of vertices to be used
MD3_SIMPLIFY_BOUNDARY_WEIGHT = 10.0
_MD3_SURF_HEADER = 108


def _md3_read(path):
    try:
        with open(path, 'rb') as f:
            d = f.read()
    except OSError:
        return None
    if len(d) < 108 or d[:4] != b'IDP3':
        return None
    return d


def _quadric_from_plane(a, b, c, dd, w=1.0):
    return [w * a * a, w * a * b, w * a * c, w * a * dd,
            w * b * b, w * b * c, w * b * dd,
            w * c * c, w * c * dd,
            w * dd * dd]


def _quadric_error(q, p):
    x, y, z = p
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x +
            q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y +
            q[7] * z * z + 2 * q[8] * z + q[9])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _simplify_surface(pos, tris, max_error):
    """Position-group half-edge quadric simplification of one MD3 surface.

    pos       : list of (x, y, z) integer MD3 coordinates, one per vertex
    tris      : list of [a, b, c] vertex indices
    max_error : stop once the cheapest collapse would exceed this (squared
                distance, MD3 units)
    Returns the surviving triangles as indices into the ORIGINAL vertex list.
    """
    nv = len(pos)
    tri = [list(t) for t in tris]
    alive = [True] * len(tri)

    # position groups
    pid_of = {}
    pid = [0] * nv
    ppos = []
    for i, p in enumerate(pos):
        k = pid_of.get(p)
        if k is None:
            k = pid_of[p] = len(ppos)
            ppos.append(p)
        pid[i] = k
    npos = len(ppos)
    verts_at = [set() for _ in range(npos)]
    for i in range(nv):
        verts_at[pid[i]].add(i)

    vtris = [set() for _ in range(nv)]
    for t, (a, b, c) in enumerate(tri):
        vtris[a].add(t)
        vtris[b].add(t)
        vtris[c].add(t)

    # Quadrics per position: every face plane once per position it touches,
    # plus boundary planes along every edge used by exactly one triangle in
    # VERTEX space (open edges, seams, creases and card outlines all look
    # like that). An edge used by three or more triangles is non-manifold:
    # its positions are locked.
    QP = [[0.0] * 10 for _ in range(npos)]
    locked = [False] * npos
    edge_use = {}
    for t, (a, b, c) in enumerate(tri):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_use.setdefault(key, []).append(t)
    face_n = []
    for t, (a, b, c) in enumerate(tri):
        n = _cross(_sub(pos[b], pos[a]), _sub(pos[c], pos[a]))
        ln = _norm(n)
        if ln <= 1e-9:
            face_n.append(None)
            continue
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        face_n.append(n)
        dd = -(n[0] * pos[a][0] + n[1] * pos[a][1] + n[2] * pos[a][2])
        k = _quadric_from_plane(n[0], n[1], n[2], dd)
        for P in {pid[a], pid[b], pid[c]}:
            q = QP[P]
            for i in range(10):
                q[i] += k[i]
    for (u, v), ts in edge_use.items():
        if len(ts) > 2:
            locked[pid[u]] = locked[pid[v]] = True
            continue
        if len(ts) != 1 or face_n[ts[0]] is None:
            continue
        e = _sub(pos[v], pos[u])
        bn = _cross(e, face_n[ts[0]])
        ln = _norm(bn)
        if ln <= 1e-9:
            continue
        bn = (bn[0] / ln, bn[1] / ln, bn[2] / ln)
        dd = -(bn[0] * pos[u][0] + bn[1] * pos[u][1] + bn[2] * pos[u][2])
        k = _quadric_from_plane(bn[0], bn[1], bn[2], dd, MD3_SIMPLIFY_BOUNDARY_WEIGHT)
        for P in (pid[u], pid[v]):
            q = QP[P]
            for i in range(10):
                q[i] += k[i]

    pdead = [False] * npos
    version = [0] * npos

    def tris_at(P):
        out = set()
        for v in verts_at[P]:
            out |= vtris[v]
        return out

    def pneighbours(P):
        out = set()
        for t in tris_at(P):
            for x in tri[t]:
                out.add(pid[x])
        out.discard(P)
        return out

    heap = []

    def push(P, R):
        if locked[P] or pdead[P] or pdead[R]:
            return
        qs = [QP[P][i] + QP[R][i] for i in range(10)]
        _heapq.heappush(heap, (_quadric_error(qs, ppos[R]), version[P], version[R], P, R))

    seen = set()
    for (u, v) in edge_use:
        P, R = pid[u], pid[v]
        if P != R and (P, R) not in seen:
            seen.add((P, R))
            seen.add((R, P))
            push(P, R)
            push(R, P)

    def plan(P, R):
        """The vertex map for collapsing P onto R, or None if not allowed."""
        tP = tris_at(P)
        shared = [t for t in tP if any(pid[x] == R for x in tri[t])]
        if not shared:
            return None
        opposite = set()
        for t in shared:
            for x in tri[t]:
                opposite.add(pid[x])
        opposite.discard(P)
        opposite.discard(R)
        if (pneighbours(P) & pneighbours(R)) != opposite:
            return None                                  # link condition
        vmap = {}
        targets = set()
        for p in verts_at[P]:
            if not vtris[p]:
                continue
            adj = {x for t in vtris[p] for x in tri[t] if pid[x] == R}
            if len(adj) != 1:
                return None                              # crease / seam would break
            r = adj.pop()
            if r in targets:
                return None                              # two wedges would merge
            targets.add(r)
            vmap[p] = r
        pr = ppos[R]
        sh = set(shared)
        for t in tP:
            if t in sh:
                continue
            a, b, c = tri[t]
            before = [ppos[pid[a]], ppos[pid[b]], ppos[pid[c]]]
            after = [pr if pid[x] == P else ppos[pid[x]] for x in (a, b, c)]
            n0 = _cross(_sub(before[1], before[0]), _sub(before[2], before[0]))
            n1 = _cross(_sub(after[1], after[0]), _sub(after[2], after[0]))
            l0, l1 = _norm(n0), _norm(n1)
            if l0 <= 1e-9 or l1 <= 1e-9:
                return None                              # would degenerate
            if (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]) / (l0 * l1) < 0.2:
                return None                              # would flip or fold
        return vmap, shared

    while heap:
        cost, vP, vR, P, R = _heapq.heappop(heap)
        if pdead[P] or pdead[R] or version[P] != vP or version[R] != vR:
            continue
        if cost > max_error:
            break                                        # everything left costs more
        res = plan(P, R)
        if res is None:
            continue
        vmap, shared = res
        for t in shared:
            alive[t] = False
            for x in tri[t]:
                vtris[x].discard(t)
        for p, r in vmap.items():
            for t in list(vtris[p]):
                tri[t] = [r if x == p else x for x in tri[t]]
                vtris[r].add(t)
            vtris[p].clear()
        verts_at[P].clear()
        pdead[P] = True
        q = QP[R]
        for i in range(10):
            q[i] += QP[P][i]
        version[R] += 1
        for X in pneighbours(R):
            push(X, R)
            push(R, X)
    return [tri[i] for i in range(len(tri)) if alive[i]]


def simplify_md3(src, dst, step):
    """Write a reduced copy of the .md3 `src` to `dst`. `step` is
    (fraction, cap): the surface may move by at most that fraction of the
    model's bounding-box diagonal, and never more than `cap` units of the
    .md3's OWN (native) scale. plan_prop_budget turns MD3_SIMPLIFY_STEPS' map
    units into these by dividing by the placements' modelscale. Returns
    (vertices, triangles) of the result, or None when the model can't be
    simplified safely (animated, malformed)."""
    d = _md3_read(src)
    if d is None:
        return None
    nfr, ntag, nsurf = struct.unpack_from('<3i', d, 76)
    ofr, otag, osurf = struct.unpack_from('<3i', d, 92)
    if nfr != 1 or nsurf < 1:
        return None                                      # animated: leave alone
    parsed = []
    so = osurf
    allpos_in = []
    for _ in range(nsurf):
        if so + _MD3_SURF_HEADER > len(d) or d[so:so + 4] != b'IDP3':
            return None
        sfr, nsh, nv, nt = struct.unpack_from('<4i', d, so + 72)
        otri, osh, ost, oxyz, oend = struct.unpack_from('<5i', d, so + 88)
        if sfr != 1:
            return None
        tris = [list(struct.unpack_from('<3i', d, so + otri + i * 12)) for i in range(nt)]
        if any(not (0 <= x < nv) for t in tris for x in t):
            return None
        xyzn = [d[so + oxyz + i * 8:so + oxyz + i * 8 + 8] for i in range(nv)]
        pos = [struct.unpack_from('<3h', xyzn[i]) for i in range(nv)]
        parsed.append((d[so:so + 72], d[so + osh:so + osh + nsh * 68], nsh, tris,
                       [d[so + ost + i * 8:so + ost + i * 8 + 8] for i in range(nv)], xyzn, pos))
        allpos_in.extend(pos)
        so += oend
    if not allpos_in:
        return None
    diag = _norm(_sub(tuple(max(p[k] for p in allpos_in) for k in range(3)),
                      tuple(min(p[k] for p in allpos_in) for k in range(3))))
    frac, cap_units = step
    max_error = min(frac * diag, cap_units * 64.0) ** 2   # MD3 stores 1/64 units

    # Surfaces sharing a shader are simplified TOGETHER. The converter splits
    # one material into chunks of under 1000 vertices (the engine's per-
    # surface cap), and those chunks meet along open seams; simplified one by
    # one, each side of a seam moved on its own and the seam cracked open -
    # de_overpass_csgo's fountain is one material in 8 chunks, and its coarse
    # copy came apart. Merged, a seam is ordinary interior mesh (the
    # simplifier collapses every vertex at a position together), and the
    # result is re-chunked under the same cap.
    merged = OrderedDict()
    for head, shaders, nsh, tris, st, xyzn, pos in parsed:
        g = merged.setdefault(bytes(shaders), [head, shaders, nsh, [], [], [], []])
        base = len(g[6])
        g[3].extend([[x + base for x in t] for t in tris])
        g[4].extend(st)
        g[5].extend(xyzn)
        g[6].extend(pos)
    chunks = []
    for head, shaders, nsh, tris, st, xyzn, pos in merged.values():
        kept = tris if len(tris) <= 4 else _simplify_surface(pos, tris, max_error)
        cur, cur_v = [], set()
        for t in kept:
            new_v = cur_v | set(t)
            if cur and (len(new_v) > MD3_SURF_MAX_VERTS or len(cur) >= MD3_SURF_MAX_TRIS):
                chunks.append((head, shaders, nsh, cur, st, xyzn, pos))
                cur, new_v = [], set(t)
            cur.append(t)
            cur_v = new_v
        if cur:
            chunks.append((head, shaders, nsh, cur, st, xyzn, pos))

    surfaces = []
    out_v = out_t = 0
    allpos = []
    for ci, (head, shaders, nsh, kept, st, xyzn, pos) in enumerate(chunks):
        head = bytearray(head)
        head[4:68] = ('surf{}'.format(ci).encode('ascii') + b'\0' * 64)[:64]
        used = sorted({x for t in kept for x in t})
        remap = {old: new for new, old in enumerate(used)}
        new_tris = [[remap[x] for x in t] for t in kept]
        allpos.extend(pos[i] for i in used)
        o_sh = _MD3_SURF_HEADER
        o_tri = o_sh + len(shaders)
        o_st = o_tri + 12 * len(new_tris)
        o_xyz = o_st + 8 * len(used)
        o_end = o_xyz + 8 * len(used)
        blob = bytearray(head)
        blob += struct.pack('<4i', 1, nsh, len(used), len(new_tris))
        blob += struct.pack('<5i', o_tri, o_sh, o_st, o_xyz, o_end)
        blob += shaders
        blob += b''.join(struct.pack('<3i', *t) for t in new_tris)
        blob += b''.join(st[i] for i in used)
        blob += b''.join(xyzn[i] for i in used)
        if len(blob) != o_end:
            return None
        surfaces.append(bytes(blob))
        out_v += len(used)
        out_t += len(new_tris)
    if not allpos:
        return None
    # Frame: same localOrigin; bounds and radius from the vertices that are
    # left (a subset of the originals, so the box can only shrink or stay).
    frame = bytearray(d[ofr:ofr + 56])
    lo = struct.unpack_from('<3f', frame, 24)
    mn = [min(p[k] for p in allpos) / 64.0 for k in range(3)]
    mx = [max(p[k] for p in allpos) / 64.0 for k in range(3)]
    r = max(math.sqrt(sum((p[k] / 64.0 - lo[k]) ** 2 for k in range(3))) for p in allpos)
    struct.pack_into('<3f', frame, 0, *mn)
    struct.pack_into('<3f', frame, 12, *mx)
    struct.pack_into('<f', frame, 36, r + 1.0 / 64.0)
    tags = d[otag:otag + ntag * 112] if ntag else b''
    header = bytearray(d[:108])
    o_frames = 108
    o_tags = o_frames + len(frame)
    o_surfs = o_tags + len(tags)
    body = b''.join(surfaces)
    struct.pack_into('<3i', header, 92, o_frames, o_tags, o_surfs)
    struct.pack_into('<i', header, 104, o_surfs + len(body))
    struct.pack_into('<i', header, 84, len(surfaces))
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(bytes(header) + bytes(frame) + tags + body)
    return out_v, out_t


def _md3_diag(path):
    """Bounding-box diagonal of an .md3's frame 0, in map units, or 0."""
    d = _md3_read(path)
    if d is None:
        return 0.0
    ofr = struct.unpack_from('<i', d, 92)[0]
    if ofr + 24 > len(d):
        return 0.0
    f = struct.unpack_from('<6f', d, ofr)
    return math.sqrt(sum((f[3 + k] - f[k]) ** 2 for k in range(3)))


def _md3_extents(path):
    """Sorted bounding-box extents (smallest first) of an .md3's frame 0."""
    d = _md3_read(path)
    if d is None:
        return (0.0, 0.0, 0.0)
    ofr = struct.unpack_from('<i', d, 92)[0]
    if ofr + 24 > len(d):
        return (0.0, 0.0, 0.0)
    f = struct.unpack_from('<6f', d, ofr)
    return tuple(sorted(abs(f[3 + k] - f[k]) for k in range(3)))


def _md3_path(md3_dir, model):
    rel = re.sub(r'^/?models/', '', model, flags=re.I)
    return os.path.join(md3_dir, *rel.split('/'))


_MODEL_STATIC_BLOCK_RE = re.compile(
    r'\{\n"classname" "model_static"\n((?:"[^"\n]*" "[^"\n]*"\n)*)\}')


def _unnetwork_excess_statics(map_text, md3_dir, model_scale):
    """--model-static-list networks small props during conversion, BEFORE the
    budget planner runs - by model identity, with no regard for the snapshot.
    On a map with enough unique models to trigger it (MODEL_STATIC_ONLY_ABOVE)
    that alone could exceed the 256-entity snapshot. When it does, every one
    is handed back as an ordinary misc_model placement so plan_prop_budget can
    decide networked / baked / simplified / dropped with all limits in view.

    The list exists to stay under classic q3map2's 512 model-file cap; the
    q3map2 this pipeline targets (netradiant-custom 2.5.17n) caches models in
    an unbounded std::map (model.cpp s_assModels), so baking them is safe.
    Returns (text, how many were handed back)."""
    statics = list(_MODEL_STATIC_BLOCK_RE.finditer(map_text))
    if not statics:
        return map_text, 0
    others = _networked_entity_count(map_text, exclude=('model_static',))
    room = (SNAPSHOT_ENTITY_LIMIT
            - SNAPSHOT_PLAYERS * (1 + SNAPSHOT_TRANSIENTS_PER_PLAYER) - others)
    if len(statics) <= room:
        return map_text, 0
    out, pos, n = [], 0, 0
    for m in statics:
        body = m.group(1)
        # "<base>_static.md3" -> "<base>.md3". Greedy, so a Source model that is
        # itself named *_static (wood_pallet_01_static_static.md3) loses only
        # the suffix this pipeline added. Recentred *_static_c copies are not
        # made by the list and are left alone.
        mm = re.search(r'"model" "([^"]*)_static\.md3"\n', body)
        if not mm or not os.path.isfile(_md3_path(md3_dir, mm.group(1) + '.md3')):
            continue
        body = body.replace(mm.group(0), '"model" "{}.md3"\n'.format(mm.group(1)))
        body = re.sub(r'"modelscale" "[^"]*"\n', '', body)
        body += '"modelscale" "{:.6g}"\n'.format(model_scale)
        out.append(map_text[pos:m.start()])
        out.append('{\n"classname" "misc_model"\n' + body + '}')
        pos = m.end()
        n += 1
    out.append(map_text[pos:])
    return ''.join(out), n


# ------------------------------------------------------- props and water fog --
#
# Source fogs only what is seen THROUGH a water surface. idTech3 fogs a whole
# surface by its depth under the fog plane once its bounds touch the fog
# volume - q3map2 decides that at compile time for baked models (fog.cpp:
# triangle surfaces are fogged whole), the renderer at run time for networked
# ones (by entity bounds). de_overpass_csgo's fountain: its basin wall stands
# around the water, so its outside below the water line drew in the brown fog
# colour. So a prop that is mostly OUT of the water is kept from the fog:
#   - never networked (run-time fog cannot be refused per prop), and
#   - baked with its shaders remapped (misc_model "_remap") to copies carrying
#     q3map_noFog, which vmt2shader.py makes from <map>_nofog_shaders.json.
# A prop mostly IN the water (rubbish on a canal bed) keeps its fog, as the
# water in Source would tint it.
PROP_FOG_INSIDE_FRACTION = 0.5
NOFOG_SHADER_TARGETS = {}      # md3 shader name -> remap target (vmt2shader adds <target>_bsp)


def _md3_shader_names(path):
    d = _md3_read(path)
    if d is None:
        return []
    nsurf = struct.unpack_from('<i', d, 84)[0]
    so = struct.unpack_from('<i', d, 100)[0]
    names = []
    for _ in range(nsurf):
        if so + 108 > len(d) or d[so:so + 4] != b'IDP3':
            break
        nsh = struct.unpack_from('<i', d, so + 76)[0]
        osh = struct.unpack_from('<i', d, so + 92)[0]
        for k in range(nsh):
            raw = d[so + osh + k * 68:so + osh + k * 68 + 64]
            nm = raw.split(b'\0')[0].decode('latin-1').strip()
            if nm and nm not in names:
                names.append(nm)
        so += struct.unpack_from('<i', d, so + 104)[0]
    return names


def _placement_box(kv, md3_dir, scale_default):
    """World-space bounding box of one model placement (map units), or None."""
    d = _md3_read(_md3_path(md3_dir, kv.get('model', '')))
    if d is None:
        return None
    ofr = struct.unpack_from('<i', d, 92)[0]
    f = struct.unpack_from('<6f', d, ofr)
    try:
        sc = float(kv.get('modelscale', scale_default) or scale_default)
        org = [float(x) for x in kv['origin'].split()]
        if 'angles' in kv:
            ang = [float(x) for x in kv['angles'].split()]
        else:
            ang = [0.0, float(kv.get('angle', '0') or 0), 0.0]
    except (KeyError, ValueError):
        return None
    ax = _q3_axis(ang)
    pts = []
    for cx in (f[0], f[3]):
        for cy in (f[1], f[4]):
            for cz in (f[2], f[5]):
                pts.append([org[k] + sc * (cx * ax[0][k] + cy * ax[1][k] + cz * ax[2][k])
                            for k in range(3)])
    return ([min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)])


def _fog_inside_fraction(box):
    """0 when the box touches no fog volume, else the largest share of its
    volume inside one (at least a sliver, so a touching box is never 0)."""
    if box is None or not FOG_BOXES:
        return 0.0
    mn, mx = box
    vol = 1.0
    for k in range(3):
        vol *= max(mx[k] - mn[k], 1.0)
    best = 0.0
    for bmn, bmx in FOG_BOXES:
        if not _box_overlap(mn, mx, bmn, bmx):
            continue
        inter = 1.0
        for k in range(3):
            inter *= max(min(mx[k], bmx[k]) - max(mn[k], bmn[k]), 1e-3)
        best = max(best, inter / vol, 1e-6)
    return best


def unnetwork_props_in_fog(map_text, md3_dir, model_scale):
    """model_statics that touch a fog volume become baked misc_models (see the
    block comment above). Run before plan_prop_budget. Returns (text, n)."""
    if not FOG_BOXES:
        return map_text, 0
    out, pos, n = [], 0, 0
    for m in _MODEL_STATIC_BLOCK_RE.finditer(map_text):
        body = m.group(1)
        kv = dict(re.findall(r'"([^"\n]*)" "([^"\n]*)"', body))
        if not _fog_inside_fraction(_placement_box(kv, md3_dir, 1.0)):
            continue
        mm = re.search(r'"model" "([^"]*)_static(?:_c)?\.md3"\n', body)
        if not mm or not os.path.isfile(_md3_path(md3_dir, mm.group(1) + '.md3')):
            continue
        if '_static_c.md3' in mm.group(0):
            continue              # recentred copy: its origin was moved; leave it
        body = body.replace(mm.group(0), '"model" "{}.md3"\n'.format(mm.group(1)))
        body = re.sub(r'"modelscale" "[^"]*"\n', '', body)
        body += '"modelscale" "{:.6g}"\n'.format(model_scale)
        out.append(map_text[pos:m.start()])
        out.append('{\n"classname" "misc_model"\n' + body + '}')
        pos = m.end()
        n += 1
    out.append(map_text[pos:])
    return ''.join(out), n


def keep_baked_props_out_of_fog(map_text, md3_dir, model_scale):
    """Add _remap keys sending every shader of a baked prop that is mostly
    out of the water to its q3map_noFog copy. Run after plan_prop_budget (the
    model may have become a LOD or simplified copy). Returns (text, n)."""
    NOFOG_SHADER_TARGETS.clear()
    if not FOG_BOXES:
        return map_text, 0
    out, pos, n = [], 0, 0
    for m in _MISC_MODEL_BLOCK_RE.finditer(map_text):
        body = m.group(1)
        kv = dict(re.findall(r'"([^"\n]*)" "([^"\n]*)"', body))
        model = kv.get('model', '')
        if not model.lower().endswith('.md3') or 'mapobjects/terrain/' in model.lower():
            continue
        frac = _fog_inside_fraction(_placement_box(kv, md3_dir, model_scale))
        if not frac or frac >= PROP_FOG_INSIDE_FRACTION:
            continue
        lines = []
        for k, nm in enumerate(_md3_shader_names(_md3_path(md3_dir, model))):
            base = re.sub(r'\.(tga|jpg|jpeg|png)$', '', nm, flags=re.I)
            target = short_qpath(base + '_nf')
            NOFOG_SHADER_TARGETS[nm] = target
            lines.append('"_remap{}" "{};{}"\n'.format(k + 1, nm, target))
        if not lines:
            continue
        out.append(map_text[pos:m.start()])
        out.append('{\n"classname" "misc_model"\n' + body + ''.join(lines) + '}')
        pos = m.end()
        n += 1
    out.append(map_text[pos:])
    return ''.join(out), n


SRC_CLASS_KEY = '_src_class'
SRC_ID_KEY = '_src_id'
SRC_SOLID_KEY = '_src_solid'
_SRC_KEY_LINE_RE = re.compile(r'^"(?:_src_class|_src_id|_src_solid)" "[^"\n]*"\n', re.M)

# Source classes whose props move or break in Source: SOF2 has no physics
# props, so these become fixed scenery. When something must go, they go first.
PHYSICS_PROP_CLASSES = {'prop_physics', 'prop_physics_multiplayer', 'prop_physics_override',
                        'prop_physics_respawnable', 'prop_ragdoll'}
DROPPED_PROP_IDS = []      # VMF entity ids plan_prop_budget dropped (for phy2clip)


def strip_source_prop_keys(map_text):
    return _SRC_KEY_LINE_RE.sub('', map_text)


def drop_tier(kv):
    """0 for a physics prop (PHYSICS_PROP_CLASSES - in Source it moves,
    breaks or pops, which SOF2 cannot do), else 1. plan_prop_budget weighs a
    physics prop's drop at PHYSICS_DROP_WEIGHT of its size."""
    cls = kv.get(SRC_CLASS_KEY, '')
    if cls in PHYSICS_PROP_CLASSES:
        return 0
    return 1


def plan_prop_budget(map_text, md3_dir, budget, model_scale, scale, offset, entity_cap=None):
    """Fit the map's props into what retail SOF2 can actually show.

    Two hard limits apply, and a prop has to satisfy one of them:
      - BAKED (misc_model): q3map2 copies it into the .bsp once per placement.
        The .bsp is loaded as one allocation, so baked vertices are capped at
        `budget` (PROP_VERTEX_BUDGET, derived from a measured size target).
      - NETWORKED (model_static): stored once, but every one is a server entity
        that shares the 256-entity snapshot (see networked_prop_cap).

    Stages, in the order that loses the least:
      1. Source's own lower LODs (<name>_lodN.md3), heaviest model first.
      2. Network the placements that save the most baked vertices per entity -
         the heaviest per placement - up to the snapshot allowance. Those are
         lit from the light grid (darker than baked props), so this only
         happens while the bake is over budget.
      3. Redo the LODs from LOD0 for what stays baked.
      4. Simplify: reduced-detail copies (simplify_md3, MD3_SIMPLIFY_STEPS) of
         the models still over, heaviest total first, measured not assumed.
      5. Drop: only if still over, the SMALLEST placements first, every one
         named in the report. 3D-skybox placements are never dropped.

    Networked props are written at the END of the entity list, largest first.
    The snapshot keeps the lowest entity numbers, so if a server has more
    players than SNAPSHOT_PLAYERS, the props that flicker are the smallest,
    and the map's doors and movers (which come earlier) are never the ones cut.

    Returns the rewritten map text and a list of report lines."""
    DROPPED_PROP_IDS[:] = []
    map_text, handed_back = _unnetwork_excess_statics(map_text, md3_dir, model_scale)
    blocks = list(_MISC_MODEL_BLOCK_RE.finditer(map_text))
    if not blocks:
        return map_text, []
    places = []                                  # [match, kv, model]
    for m in blocks:
        kv = dict(re.findall(r'"([^"\n]*)" "([^"\n]*)"', m.group(1)))
        model = kv.get('model', '').replace('\\', '/')
        if not model.lower().endswith('.md3') or 'mapobjects/terrain/' in model.lower():
            continue
        places.append((m, kv, model))
    chains = {mdl: _md3_lod_chain(md3_dir, mdl) for mdl in {p[2] for p in places}}
    chains = {mdl: c for mdl, c in chains.items() if c}
    places = [p for p in places if p[2] in chains]
    if not places:
        return map_text, []
    n_source = {mdl: len(c) for mdl, c in chains.items()}
    by_model = {}
    for idx, p in enumerate(places):
        by_model.setdefault(p[2], []).append(idx)
    lod = {mdl: 0 for mdl in chains}
    fate = ['bake'] * len(places)                # 'bake' | 'net' | 'drop'
    net_kind = {}                                # idx -> ('static', None) | ('recentred', (model, origin))

    def n_baked(mdl):
        return sum(1 for idx in by_model[mdl] if fate[idx] == 'bake')

    def cost(mdl):
        return n_baked(mdl) * chains[mdl][lod[mdl]][1]

    def total():
        return sum(cost(mdl) for mdl in chains)

    def step_rounds(limit):
        """One chain step per model per round, heaviest total first, until
        under budget or nothing can step. limit(mdl) = chain entries usable."""
        progressed = True
        while total() > budget and progressed:
            progressed = False
            for mdl in sorted(chains, key=cost, reverse=True):
                if not cost(mdl):
                    continue
                if lod[mdl] + 1 < limit(mdl):
                    lod[mdl] += 1
                    progressed = True
                    if total() <= budget:
                        return

    start = total()
    report = []
    if budget <= 0:
        # --prop-vertex-budget 0 (the GUI's "Props at original quality"):
        # nothing simplified, nothing dropped - only say what it costs.
        est = start * BSP_BYTES_PER_BAKED_VERTEX / 1048576.0
        report.append(
            "Prop budget OFF (original quality): {:,} placement(s), {:,} baked vertices at full "
            "detail, ~{:.1f} MB of the .bsp from props alone (the size target is {:.0f} MB "
            "for the whole .bsp). SOF2 reads the .bsp into one block of memory; if the "
            "finished map fails with Z_Malloc(), reduce props by hand or convert without "
            "this option.".format(len(places), start, est, BSP_SIZE_TARGET / 1048576.0))
        return map_text, report
    if start <= budget:
        return map_text, report

    # 1. Source's own LODs
    step_rounds(lambda mdl: n_source[mdl])

    # 2. network the heaviest placements, up to the snapshot allowance
    ent_cap = networked_prop_cap(map_text) if entity_cap is None else entity_cap
    ent_left = ent_cap
    not_networkable = Counter()                  # reason -> placements
    if total() > budget:
        offsets_keys = MODEL_ORIGIN_OFFSETS_KEYS()
        for mdl in sorted(chains, key=lambda m_: (chains[m_][0][1], len(by_model[m_])), reverse=True):
            if total() <= budget:
                break
            static_path = _md3_path(md3_dir, mdl[:-4] + '_static.md3')
            if not os.path.isfile(static_path) or mdl.lower() in offsets_keys:
                not_networkable['no _static model'] += n_baked(mdl)
                continue
            for idx in by_model[mdl]:
                if total() <= budget:
                    break
                if fate[idx] != 'bake':
                    continue
                kv = places[idx][1]
                if _fog_inside_fraction(_placement_box(kv, md3_dir, model_scale)):
                    not_networkable['touches water (fogged whole when networked)'] += 1
                    continue
                why = _static_block_reason(kv, model_scale, scale, offset)
                if why == 'origin in wall':
                    rec = _recentred_static(kv, mdl, md3_dir, scale, offset)
                    if rec is None:
                        not_networkable['origin in wall'] += 1
                        continue
                    kind = ('recentred', rec)
                elif why is None:
                    kind = ('static', None)
                else:
                    not_networkable[why] += 1
                    continue
                if ent_left <= 0:
                    not_networkable['snapshot allowance spent'] += 1
                    continue
                fate[idx] = 'net'
                net_kind[idx] = kind
                ent_left -= 1

    # 3. redo the LODs from scratch for what stays baked
    for mdl in lod:
        lod[mdl] = 0
    step_rounds(lambda mdl: n_source[mdl])

    # 4 + 5. Simplify or drop, whichever loses the least - one measure for
    #    both: how far the surface moves, in map units, per vertex saved.
    #    A simplified copy moves it by at most its step's error bound
    #    (MD3_SIMPLIFY_STEPS); a dropped placement loses all of it, so its
    #    "error" is its whole size. A 15-unit wrench of 400 vertices goes
    #    before a 70-unit railing of 300, and a coarser copy of a 300-unit
    #    bush before either, if that is what costs least on screen.
    #    Physics props (PHYSICS_PROP_CLASSES) count at PHYSICS_DROP_WEIGHT of
    #    their size: in Source they get knocked away, broken or popped, and
    #    SOF2 cannot do any of that, so losing one costs less. 3D-skybox
    #    placements are never dropped (they are the skyline).
    size = {}
    diag_cache = {}
    for idx, (_m, kv, mdl) in enumerate(places):
        try:
            sc = abs(float(kv.get('modelscale', model_scale) or model_scale))
        except ValueError:
            sc = model_scale
        if mdl not in diag_cache:
            diag_cache[mdl] = _md3_diag(_md3_path(md3_dir, mdl))
        size[idx] = diag_cache[mdl] * sc
    ext_cache = {}
    protected = set()
    for idx, (_m, kv, mdl) in enumerate(places):
        if mdl not in ext_cache:
            ext_cache[mdl] = _md3_extents(_md3_path(md3_dir, mdl))
        e = ext_cache[mdl]
        sc = size[idx] / diag_cache[mdl] if diag_cache.get(mdl) else model_scale
        if size[idx] > DROP_PROTECT_SIZE or (
                e[2] * sc >= DROP_LINEAR_MIN_LENGTH and e[0] < DROP_LINEAR_RATIO * e[2]):
            protected.add(idx)
    tier = {i_: drop_tier(places[i_][1]) for i_ in range(len(places))}
    weight = {i_: (PHYSICS_DROP_WEIGHT if tier[i_] == 0 else 1.0) for i_ in range(len(places))}
    droppable = {i_ for i_ in range(len(places))
                 if places[i_][1].get('spawnflags', '0') in ('0', '')}

    simplified = {}
    next_step = {}
    err = {}                      # (model, chain index) -> map units the surface may move

    def smax_of(mdl):
        # The step's absolute cap is in MAP units, but the .md3 is at native
        # scale and q3map2 multiplies it by each placement's modelscale. Divide
        # by the largest one baked, so no placement moves more than the cap -
        # without this a 3D-skybox prop moved into the world at 16x (modelscale
        # 20 on de_overpass_csgo's crane) would have been allowed 50 units.
        scales = []
        for idx in by_model[mdl]:
            if fate[idx] != 'bake':
                continue
            try:
                scales.append(abs(float(places[idx][1].get('modelscale', model_scale) or model_scale)))
            except ValueError:
                scales.append(model_scale)
        return max(scales) if scales else model_scale

    def extend_one(mdl):
        """Add the next simplified copy to mdl's chain. False when none is left."""
        simplified.setdefault(mdl, [])
        base_model = chains[mdl][n_source[mdl] - 1][0]
        src = _md3_path(md3_dir, base_model)
        while next_step.get(mdl, 0) < len(MD3_SIMPLIFY_STEPS):
            k = next_step.get(mdl, 0)
            next_step[mdl] = k + 1
            frac, cap = MD3_SIMPLIFY_STEPS[k]
            smax = smax_of(mdl)
            new_model = base_model[:-4] + '_simp{}.md3'.format(k + 1)
            dst = _md3_path(md3_dir, new_model)
            res = simplify_md3(src, dst, (frac, cap / smax if smax > 0 else cap))
            if res is None:
                next_step[mdl] = len(MD3_SIMPLIFY_STEPS)
                return False
            if res[0] > chains[mdl][-1][1] * (1.0 - MD3_SIMPLIFY_MIN_GAIN):
                try:
                    os.remove(dst)              # not worth a step of its own
                except OSError:
                    pass
                continue
            chains[mdl].append((new_model, res[0]))
            err[(mdl, len(chains[mdl]) - 1)] = min(frac * diag_cache.get(mdl) * smax
                                                   if diag_cache.get(mdl) else cap, cap)
            simplified[mdl].append(new_model)
            return True
        return False

    for mdl in chains:
        diag_cache.setdefault(mdl, _md3_diag(_md3_path(md3_dir, mdl)))

    def err_at(mdl, k):
        return err.get((mdl, k), 0.0)  # Source's own LODs: the model's author's call

    dropped = []
    while total() > budget:
        best = None
        for mdl in chains:
            if not n_baked(mdl):
                continue
            k = lod[mdl]
            if k + 1 >= len(chains[mdl]) and not extend_one(mdl):
                continue
            dv = chains[mdl][k][1] - chains[mdl][k + 1][1]
            if dv <= 0:
                continue
            c = (err_at(mdl, k + 1) - err_at(mdl, k)) / float(dv)
            if best is None or c < best[0]:
                best = (c, 'step', mdl)
        for idx in droppable:
            if fate[idx] != 'bake' or idx in protected or ARCHITECTURE_PROP_RE.search(places[idx][2]):
                continue
            mdl = places[idx][2]
            v = chains[mdl][lod[mdl]][1]
            if v <= 0:
                continue
            c = (size[idx] * weight[idx] - err_at(mdl, lod[mdl])) / float(v)
            if best is None or c < best[0]:
                best = (c, 'drop', idx)
        if best is None:
            # Only protected props are left to cut: the smallest goes first,
            # so the fountain is the very last thing to go. Architecture is
            # never cut (ARCHITECTURE_PROP_RE) - the budget is overshot.
            left = [idx for idx in protected if idx in droppable and fate[idx] == 'bake'
                    and not ARCHITECTURE_PROP_RE.search(places[idx][2])
                    and chains[places[idx][2]][lod[places[idx][2]]][1] > 0]
            if left:
                best = (0.0, 'drop', min(left, key=lambda i_: size[i_]))
        if best is None:
            break
        if best[1] == 'step':
            lod[best[2]] += 1
        else:
            fate[best[2]] = 'drop'
            dropped.append(best[2])
    DROPPED_PROP_IDS[:] = sorted({places[i_][1][SRC_ID_KEY] for i_ in dropped
                                  if places[i_][1].get(SRC_ID_KEY)})

    # Rewrite. Networked blocks are collected and appended at the end, largest
    # first (see the docstring for why).
    repl = {}
    statics = []
    for idx, (m, kv, mdl) in enumerate(places):
        body = m.group(1)
        if fate[idx] == 'drop':
            repl[m.start()] = (m.end(), '')
        elif fate[idx] == 'net':
            kind, rec = net_kind[idx]
            if kind == 'recentred':
                rmodel, rorigin = rec
                new = re.sub(r'"model" "[^"]*"', '"model" "{}"'.format(rmodel), body)
                new = re.sub(r'"(modelscale|spawnflags)" "[^"]*"\n', '', new)
                new = re.sub(r'"origin" "[^"]*"',
                             '"origin" "{:.6g} {:.6g} {:.6g}"'.format(*rorigin), new)
                smodel = rmodel
            else:
                smodel = mdl[:-4] + '_static.md3'
                new = re.sub(r'"model" "[^"]*"', '"model" "{}"'.format(smodel), body)
                new = re.sub(r'"(modelscale|spawnflags)" "[^"]*"\n', '', new)
                o_src = tuple((float(c) + offset[k]) / scale
                              for k, c in enumerate(kv['origin'].split()))
                pen = world_brush_penetration(o_src)
                if pen is not None:
                    o = add(o_src, scale_vec(pen[1], pen[0] + 0.25))
                    no = tuple(o[k] * scale - offset[k] for k in range(3))
                    new = re.sub(r'"origin" "[^"]*"',
                                 '"origin" "{:.6g} {:.6g} {:.6g}"'.format(*no), new)
            statics.append((_md3_diag(_md3_path(md3_dir, smodel)), idx,
                            '{\n"classname" "model_static"\n' + new + '}'))
            repl[m.start()] = (m.end(), '')
        elif lod[mdl]:
            new = body.replace('"model" "{}"'.format(mdl),
                               '"model" "{}"'.format(chains[mdl][lod[mdl]][0]))
            repl[m.start()] = (m.end(), '{\n"classname" "misc_model"\n' + new + '}')
    out, pos = [], 0
    for st in sorted(repl):
        end, txt = repl[st]
        out.append(map_text[pos:st])
        out.append(txt)
        pos = end
    out.append(map_text[pos:])
    text = ''.join(out)
    if statics:
        statics.sort(key=lambda s: (-s[0], s[1]))
        if not text.endswith('\n'):
            text += '\n'
        text += '\n'.join(s[2] for s in statics) + '\n'

    # Report
    final = total()
    if budget and final > budget:
        report.append(
            "OVER BUDGET: {:,} baked vertices against {:,} (~{:.1f} MB more .bsp than the {:.0f} "
            "MB target). Everything that may be dropped was; what is left is architecture "
            "(ARCHITECTURE_PROP_RE - windows, arches, trims, doors...) and is kept so walls have "
            "no holes. If the map then fails to load with Z_Malloc(), the player's "
            "com_hunkMegs is too large for it (see BSP_SIZE_TARGET).".format(
                final, budget, (final - budget) * BSP_BYTES_PER_BAKED_VERTEX / 1048576.0,
                BSP_SIZE_TARGET / 1048576.0))
    n_net = sum(1 for f in fate if f == 'net')
    report.append("Prop budget: {:,} placement(s) of {} model(s), {:,} baked vertices at full "
                  "detail against a budget of {:,} ({:.0f} MB .bsp target, {:.1f} bytes per "
                  "vertex measured).".format(len(places), len(chains), start, budget,
                                             BSP_SIZE_TARGET / 1048576.0,
                                             BSP_BYTES_PER_BAKED_VERTEX))
    if handed_back:
        report.append("  {} prop(s) networked by --model-static-list would on their own overflow "
                      "the snapshot, so they were planned like every other placement.".format(
                          handed_back))
    stepped = sorted((mdl for mdl in chains
                      if 0 < lod[mdl] < n_source[mdl] and n_baked(mdl)),
                     key=lambda mdl_: -cost(mdl_))
    if stepped:
        report.append("  Baked from Source's own lower LODs ({} model(s)):".format(len(stepped)))
        for mdl in stepped:
            report.append("    LOD{}  {:>6,} -> {:>6,} verts x {:>3}  {}".format(
                lod[mdl], chains[mdl][0][1], chains[mdl][lod[mdl]][1], n_baked(mdl), mdl))
    if n_net or ent_cap == 0:
        report.append("  Networked (model_static): {} of an allowance of {} - retail SOF2 truncates "
                      "every snapshot to {} entities, kept for {} players plus {} transient each "
                      "and the map's own networked entities. Stored once, lit from the light "
                      "grid (darker than baked props).".format(
                          n_net, ent_cap, SNAPSHOT_ENTITY_LIMIT, SNAPSHOT_PLAYERS,
                          SNAPSHOT_TRANSIENTS_PER_PLAYER))
        per = Counter(places[idx][2] for idx in range(len(places)) if fate[idx] == 'net')
        for mdl, n in sorted(per.items(), key=lambda kv_: -chains[kv_[0]][0][1] * kv_[1])[:15]:
            report.append("    {:>6,} verts x {:>3}  {}".format(chains[mdl][0][1], n, mdl))
        if len(per) > 15:
            report.append("    ... +{} more model(s)".format(len(per) - 15))
    if not_networkable:
        report.append("  Could not be networked (stayed baked): " + ", ".join(
            "{} {}".format(n, r) for r, n in not_networkable.most_common()))
    simp_used = sorted((mdl for mdl in chains if lod[mdl] >= n_source[mdl] and n_baked(mdl)),
                       key=lambda mdl_: -cost(mdl_))
    if simp_used:
        report.append("  Simplified (reduced-detail copies, shape kept within the error bound; "
                      "seams, creases and outlines preserved) - {} model(s):".format(len(simp_used)))
        for mdl in simp_used:
            report.append("    {:>6,} -> {:>6,} verts x {:>3}  {}".format(
                chains[mdl][n_source[mdl] - 1][1], chains[mdl][lod[mdl]][1], n_baked(mdl),
                chains[mdl][lod[mdl]][0]))
    if dropped:
        tier_names = {0: 'physics props (move/break in Source, fixed scenery here)',
                      1: 'scenery (collision dropped with it - phy2clip --skip-ids)'}
        report.append("  DROPPED - where that cost less on screen than simplifying further: {} "
                      "placement(s):".format(
            len(dropped)))
        for t_ in (0, 1):
            ids_ = [idx for idx in dropped if tier[idx] == t_]
            if not ids_:
                continue
            per = Counter(places[idx][2] for idx in ids_)
            report.append("   {} - {} placement(s) of {} model(s):".format(
                tier_names[t_], len(ids_), len(per)))
            across = {}
            for idx in ids_:
                across[places[idx][2]] = max(across.get(places[idx][2], 0.0), size[idx])
            for mdl, n in sorted(per.items(), key=lambda kv_: across[kv_[0]]):
                report.append("    {:>3} x  {}  ({:.0f} units across)".format(n, mdl, across[mdl]))
    report.append("  Result: {:,} baked vertices (~{:.1f} MB of the .bsp), {} networked, {} "
                  "dropped.".format(final, final * BSP_BYTES_PER_BAKED_VERTEX / 1048576.0,
                                    n_net, len(dropped)))
    if final > budget:
        report.append("  Still {:,} over budget with every stage used (all remaining placements "
                      "are in the 3D skybox). The .bsp may be too big for SOF2 to load.".format(
                          final - budget))
    return text, report



def _q3_axis(angles):
    """idTech3 AnglesToAxis for "pitch yaw roll": the model's local x, y, z
    axes in world space (forward, left, up)."""
    p, y, r = (math.radians(a) for a in angles)
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, -sp)
    right = (-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, -sr * cp)
    up = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
    return forward, (-right[0], -right[1], -right[2]), up


_RECENTRED_CACHE = {}


def _md3_write_recentred(src, dst):
    """Copy an MD3 with every frame shifted so its frame-0 bounding-box centre
    is the model origin. The shift is a whole number of MD3 units (1/64), so
    the copy is exact. Returns the shift c in model units, or None."""
    try:
        d = bytearray(open(src, 'rb').read())
    except OSError:
        return None
    if d[:4] != b'IDP3':
        return None
    nfr, ntag, nsurf = struct.unpack_from('<3i', d, 76)
    ofr, otag, osurf = struct.unpack_from('<3i', d, 92)
    if nfr < 1:
        return None
    mins = struct.unpack_from('<3f', d, ofr)
    maxs = struct.unpack_from('<3f', d, ofr + 12)
    ci = [int(round((mins[k] + maxs[k]) * 0.5 * 64)) for k in range(3)]
    c = tuple(v / 64.0 for v in ci)
    # Shift each frame's box and cull-sphere centre. The radius is deliberately
    # NOT written here: it has to be measured from the *shifted* localOrigin,
    # and the only truthful source for that is the shifted vertex data that the
    # surface loop below rewrites. Measuring it from (0,0,0) - as this used to -
    # leaves the sphere centred at -c with a radius taken from the origin, so on
    # any model whose geometry was offset from its origin (every prop whose
    # Source origin sits at its base: a fence post gives c = (0,0,80)) the
    # sphere ends up covering z -161..+1 while the geometry spans -80..+80.
    # Measured on de_overpass, 22 of 29 recentred copies were up to 2.00x short.
    #
    # Scope, so nobody chases the wrong bug with this: frame->radius is only
    # read by R_CullModel's sphere test (rd-vanilla/tr_mesh.cpp), which sits
    # behind `if ( !ent->e.nonNormalizedAxes )`. CG_DrawMiscStaticModels sets
    # nonNormalizedAxes = qtrue, so world props never reach it - they cull on
    # R_CullLocalBox( frame->bounds ) instead, and those bounds were already
    # correct. And in RETAIL SOF2 that client-side draw loop does not exist at
    # all: a model_static is a networked server entity (see networked_prop_cap).
    # So this fix does NOT explain props vanishing in game - the 256-entity
    # snapshot limit does. It matters for anything rendered with normalized
    # axes, and a wrong field is worth not shipping either way.
    centres = []
    for f in range(nfr):
        o = ofr + f * 56
        mn = struct.unpack_from('<3f', d, o)
        mx = struct.unpack_from('<3f', d, o + 12)
        lo = struct.unpack_from('<3f', d, o + 24)
        mn2 = [mn[k] - c[k] for k in range(3)]
        mx2 = [mx[k] - c[k] for k in range(3)]
        lo2 = [lo[k] - c[k] for k in range(3)]
        struct.pack_into('<3f', d, o, *mn2)
        struct.pack_into('<3f', d, o + 12, *mx2)
        struct.pack_into('<3f', d, o + 24, *lo2)
        centres.append(lo2)
    far = [0.0] * nfr
    for t in range(ntag * nfr):
        o = otag + t * 112 + 64
        org = struct.unpack_from('<3f', d, o)
        struct.pack_into('<3f', d, o, *[org[k] - c[k] for k in range(3)])
    so = osurf
    for _ in range(nsurf):
        if d[so:so + 4] != b'IDP3':
            return None
        sfr, nsh, nv = struct.unpack_from('<3i', d, so + 72)
        oxyz, oend = struct.unpack_from('<2i', d, so + 100)
        for v in range(sfr * nv):
            o = so + oxyz + v * 8
            x, y, z = struct.unpack_from('<3h', d, o)
            nx, ny, nz = x - ci[0], y - ci[1], z - ci[2]
            if not all(-32768 <= q <= 32767 for q in (nx, ny, nz)):
                return None
            struct.pack_into('<3h', d, o, nx, ny, nz)
            # farthest vertex from this frame's cull-sphere centre, measured on
            # the quantised values that actually land in the file
            f = v // nv if nv else 0
            if f < nfr:
                cen = centres[f]
                dx = nx / 64.0 - cen[0]
                dy = ny / 64.0 - cen[1]
                dz = nz / 64.0 - cen[2]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > far[f]:
                    far[f] = r2
        so += oend
    # One 1/64 unit of slack so a vertex exactly on the sphere is inside it.
    # A frame with no vertices falls back to the box corner, which can only be
    # an over-estimate - never a sphere too small to contain the geometry.
    for f in range(nfr):
        o = ofr + f * 56
        if far[f] > 0.0:
            r = math.sqrt(far[f])
        else:
            mn2 = struct.unpack_from('<3f', d, o)
            mx2 = struct.unpack_from('<3f', d, o + 12)
            cen = struct.unpack_from('<3f', d, o + 24)
            r = max(math.sqrt(sum((p[k] - cen[k]) ** 2 for k in range(3)))
                    for p in ((mn2[0], mn2[1], mn2[2]), (mn2[0], mn2[1], mx2[2]),
                              (mn2[0], mx2[1], mn2[2]), (mn2[0], mx2[1], mx2[2]),
                              (mx2[0], mn2[1], mn2[2]), (mx2[0], mn2[1], mx2[2]),
                              (mx2[0], mx2[1], mn2[2]), (mx2[0], mx2[1], mx2[2])))
        struct.pack_into('<f', d, o + 36, r + (1.0 / 64.0))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(d)
    return c


def _recentred_static(kv, mdl, md3_dir, scale, offset):
    """(model path, new origin) for a model_static placement whose origin is
    moved to the model's bounding-box centre, or None when there is no _static
    model to copy or the centre is itself inside a wall."""
    key = mdl.lower()
    if key not in _RECENTRED_CACHE:
        rel = re.sub(r'^/?models/', '', mdl[:-4], flags=re.I)
        src = os.path.join(md3_dir, *(rel + '_static.md3').split('/'))
        dst = os.path.join(md3_dir, *(rel + '_static_c.md3').split('/'))
        c = _md3_write_recentred(src, dst) if os.path.isfile(src) else None
        _RECENTRED_CACHE[key] = (mdl[:-4] + '_static_c.md3', c) if c is not None else None
    hit = _RECENTRED_CACHE[key]
    if hit is None:
        return None
    rmodel, c = hit
    try:
        o = [float(v) for v in kv['origin'].split()]
        ang = [float(v) for v in (kv.get('angles') or '0 {} 0'.format(kv.get('angle', '0'))).split()]
    except (KeyError, ValueError):
        return None
    ax = _q3_axis(ang)
    no = tuple(o[k] + c[0] * ax[0][k] + c[1] * ax[1][k] + c[2] * ax[2][k] for k in range(3))
    pen = world_brush_penetration(tuple((no[k] + offset[k]) / scale for k in range(3)))
    if pen is not None and pen[0] >= 1.0:
        return None
    return rmodel, no

def MODEL_ORIGIN_OFFSETS_KEYS():
    return {k.lower() for k in MODEL_ORIGIN_OFFSETS}


def _static_block_reason(kv, model_scale, scale, offset):
    """Why a misc_model placement can't become model_static as it stands, or
    None if it can."""
    if kv.get('spawnflags', '0') not in ('0', ''):
        return 'skybox'                    # skybox-marked (4): must stay a baked surface
    try:
        if abs(float(kv.get('modelscale', model_scale)) - model_scale) > 1e-3:
            return 'scaled'                # _static.md3 is baked at --model-scale only
        o = tuple((float(c) + offset[i]) / scale for i, c in enumerate(kv['origin'].split()))
    except (KeyError, ValueError):
        return 'no origin'
    pen = world_brush_penetration(o)
    if pen is None or pen[0] < 1.0:
        return None
    return 'origin in wall'                # drawn only from inside the wall (see convert_entity)


def _static_ok(kv, model_scale, scale, offset):
    """A misc_model placement that can become model_static as it stands."""
    return _static_block_reason(kv, model_scale, scale, offset) is None


def shorten_map_texture_names(map_text):
    """Every brush face and patch whose shader name is too long for idTech3 gets
    its short form. Returns (text, {old: new})."""
    changed = {}

    def sub(m):
        new = _short_map_texture(m.group(2))
        if new != m.group(2):
            changed[m.group(2)] = new
        return m.group(1) + new

    map_text = _FACE_TEX_RE.sub(sub, map_text)
    map_text = _PATCH_TEX_RE.sub(sub, map_text)
    return map_text, changed


# ==============================================================================
# VDF / KeyValues parsing
# ==============================================================================

_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(//[^\n]*)|([{}])|(\S+)')


def tokenize(text):
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            tokens.append(m.group(1).replace('\\"', '"'))
        elif m.group(2) is not None:
            continue  # line comment - skip entirely (see vmt2shader.py for why
                      # this matters: without it a leading comment becomes a
                      # real token and shifts the whole parse)
        elif m.group(3) is not None:
            tokens.append(m.group(3))
        else:
            tokens.append(m.group(4))
    return tokens


def parse_vdf(tokens):
    """Parse tokens into a tree of dicts. Each node maps key -> list of
    values, where a value is either a string or another node (dict), since
    VMF allows repeated keys (multiple 'solid', 'side', 'entity', etc.)."""
    pos = [0]

    def parse_block():
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
                value = parse_block()
            else:
                value = nxt
                pos[0] += 1
                if key.lower() in _PATH_KEYS:
                    # Hammer can save a trailing space after a path (de_mirage_cs2:
                    # "de_mirage/base/de_mirage_base_trim_ver1_diffuse "). Source
                    # finds the file anyway (Windows ignores trailing spaces in
                    # file names); here it made the .vmt lookups miss, so a
                    # blend material was built as a plain one.
                    value = value.strip()
            node.setdefault(key, []).append(value)
        return node

    return parse_block()


_PATH_KEYS = {'material', 'model', 'texture', 'skyname', 'detailmaterial'}


def get1(node, key, default=None):
    v = node.get(key)
    return v[0] if v else default


def get_solid_blocks(node):
    """Return only the nested 'solid { ... }' brush blocks under this node,
    ignoring any plain 'solid' "N" keyvalue (Source's collision-type flag,
    which happens to share the same key name as brush blocks)."""
    return [v for v in node.get('solid', []) if isinstance(v, dict)]


# --------------------------------------------------------------- sounds --
#
# Source's ambience -> SOF2 target_speakers; see sof2_sounds.py for the
# mapping. The converted files are written under SOUND_OUT_DIR (set by main
# to <output folder>/sound_sof2; install_to_sof2.py installs it as sound/),
# never next to the extracted originals, which share the same folder.
SOUND_OUT_DIR = [None]
# (Sounds SOF2 has as one of its own sound sets are exempt from all of this:
# they become "soundSet" speakers, which are never in a snapshot - see
# sof2_sounds.RAVEN_SET_RULES. What follows is about the rest.)
# Every "noise" target_speaker is a networked entity: it is sent in snapshots and
# comes out of the same 256-entity room as networked props (networked_prop_cap
# counts them). de_overpass_csgo's soundscapes, converted one for one, gave 203
# speakers - more than the whole room. So sounds get a budget, filled in order
# of what is lost least: positioned loops (the fan, the generator, the fountain
# - a sound you walk up to), then ambient loops (one per area), then positioned
# random sounds, then ambient random ones. Each random list gets ONE speaker,
# its wave rotating through the list from one placement to the next.
SOUND_SPEAKER_BUDGET = 32
RANDOM_MERGE_RANGE_UNITS = 1536.0
SOUND_SPEAKERS = []          # (noise, origin) of every speaker written, for the report


def _load_sibling_module(name):
    try:
        return __import__(name)
    except ImportError:
        pass
    import importlib.util
    for d in (os.path.dirname(os.path.abspath(__file__)),
              os.path.dirname(os.path.abspath(sys.argv[0])), getattr(sys, '_MEIPASS', '')):
        path = os.path.join(d, name + '.py') if d else ''
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def build_sound_speakers(root, scale, offset, materials_dir, warnings):
    """env_soundscape + ambient_generic -> target_speaker entity blocks, with
    each sound converted into SOUND_OUT_DIR. Only sounds that were extracted
    and converted get a speaker (SOF2 would complain about a missing file)."""
    SOUND_SPEAKERS[:] = []
    if not materials_dir or not SOUND_OUT_DIR[0]:
        return []
    snd = _load_sibling_module('sof2_sounds')
    if snd is None:
        warnings.append("sof2_sounds.py not found next to vmf2map.py - no sounds converted.")
        return []
    src_root = os.path.dirname(os.path.abspath(materials_dir))
    if os.path.isdir(SOUND_OUT_DIR[0]):
        shutil.rmtree(SOUND_OUT_DIR[0], ignore_errors=True)   # our own output only
    scripts_dir = os.path.join(src_root, 'scripts')
    scripts = []
    if os.path.isdir(scripts_dir):
        for fn in sorted(os.listdir(scripts_dir)):
            if fn.lower().startswith('soundscapes') and fn.lower().endswith('.txt'):
                scripts.append(os.path.join(scripts_dir, fn))
    # the map's own soundscape file wins, as in Source: load it last
    scripts.sort(key=lambda p_: 'manifest' not in p_.lower() and
                 os.path.basename(p_).lower() != 'soundscapes.txt')
    defs = snd.load_soundscapes(scripts)

    def mpos(o):
        return tuple(o[k] * scale - offset[k] for k in range(3))

    named = {}
    for e in root.get('entity', []):
        if isinstance(e, dict) and get1(e, 'targetname'):
            o = _vec_key(e, 'origin')
            if o:
                named.setdefault(get1(e, 'targetname').strip().lower(), o)

    converted = {}       # (wave, volume, looped) -> noise path or None
    notes = Counter()

    def convert(wave, volume, looped):
        key = (wave.lower(), round(volume, 2), looped)
        if key in converted:
            return converted[key]
        src = os.path.join(src_root, *wave.split('/'))
        noise = None
        if os.path.isfile(src):
            try:
                with open(src, 'rb') as f:
                    data, ext, n = snd.convert_sound(f.read(), volume, looped)
                noise = snd.variant_name(wave, volume, looped, ext)
                if len(noise) > 63:                  # MAX_QPATH, the engine's path buffer
                    noise = short_qpath(os.path.splitext(noise)[0], 63 - len(ext)) + ext
                dst = os.path.join(SOUND_OUT_DIR[0], *noise.split('/'))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as f:
                    f.write(data)
                for x in n:
                    notes[x] += 1
            except (ValueError, struct.error, OSError) as ex:
                notes['not converted ({})'.format(ex)] += 1
                noise = None
        else:
            notes['not extracted (missing from the game content)'] += 1
        converted[key] = noise
        return noise

    candidates = []        # (priority, order, group key, waves, volume, pos, looped, wait, rand, merge range)

    set_candidates = []    # (positioned first, order, set name, radius, pos, positioned, source waves)

    def add(waves, volume, pos, looped, wait=0.0, rand=0.0, merge=0.0, priority=0):
        # One-off gags (SOUND_SKIP_KEYWORDS) never get a speaker.
        kept = [w for w in waves if not snd.is_skipped_sound(w)]
        if len(kept) < len(waves):
            notes['sound left out by SOUND_SKIP_KEYWORDS ({})'.format(
                ', '.join(snd.SOUND_SKIP_KEYWORDS))] += len(waves) - len(kept)
        if not kept:
            return
        waves = kept
        # A sound SOF2 already has as one of its own ambient sets plays from
        # a soundSet speaker: free against the snapshot room, no file to
        # ship (see sof2_sounds.RAVEN_SET_RULES).
        rset = snd.raven_set_for(waves)
        if rset is not None:
            positioned = priority in (0, 2)
            set_candidates.append((0 if positioned else 1, len(set_candidates), rset[0], rset[1],
                                   pos, positioned, tuple(sorted(set(waves)))))
            return
        candidates.append((priority, len(candidates), tuple(w.lower() for w in waves), list(waves),
                           volume, pos, looped, wait, rand, merge))

    missing_defs, missing_pos = set(), Counter()
    for e in root.get('entity', []):
        if not isinstance(e, dict):
            continue
        cls = (get1(e, 'classname') or '').lower()
        o = _vec_key(e, 'origin')
        if not o:
            continue
        if cls == 'env_soundscape':
            name = get1(e, 'soundscape') or ''
            items = snd.soundscape_items(name, defs, missing=missing_defs)
            for it in items:
                if it['position'] is None:
                    at, merge = mpos(o), True
                else:
                    tgt = (get1(e, 'position%d' % it['position']) or '').strip().lower()
                    if tgt not in named:
                        missing_pos[name] += 1
                        continue
                    at, merge = mpos(named[tgt]), False
                if it['kind'] == 'loop':
                    add(it['waves'][:1], it['volume'], at, True,
                        merge=AMBIENT_MERGE_RANGE_UNITS if merge else 1.0,
                        priority=1 if merge else 0)
                else:
                    a_, b_ = it['time']
                    add(it['waves'], it['volume'], at, False, wait=(a_ + b_) / 2.0,
                        rand=(b_ - a_) / 2.0, merge=RANDOM_MERGE_RANGE_UNITS if merge else 1.0,
                        priority=3 if merge else 2)
        elif cls == 'ambient_generic':
            msg = (get1(e, 'message') or '').strip()
            try:
                flags = int(get1(e, 'spawnflags', '0') or 0)
                vol = max(0.0, min(1.0, float(get1(e, 'health', '10') or 10) / 10.0))
            except ValueError:
                flags, vol = 0, 1.0
            if not msg.lower().endswith(('.wav', '.mp3')):
                notes['ambient_generic naming a sound script entry, not a file (skipped)'] += 1
                continue
            if flags & 16 or flags & 32:
                # starts silent (switched on by Source I/O, which SOF2 has
                # not) or plays once - nothing to hear in SOF2
                notes['ambient_generic that only plays when triggered (skipped)'] += 1
                continue
            add([snd.clean_wave(msg)], vol, mpos(o), True, merge=1.0, priority=0)

    speakers = []          # (noise, pos, looped, wait, random)
    placed = []            # (group key, pos)
    rotation = Counter()   # random list -> next wave to use
    over_budget = Counter()
    for (prio, _o, gkey, waves, volume, pos, looped, wait, rand, merge) in sorted(candidates):
        # the same sound (or random list) already placed within its merge
        # range - at any volume - is not placed again: identical loops only
        # add up (target_speaker's own documentation says so)
        if any(g_ == gkey and math.dist(p_, pos) < merge for g_, p_ in placed):
            continue
        if len(speakers) >= SOUND_SPEAKER_BUDGET:
            over_budget[('positioned ' if prio in (0, 2) else 'ambient ') +
                        ('loop' if looped else 'random sound')] += 1
            continue
        wave = waves[rotation[gkey] % len(waves)]
        rotation[gkey] += 1
        noise = convert(wave, volume, looped)
        if noise is None:
            continue
        placed.append((gkey, pos))
        speakers.append((noise, pos, looped, wait, rand))
    for k, v in over_budget.items():
        notes['{} left out - over the {}-speaker budget'.format(k, SOUND_SPEAKER_BUDGET)] += v

    # SOF2 sound-set speakers. The same set is not placed again within its
    # own radius (an ambient one, standing in for "everywhere in this area")
    # or half of it (one tied to a spot - two fans 300 units apart are two
    # fans). Each identical set playing at once only adds up.
    set_speakers = []      # (set name, pos)
    set_sources = {}       # set name -> Source waves it replaces
    for (_pf, _o, sname, radius, pos, positioned, waves) in sorted(set_candidates):
        merge = radius * (0.5 if positioned else 1.0)
        if any(n_ == sname and math.dist(p_, pos) < merge for n_, p_ in set_speakers):
            continue
        set_speakers.append((sname, pos))
        set_sources.setdefault(sname, set()).update(waves)

    blocks = []
    for sname, pos in set_speakers:
        blocks.append('\n'.join(['{', '"classname" "target_speaker"',
                                 '"origin" "{:.6g} {:.6g} {:.6g}"'.format(*pos),
                                 '"soundSet" "{}"'.format(sname), '}']))
    if set_speakers:
        warnings.append(
            "Sounds: {} target_speaker(s) play SOF2's own sound sets instead of a CS sound (sent "
            "once at map load, not in snapshots, so free against the networked props): {}. "
            "Change sof2_sounds.RAVEN_SET_RULES to keep any of these as the CS sound.".format(
                len(set_speakers), "; ".join(
                    "{} x{} <- {}".format(n_, sum(1 for s_ in set_speakers if s_[0] == n_),
                                          ", ".join(os.path.basename(w) for w in sorted(set_sources[n_])[:4]) +
                                          (" ..." if len(set_sources[n_]) > 4 else ""))
                    for n_ in sorted(set_sources))))
    for noise, pos, looped, wait, rand in speakers:
        lines = ['{', '"classname" "target_speaker"',
                 '"origin" "{:.6g} {:.6g} {:.6g}"'.format(*pos),
                 '"noise" "{}"'.format(noise)]
        if looped:
            lines.append('"spawnflags" "1"')
        else:
            lines.append('"wait" "{:.3g}"'.format(max(1.0, wait)))
            lines.append('"random" "{:.3g}"'.format(max(0.0, rand)))
        lines.append('}')
        blocks.append('\n'.join(lines))
        SOUND_SPEAKERS.append((noise, pos))
    if speakers or notes or missing_defs:
        n_loop = sum(1 for s_ in speakers if s_[2])
        warnings.append(
            "Sounds: {} target_speaker(s) with a CS sound ({} looping, {} repeating) from the "
            "map's soundscapes and ambient_generics, {} sound file(s) converted into sound_sof2/. "
            "Each of these is a networked entity and comes out of the same snapshot room as "
            "networked props.{}{}{}"
            .format(len(speakers), n_loop, len(speakers) - n_loop,
                    sum(1 for v in converted.values() if v),
                    "".join("; {} x {}".format(v, k) for k, v in notes.most_common()),
                    "; soundscapes not found in any script: " + ", ".join(sorted(missing_defs))
                    if missing_defs else "",
                    "; positions with no named entity: " + ", ".join(
                        "{} x{}".format(k, v) for k, v in missing_pos.items()) if missing_pos else ""))
    return blocks


AMBIENT_MERGE_RANGE_UNITS = 768.0


def find_sky_camera_origin(root):
    """Source's 3D skybox (a separate, far-away, scaled-down copy of the
    map's own geometry, rendered as a distant backdrop) is anchored by a
    'sky_camera' entity's own origin - confirmed by a real map where every
    skybox-only prop (name containing '_skybx') clustered tightly around
    one distant point (~10000-12000 units out), far outside the main map's
    own coordinate range. idTech3 has no equivalent rendering concept at
    all (its sky is a flat shader property, not a separate rendered
    world), so this whole area serves no purpose in the converted map -
    confirmed floating/misplaced props were coming from exactly this
    region. Returns the origin as (x, y, z) or None if no sky_camera
    entity exists in this VMF."""
    for entity in root.get('entity', []):
        if get1(entity, 'classname', '').strip().lower() == 'sky_camera':
            origin_str = get1(entity, 'origin')
            if origin_str:
                parts = origin_str.split()
                if len(parts) == 3:
                    try:
                        return tuple(float(p) for p in parts)
                    except ValueError:
                        pass
    return None


# Nominal model half-extent, in units, multiplied by an entity's own
# 'modelscale' to reserve room for its mesh inside the sealing box. 128 is a
# generous half-size for an unscaled prop; at the sky scale of 16 that
# becomes 2048 units of headroom for a worldized skybox tree.
ENTITY_MODEL_PAD = 128.0


def _box_brush(mn, mx, texture="tools/_caulk"):
    """One axis-aligned solid brush spanning mn..mx, as six .map faces.

    Windings follow q3map2's own PlaneFromPoints (normal = (p0-p1) x (p2-p1))
    so every face normal points OUT of the box."""
    x0, y0, z0 = mn
    x1, y1, z1 = mx

    def face(p0, p1, p2):
        return ("( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) "
                "( {:.10g} {:.10g} {:.10g} ) {} 0 0 0 0.500000 0.500000 0 0 0").format(
                    p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], texture)

    return "\n".join([
        "{",
        face((x0, y1, z1), (x1, y1, z1), (x1, y0, z1)),   # +z
        face((x0, y0, z0), (x1, y0, z0), (x1, y1, z0)),   # -z
        face((x0, y1, z0), (x1, y1, z0), (x1, y1, z1)),   # +y
        face((x1, y0, z0), (x0, y0, z0), (x0, y0, z1)),   # -y
        face((x1, y1, z0), (x1, y0, z0), (x1, y0, z1)),   # +x
        face((x0, y0, z0), (x0, y1, z0), (x0, y1, z1)),   # -x
        "}",
    ])


def find_geometry_clusters(map_blocks, gap=400.0):
    """Group everything the compiler will see into bodies separated by a large
    vertical gap, and return each body's bounding box in final map units.

    This reads the GENERATED map text rather than the source VMF on purpose.
    The .map is what q3map2 actually loads, so brush points, patch control
    points and entity origins are all measured after recentering, after
    --skip-3d-skybox has dropped things, and after worldize_skybox_models has
    rewritten the skybox props' positions - no scale/offset/skip logic to keep
    in sync, and nothing can be missed.

    Entity origins count, and that is not a detail: FloodEntities seeds the
    flood from entity origins, so a single entity outside the sealed hull
    reports a leak no matter how well the brushwork closes. On de_dust2, 18
    worldized skybox props landed outside a hull fitted to brushwork alone -
    which would also have left them in the void, not rendering at all.

    A converted Source map is usually two separate bodies: the playable map,
    and the 3D skybox's miniature copy parked far away. Measured on de_dust2
    they separate cleanly on z (playable -470..1527, skybox -2787..-1236, with
    nothing between), so one box around everything would enclose a vast empty
    slab and make vis process it. Splitting on the largest vertical gap gives
    each body its own tight box."""
    face_re = re.compile(
        r'^\s*\(([^()]*)\)\s*\(([^()]*)\)\s*\(([^()]*)\)\s+\S')
    tuple5_re = re.compile(r'\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)'
                           r'\s+-?[\d.eE+-]+\s+-?[\d.eE+-]+\s*\)')
    origin_re = re.compile(r'^\s*"origin"\s*"\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)'
                           r'\s+(-?[\d.eE+-]+)')

    scale_re = re.compile(r'^\s*"modelscale"\s*"\s*(-?[\d.eE+-]+)')

    # Each entry is (raw z used for clustering, padded lo, padded hi). Clustering
    # deliberately uses the RAW point, never the padded box: on the first
    # attempt, a 16x-scaled prop sitting near the playable map's floor padded
    # 2048 units DOWNWARD, bridged the gap to the 3D skybox below, and merged
    # the two bodies into one box - which would have driven a caulk slab straight
    # through the skybox buildings. Padding also only reaches sideways and UP,
    # because props stand on the ground and grow upward.
    spans = []

    def add(p, pad=0.0):
        spans.append((p[2],
                      (p[0] - pad, p[1] - pad, p[2]),
                      (p[0] + pad, p[1] + pad, p[2] + pad)))

    for block in map_blocks:
        lines = block.splitlines()
        # a model's own mesh reaches well past its origin, and the converter has
        # no access to its bounds - those live in the MD3, built by a different
        # stage. modelscale is the one proxy available, and it is the one that
        # matters here: worldize_skybox_models multiplies it by the sky scale
        # (16 on de_dust2), so the distant skybox trees are the very models that
        # overhang furthest. Padding each entity by its own scale keeps them
        # inside the hull; a model left outside would not merely leak, it would
        # sit in the void and not render at all.
        model_pad = 0.0
        for line in lines:
            m = scale_re.match(line)
            if m:
                try:
                    model_pad = abs(float(m.group(1))) * ENTITY_MODEL_PAD
                except ValueError:
                    pass
                break
        for line in lines:
            m = face_re.match(line)
            if m:
                for group in m.groups():
                    parts = group.split()
                    if len(parts) == 3:
                        try:
                            add([float(v) for v in parts])
                        except ValueError:
                            pass
                continue
            m = origin_re.match(line)
            if m:
                add([float(v) for v in m.groups()], model_pad)
                continue
            # patch control rows carry several ( x y z u v ) tuples; the lone
            # five-number patch DIMENSION row ( 3 3 0 0 0 ) must not be read as
            # a point, hence the "more than one on the line" test
            found = tuple5_re.findall(line)
            if len(found) > 1:
                for g in found:
                    add([float(v) for v in g])
    if not spans:
        return []

    spans.sort(key=lambda sp: sp[0])
    clusters = [[spans[0]]]
    reach = spans[0][0]
    for sp in spans[1:]:
        if sp[0] - reach > gap:
            clusters.append([sp])
        else:
            clusters[-1].append(sp)
        reach = max(reach, sp[0])

    out = []
    for group in clusters:
        mn = [min(g[1][i] for g in group) for i in range(3)]
        mx = [max(g[2][i] for g in group) for i in range(3)]
        raw_top = max(g[0] for g in group)
        raw_bottom = min(g[0] for g in group)
        out.append([mn, mx, len(group), raw_top, raw_bottom])

    # A body's upward padding must not reach into the body above it, or its lid
    # would be built through that body's contents. Stop each lid at the midpoint
    # of the real gap; the padding is headroom for overhanging models, not a
    # measurement, so losing some of it is the safe trade.
    for i in range(len(out) - 1):
        limit = (out[i][3] + out[i + 1][4]) * 0.5
        if out[i][1][2] > limit:
            out[i][1][2] = max(out[i][3], limit)
    return [(o[0], o[1], o[2]) for o in out]


def build_seal_boxes(map_blocks, margin, warnings, thickness=64.0):
    """A tight closed caulk box around each body of geometry, to seal the map.

    Why a box rather than a shell that follows the ground. Source seals a map
    with its displacement surfaces; converted to a model, they seal nothing,
    and the flood escapes underneath the terrain and out. An earlier attempt at
    a terrain-following prism shell was a measured failure: 552 thin sloped
    structural brushes roughly DOUBLED the leaf count (8127 -> 15234), turned
    67 tiny portals into 1416, pushed references UP from 1.62M to 1.79M, and
    still leaked because the flood simply passed beneath it. Six large
    axis-aligned brushes cost the BSP almost nothing by comparison, and a
    closed box cannot be gone around.

    The boxes are tight: `margin` units clear of everything they wrap, so the
    enclosed dead space stays small. Measured on de_dust2, the leak ran at
    z -512..-674, BELOW the playable map's own lowest brush at -406 - so the
    floor alone puts the entire escape route inside solid.

    Faces are caulk (nodraw, solid) and the brushes carry no detail flag, which
    matters: q3map2 filters detail brushes in AFTER FloodEntities, so a detail
    box would seal nothing."""
    clusters = find_geometry_clusters(map_blocks)
    if not clusters:
        return []

    blocks = []
    for mn, mx, count in clusters:
        lo = [mn[i] - margin for i in range(3)]
        hi = [mx[i] + margin for i in range(3)]
        t = thickness
        # six slabs forming a closed shell: floor and lid span the whole outer
        # footprint, the four walls fill between them, so every edge abuts and
        # there is no seam for the flood to squeeze through
        blocks.append(_box_brush((lo[0] - t, lo[1] - t, lo[2] - t), (hi[0] + t, hi[1] + t, lo[2])))
        blocks.append(_box_brush((lo[0] - t, lo[1] - t, hi[2]), (hi[0] + t, hi[1] + t, hi[2] + t)))
        blocks.append(_box_brush((lo[0] - t, lo[1] - t, lo[2]), (hi[0] + t, lo[1], hi[2])))
        blocks.append(_box_brush((lo[0] - t, hi[1], lo[2]), (hi[0] + t, hi[1] + t, hi[2])))
        blocks.append(_box_brush((lo[0] - t, lo[1], lo[2]), (lo[0], hi[1], hi[2])))
        blocks.append(_box_brush((hi[0], lo[1], lo[2]), (hi[0] + t, hi[1], hi[2])))
        warnings.append(
            "--seal-map-box {:g}: wrapped a body of geometry in a closed caulk box, "
            "{:.0f} x {:.0f} x {:.0f} units (x {:.0f}..{:.0f}, y {:.0f}..{:.0f}, z {:.0f}..{:.0f}), "
            "{} point(s) of brushwork, patches and entity origins enclosed. Six structural "
            "brushes - a detail box would seal nothing, since q3map2 filters detail in after "
            "FloodEntities.".format(
                margin, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2],
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], count))
    if len(clusters) > 1:
        warnings.append(
            "--seal-map-box: the map's geometry forms {} separate bodies (typically the playable "
            "map and the 3D skybox's miniature copy, which sit far apart), so each got its own "
            "tight box rather than one enormous box enclosing the empty space between "
            "them.".format(len(clusters)))
    return ["\n".join(["{", '"classname" "func_group"'] + blocks + ["}"])]


def find_sky_camera_scale(root, default=16.0):
    """The sky_camera's own 'scale' key: how much SMALLER the 3D skybox copy
    is than the real map (Source's own docs: "a value of 16 means the skybox
    is 1/16th scale"). de_dust2 uses 4. Returns `default` if there's no
    sky_camera or no readable scale."""
    for entity in root.get('entity', []):
        if get1(entity, 'classname', '').strip().lower() == 'sky_camera':
            try:
                s = float(get1(entity, 'scale', ''))
                if s > 0:
                    return s
            except (TypeError, ValueError):
                pass
            return default
    return default


def playable_bounds(root):
    """Bounding box (Source units) of every brush outside the 3D skybox area -
    the playable map. None when there is no brush to measure."""
    mn = [float('inf')] * 3
    mx = [float('-inf')] * 3
    for holder in list(root.get('world', [])) + list(root.get('entity', [])):
        for solid in get_solid_blocks(holder):
            pb = _brush_planes_and_bounds(solid)
            if not pb:
                continue
            _pl, bmn, bmx = pb
            centre = tuple((bmn[i] + bmx[i]) / 2.0 for i in range(3))
            if in_skybox_area(centre):
                continue
            for i in range(3):
                mn[i] = min(mn[i], bmn[i])
                mx[i] = max(mx[i], bmx[i])
    if mn[0] == float('inf'):
        return None
    return tuple(mn), tuple(mx)


def worldize_skybox_models(root, warnings, main_bounds=None, map_scale=1.0):
    """Move every model entity in the 3D skybox area into REAL map space, at
    real scale, instead of leaving it for q3map2's _skybox to render as a
    backdrop (see --skybox-models-to-world).

    THE TRANSFORM, and why it's exact rather than a guess:

        P_world = (P_sky - sky_camera.origin) * sky_camera.scale

    Source renders the 3D skybox with the view at the sky_camera's origin,
    so a point sitting `d` away from the sky_camera appears to the player
    `d * scale` away in real world terms - that's the whole trick, and the
    same relation run backwards gives the real position a piece of skybox
    content is pretending to occupy.

    VERIFIED directly against de_dust2's own data, not derived on paper.
    That map has 8 'palm_tree_head_skybox' props in its 3D skybox and 8
    full-size 'palm_tree_trunk' props in the playable map: Valve put the
    CROWNS of the tall palms in the skybox so they'd read as very tall
    trees. Running this transform on the 8 crowns lands every one of them
    on a trunk, with the SAME residual on all 8 (dx=-3.1, dy=-1.2,
    dz=+1.4) - a constant offset, i.e. Valve's grid snapping at 1/4 scale
    (0.78 units * 4), not a fitting error. A transform that reproduces 8
    independent prop positions to ~3 units with a single rigid mapping is
    the transform.

    Origins are rewritten in SOURCE units (the ordinary --scale/--recenter
    handling downstream still applies), and modelscale is multiplied by the
    skybox scale, since the model FILE is built at skybox size.

    Each rewritten entity is tagged with WORLDIZED_MARKER_KEY so the later
    skybox passes leave it alone - its new origin can still fall inside
    --skybox-radius.

    Returns (count, n_inside_main_map). Mutates `root` in place (replacing
    the entity list with new dicts; the originals aren't modified)."""
    sky_origin = find_sky_camera_origin(root)
    if sky_origin is None:
        warnings.append(
            "--skybox-models-to-world was given but no sky_camera entity was found in this VMF - "
            "nothing moved (this map may not use a 3D skybox at all).")
        return 0, 0
    sky_scale = find_sky_camera_scale(root)

    moved = 0
    inside = 0
    moved_models = {}
    new_entities = []
    too_far = []
    outside = []
    playable = playable_bounds(root)
    for entity in root.get('entity', []):
        origin_str = get1(entity, 'origin')
        model_val = get1(entity, 'model') or get1(entity, 'model2')
        pos = None
        if origin_str:
            parts = origin_str.split()
            if len(parts) == 3:
                try:
                    pos = tuple(float(p) for p in parts)
                except ValueError:
                    pos = None
        # Only point-placed MODEL entities: brush entities have no single
        # origin to scale about, and scaling brush geometry about a point
        # is a different (and much riskier) operation than moving a prop.
        if pos is None or not model_val or get_solid_blocks(entity):
            new_entities.append(entity)
            continue
        if not in_skybox_area(pos):
            new_entities.append(entity)
            continue

        new_pos = tuple((pos[i] - sky_origin[i]) * sky_scale for i in range(3))
        # Horizon scenery lands outside the playable map at real scale - past
        # its outer walls and sky, where no player can ever see it. Emitting it
        # there anyway is not free: the map has to be sealed around it, and
        # every cubic unit of that sealed space becomes vis clusters
        # (de_mirage_cs2: 18 horizon props stretched the sealed volume from
        # ~8k x 8k to 23.6k x 26k units, giving 19940 clusters - over
        # q3map2's 16384 MAX_MAP_VISCLUSTERS, "MAX_MAP_VISIBILITY exceeded").
        # Such a prop is left out.
        if playable is not None and not all(
                playable[0][i] <= new_pos[i] <= playable[1][i] for i in range(3)):
            outside.append(convert_model_path(model_val))
            continue
        # q3map2 drops anything past +-65536 (MIN/MAX_WORLD_COORD), and the
        # seal box around it would lose its walls and leak. Such a prop stays
        # in the 3D skybox, where Source draws it anyway.
        if any(abs(c) * map_scale > WORLDIZE_COORD_LIMIT for c in new_pos):
            too_far.append(convert_model_path(model_val))
            new_entities.append(entity)
            continue
        e = dict(entity)
        e['origin'] = ['{:.10g} {:.10g} {:.10g}'.format(*new_pos)]
        try:
            existing_ms = float(get1(entity, 'modelscale', '1'))
        except ValueError:
            existing_ms = 1.0
        e['modelscale'] = ['{:.6g}'.format(existing_ms * sky_scale)]
        e[WORLDIZED_MARKER_KEY] = ['1']
        new_entities.append(e)
        moved += 1
        moved_models[convert_model_path(model_val)] = moved_models.get(
            convert_model_path(model_val), 0) + 1
        if main_bounds is not None:
            (mn, mx) = main_bounds
            if all(mn[i] <= new_pos[i] <= mx[i] for i in range(3)):
                inside += 1

    root['entity'] = new_entities
    if outside:
        counts = Counter(outside)
        warnings.append(
            "--skybox-models-to-world: left out {} 3D-skybox prop(s) that land outside the playable "
            "map at real scale (playable area x {:.0f}..{:.0f}, y {:.0f}..{:.0f}, z {:.0f}..{:.0f}). "
            "They would sit beyond the outer walls and sky where they can never be seen, and the "
            "map would have to be sealed around them - on de_mirage_cs2 that empty sealed space "
            "pushed the vis cluster count past q3map2's limit. Left out: {}.".format(
                len(outside), playable[0][0], playable[1][0], playable[0][1], playable[1][1],
                playable[0][2], playable[1][2],
                ", ".join("{}x {}".format(n, m) for m, n in counts.most_common())))
    if too_far:
        warnings.append(
            "--skybox-models-to-world: {} prop(s) would land past q3map2's +-65536 world limit "
            "at real scale, so they stay in the 3D skybox: {}".format(
                len(too_far), ", ".join(sorted(set(too_far)))))
    if moved:
        detail = ", ".join("{}x {}".format(n, m) for m, n in
                           sorted(moved_models.items(), key=lambda kv: -kv[1]))
        warnings.append(
            "--skybox-models-to-world: moved {} model(s) out of the 3D skybox area into real map "
            "space at real scale, using P_world = (P_sky - sky_camera_origin) * {:g} (the "
            "sky_camera's own scale) and modelscale x {:g}. These are ordinary main-map models "
            "now - NOT skybox surfaces - so they cost nothing against MAX_MAP_LEAFFACES and do "
            "not need --skybox-entity at all. Moved: {}.".format(
                moved, sky_scale, sky_scale, detail))
        if main_bounds is not None and inside < moved:
            warnings.append(
                "--skybox-models-to-world: {} of the {} moved model(s) landed OUTSIDE the main "
                "map's own bounding box - expected for a map whose 3D skybox includes distant "
                "horizon scenery (those pieces are pretending to be far away, so at real scale "
                "they genuinely are). They may also land on top of the miniature skybox "
                "brushwork, which is still in the map unless you also pass --skip-3d-skybox: "
                "combining the two flags gives real-scale props with no leftover miniature copy "
                "underneath them.".format(moved - inside, moved))
    return moved, inside


# Mutable single-element holder so --skybox-radius can drive
# worldize_skybox_models' own region test without threading the value
# through every call site (set once in main()).
SKYBOX_WORLDIZE_RADIUS = [4096.0]
# q3map2 MIN/MAX_WORLD_COORD is +-65536; keep a margin for the seal box
WORLDIZE_COORD_LIMIT = 60000.0


def brush_approx_position(solid):
    """A cheap, approximate world position for a brush, for skybox-area
    distance filtering only - not precise geometry, just enough to tell
    whether this brush sits near the main map or off in the skybox's own
    far-away cluster. Uses the first plane point found on the first real
    side block."""
    for side in solid.get('side', []):
        if not isinstance(side, dict):
            continue
        plane = get1(side, 'plane')
        if plane:
            m = re.match(r'\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*\)', plane)
            if m:
                return tuple(float(m.group(i)) for i in (1, 2, 3))
    return None


# ==============================================================================
# The 3D skybox area: the sealed room around the sky_camera
# ==============================================================================
# Every "is this part of the 3D skybox?" test used to be a 4096-unit sphere
# around the sky_camera (--skybox-radius). That only works when the skybox room
# sits far from the playable map. de_mirage_cs2's sky_camera is at x 5208 and
# its skybox room is a 2064 x 2064 x 528 toolsskybox box (x 4088..6152), but the
# playable map runs up to x ~2048, so the sphere reached 3000+ units into T
# spawn: 37 playable props (lamps, windows, doors) were treated as skybox props
# and flung 16x out to x -85000 (past q3map2's +-65536 world limit), 87 playable
# displacements were rebuilt as coarse skybox landscape prisms, 20 overlays and
# nearby misc_models were skybox-marked.
#
# What Source itself means by "the 3D skybox" is the sealed region the
# sky_camera stands in - VBSP gives it its own area, and the engine draws the
# skybox from that area alone. find_skybox_area() finds that region directly:
# a flood fill on a grid from the sky_camera, where a step between two
# neighbouring cells is blocked if the segment between their centres passes
# through any solid brush (so walls of any thickness block it, unlike a
# point-in-brush test). The result is the bounding box of the space reached.
# If the flood escapes (an unsealed skybox, or no walls at all), it falls back
# to the old sphere and says so.

SKYBOX_AREA = [None]            # (mins, maxs) in Source units, or None = sphere fallback
SKYBOX_AREA_INFO = [None]       # text for the report
SKYBOX_FLOOD_CELL = 64.0
SKYBOX_FLOOD_MAX_CELLS = 2000000


def _brush_planes_and_bounds(solid):
    planes = []
    for side in solid.get('side', []):
        if not isinstance(side, dict):
            continue
        tri = parse_plane(get1(side, 'plane') or '')
        if len(tri) != 3:
            continue
        nrm = cross(sub(tri[0], tri[1]), sub(tri[2], tri[1]))
        if length(nrm) < 1e-9:
            continue
        nrm = normalize(nrm)
        planes.append((nrm, dot(nrm, tri[0])))
    if len(planes) < 4:
        return None
    verts = []
    n = len(planes)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (n1, d1), (n2, d2), (n3, d3) = planes[i], planes[j], planes[k]
                c23 = cross(n2, n3)
                det = dot(n1, c23)
                if abs(det) < 1e-9:
                    continue
                p = scale_vec(add(add(scale_vec(c23, d1), scale_vec(cross(n3, n1), d2)),
                                  scale_vec(cross(n1, n2), d3)), 1.0 / det)
                if all(dot(pn, p) - pd <= 0.01 for pn, pd in planes):
                    verts.append(p)
    if not verts:
        return None
    mins = tuple(min(v[a] for v in verts) for a in range(3))
    maxs = tuple(max(v[a] for v in verts) for a in range(3))
    return planes, mins, maxs


def _segment_hits_brush(a, b, planes, eps=0.01):
    """True if the open segment a-b passes through the brush's interior."""
    t0, t1 = 0.0, 1.0
    d = sub(b, a)
    for nrm, dist in planes:
        da = dot(nrm, a) - dist
        dd = dot(nrm, d)
        if abs(dd) < 1e-12:
            if da >= -eps:
                return False
            continue
        t = (-eps - da) / dd
        if dd > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
        if t0 >= t1:
            return False
    return t1 - t0 > 1e-6


def find_skybox_area(root):
    """-> (mins, maxs) of the sealed room around the sky_camera, Source units,
    or None (no sky_camera, or the flood leaked)."""
    cam = find_sky_camera_origin(root)
    if cam is None:
        return None
    blockers = []
    holders = list(root.get('world', [])) + list(root.get('entity', []))
    for holder in holders:
        for solid in get_solid_blocks(holder):
            mats = {(get1(s, 'material', '') or '').upper()
                    for s in solid.get('side', []) if isinstance(s, dict)}
            if not mats or mats <= _NONSOLID_TOOL_MATERIALS:
                continue
            pb = _brush_planes_and_bounds(solid)
            if pb:
                blockers.append(pb)
    if not blockers:
        return None
    world_min = tuple(min(b[1][a] for b in blockers) for a in range(3))
    world_max = tuple(max(b[2][a] for b in blockers) for a in range(3))

    bucket = 256.0
    grid = {}
    for idx, (_pl, mn, mx) in enumerate(blockers):
        for bx in range(int(math.floor(mn[0] / bucket)), int(math.floor(mx[0] / bucket)) + 1):
            for by in range(int(math.floor(mn[1] / bucket)), int(math.floor(mx[1] / bucket)) + 1):
                for bz in range(int(math.floor(mn[2] / bucket)), int(math.floor(mx[2] / bucket)) + 1):
                    grid.setdefault((bx, by, bz), []).append(idx)

    C = SKYBOX_FLOOD_CELL

    def centre(c):
        return (cam[0] + c[0] * C, cam[1] + c[1] * C, cam[2] + c[2] * C)

    def blocked(pa, pb):
        keys = set()
        for p in (pa, pb):
            keys.add((int(math.floor(p[0] / bucket)), int(math.floor(p[1] / bucket)),
                      int(math.floor(p[2] / bucket))))
        seen = set()
        for k in keys:
            for idx in grid.get(k, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                planes, mn, mx = blockers[idx]
                if (max(pa[0], pb[0]) < mn[0] or min(pa[0], pb[0]) > mx[0] or
                        max(pa[1], pb[1]) < mn[1] or min(pa[1], pb[1]) > mx[1] or
                        max(pa[2], pb[2]) < mn[2] or min(pa[2], pb[2]) > mx[2]):
                    continue
                if _segment_hits_brush(pa, pb, planes):
                    return True
        return False

    start = (0, 0, 0)
    visited = {start}
    stack = [start]
    lo = list(centre(start))
    hi = list(centre(start))
    steps = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while stack:
        c = stack.pop()
        pc = centre(c)
        for s in steps:
            nc = (c[0] + s[0], c[1] + s[1], c[2] + s[2])
            if nc in visited:
                continue
            pn = centre(nc)
            if any(pn[a] < world_min[a] - C or pn[a] > world_max[a] + C for a in range(3)):
                return None     # reached the outside: the skybox room isn't sealed
            if blocked(pc, pn):
                continue
            visited.add(nc)
            if len(visited) > SKYBOX_FLOOD_MAX_CELLS:
                return None
            for a in range(3):
                lo[a] = min(lo[a], pn[a])
                hi[a] = max(hi[a], pn[a])
            stack.append(nc)
    # cell centres stop up to one cell short of the walls
    return (tuple(v - C for v in lo), tuple(v + C for v in hi))


def setup_skybox_area(root, warnings, radius):
    area = find_skybox_area(root)
    SKYBOX_AREA[0] = area
    cam = find_sky_camera_origin(root)
    SKYBOX_FALLBACK_CENTRE[0] = cam
    if cam is None:
        return
    if area:
        mn, mx = area
        SKYBOX_AREA_INFO[0] = "x {:.0f}..{:.0f}, y {:.0f}..{:.0f}, z {:.0f}..{:.0f}".format(
            mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
        warnings.append(
            "3D skybox area: the sealed room around the sky_camera at ({:g} {:g} {:g}) is {} "
            "(Source units, found by flood-filling from the sky_camera to its walls). Everything "
            "skybox-related - --skybox-models-to-world, --skip-3d-skybox, skybox-marked props, "
            "overlays and terrain - uses this box.".format(cam[0], cam[1], cam[2], SKYBOX_AREA_INFO[0]))
    else:
        SKYBOX_AREA_INFO[0] = "within {:g} units of the sky_camera".format(radius)
        warnings.append(
            "3D skybox area: could not find a sealed room around the sky_camera at ({:g} {:g} {:g}) "
            "(the flood from it reached the outside of the map), so falling back to everything "
            "within --skybox-radius {:g} of it. If playable content near the skybox gets treated "
            "as skybox, lower --skybox-radius.".format(cam[0], cam[1], cam[2], radius))


def in_skybox_area(p):
    """p in Source units."""
    if p is None:
        return False
    area = SKYBOX_AREA[0]
    if area is not None:
        mn, mx = area
        return all(mn[a] <= p[a] <= mx[a] for a in range(3))
    return False if SKYBOX_FALLBACK_CENTRE[0] is None else (
        sum((p[a] - SKYBOX_FALLBACK_CENTRE[0][a]) ** 2 for a in range(3))
        <= SKYBOX_WORLDIZE_RADIUS[0] ** 2)


SKYBOX_FALLBACK_CENTRE = [None]


def in_skybox_area_mapspace(p, scale, offset):
    """p in converted .map units (src * scale - offset)."""
    return in_skybox_area(tuple((p[a] + offset[a]) / scale for a in range(3)))


# ==============================================================================
# Vector / geometry helpers
# ==============================================================================

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale_vec(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


# ==============================================================================
# Displacements (dispinfo) -> idTech3 patchDef2 bezier patch meshes
# ==============================================================================
#
# Source's displacement grid (power 2/3/4 -> 5x5/9x9/17x17 exact mesh vertices)
# and Quake3's patchDef2 (an MxN grid of bezier CONTROL points, M/N must be odd)
# happen to have compatible grid dimensions, which is what makes this
# conversion possible at all. But they are not the same kind of surface:
# patchDef2 is a quadratic bezier - only each 3x3 block's CORNERS sit exactly
# on the rendered surface, the rest are curve handles that pull the surface
# toward them without necessarily reaching them. Source's displacement is a
# literal flat-shaded mesh where every single grid point is a real vertex.
# For gentle terrain this converts closely; sharp/jagged detail will look
# visibly smoothed compared to the original - there's no way around that
# within what a bezier patch fundamentally is.

DEFAULT_TEXTURE_PIXEL_SIZE = (512, 512)
TEXTURE_PIXEL_SIZES = {}  # material (lowercase) -> (width, height), populated in main() from --texture-sizes


def parse_grid_rows(node, expected_rows):
    """Parse a VMF 'normals'/'distances'/'offsets' block's "row0".."rowN-1"
    string values into a list of lists of floats."""
    rows = []
    for i in range(expected_rows):
        raw = get1(node, 'row{}'.format(i))
        if raw is None:
            rows.append(None)
            continue
        rows.append([float(x) for x in raw.split()])
    return rows


def bilinear_upsample_grid(grid, old_n, new_n, is_vector, normalize=False):
    """Resample an old_n x old_n grid (of scalars, or 3-tuples if
    is_vector) up to new_n x new_n via bilinear interpolation - matching
    Source's own linear displacement semantics exactly, so the upsampled
    grid represents the SAME underlying shape as the original, just
    sampled more finely (not a different, smoothed or bulged surface).

    normalize=True renormalizes vector results to unit length after
    interpolating (for normals: interpolating two unit vectors doesn't
    generally produce one). normalize=False (the default, and what
    offsets must use) leaves interpolated vectors as plain, un-renormalized
    values - confirmed as a real, previously-shipped bug to get this
    wrong: offsets are frequently and legitimately (0, 0, 0), and
    renormalizing would replace that exact zero with a fabricated
    fallback unit vector (0, 0, 1), silently adding a full extra unit of
    height at every upsampled vertex whose interpolated offset happened to
    round near zero - caught by directly reproducing this function's own
    output against the real pipeline and finding an unexplained ~1.27
    unit Z discrepancy that traced back to exactly this fallback firing.
    """
    if old_n == new_n:
        return grid

    def sample(r, c):
        # map new-grid (r, c) to the equivalent fractional old-grid position
        old_r = r * (old_n - 1) / (new_n - 1)
        old_c = c * (old_n - 1) / (new_n - 1)
        r0, c0 = int(old_r), int(old_c)
        r1, c1 = min(r0 + 1, old_n - 1), min(c0 + 1, old_n - 1)
        tr, tc = old_r - r0, old_c - c0
        v00, v01 = grid[r0][c0], grid[r0][c1]
        v10, v11 = grid[r1][c0], grid[r1][c1]
        if is_vector:
            top = tuple(v00[i] + (v01[i] - v00[i]) * tc for i in range(3))
            bot = tuple(v10[i] + (v11[i] - v10[i]) * tc for i in range(3))
            val = tuple(top[i] + (bot[i] - top[i]) * tr for i in range(3))
            return val
        else:
            top = v00 + (v01 - v00) * tc
            bot = v10 + (v11 - v10) * tc
            return top + (bot - top) * tr

    out = []
    for r in range(new_n):
        row = []
        for c in range(new_n):
            v = sample(r, c)
            if is_vector and normalize:
                length = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
                v = (v[0] / length, v[1] / length, v[2] / length) if length > 1e-9 else (0.0, 0.0, 1.0)
            row.append(v)
        out.append(row)
    return out


def parse_dispinfo(dispinfo, upsample=True):
    power = int(float(get1(dispinfo, 'power', '3')))
    n = (1 << power) + 1  # 2^power + 1 vertices per side

    startpos_str = get1(dispinfo, 'startposition', '[0 0 0]')
    startpos = tuple(float(x) for x in startpos_str.strip('[]').split())

    normals_node = get1(dispinfo, 'normals', {}) or {}
    distances_node = get1(dispinfo, 'distances', {}) or {}
    offsets_node = get1(dispinfo, 'offsets', {}) or {}
    alphas_node = get1(dispinfo, 'alphas', {}) or {}

    normal_rows_flat = parse_grid_rows(normals_node, n) if isinstance(normals_node, dict) else [None] * n
    dist_rows = parse_grid_rows(distances_node, n) if isinstance(distances_node, dict) else [None] * n
    offset_rows_flat = parse_grid_rows(offsets_node, n) if isinstance(offsets_node, dict) else [None] * n
    alpha_rows = parse_grid_rows(alphas_node, n) if isinstance(alphas_node, dict) else [None] * n

    normals = []
    distances = []
    offsets = []
    alphas = []
    for r in range(n):
        nrow, drow, orow, arow = [], [], [], []
        nflat = normal_rows_flat[r]
        dflat = dist_rows[r]
        oflat = offset_rows_flat[r]
        aflat = alpha_rows[r]
        for c in range(n):
            if nflat and len(nflat) >= (c + 1) * 3:
                nrow.append((nflat[c * 3], nflat[c * 3 + 1], nflat[c * 3 + 2]))
            else:
                nrow.append((0.0, 0.0, 1.0))
            drow.append(dflat[c] if dflat and c < len(dflat) else 0.0)
            if oflat and len(oflat) >= (c + 1) * 3:
                orow.append((oflat[c * 3], oflat[c * 3 + 1], oflat[c * 3 + 2]))
            else:
                orow.append((0.0, 0.0, 0.0))
            arow.append(aflat[c] if aflat and c < len(aflat) else 0.0)
        normals.append(nrow)
        distances.append(drow)
        offsets.append(orow)
        alphas.append(arow)

    # Upsample to the map-wide target power (see TERRAIN_TARGET_POWER's own
    # comment) so every displacement's edges match its neighbors' Bezier
    # segment structure, regardless of what power this ONE displacement
    # was originally authored at.
    if upsample and TERRAIN_TARGET_POWER is not None and TERRAIN_TARGET_POWER > power:
        new_power = TERRAIN_TARGET_POWER
        new_n = (1 << new_power) + 1
        normals = bilinear_upsample_grid(normals, n, new_n, is_vector=True, normalize=True)
        distances = bilinear_upsample_grid(distances, n, new_n, is_vector=False)
        offsets = bilinear_upsample_grid(offsets, n, new_n, is_vector=True, normalize=False)
        alphas = bilinear_upsample_grid(alphas, n, new_n, is_vector=False)
        power, n = new_power, new_n

    return power, n, startpos, normals, distances, offsets, alphas


def plane_from_points(p1, p2, p3):
    # NOTE: deliberately (p1-p2, p3-p2), NOT (p2-p1, p3-p1) - verified
    # against real VMF solid geometry that THIS winding gives Source's
    # actual outward-facing normal (needed here for genuine inside/outside
    # clipping); the existing plane_normal() elsewhere in this file uses
    # the other winding, which has always been fine there only because
    # dominant-axis selection for texture alignment doesn't care about
    # normal sign - it does matter here.
    n = normalize(cross(sub(p1, p2), sub(p3, p2)))
    d = dot(n, p1)
    return n, d


def clip_polygon_to_halfspace(poly, plane_n, plane_d, epsilon=1e-6):
    """Sutherland-Hodgman clip: keeps the part of poly where dot(p,n) <= d
    (matches a Source solid's convention of outward face normals bounding
    the interior from the negative side)."""
    if not poly:
        return poly
    out = []
    count = len(poly)
    for i in range(count):
        cur = poly[i]
        prev = poly[i - 1]
        cur_in = dot(cur, plane_n) - plane_d <= epsilon
        prev_in = dot(prev, plane_n) - plane_d <= epsilon
        if cur_in:
            if not prev_in:
                out.append(_segment_plane_intersect(prev, cur, plane_n, plane_d))
            out.append(cur)
        elif prev_in:
            out.append(_segment_plane_intersect(prev, cur, plane_n, plane_d))
    return out


def _segment_plane_intersect(a, b, plane_n, plane_d):
    da = dot(a, plane_n) - plane_d
    db = dot(b, plane_n) - plane_d
    t = da / (da - db) if (da - db) != 0 else 0.0
    return lerp3(a, b, t)


def get_face_polygon(target_side, all_sides, radius=1e6):
    """Build the actual bounded polygon for one brush face by clipping a
    huge quad on its plane against every OTHER face's plane in the same
    solid - the same CSG process a Source/Quake engine performs at runtime,
    needed here because a VMF/.map face is stored as just 3 plane-defining
    points, not an explicit polygon boundary."""
    target_pts = parse_plane(get1(target_side, 'plane'))
    n, d = plane_from_points(*target_pts)

    # build two in-plane basis vectors, then a huge quad centered on a
    # point known to be on the plane
    tx, ty = texture_axis_from_plane(n)
    # texture_axis_from_plane's tx/ty aren't necessarily IN the plane for
    # non-axis-aligned faces (that's the whole "tilted" issue elsewhere in
    # this file) - build a true in-plane basis via Gram-Schmidt instead.
    arbitrary = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = normalize(sub(arbitrary, scale_vec(n, dot(arbitrary, n))))
    v = cross(n, u)
    center = target_pts[0]
    poly = [
        add(add(center, scale_vec(u, -radius)), scale_vec(v, -radius)),
        add(add(center, scale_vec(u, radius)), scale_vec(v, -radius)),
        add(add(center, scale_vec(u, radius)), scale_vec(v, radius)),
        add(add(center, scale_vec(u, -radius)), scale_vec(v, radius)),
    ]

    for other in all_sides:
        if other is target_side or not isinstance(other, dict):
            continue
        other_pts = parse_plane(get1(other, 'plane'))
        on, od = plane_from_points(*other_pts)
        poly = clip_polygon_to_halfspace(poly, on, od)
        if len(poly) < 3:
            return None
    return poly


def order_quad_from_startpos(poly, startpos):
    """Reorder a clipped quad's 4 corners so index 0 is whichever corner is
    closest to Source's startposition (the dispinfo's declared row0/col0
    anchor), preserving the polygon's existing winding direction."""
    if not poly or len(poly) != 4:
        return None
    best_i = min(range(4), key=lambda i: length(sub(poly[i], startpos)))
    return poly[best_i:] + poly[:best_i]


def disp_elevation_vector(side, dispinfo):
    """Source's 'elevation' term, UNSCALED: the displacement's whole surface
    is pushed along the face's outward normal by this many units, on top of
    normals*distances and offsets. Taken verbatim from Source SDK 2013
    builddisp.cpp, CCoreDispInfo::GenerateDispSurf():
        vert = flatVert + surfNormal*m_Elevation + subdivPos(offset)
               + fieldVector*fieldDistance
    Previously ignored entirely by every displacement path in this file -
    harmless on maps where every elevation is 0 (true for de_dust2), wrong
    on any map that uses it."""
    try:
        elevation = float(get1(dispinfo, 'elevation', '0') or 0.0)
    except (TypeError, ValueError):
        elevation = 0.0
    if elevation == 0.0:
        return (0.0, 0.0, 0.0)
    fn, _ = plane_from_points(*parse_plane(get1(side, 'plane')))
    return scale_vec(fn, elevation)


def build_patch_uv(point, u_dir, u_shift, u_scale, v_dir, v_shift, v_scale, tex_w, tex_h):
    tex_u = dot(point, u_dir) / u_scale + u_shift
    tex_v = dot(point, v_dir) / v_scale + v_shift
    return tex_u / tex_w, tex_v / tex_h


def project_alphamap_by_world_position(flat_pos, grid_layers, n):
    """Build the alphamap grid to WRITE by projecting each vertex's alpha
    value according to its own FLAT (undisplaced) world-space position,
    not its [row][col] index in our own grid construction - matching the
    user's own description of how the compiler actually applies it: a
    simple top-down projection onto the displacement's own world-space
    bounding box, independent of internal vertex traversal order.

    Default: project onto (X, Y) - column tracks X increasing, row tracks
    Y decreasing (row 0 = max Y). This default, and that exact row/column
    orientation, is directly confirmed correct against real, independently
    tested displacements (ids 17549, 17537, 8951) of differing shapes and
    orientations - not inferred.

    Falls back to a different axis pair ONLY when the default's chosen
    axes don't each vary independently with one of this grid's own row/
    column directions (see the axis-pick step below) - this happens for a
    displacement steep/oriented enough that X or Y barely varies across
    it (e.g. a wall running almost exactly along one world axis).

    IMPORTANT: this maps grid index directly (r -> output row or column,
    c -> the other, each possibly reversed based on whether that axis
    increases or decreases with the index) - it does NOT round each
    vertex's absolute (u, v) position to a pixel coordinate. An earlier
    version did exactly that, and was confirmed - via direct, real-data
    collision counting on a real displacement (id 8853) - to leave some
    output cells completely empty while others silently got overwritten
    by more than one source vertex, whenever the quad wasn't a perfect
    axis-aligned rectangle (e.g. a trapezoid, or any quad with a slight
    skew - which is common). Index-based mapping is a straight bijection
    for a regular n x n grid: every output cell gets exactly one value,
    always, regardless of the quad's exact shape. Two earlier attempts to
    guess the row/column axis choice from "largest range" or "best
    independent row/column variation" were each individually falsified by
    real per-displacement test data - if a differently shaped displacement
    still comes out wrong, that is real evidence this logic itself needs
    revisiting with fresh test data, not reason to swap in another guess.
    """
    def variation(axis, along_rows):
        # average spread of `axis` when moving along rows (fixed column) or
        # along columns (fixed row) - whichever `along_rows` selects.
        if along_rows:
            return sum(
                max(flat_pos[r][c][axis] for r in range(n)) - min(flat_pos[r][c][axis] for r in range(n))
                for c in range(n)
            ) / n
        else:
            return sum(
                max(flat_pos[r][c][axis] for c in range(n)) - min(flat_pos[r][c][axis] for c in range(n))
                for r in range(n)
            ) / n

    def try_axes(u_axis, v_axis):
        """Returns (out_grid, is_valid). Valid only if u_axis and v_axis
        each align distinctly with one of {row, column} (one of the two
        possible pairings, whichever fits better) - i.e. this quad isn't
        so skewed that an axis fails to consistently track either grid
        direction."""
        u_row_var, u_col_var = variation(u_axis, True), variation(u_axis, False)
        v_row_var, v_col_var = variation(v_axis, True), variation(v_axis, False)
        # Two possible pairings: (u->column, v->row) or (u->row, v->column).
        # Pick whichever pairing gives the larger combined variation -
        # confirmed against real data (build_displacement_terrain_patch's
        # own row0/col0 convention) that "u tracks column, v tracks row" is
        # the right default whenever both have real column/row variation
        # respectively.
        score_normal = u_col_var + v_row_var
        score_swapped = u_row_var + v_col_var
        if max(score_normal, score_swapped) < 1e-6:
            return None, False  # neither axis varies meaningfully at all
        swapped = score_swapped > score_normal

        out = [[0 for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if not swapped:
                    # u_axis tracks column (c), v_axis tracks row (r)
                    col_idx, row_idx = c, r
                else:
                    # u_axis tracks row (r), v_axis tracks column (c)
                    col_idx, row_idx = r, c
                out[row_idx][col_idx] = None  # placeholder, filled below via direction check
        # Determine direction: does u_axis increase or decrease along its
        # tracked index (0 -> n-1)? Row 0 = max v_axis (matching the
        # confirmed default); column 0 = min u_axis.
        if not swapped:
            u_first, u_last = flat_pos[0][0][u_axis], flat_pos[0][n - 1][u_axis]
            v_first, v_last = flat_pos[0][0][v_axis], flat_pos[n - 1][0][v_axis]
        else:
            u_first, u_last = flat_pos[0][0][u_axis], flat_pos[n - 1][0][u_axis]
            v_first, v_last = flat_pos[0][0][v_axis], flat_pos[0][n - 1][v_axis]
        u_increasing = u_last >= u_first
        v_increasing = v_last >= v_first  # row should DECREASE as v increases -> row0=max v

        for r in range(n):
            for c in range(n):
                if not swapped:
                    col_idx = c if u_increasing else (n - 1 - c)
                    row_idx = (n - 1 - r) if v_increasing else r
                else:
                    col_idx = r if u_increasing else (n - 1 - r)
                    row_idx = (n - 1 - c) if v_increasing else c
                out[row_idx][col_idx] = grid_layers[r][c]
        return out, True

    # (X, Y) first - the confirmed default - then fallbacks, in that order.
    candidates = [(0, 1), (0, 2), (1, 2)]
    for u_axis, v_axis in candidates:
        out, valid = try_axes(u_axis, v_axis)
        if valid:
            return out
    # Nothing varied meaningfully on any plane (a degenerate/near-zero-size
    # displacement) - fall back to direct index passthrough rather than
    # crash; better than nothing for a case that shouldn't occur in
    # practice.
    return [row[:] for row in grid_layers]


def write_pcx_indexed(width, height, pixel_rows):
    """Write a minimal 8-bit palette-indexed PCX (v5) image, RLE-encoded,
    from pixel_rows (list of `height` rows, each a list of `width` byte
    values 0-255). SOF2's terrain compiler reads each pixel's raw palette
    INDEX directly as a layer number - it never displays this image itself -
    but the palette still needs genuinely distinct colors per layer so a
    person CAN see and hand-paint it in a normal image editor, which is the
    whole point of this being an editable file at all. Uses the same
    teal/green/blue/orange set as a confirmed-working real SOF2 terrain
    alphamap (see LAYER_COLORS below).

    No use of PIL/Pillow here deliberately: this keeps vmf2map.py free of
    any third-party dependency, matching every other function in this file."""
    bytes_per_line = width + (width % 2)  # PCX requires an even scanline byte count

    header = bytearray(128)
    header[0] = 10          # manufacturer: ZSoft
    header[1] = 5           # version 5
    header[2] = 1           # RLE encoding
    header[3] = 8           # bits per pixel
    header[4:6] = (0).to_bytes(2, 'little')            # xmin
    header[6:8] = (0).to_bytes(2, 'little')            # ymin
    header[8:10] = (width - 1).to_bytes(2, 'little')   # xmax
    header[10:12] = (height - 1).to_bytes(2, 'little')  # ymax
    header[12:14] = (72).to_bytes(2, 'little')  # hdpi
    header[14:16] = (72).to_bytes(2, 'little')  # vdpi
    # bytes 16-63: 16-color EGA palette, unused at 8bpp - left zeroed
    header[64] = 0           # reserved
    header[65] = 1           # number of color planes
    header[66:68] = bytes_per_line.to_bytes(2, 'little')
    header[68:70] = (1).to_bytes(2, 'little')  # palette info: 1 = color/BW
    # remaining header bytes (screen size etc.) left zeroed - not required for reading

    body = bytearray()
    for row in pixel_rows:
        row = list(row) + [0] * (bytes_per_line - len(row))  # pad to even width if needed
        i = 0
        while i < len(row):
            value = row[i]
            run = 1
            while i + run < len(row) and row[i + run] == value and run < 63:
                run += 1
            if run > 1 or value >= 0xC0:
                body.append(0xC0 | run)
                body.append(value)
            else:
                body.append(value)
            i += run

    # 256-entry VGA palette block. The engine reads each pixel's raw index
    # directly as a layer number, so the RGB values are cosmetic to the
    # engine - but NOT to a person opening this file to hand-paint the
    # blend boundaries, which is the entire point of it being a normal,
    # editable image format. A grayscale ramp put indices 0/1 at
    # near-black-on-black, making it unreadable/unpaintable in practice.
    # These first four colors are copied directly from a confirmed-working
    # real SOF2 terrain alphamap (mif_bridge's bridgea.pcx) - teal/green/
    # blue/orange, one clearly distinct color per layer index.
    LAYER_COLORS = [
        (64, 128, 128),   # layer 0
        (0, 128, 0),      # layer 1
        (0, 128, 192),    # layer 2
        (255, 128, 0),    # layer 3
    ]
    palette = bytearray([0x0C])
    for i in range(256):
        if i < len(LAYER_COLORS):
            r, g, b = LAYER_COLORS[i]
        else:
            r = g = b = 0
        palette += bytes([r, g, b])

    return bytes(header) + bytes(body) + bytes(palette)


_VMT_CACHE = {}  # material name (lowercase) -> (shader_class, basetexture2) or None if not found/no blend


def read_vmt_blend_info(material, materials_dir):
    """For a WorldVertexTransition (Source's displacement blend) material,
    return (shader_class, basetexture2_or_None). Returns None entirely if
    materials_dir isn't given, the .vmt can't be found, or it isn't a
    WorldVertexTransition material at all. VMT syntax is the same
    KeyValues format as VMF, so this reuses tokenize/parse_vdf rather than
    a separate parser."""
    key = material.strip().lower().replace('\\', '/')
    if key in _VMT_CACHE:
        return _VMT_CACHE[key]

    result = None
    if materials_dir:
        vmt_path = os.path.join(materials_dir, *key.split('/')) + '.vmt'
        if os.path.isfile(vmt_path):
            try:
                with open(vmt_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                tokens = tokenize(text)
                root = parse_vdf(tokens)
                if root:
                    shader_class = list(root.keys())[0]
                    if shader_class.strip().lower() == 'worldvertextransition':
                        body = root[shader_class][0]
                        basetexture2 = get1(body, 'basetexture2')
                        result = (shader_class, basetexture2)
            except (OSError, IndexError, KeyError):
                result = None

    _VMT_CACHE[key] = result
    return result


def terrain_scale_key(u_scale, v_scale):
    """Short, deterministic, shader-name-safe string from a displacement's
    own uaxis/vaxis scale, rounded to 4 decimals so floating-point noise
    doesn't split otherwise-identical scales into separate shaders.
    Displacements of the same material with the SAME scale share one
    shader name (and therefore one set of worldspawn vertexremapshader
    entries); only a genuinely different scale gets its own. Confirmed
    necessary: making every displacement's shader name unique regardless
    of scale caused a real map's worldspawn entity text to exceed the
    engine's own per-entity size limit (G_AddSpawnVarToken: MAX_SPAWN_CHARS)
    once there were enough displacements - grouping by actual scale value
    instead keeps the common case (most displacements sharing the same,
    default scale) down to one shader, only splitting out the rare
    displacement that's genuinely different.
    """
    return '{:.4f}x{:.4f}'.format(u_scale, v_scale).replace('.', '_').replace('-', 'n')


def build_displacement_terrain(side, all_sides, texture_map, unmapped_counter,
                                scale, offset, warnings, material):
    """Build a func_group 'terrain' entity for a WorldVertexTransition
    displacement: one full-depth triangular-prism brush per displacement
    triangle (top face 'tools/_terrain', sides/bottom 'tools/_caulk'), plus
    a palette-indexed .pcx alphamap encoding each grid vertex's dominant
    layer. Mirrors the exact structure of a real, confirmed-working SOF2
    terrain entity (id 27, key content-flags value 134217728 - preserved
    verbatim from that reference map, since its exact bit meaning isn't
    independently documented anywhere accessible here).

    Returns (entity_text, pcx_bytes, pcx_filename, base_shader_name, material,
    u_scale, v_scale) or None if the displacement's face geometry couldn't
    be resolved. u_scale/v_scale are the displacement's own uaxis/vaxis
    scale values, needed so vmt2shader.py can compute a tcmod scale that
    matches THIS specific displacement's texture density, not a single
    value assumed for the whole map.
    """
    dispinfo = get1(side, 'dispinfo')
    if not isinstance(dispinfo, dict):
        return None

    side_id = get1(side, 'id') or 'unknown'
    uaxis_str = get1(side, 'uaxis')
    vaxis_str = get1(side, 'vaxis')
    _, _, u_scale = parse_axis(uaxis_str) if uaxis_str else ((1, 0, 0), 0.0, 0.25)
    _, _, v_scale = parse_axis(vaxis_str) if vaxis_str else ((0, 1, 0), 0.0, 0.25)
    # Shader name is shared by every displacement of this material that
    # uses the SAME uaxis/vaxis scale (see terrain_scale_key) - not unique
    # per-displacement, which blew up worldspawn's entity size once there
    # were enough displacements. The .pcx alphamap filename below still
    # gets its own, separate per-displacement uniqueness via side_id,
    # since that part genuinely must never be shared (confirmed critical
    # bug, fixed earlier).
    base_shader_name = (material.strip().lower().replace('\\', '/') + '/terrain_' +
                         terrain_scale_key(u_scale, v_scale))
    num_layers = 2

    poly = get_face_polygon(side, all_sides)
    if not poly or len(poly) != 4:
        warnings.append(
            "terrain displacement on material '{}' didn't resolve to a clean quad "
            "(got {} corners) - skipped, base face left flat".format(
                material, len(poly) if poly else 0))
        return None

    power, n, startpos, normals, distances, offsets, alphas = parse_dispinfo(dispinfo)
    quad = order_quad_from_startpos(poly, startpos)
    if quad is None:
        warnings.append("terrain displacement on material '{}' - couldn't match startposition "
                         "to a quad corner, skipped".format(material))
        return None
    c00, c01, c11, c10 = quad
    elev_unscaled = disp_elevation_vector(side, dispinfo)
    elev_scaled = scale_vec(elev_unscaled, scale)

    grid_points = []   # [row][col] -> world-space (x, y, z), already scale/offset applied
    grid_layers = []   # [row][col] -> 0 or 1 (thresholded from alpha)
    flat_pos = []      # [row][col] -> (x, y, z) of the FLAT, undisplaced vertex (no normal offset)
    material_key = (material or '').strip().lower()
    for r in range(n):
        t = r / (n - 1)
        row_start = lerp3(c00, c10, t)
        row_end = lerp3(c01, c11, t)
        pt_row, layer_row, flat_row = [], [], []
        for c in range(n):
            s = c / (n - 1)
            base_unscaled = lerp3(row_start, row_end, s)
            base = (base_unscaled[0] * scale - offset[0], base_unscaled[1] * scale - offset[1],
                    base_unscaled[2] * scale - offset[2])
            flat_row.append(base)
            disp_normal = normals[r][c]
            disp_dist = distances[r][c] * scale
            off = scale_vec(offsets[r][c], scale)
            final = add(add(add(base, elev_scaled), scale_vec(disp_normal, disp_dist)), off)
            pt_row.append(final)
            if DEBUG_TERRAIN_CORNER_MARKER:
                layer_row.append(1 if r == 0 else 0)
            else:
                recon_key = (material_key, round(base_unscaled[0], 1), round(base_unscaled[1], 1))
                if TERRAIN_ALPHA_RECONCILIATION and recon_key in TERRAIN_ALPHA_RECONCILIATION:
                    layer_row.append(1 if TERRAIN_ALPHA_RECONCILIATION[recon_key] >= 128.0 else 0)
                elif TERRAIN_ALPHA_FEATHER and recon_key in TERRAIN_ALPHA_FEATHER:
                    layer_row.append(1 if TERRAIN_ALPHA_FEATHER[recon_key] >= 128.0 else 0)
                else:
                    layer_row.append(1 if alphas[r][c] >= 128.0 else 0)
        grid_points.append(pt_row)
        grid_layers.append(layer_row)
        flat_pos.append(flat_row)

    # Winding fix for the visible geometry - the row traversal order above
    # renders back-facing without this.
    grid_points = grid_points[::-1]

    PRISM_DEPTH = 4096.0  # extruded straight down from each triangle's own lowest vertex

    def emit_prism(p1, p2, p3, layer_pair):
        """One watertight triangular-prism brush: top face (real terrain
        triangle, 'tools/_terrain') + a flat bottom + 3 vertical sides
        (all 'tools/_caulk'), matching the reference map's own structure."""
        min_z = min(p1[2], p2[2], p3[2]) - PRISM_DEPTH
        b1 = (p1[0], p1[1], min_z)
        b2 = (p2[0], p2[1], min_z)
        b3 = (p3[0], p3[1], min_z)

        # ONE mixed layer pair on this triangle picks the blend shader;
        # a uniform triangle just gets its single layer's plain shader -
        # the content-flags value (134217728) is preserved from the
        # reference map on every side, matching that map exactly.
        CF = 134217728
        lines = ["{"]

        def side_line(a, b, c, tex):
            return "( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) {} 0 0 0 0.500000 0.500000 {} 0 0".format(
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], tex, CF)

        lines.append(side_line(p1, p2, p3, "tools/_terrain"))
        lines.append(side_line(b1, b3, b2, "tools/_caulk"))       # bottom (reversed winding)
        lines.append(side_line(p1, b1, b2, "tools/_caulk"))       # side 1-2
        lines.append(side_line(p2, b2, b3, "tools/_caulk"))       # side 2-3
        lines.append(side_line(p3, b3, b1, "tools/_caulk"))       # side 3-1
        lines.append("}")
        return "\n".join(lines)

    brushes = []
    for r in range(n - 1):
        for c in range(n - 1):
            p00, p01 = grid_points[r][c], grid_points[r][c + 1]
            p10, p11 = grid_points[r + 1][c], grid_points[r + 1][c + 1]
            l00, l01 = grid_layers[r][c], grid_layers[r][c + 1]
            l10, l11 = grid_layers[r + 1][c], grid_layers[r + 1][c + 1]
            brushes.append(emit_prism(p00, p10, p11, (l00, l10, l11)))
            brushes.append(emit_prism(p00, p11, p01, (l00, l11, l01)))

    # base_shader_name is now shared by scale, NOT unique per-displacement
    # (see terrain_scale_key) - side_id must be appended explicitly here
    # so the alphamap filename itself stays unique per-displacement,
    # regardless of whether its shader name is shared with others.
    pcx_filename = base_shader_name.replace('/', '_') + '_' + str(side_id) + '.pcx'

    entity_lines = [
        "{",
        '"terrain" "1"',
        '"layers" "{}"'.format(num_layers),
        '"shader" "{}"'.format(base_shader_name),
        '"alphamap" "{}"'.format(pcx_filename),
        '"classname" "func_group"',
    ]
    entity_lines.extend(brushes)
    entity_lines.append("}")

    # Alphamap image: one pixel per displacement grid vertex, projected
    # onto the output grid by each vertex's own real world-space position
    # (see project_alphamap_by_world_position) rather than any fixed
    # row/column rule. A simple row-reversal was confirmed correct for one
    # displacement (17549) via a direct in-game/manual-fix comparison, but
    # a second displacement (17537) at a different orientation broke under
    # that same fixed rule - confirmed directly by comparing real flat
    # vertex coordinates: world Y tracked this code's "row" index for
    # 17549 but tracked "column" for 17537, a 90-degree relative
    # difference in how each was originally placed in Hammer. A per-
    # displacement, position-based projection is the only way that
    # generalizes correctly to every orientation.
    _alpha_out = project_alphamap_by_world_position(flat_pos, grid_layers, n)
    pcx_bytes = write_pcx_indexed(n, n, _alpha_out)

    return ("\n".join(entity_lines), pcx_bytes, pcx_filename, base_shader_name,
            material, u_scale, v_scale)


def build_displacement_terrain_patch(side, all_sides, texture_map, unmapped_counter,
                                      scale, offset, warnings, material):
    """Alternative to build_displacement_terrain: wraps a single patchDef2
    (the SAME proven geometry/UV grid logic as build_displacement_patch)
    inside the terrain func_group entity, instead of per-triangle prism
    brushes. Same alphamap/shader/layers keys, same generated .pcx.

    This is a genuine, uncertain experiment, not a proven-working path:
    the one terrain-entity reference found for this system describes it
    as "a func_group entity (brushes only)" - suggesting q3map2's
    alphamap/vertex-color-baking pass may specifically target brush BSP
    surfaces, since patches compile through q3map2's separate bezier-
    tessellation path entirely. Whether the terrain vertex-color bake
    applies to patch geometry at all is unverified either way - worth
    testing directly against the brush-based approach's own separate,
    unresolved problem rather than assuming either one is right.

    Returns (entity_text, pcx_bytes, pcx_filename, base_shader_name, material,
    u_scale, v_scale) or None on the same failure conditions as the other
    two builders.
    """
    dispinfo = get1(side, 'dispinfo')
    if not isinstance(dispinfo, dict):
        return None

    side_id = get1(side, 'id') or 'unknown'
    uaxis_str = get1(side, 'uaxis')
    vaxis_str = get1(side, 'vaxis')
    u_dir, u_shift, u_scale = parse_axis(uaxis_str) if uaxis_str else ((1, 0, 0), 0.0, 0.25)
    v_dir, v_shift, v_scale = parse_axis(vaxis_str) if vaxis_str else ((0, 1, 0), 0.0, 0.25)
    # Shared by scale, not unique per-displacement - see terrain_scale_key
    # and build_displacement_terrain's own matching comment (confirmed
    # necessary: MAX_SPAWN_CHARS on worldspawn once made unique regardless
    # of scale).
    base_shader_name = (material.strip().lower().replace('\\', '/') + '/terrain_' +
                         terrain_scale_key(u_scale, v_scale))
    num_layers = 2

    poly = get_face_polygon(side, all_sides)
    if not poly or len(poly) != 4:
        warnings.append(
            "terrain displacement (patch mode) on material '{}' didn't resolve to a clean "
            "quad (got {} corners) - skipped, base face left flat".format(
                material, len(poly) if poly else 0))
        return None

    power, n, startpos, normals, distances, offsets, alphas = parse_dispinfo(dispinfo)
    quad = order_quad_from_startpos(poly, startpos)
    if quad is None:
        warnings.append("terrain displacement (patch mode) on material '{}' - couldn't match "
                         "startposition to a quad corner, skipped".format(material))
        return None
    c00, c01, c11, c10 = quad
    elev_unscaled = disp_elevation_vector(side, dispinfo)
    elev_scaled = scale_vec(elev_unscaled, scale)

    tex_w, tex_h = TEXTURE_PIXEL_SIZES.get(material.lower(), DEFAULT_TEXTURE_PIXEL_SIZE)

    grid_points = []   # [row][col] -> (final_scaled_xyz, u, v)
    grid_layers = []   # [row][col] -> 0 or 1 (thresholded from alpha) - same as brush mode
    flat_pos = []      # [row][col] -> (x, y, z) of the FLAT, undisplaced vertex (no normal offset)
    material_key = (material or '').strip().lower()
    for r in range(n):
        t = r / (n - 1)
        row_start = lerp3(c00, c10, t)
        row_end = lerp3(c01, c11, t)
        pt_row, layer_row, flat_row = [], [], []
        for c in range(n):
            s = c / (n - 1)
            base_unscaled = lerp3(row_start, row_end, s)
            base = (base_unscaled[0] * scale - offset[0], base_unscaled[1] * scale - offset[1],
                    base_unscaled[2] * scale - offset[2])
            flat_row.append(base)
            disp_normal = normals[r][c]
            disp_dist = distances[r][c] * scale
            off = scale_vec(offsets[r][c], scale)
            final = add(add(add(base, elev_scaled), scale_vec(disp_normal, disp_dist)), off)
            # Same scale-vs-UV fix as build_displacement_patch: UV comes
            # from the unscaled position, not the scaled one.
            final_unscaled = add(add(add(base_unscaled, elev_unscaled), scale_vec(disp_normal, distances[r][c])),
                                  offsets[r][c])
            u, v = build_patch_uv(base_unscaled, u_dir, u_shift, u_scale, v_dir, v_shift,
                                   v_scale, tex_w, tex_h)
            pt_row.append((final, u, v))
            if DEBUG_TERRAIN_CORNER_MARKER:
                layer_row.append(1 if r == 0 else 0)
            else:
                recon_key = (material_key, round(base_unscaled[0], 1), round(base_unscaled[1], 1))
                if TERRAIN_ALPHA_RECONCILIATION and recon_key in TERRAIN_ALPHA_RECONCILIATION:
                    layer_row.append(1 if TERRAIN_ALPHA_RECONCILIATION[recon_key] >= 128.0 else 0)
                elif TERRAIN_ALPHA_FEATHER and recon_key in TERRAIN_ALPHA_FEATHER:
                    layer_row.append(1 if TERRAIN_ALPHA_FEATHER[recon_key] >= 128.0 else 0)
                else:
                    layer_row.append(1 if alphas[r][c] >= 128.0 else 0)
        grid_points.append(pt_row)
        grid_layers.append(layer_row)
        flat_pos.append(flat_row)

    # Winding fix for the visible geometry.
    grid_points = grid_points[::-1]

    patch_lines = ["{", "patchDef2", "{", "tools/_terrain", "( {} {} 0 0 0 )".format(n, n), "("]
    for row in grid_points:
        row_str = " ".join(
            "( {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} )".format(p[0], p[1], p[2], u, v)
            for p, u, v in row
        )
        patch_lines.append("( {} )".format(row_str))
    patch_lines.append(")")
    patch_lines.append("}")
    patch_lines.append("}")

    # base_shader_name is scale-grouped now, not unique per-displacement -
    # side_id must be appended explicitly (same reasoning as brush mode).
    pcx_filename = base_shader_name.replace('/', '_') + '_patchb_' + str(side_id) + '.pcx'

    entity_lines = [
        "{",
        '"terrain" "1"',
        '"layers" "{}"'.format(num_layers),
        '"shader" "{}"'.format(base_shader_name),
        '"alphamap" "{}"'.format(pcx_filename),
        '"classname" "func_group"',
    ]
    entity_lines.extend(patch_lines)
    entity_lines.append("}")

    # World-space position projection - see build_displacement_terrain's
    # own matching comment for why a fixed transform can't generalize.
    _alpha_out = project_alphamap_by_world_position(flat_pos, grid_layers, n)
    pcx_bytes = write_pcx_indexed(n, n, _alpha_out)

    return ("\n".join(entity_lines), pcx_bytes, pcx_filename, base_shader_name,
            material, u_scale, v_scale)


_VMT_BODY_CACHE = {}


def read_vmt_param(material, materials_dir, param):
    """Return one $parameter's value from a material's .vmt (any shader
    class), or None. Keys are matched case-insensitively with or without
    the leading '$'."""
    key = (material or '').strip().lower().replace('\\', '/')
    if key not in _VMT_BODY_CACHE:
        body = None
        if materials_dir:
            vmt_path = os.path.join(materials_dir, *key.split('/')) + '.vmt'
            if os.path.isfile(vmt_path):
                try:
                    with open(vmt_path, 'r', encoding='utf-8', errors='replace') as f:
                        root = parse_vdf(tokenize(f.read()))
                    if root:
                        body = root[list(root.keys())[0]][0]
                except (OSError, IndexError, KeyError):
                    body = None
        _VMT_BODY_CACHE[key] = body
    body = _VMT_BODY_CACHE[key]
    if not isinstance(body, dict):
        return None
    want = param.lower().lstrip('$')
    for k, vals in body.items():
        if k.lower().lstrip('$') == want and vals and not isinstance(vals[0], dict):
            return vals[0]
    return None


def compute_displacement_grid_lwo(side, all_sides, scale, offset, warnings, material,
                                  is_blend=True, texture_map=None, unmapped_counter=None,
                                  materials_dir=None):
    """One displacement, built EXACTLY the way Source builds it (Source SDK
    2013, public/builddisp.cpp), for --terrain-mode lwo:

      * NATIVE power - no upsampling. Source never resamples a displacement;
        mixed-power neighbours are reconciled at the shared EDGE only (see
        LWOTerrainBuilder.finalize_points). The old path bilinearly upsampled
        normals and distances separately, measured on real de_dust2 data to
        put some rock displacements up to ~157 units off Source's surface.
      * Vertex = flat + faceNormal*elevation + offset + normal*distance
        (CCoreDispInfo::GenerateDispSurf). elevation used to be ignored.
      * Row i runs corner0 -> corner1 (next point of Source's clockwise face
        winding), column j runs corner0 -> corner3. Our clipped polygon is
        counter-clockwise, so that is quad[0]->quad[3] for i and
        quad[0]->quad[1] for j - i.e. c00->c10 and c00->c01, as before.
      * UVs from the FLAT position (CalcDispSurfCoords bilinearly
        interpolates the flat face's corner texcoords, so texture does NOT
        follow the displaced surface - that is Source's own look, including
        its familiar stretching on steep displacement faces).

    Returns a record dict for LWOTerrainBuilder.add_displacement, or None on
    the same failure conditions as the other displacement builders.
    """
    dispinfo = get1(side, 'dispinfo')
    if not isinstance(dispinfo, dict):
        return None

    uaxis_str = get1(side, 'uaxis')
    vaxis_str = get1(side, 'vaxis')
    u_dir, u_shift, u_scale = parse_axis(uaxis_str) if uaxis_str else ((1, 0, 0), 0.0, 0.25)
    v_dir, v_shift, v_scale = parse_axis(vaxis_str) if vaxis_str else ((0, 1, 0), 0.0, 0.25)

    poly = get_face_polygon(side, all_sides)
    if not poly or len(poly) != 4:
        warnings.append(
            "terrain displacement (lwo mode) on material '{}' didn't resolve to a clean "
            "quad (got {} corners) - skipped, base face left flat".format(
                material, len(poly) if poly else 0))
        return None

    power, n, startpos, normals, distances, offsets, alphas = parse_dispinfo(dispinfo, upsample=False)
    quad = order_quad_from_startpos(poly, startpos)
    if quad is None:
        warnings.append("terrain displacement (lwo mode) on material '{}' - couldn't match "
                         "startposition to a quad corner, skipped".format(material))
        return None
    c00, c01, c11, c10 = quad
    elev_unscaled = disp_elevation_vector(side, dispinfo)
    face_n, _ = plane_from_points(*parse_plane(get1(side, 'plane')))

    # Texture pixel size: look up the texture actually shown ($basetexture),
    # not the material name - a blend material's own name is never a
    # texture-size key, so this used to silently fall back to 512x512.
    size_key = material.lower()
    basetexture = read_vmt_param(material, materials_dir, 'basetexture')
    if basetexture:
        bt = basetexture.strip().lower().replace('\\', '/')
        if bt in TEXTURE_PIXEL_SIZES:
            size_key = bt
    tex_w, tex_h = TEXTURE_PIXEL_SIZES.get(size_key, DEFAULT_TEXTURE_PIXEL_SIZE)

    def to_world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])

    pos, pos_unscaled, flat, uv, alpha = [], [], [], [], []
    for i in range(n):
        t = i / (n - 1)
        row_start = lerp3(c00, c10, t)
        row_end = lerp3(c01, c11, t)
        prow, purow, frow, uvrow, arow = [], [], [], [], []
        for j in range(n):
            flat_unscaled = lerp3(row_start, row_end, j / (n - 1))
            final_unscaled = add(add(add(flat_unscaled, elev_unscaled), offsets[i][j]),
                                 scale_vec(normals[i][j], distances[i][j]))
            prow.append(to_world(final_unscaled))
            purow.append(final_unscaled)
            frow.append(to_world(flat_unscaled))
            uvrow.append(build_patch_uv(flat_unscaled, u_dir, u_shift, u_scale, v_dir, v_shift,
                                        v_scale, tex_w, tex_h))
            arow.append(max(0.0, min(1.0, alphas[i][j] / 255.0)) if is_blend else 1.0)
        pos.append(prow); pos_unscaled.append(purow); flat.append(frow)
        uv.append(uvrow); alpha.append(arow)

    if is_blend:
        # Same material+scale -> same shader, exactly as the other terrain
        # modes group them.
        nested_key = (material.strip().lower().replace('\\', '/') + '/terrain_' +
                      terrain_scale_key(u_scale, v_scale))
        surface = None
    else:
        # A plain single-texture displacement uses the material's ORDINARY
        # shader - the same one vmt2shader already generates for flat brush
        # faces with this material - so it looks identical to them. Needs
        # the full shader name (with textures/), since a model's surface name
        # is looked up as an exact shader name.
        mapped = convert_texture_name(material, texture_map or {},
                                      unmapped_counter if unmapped_counter is not None else Counter())
        mapped = mapped.replace('\\', '/')
        surface = mapped if mapped.lower().startswith('textures/') else 'textures/' + mapped
        nested_key = None

    return {
        'n': n, 'power': power, 'pos': pos, 'pos_unscaled': pos_unscaled, 'flat': flat,
        'uv': uv, 'alpha': alpha, 'face_n': face_n, 'is_blend': is_blend,
        'nested_key': nested_key, 'surface': surface, 'material': material,
        'u_scale': u_scale, 'v_scale': v_scale,
        'uv_axes': (u_dir, u_shift, v_dir, v_shift, tex_w, tex_h),
    }


def build_skybox_landscape_brushes(recs, cells, texture_map, unmapped_counter, warnings):
    """A COARSE, solid brush landscape approximating the 3D skybox area's
    displacement terrain, for use with --skybox-entity (see
    --skybox-landscape-brushes).

    Why brushes, and why coarse. q3map2's _skybox replicates every skybox
    drawsurface into every leaf that can see sky, and on de_dust2 that
    multiplier measured 1241 leaves per surface against the 1048576
    MAX_MAP_LEAFFACES budget - i.e. the WHOLE 3D skybox gets about 827
    surfaces to spend. The skybox terrain at native displacement resolution
    is 1024 triangles on its own, which is most of that budget gone on
    ground the player only ever sees as a distant silhouette. Decimating to
    `cells` x `cells` quads per displacement spends a small, predictable
    fraction of it instead: 2 triangles per cell, so cells=3 over 8
    displacements is 144 surfaces.

    Two earlier attempts at skybox terrain both failed for reasons this
    avoids:
      - one patchDef2 per displacement: crashed the GAME on .bsp load
        (MAX_PATCH_PLANES - the engine's per-patch collision facet array,
        blown by a dense bumpy Bezier patch), and reintroduced exactly the
        Bezier seam problems --terrain-mode lwo exists to avoid.
      - the LWO mesh as a forced-meta misc_model: correct, but 1024
        triangles against an 827-surface budget, so MAX_MAP_LEAFFACES.
    A prism brush carries no Bezier surface and no patch collision data at
    all, and the count here is chosen rather than inherited.

    Only the TOP face gets the real material; the sides and bottom are
    caulk. NOTE these prisms carry CONTENTS_DETAIL (like the brush-mode
    terrain they copy), so they do NOT seal the map - q3map2's FloodEntities
    ignores detail brushes. They are a visual backdrop only; sealing is a
    separate problem (see --seal-leak-from-lin).

    COST PER PRISM, measured rather than assumed: a real compile of
    de_dust2 with cells=3 (144 prisms) reported 1290 skybox surfaces, where
    the buildings and sky overlay account for roughly 450 of them - about
    SIX skybox surfaces per prism, not the one its single drawn face would
    suggest. Either the caulk sides still count or the tops get fragmented
    by the tree; either way, budget ~6 surfaces per prism when choosing
    cells, and prefer the smallest value that still reads as a landscape.

    Returns (func_group_text, n_brushes, n_surfaces) or None if there's
    nothing in the skybox area to build from."""
    if not recs or cells < 1:
        return None

    # Extrude every prism down to one shared floor just under the lowest
    # point of the whole skybox landscape, so the result is a single solid
    # mass rather than a field of loose spikes with gaps between them.
    all_z = [rec['pos'][i][j][2] for rec in recs for i in range(rec['n']) for j in range(rec['n'])]
    floor_z = min(all_z) - 64.0

    CF = 134217728  # same content-flags value the brush-mode terrain uses

    def side_line(a, b, c, tex):
        return ("( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) "
                "( {:.10g} {:.10g} {:.10g} ) {} 0 0 0 0.500000 0.500000 {} 0 0").format(
                    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], tex, CF)

    def emit_prism(p1, p2, p3, top_tex):
        b1 = (p1[0], p1[1], floor_z)
        b2 = (p2[0], p2[1], floor_z)
        b3 = (p3[0], p3[1], floor_z)
        return "\n".join([
            "{",
            side_line(p1, p2, p3, top_tex),
            side_line(b1, b3, b2, "tools/_caulk"),
            side_line(p1, b1, b2, "tools/_caulk"),
            side_line(p2, b2, b3, "tools/_caulk"),
            side_line(p3, b3, b1, "tools/_caulk"),
            "}",
        ])

    brushes = []
    degenerate = 0
    for rec in recs:
        n = rec['n']
        pos = rec['pos']
        mapped = convert_texture_name(rec.get('material') or 'tools/toolsnodraw',
                                      texture_map or {}, unmapped_counter)
        # Exactly `cells` intervals, evenly spaced over this displacement's
        # own grid, so CELLS means what it says regardless of the
        # displacement's power (n is 5/9/17 for power 2/3/4). Integer
        # stepping with a remainder row silently gives CELLS+1 instead.
        k = min(cells, n - 1)
        rows = sorted(set(int(round(i * (n - 1) / float(k))) for i in range(k + 1)))
        for ri in range(len(rows) - 1):
            for ci in range(len(rows) - 1):
                r0, r1 = rows[ri], rows[ri + 1]
                c0, c1 = rows[ci], rows[ci + 1]
                p00, p01 = pos[r0][c0], pos[r0][c1]
                p10, p11 = pos[r1][c0], pos[r1][c1]
                for tri in ((p00, p10, p11), (p00, p11, p01)):
                    a, b, c = _orient_top_ccw(*tri)
                    # Reject a decimated cell whose corners are collinear in
                    # plan view - it would make an invalid (zero-area top)
                    # brush that q3map2 reports as a degenerate plane.
                    ax = (b[0] - a[0], b[1] - a[1])
                    bx = (c[0] - a[0], c[1] - a[1])
                    if abs(ax[0] * bx[1] - ax[1] * bx[0]) < 1e-6:
                        degenerate += 1
                        continue
                    brushes.append(emit_prism(a, b, c, mapped))

    if not brushes:
        return None
    if degenerate:
        warnings.append(
            "--skybox-landscape-brushes: skipped {} decimated cell(s) whose corners were "
            "collinear in plan view (they'd have made zero-area brush tops).".format(degenerate))

    text = "\n".join(["{", '"classname" "func_group"'] + brushes + ["}"])
    return text, len(brushes), len(brushes)


# ==============================================================================
# Leak sealing from a q3map2 pointfile (--seal-leak-from-lin)
# ==============================================================================

def solid_planes_source(solid):
    """Outward-facing planes of one VMF solid, in SOURCE units, as
    [(normal, dist), ...] - or None if the solid has too few usable sides.

    Orientation is self-calibrated per brush rather than assumed from the
    winding convention: the plane set is built either way round, then every
    normal is flipped as needed so the brush's own centroid ends up INSIDE
    all of them. That makes the test independent of how a given .vmf happens
    to order its plane points."""
    pl = []
    for side in solid.get('side', []):
        if not isinstance(side, dict):
            continue
        plane_str = get1(side, 'plane')
        if not plane_str:
            continue
        tri = parse_plane(plane_str)
        if len(tri) != 3:
            continue
        p0, p1, p2 = tri
        t1 = [p0[i] - p1[i] for i in range(3)]
        t2 = [p2[i] - p1[i] for i in range(3)]
        n = [t1[1] * t2[2] - t1[2] * t2[1],
             t1[2] * t2[0] - t1[0] * t2[2],
             t1[0] * t2[1] - t1[1] * t2[0]]
        ln = math.sqrt(sum(c * c for c in n))
        if ln < 1e-9:
            continue
        n = [c / ln for c in n]
        pl.append((n, sum(n[i] * p1[i] for i in range(3)), tri))
    if len(pl) < 4:
        return None
    verts = [pt for _, _, tri in pl for pt in tri]
    cen = [sum(p[i] for p in verts) / len(verts) for i in range(3)]
    out = []
    for n, d, _ in pl:
        if sum(n[i] * cen[i] for i in range(3)) - d > 0:
            n = [-c for c in n]
            d = -d
        out.append((n, d))
    return out


def point_in_planes(planes, p, eps=0.5):
    """True if p is strictly inside every plane (by at least eps)."""
    for n, d in planes:
        if sum(n[i] * p[i] for i in range(3)) - d >= -eps:
            return False
    return True


def read_leak_pointfile(path, scale, offset, warnings, sample_step=8.0):
    """Read a q3map2 .lin leak pointfile (written in the COMPILED map's own
    units) and return a list of points in SOURCE units, densely sampled
    along each segment.

    Sampling matters: the pointfile's own vertices are leaf-portal crossings,
    so a brush can be passed straight THROUGH between two consecutive
    vertices without either vertex landing inside it. Walking each segment in
    small steps catches those."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw = []
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        raw.append(tuple(float(x) for x in parts))
                    except ValueError:
                        pass
    except OSError as e:
        warnings.append("--seal-leak-from-lin: couldn't read '{}' ({}) - nothing sealed.".format(
            path, e))
        return []
    if len(raw) < 2:
        warnings.append("--seal-leak-from-lin: '{}' had fewer than 2 points - nothing sealed.".format(
            path))
        return []

    def to_source(p):
        return ((p[0] + offset[0]) / scale, (p[1] + offset[1]) / scale, (p[2] + offset[2]) / scale)

    pts = []
    for i in range(len(raw) - 1):
        a, b = to_source(raw[i]), to_source(raw[i + 1])
        seg = math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3)))
        steps = max(1, int(seg / sample_step))
        for st in range(steps + 1):
            t = st / float(steps)
            pts.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
    return pts


def find_detail_solids_on_leak_path(root, leak_pts, detail_classes):
    """Which DETAIL-class brushes does the leak path run through?

    This is the whole trick behind --seal-leak-from-lin. q3map2 runs
    FloodEntities BEFORE FilterDetailBrushesIntoTree, so a detail brush is
    invisible to the leak flood - the geometry is there, it just doesn't
    seal. Source's func_detail, which this converter maps to q3map2 detail
    (CONTENTS_DETAIL) because that's the semantically right match for "don't
    split visleaves", can therefore be load-bearing for sealing in the
    original map while sealing nothing in the converted one.

    VERIFIED on de_dust2: of the 50 points in its own pointfile, exactly two
    lie inside Source solids, and both are func_detail (ids 10776 and 12338)
    - i.e. the flood walks straight through brushwork that is solid in
    Source. Promoting those to structural is what seals the map, and it's a
    far smaller change than making every func_detail structural (that map
    has 8354 detail brushes; turning them all structural would bloat the BSP
    tree for no benefit).

    Returns the set of solid ids to promote."""
    promote = set()
    for entity in root.get('entity', []):
        if get1(entity, 'classname', '').strip().lower() not in detail_classes:
            continue
        for solid in get_solid_blocks(entity):
            sid = get1(solid, 'id')
            if sid is None or sid in promote:
                continue
            planes = solid_planes_source(solid)
            if not planes:
                continue
            for p in leak_pts:
                if point_in_planes(planes, p):
                    promote.add(sid)
                    break
    return promote


def _orient_top_ccw(a, b, c):
    """Order three top-face points so q3map2's own PlaneFromPoints
    (normal = (p0-p1) x (p2-p1)) yields an UPWARD normal.

    Worked out from that formula rather than guessed: for a,b,c laid out
    counter-clockwise seen from above, it returns (0,0,-1) - so an up-facing
    top needs CLOCKWISE plan-view order, i.e. a negative 2D cross product.
    Displacement grids are not all wound the same way (on de_dust2, 5 of 552
    sealing cells came out inverted before this was applied), so the winding
    is normalised per triangle instead of assumed from the grid."""
    if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0:
        return a, c, b
    return a, b, c


def build_terrain_seal_brushes(recs, cells, margin=16.0):
    """A thin STRUCTURAL shell just under the displacement terrain, whose only
    job is to seal the map (--seal-terrain-under).

    Why this is needed at all. In Source, a displacement SURFACE seals the
    world - VBSP deletes the brush that carried it and the displacement
    itself closes the map. In idTech3 there is no such thing: with
    --terrain-mode lwo the terrain becomes a misc_model, and a model seals
    nothing, while --drop-terrain-backing-brush removes the only brush that
    was still standing in for it. The result is a map whose ground is
    visually and physically present but completely open to the flood, which
    then wanders underneath the terrain and out into the void.

    VERIFIED on de_dust2: of the 1732 points sampled along its leak path,
    207 sit directly beneath a displacement footprint, under 5 distinct
    displacements - and the whole path is empty space in Source too, i.e.
    nothing was "lost" in conversion; the sealing role of the displacements
    simply has no equivalent here.

    Geometry. One triangular prism per decimated grid cell, tops taken from
    the displacement's own welded grid so adjacent cells - and adjacent
    displacements, which share their edge vertices - meet EXACTLY and form a
    continuous shell with no seams to leak through. The whole shell is
    offset DOWN by `margin` so it can never poke above the visible terrain
    (a flat cell between two sampled vertices would otherwise rise above a
    dip in the real surface), and every prism runs 1024 units deep - far
    enough below that the flood cannot slip underneath (it was measured
    reaching 695 units below the terrain) while staying local to its own
    displacement.

    Faces use 'vmf2map/terrain_base' (nodraw + nonsolid, generated by
    vmt2shader.py), which is exactly right here: q3map2 decides whether a
    brush seals from C_TRANSLUCENT, NOT from solidity, so a nonsolid opaque
    brush still seals while adding no collision - the LWO model keeps
    providing the surface the player actually walks on. These are emitted
    WITHOUT the detail flag on purpose: a detail brush would be filtered in
    after FloodEntities and seal nothing, which is the whole bug this
    function exists to fix.

    Returns (func_group_text, n_brushes) or None."""
    if not recs or cells < 1:
        return None

    # Each displacement gets ONE floor of its own, SEAL_DEPTH below its own
    # lowest point - not a single floor shared by the whole map.
    #
    # This is the fix for a measured failure: a first version made each prism
    # a fixed 192 units deep, which produced a floating slab with open space
    # underneath it - and the flood simply went under. Measured on de_dust2's
    # next pointfile, the escape route ran 131 to 547 source units below the
    # terrain while that shell only occupied 13 to 164 units below it, so 442
    # of 1394 sampled points were still beneath the ground. With every prism
    # reaching the same floor there is no space left to pass through, and
    # neighbouring prisms are guaranteed to overlap vertically no matter how
    # much their tops differ in height.
    # A genuinely global floor was the obvious next idea and is wrong here:
    # the record set includes the 3D skybox's own displacements, which sit
    # thousands of units below the playable map, so one shared floor turns
    # every main-map prism into a ~3000-unit column and merges the space under
    # the map with the skybox region. Per-displacement floors stay local while
    # still being deep enough that neighbours overlap vertically whatever
    # their heights, and deeper than the 695 map units below the terrain that
    # the flood was measured reaching.
    SEAL_DEPTH = 1024.0

    def side_line(a, b, c):
        return ("( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) "
                "( {:.10g} {:.10g} {:.10g} ) vmf2map/terrain_base 0 0 0 "
                "0.500000 0.500000 0 0 0").format(
                    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])

    brushes = []
    degenerate = 0
    for rec in recs:
        n = rec['n']
        pos = rec['pos']
        rec_floor = min(pos[i][j][2] for i in range(n) for j in range(n)) - SEAL_DEPTH
        k = min(cells, n - 1)
        idxs = sorted(set(int(round(i * (n - 1) / float(k))) for i in range(k + 1)))
        for ri in range(len(idxs) - 1):
            for ci in range(len(idxs) - 1):
                r0, r1 = idxs[ri], idxs[ri + 1]
                c0, c1 = idxs[ci], idxs[ci + 1]
                # Winding copied verbatim from build_skybox_landscape_brushes,
                # which validated clean (all tops facing up, every centroid
                # inside all five planes). An earlier version of this
                # function tried to pick the winding per-triangle from a
                # plan-view cross product and produced 552 inverted brushes
                # out of 552 - so this mirrors the known-good pattern rather
                # than deriving it again.
                p00, p01 = pos[r0][c0], pos[r0][c1]
                p10, p11 = pos[r1][c0], pos[r1][c1]
                for tri in ((p00, p10, p11), (p00, p11, p01)):
                    a, b, c = _orient_top_ccw(
                        *[(q[0], q[1], q[2] - margin) for q in tri])
                    ax = (b[0] - a[0], b[1] - a[1])
                    bx = (c[0] - a[0], c[1] - a[1])
                    if abs(ax[0] * bx[1] - ax[1] * bx[0]) < 1e-6:
                        degenerate += 1
                        continue
                    b1 = (a[0], a[1], rec_floor)
                    b2 = (b[0], b[1], rec_floor)
                    b3 = (c[0], c[1], rec_floor)
                    brushes.append("\n".join([
                        "{",
                        side_line(a, b, c),
                        side_line(b1, b3, b2),
                        side_line(a, b1, b2),
                        side_line(b, b2, b3),
                        side_line(c, b3, b1),
                        "}",
                    ]))
    if not brushes:
        return None
    return "\n".join(["{", '"classname" "func_group"'] + brushes + ["}"]), len(brushes)


def lwo_weld_key(xyz, shader_flat_name, precision=1):
    """3D (vertical displacements exist) and shader-scoped. Different
    shaders keep separate alpha at the same position: CONFIRMED by a real
    report of sand-looking ground with a rock footstep sound, caused by
    alpha averaged across two shaders that share one texture. Positions
    still coincide either way, so this never opens a crack."""
    return (round(xyz[0], precision), round(xyz[1], precision), round(xyz[2], precision),
            shader_flat_name)


# Off by default: re-project UVs of steep triangles by dominant axis. NOT how
# Source does it (Source stretches textures on steep displacement faces), so
# only enabled by --terrain-lwo-steep-reproject for users who prefer it.
LWO_STEEP_REPROJECT = False

# Flat-edge tolerance (post-scale units) for deciding two displacements share
# an edge. Source decides neighbours by matching the flat face corners, so
# this compares FLAT positions, never displaced ones.
LWO_EDGE_TOL = 0.5


class LWOTerrainBuilder:
    """Collects every displacement in the map (as Source-exact records from
    compute_displacement_grid_lwo) and, in finalize_points, turns them into
    ONE welded triangle mesh:

      1. Mixed-power edges, the way Source does it (disp_common.cpp
         SetupAllowedVerts + vbsp SnapRemainingVertsToSurface): a vertex on
         the higher-power displacement's edge that the lower-power
         neighbour doesn't have is snapped onto the neighbour's edge. The
         neighbour's boundary triangle is then split at that exact point
         instead of leaving a T-junction. The lower-power displacement's own
         surface is untouched. (Source additionally coarsens a band of the
         higher-power displacement's interior next to such an edge; here
         that interior keeps its authored detail - the only intended
         difference from Source.)
      2. Source's own triangulation: a checkerboard of diagonals
         (CCoreDispInfo::GenerateCollisionSurface - cell (i,j) uses the
         (i,j+1)-(i+1,j) diagonal when i+j is odd, (i,j)-(i+1,j+1) when
         even). The old path used one diagonal everywhere, which on
         non-planar quads puts half the surface at the wrong height.
      3. Welding: same position + same shader -> one point, alpha averaged
         (fixes Source's own ~30 unsewn alpha disagreements in de_dust2).
    """
    def __init__(self):
        self.records = []
        self._flat_name_map = {}
        self.shader_scales = {}
        self.shader_triangles = {}
        self.total_vertex_adds = 0
        self.snapped_edge_verts = 0
        self.split_triangles = 0

    def _flat_shader_name(self, nested_key):
        flat = self._flat_name_map.get(nested_key)
        if flat is None:
            flat = '{}terrain{}'.format(MAP_SHADER_TAG, len(self._flat_name_map))
            self._flat_name_map[nested_key] = flat
        return flat

    def add_displacement(self, rec):
        if rec['is_blend']:
            rec['shader'] = self._flat_shader_name(rec['nested_key'])
            if rec['shader'] not in self.shader_scales:
                self.shader_scales[rec['shader']] = {
                    'material': rec['material'], 'u_scale': rec['u_scale'],
                    'v_scale': rec['v_scale'],
                }
        else:
            rec['shader'] = rec['surface']
        self.records.append(rec)

    def is_empty(self):
        return not self.records

    @staticmethod
    def _edges(n):
        """The 4 edges as lists of (i,j) grid ids, each in increasing order."""
        return [
            [(0, j) for j in range(n)],
            [(n - 1, j) for j in range(n)],
            [(i, 0) for i in range(n)],
            [(i, n - 1) for i in range(n)],
        ]

    @staticmethod
    def _source_triangles(n):
        tris = []
        for i in range(n - 1):
            for j in range(n - 1):
                if (i + j) % 2 == 1:   # BuildTriTLtoBR
                    tris.append([(i, j), (i + 1, j), (i, j + 1)])
                    tris.append([(i, j + 1), (i + 1, j), (i + 1, j + 1)])
                else:                  # BuildTriBLtoTR
                    tris.append([(i, j), (i + 1, j), (i + 1, j + 1)])
                    tris.append([(i, j), (i + 1, j + 1), (i, j + 1)])
        return tris

    def _resolve_edges(self):
        """Step 1 above. Returns {rec_index: {frozenset(seg ids): [(t, xyz,
        uv, alpha), ...]}} - points to insert into that record's boundary
        segment - and snaps the higher-power side's vertices in place."""
        tol = LWO_EDGE_TOL
        edge_info = []
        for rec in self.records:
            n = rec['n']
            infos = []
            for edge in self._edges(n):
                a = rec['flat'][edge[0][0]][edge[0][1]]
                b = rec['flat'][edge[-1][0]][edge[-1][1]]
                lo = tuple(min(a[k], b[k]) - tol for k in range(3))
                hi = tuple(max(a[k], b[k]) + tol for k in range(3))
                infos.append((edge, a, b, lo, hi))
            edge_info.append(infos)

        inserts = {}
        for ai, arec in enumerate(self.records):
            for (a_edge, _, _, alo, ahi) in edge_info[ai]:
                for bi, brec in enumerate(self.records):
                    if bi == ai or brec['n'] >= arec['n']:
                        continue  # only the higher-power side gets snapped
                    for (b_edge, bpa, bpb, blo, bhi) in edge_info[bi]:
                        if any(ahi[k] < blo[k] or alo[k] > bhi[k] for k in range(3)):
                            continue
                        seg = sub(bpb, bpa)
                        seg_len2 = dot(seg, seg)
                        if seg_len2 <= 0:
                            continue
                        nb = brec['n']
                        for (vi, vj) in a_edge:
                            pf = arec['flat'][vi][vj]
                            t = dot(sub(pf, bpa), seg) / seg_len2
                            if t < -1e-6 or t > 1 + 1e-6:
                                continue
                            if length(sub(pf, add(bpa, scale_vec(seg, t)))) > tol:
                                continue
                            param = t * (nb - 1)
                            k0 = int(math.floor(param + 1e-6))
                            if abs(param - round(param)) < 1e-4:
                                continue  # coincides with a real neighbour vertex
                            k0 = max(0, min(nb - 2, k0))
                            f = param - k0
                            p0 = brec['pos'][b_edge[k0][0]][b_edge[k0][1]]
                            p1 = brec['pos'][b_edge[k0 + 1][0]][b_edge[k0 + 1][1]]
                            snapped = lerp3(p0, p1, f)
                            arec['pos'][vi][vj] = snapped
                            self.snapped_edge_verts += 1
                            uv0 = brec['uv'][b_edge[k0][0]][b_edge[k0][1]]
                            uv1 = brec['uv'][b_edge[k0 + 1][0]][b_edge[k0 + 1][1]]
                            al0 = brec['alpha'][b_edge[k0][0]][b_edge[k0][1]]
                            al1 = brec['alpha'][b_edge[k0 + 1][0]][b_edge[k0 + 1][1]]
                            key = frozenset((b_edge[k0], b_edge[k0 + 1]))
                            inserts.setdefault(bi, {}).setdefault(key, []).append(
                                (f, b_edge[k0], snapped,
                                 (uv0[0] + (uv1[0] - uv0[0]) * f, uv0[1] + (uv1[1] - uv0[1]) * f),
                                 al0 + (al1 - al0) * f))
        return inserts

    def finalize_points(self):
        inserts = self._resolve_edges()

        point_index = {}
        pos_sum, pos_cnt, a_sum, a_cnt = [], [], [], []

        def weld(xyz, alpha01, shader):
            self.total_vertex_adds += 1
            key = lwo_weld_key(xyz, shader)
            idx = point_index.get(key)
            if idx is None:
                idx = len(pos_sum)
                point_index[key] = idx
                pos_sum.append([0.0, 0.0, 0.0]); pos_cnt.append(0)
                a_sum.append(0.0); a_cnt.append(0)
            s = pos_sum[idx]
            s[0] += xyz[0]; s[1] += xyz[1]; s[2] += xyz[2]
            pos_cnt[idx] += 1
            a_sum[idx] += alpha01; a_cnt[idx] += 1
            return idx

        for ri, rec in enumerate(self.records):
            n = rec['n']
            shader = rec['shader']
            ids = [[weld(rec['pos'][i][j], rec['alpha'][i][j], shader) for j in range(n)]
                   for i in range(n)]

            def corner(gid):
                i, j = gid
                return (ids[i][j], rec['uv'][i][j][0], rec['uv'][i][j][1])

            tris = self._source_triangles(n)
            # Keep the facing that is already confirmed rendering correctly
            # in SOF2 (right-hand normal pointing AWAY from the face's
            # outward normal, per LWO's clockwise-front convention). Checked
            # on the flat grid, so real overhangs keep their true facing.
            f00, f10, f01 = rec['flat'][0][0], rec['flat'][1][0], rec['flat'][0][1]
            if dot(cross(sub(f10, f00), sub(f01, f00)), rec['face_n']) > 0:
                tris = [[t[0], t[2], t[1]] for t in tris]

            rec_inserts = inserts.get(ri, {})
            out = self.shader_triangles.setdefault(shader, [])
            steep = self._steep_uvs(rec) if LWO_STEEP_REPROJECT else None
            for tri in tris:
                split = None
                for e in range(3):
                    key = frozenset((tri[e], tri[(e + 1) % 3]))
                    if key in rec_inserts:
                        split = (e, rec_inserts[key])
                        break
                if split is None:
                    corners = [corner(g) for g in tri]
                    if steep is not None:
                        corners = steep(tri, corners)
                    out.append(corners)
                    continue
                e, pts = split
                a, b, c = tri[e], tri[(e + 1) % 3], tri[(e + 2) % 3]
                # pts are stored with f measured from their segment's
                # FIRST grid id; order them along a -> b.
                chain = []
                for (f, start_gid, xyz, uvp, alp) in pts:
                    tt = f if start_gid == a else 1.0 - f
                    chain.append((tt, (weld(xyz, alp, shader), uvp[0], uvp[1])))
                chain.sort(key=lambda x: x[0])
                seq = [corner(a)] + [cpt for _, cpt in chain] + [corner(b)]
                for m in range(len(seq) - 1):
                    out.append([seq[m], seq[m + 1], corner(c)])
                self.split_triangles += 1

        points = [(s[0] / c, s[1] / c, s[2] / c) for s, c in zip(pos_sum, pos_cnt)]
        alphas = [s / c for s, c in zip(a_sum, a_cnt)]
        return points, alphas

    @staticmethod
    def _steep_uvs(rec):
        """Optional (--terrain-lwo-steep-reproject): dominant-axis UVs for
        triangles tilted > ~60 degrees from the face's projection plane."""
        u_dir, u_shift, v_dir, v_shift, tex_w, tex_h = rec['uv_axes']
        orig_normal = normalize(cross(u_dir, v_dir))
        def apply(tri, corners):
            p0, p1, p2 = (rec['pos'][g[0]][g[1]] for g in tri)
            tn = normalize(cross(sub(p1, p0), sub(p2, p0)))
            if tn == (0.0, 0.0, 0.0) or abs(dot(orig_normal, tn)) >= 0.5:
                return corners
            tx, ty = texture_axis_from_plane(tn)
            outc = []
            for g, (pidx, _, _) in zip(tri, corners):
                u, v = build_patch_uv(rec['pos_unscaled'][g[0]][g[1]], tx, u_shift,
                                      rec['u_scale'], ty, v_shift, rec['v_scale'], tex_w, tex_h)
                outc.append((pidx, u, v))
            return outc
        return apply


def _lwo_s0(s):
    b = s.encode('ascii', 'replace') + b'\x00'
    if len(b) % 2 == 1:
        b += b'\x00'
    return b


def _lwo_vx(i):
    if i < 0xFF00:
        return struct.pack('>H', i)
    return struct.pack('>I', 0xFF000000 | i)


def _lwo_subchunk(tag, data):
    out = tag.encode('ascii') + struct.pack('>H', len(data)) + data
    if len(data) % 2 == 1:
        out += b'\x00'
    return out


def _lwo_chunk(tag, data):
    out = tag.encode('ascii') + struct.pack('>I', len(data)) + data
    if len(data) % 2 == 1:
        out += b'\x00'
    return out


# idTech3's renderer batches all triangles sharing one shader into a single
# "tess" buffer per draw call, hard-capped at SHADER_MAX_VERTEXES=1000 /
# SHADER_MAX_INDEXES=6000 (checked directly in the original Quake3 engine
# source, qfiles.h - the same codebase era SOF2 is built on). Exceeding it
# is a runtime crash: "RB_CheckOverflow: indices > MAX". Confirmed as a
# real, direct consequence of welding/merging many displacements or
# overlays under one shader tag for the MAX_SHADER_INFO fix - a popular
# material (groundsand_blend on the terrain, brickroad01 on overlays) can
# easily exceed 2000 triangles once merged. This splits any oversized group
# into multiple sub-tags, each safely under the limit, all still mapping
# back to the SAME material for shader generation (so this doesn't
# reintroduce the "too many total shader definitions" problem - only
# materials that actually need more than one tag get more than one).
LWO_MAX_TRIANGLES_PER_GROUP = 1800

# For LWO models emitted as model_static instead of misc_model (currently:
# the 3D skybox area's terrain/overlays, when a leak stops misc_model's own
# forced-meta pipeline from working - see the --skybox-entity help text).
# model_static bypasses q3map2's compile-time meta pipeline entirely, so it
# never gets misc_model's automatic per-surface splitting, and its own
# runtime limit is vertex-based (confirmed via a real "RB_CheckOverflow:
# verts > MAX" test) rather than the index-based one misc_model hits. Kept
# a margin under the true ~1000 (mdl_to_md3.py's own MD3_MAX_VERTS_PER_SURFACE
# is 999, one under a boundary that errored at exactly 1000).
LWO_MAX_VERTS_PER_GROUP_STATIC = 950


def split_mesh_by_region(points, point_alphas, shader_triangles, in_sky):
    """Partition a welded terrain mesh into (main_map, skybox_area) by which
    side of the 3D skybox area boundary each triangle's centre falls on
    (in_sky(point) - see find_skybox_area), each with its own remapped point
    list.

    Source keeps the 3D skybox as a separate, far-away room inside the same
    VMF, so displacements from BOTH regions were landing in one welded
    model spanning the whole map - measured on de_dust2: the terrain model
    ran X -2926 (main map) to 12963 (skybox area), with 577 of its 4488
    points inside the skybox region. q3map2's _skybox needs the skybox to
    be a wholly contained, SEPARATE area; one misc_model covering both
    can't be separated, which is why enabling _skybox generated 2749
    skybox surfaces and blew MAX_MAP_LEAFFACES (3438492 references against
    a 1048576 cap). Splitting them into two models gives q3map2 exactly the
    separation it expects.

    Returns two (points, alphas, shader_triangles) tuples; either may be
    empty.
    """
    def centre_in_sky(tri):
        cx = sum(points[i][0] for i, _u, _v in tri) / 3.0
        cy = sum(points[i][1] for i, _u, _v in tri) / 3.0
        cz = sum(points[i][2] for i, _u, _v in tri) / 3.0
        return in_sky((cx, cy, cz))

    out = []
    for want_sky in (False, True):
        sub_tris, remap, sub_pts, sub_alphas = {}, {}, [], []
        for name, tris in shader_triangles.items():
            kept = []
            for tri in tris:
                if centre_in_sky(tri) != want_sky:
                    continue
                new_tri = []
                for idx, u, v in tri:
                    if idx not in remap:
                        remap[idx] = len(sub_pts)
                        sub_pts.append(points[idx])
                        sub_alphas.append(point_alphas[idx])
                    new_tri.append((remap[idx], u, v))
                kept.append(new_tri)
            if kept:
                sub_tris[name] = kept
        out.append((sub_pts, sub_alphas, sub_tris))
    return out[0], out[1]


# ------------------------------------------------------------ fog volumes --
#
# Why the welded terrain/overlay models have to be cut up around water.
#
# idTech3 fogs a surface by its depth below the fog volume's top plane, and
# that plane has no edges: whether a surface is fogged at all is decided at
# compile time, by q3map2's FogDrawSurfaces. For a brush face it chops the
# face along the fog brush, so only the part inside gets the fog. For any
# other surface - which is what a misc_model's triangles are - it only tests
# the surface's bounding box against the fog brush's, and if they touch, the
# WHOLE surface gets that fog. The welded terrain model has one surface per
# material spanning the whole map, so one small water brush fogged every
# terrain triangle below its surface level, everywhere.
# Measured on de_overpass_csgo: the fountain's aztecwater (180 x 180 units,
# top at z 630) fogged 104 terrain/overlay surfaces from x -4975 to -43; the
# park's gravel sits at z ~590, 40 units under that plane, so it drew as
# solid fog colour (0.15 0.1 0) - the "brown floor".
#
# The fix keeps q3map2's own rule and makes it come out right: every
# triangle is clipped against each fog brush's bounding box (grown by
# FOG_SPLIT_MARGIN), the parts inside go in a model of their own (fogged, as
# they should be - that is Source's water fog too), and the parts outside
# are grouped so that no output model's bounding box touches a fog box it
# is not inside. Same triangles, same shaders; q3map2's meta stage merges
# them back into surfaces across the models.

FOG_SPLIT_MARGIN = 1.0      # map units; covers snap_brush_planes_to_grid's 1/8 moves
FOG_BOXES = []              # set per run by convert_vmf (find_fog_volume_boxes)
FOG_BRUSH_PLANES = []       # the same brushes' planes, same order
TERRAIN_NOFOG_TARGETS = {}  # LWO tag -> remap target, for ground in a fog box but out of its water
FOG_SPLIT_LWO_FILES = []    # (bytes, file name) of the extra models, written by main()


def find_fog_volume_boxes(root, materials_dir, scale, offset):
    """Map-space (mins, maxs) of every brush that becomes a fog volume in
    SOF2: any brush with a face whose Source material is a 'Water' class
    material - vmt2shader.py gives every one of those a fogparms."""
    boxes = []
    if not materials_dir:
        return boxes
    water_cache = {}

    def is_water(material):
        key = (material or '').strip().lower().replace('\\', '/')
        if key not in water_cache:
            cls = None
            vmt_path = os.path.join(materials_dir, *key.split('/')) + '.vmt'
            if key and os.path.isfile(vmt_path):
                try:
                    with open(vmt_path, 'r', encoding='utf-8', errors='replace') as f:
                        r = parse_vdf(tokenize(f.read()))
                    if r:
                        cls = list(r.keys())[0]
                except (OSError, IndexError, KeyError):
                    cls = None
            water_cache[key] = bool(cls) and cls.strip().lower() == 'water'
        return water_cache[key]

    for holder in list(root.get('world', [])) + list(root.get('entity', [])):
        if not isinstance(holder, dict):
            continue
        for solid in get_solid_blocks(holder):
            if not any(isinstance(sd, dict) and is_water(get1(sd, 'material', ''))
                       for sd in solid.get('side', [])):
                continue
            pb = _brush_planes_and_bounds(solid)
            if not pb:
                continue
            _pl, mn, mx = pb
            a = [mn[i] * scale - offset[i] for i in range(3)]
            b = [mx[i] * scale - offset[i] for i in range(3)]
            boxes.append((tuple(min(a[i], b[i]) - FOG_SPLIT_MARGIN for i in range(3)),
                          tuple(max(a[i], b[i]) + FOG_SPLIT_MARGIN for i in range(3))))
            # the brush itself, map space: inside is n . p <= d
            FOG_BRUSH_PLANES.append([(n, d * scale - dot(n, offset)) for n, d in _pl])
    return boxes


def _clip_poly_axis(poly, axis, value, keep_below):
    """Split a convex polygon (list of vertex tuples whose first three
    components are xyz; the rest interpolate linearly) at axis = value.
    Returns (part on the kept side, the other part)."""
    keep, other = [], []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da = (value - a[axis]) if keep_below else (a[axis] - value)
        db = (value - b[axis]) if keep_below else (b[axis] - value)
        if da >= 0:
            keep.append(a)
        if da <= 0:
            other.append(a)
        if (da > 0 > db) or (da < 0 < db):
            t = da / (da - db)
            v = tuple(a[k] + (b[k] - a[k]) * t for k in range(len(a)))
            v = v[:axis] + (value,) + v[axis + 1:]
            keep.append(v)
            other.append(v)
    keep = keep if len(keep) >= 3 else []
    other = other if len(other) >= 3 else []
    return keep, other


def _poly_bounds(poly):
    return (tuple(min(v[a] for v in poly) for a in range(3)),
            tuple(max(v[a] for v in poly) for a in range(3)))


def _box_overlap(mn, mx, bmn, bmx):
    return all(mx[a] > bmn[a] and mn[a] < bmx[a] for a in range(3))


def split_polys_by_fog_boxes(polys, boxes):
    """polys: iterable of (group_name, [vertex tuples]). Returns a list of
    buckets, each {'inside': frozenset of fog-box indices, 'polys':
    [(group_name, poly)...]}; the first bucket is the largest one that is
    inside no fog box. No bucket's bounds touch a box it is not inside."""
    # Height first: water boxes are thin and most ground lies wholly above
    # or below one, so z slabs hold large connected areas and merge well.
    planes = [(2, False), (2, True), (0, True), (0, False), (1, True), (1, False)]
    groups = {}
    for name, poly in polys:
        pieces = [(poly, ())]
        for (bmn, bmx) in boxes:
            nxt = []
            for pc, key in pieces:
                mn, mx = _poly_bounds(pc)
                slab = None
                for si, (ax, below) in enumerate(planes):
                    if (mx[ax] <= bmn[ax]) if below else (mn[ax] >= bmx[ax]):
                        slab = si
                        break
                if slab is not None:
                    nxt.append((pc, key + (slab,)))
                    continue
                rest = pc
                for si, (ax, below) in enumerate(planes):
                    lim = bmn[ax] if below else bmx[ax]
                    out_part, rest = _clip_poly_axis(rest, ax, lim, below)
                    if out_part:
                        nxt.append((out_part, key + (si,)))
                    if not rest:
                        break
                if rest:
                    nxt.append((rest, key + ('in',)))
            pieces = nxt
        for pc, key in pieces:
            g = groups.setdefault(key, {'polys': [], 'mn': [float('inf')] * 3,
                                        'mx': [float('-inf')] * 3})
            g['polys'].append((name, pc))
            mn, mx = _poly_bounds(pc)
            for a in range(3):
                g['mn'][a] = min(g['mn'][a], mn[a])
                g['mx'][a] = max(g['mx'][a], mx[a])

    # Merge the slab groups back together wherever the merged bounds still
    # stay clear of every fog box the group is not inside - fewer models.
    buckets = []
    for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1]['polys'])):
        inside = frozenset(i for i, k in enumerate(key) if k == 'in')
        placed = False
        for bk in buckets:
            if bk['inside'] != inside:
                continue
            mn = [min(bk['mn'][a], g['mn'][a]) for a in range(3)]
            mx = [max(bk['mx'][a], g['mx'][a]) for a in range(3)]
            if any(_box_overlap(mn, mx, bmn, bmx) for i, (bmn, bmx) in enumerate(boxes)
                   if i not in inside):
                continue
            bk['mn'], bk['mx'] = mn, mx
            bk['polys'].extend(g['polys'])
            placed = True
            break
        if not placed:
            buckets.append({'inside': inside, 'mn': list(g['mn']), 'mx': list(g['mx']),
                            'polys': list(g['polys'])})
    # then any two buckets that can still be joined, until none can
    merged = True
    while merged:
        merged = False
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                bi, bj = buckets[i], buckets[j]
                if bi['inside'] != bj['inside']:
                    continue
                mn = [min(bi['mn'][a], bj['mn'][a]) for a in range(3)]
                mx = [max(bi['mx'][a], bj['mx'][a]) for a in range(3)]
                if any(_box_overlap(mn, mx, bmn, bmx) for k, (bmn, bmx) in enumerate(boxes)
                       if k not in bi['inside']):
                    continue
                bi['mn'], bi['mx'] = mn, mx
                bi['polys'].extend(bj['polys'])
                del buckets[j]
                merged = True
                break
            if merged:
                break
    buckets.sort(key=lambda bk: (bool(bk['inside']), -len(bk['polys'])))
    return buckets


# Two pieces go into the same model only when they are this close (bounding
# box to bounding box, map units): each model is then one area of the map.
AREA_JOIN_GAP = 512.0


def _polys_bounds(polys):
    mn = [float('inf')] * 3
    mx = [float('-inf')] * 3
    for _name, poly in polys:
        pmn, pmx = _poly_bounds(poly)
        for a in range(3):
            mn[a] = min(mn[a], pmn[a])
            mx[a] = max(mx[a], pmx[a])
    return mn, mx


def _box_gap(amn, amx, bmn, bmx):
    return math.sqrt(sum(max(0.0, bmn[a] - amx[a], amn[a] - bmx[a]) ** 2 for a in range(3)))


def _spatial_clusters(plist, gap=AREA_JOIN_GAP):
    """Split [(name, poly)...] into groups that are each one connected area:
    two polygons are in the same group when their bounding boxes, grown by
    gap / 2, touch through a chain of others. (split_polys_by_fog_boxes
    merges a material's pieces by fog rules only, so one of its buckets can
    hold ground from both ends of the map.)"""
    if len(plist) <= 1:
        return [plist]
    cell = max(gap, 1.0)
    parent = list(range(len(plist)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    half = gap * 0.5
    for i, (_n, poly) in enumerate(plist):
        mn, mx = _poly_bounds(poly)
        lo = [int(math.floor((mn[a] - half) / cell)) for a in range(3)]
        hi = [int(math.floor((mx[a] + half) / cell)) for a in range(3)]
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    j = owner.setdefault((x, y, z), i)
                    if j != i:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
    groups = OrderedDict()
    for i, item in enumerate(plist):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


def split_polys_by_fog_boxes_per_group(polys, boxes):
    """q3map2 tests fog per SURFACE, and a model has one surface per
    material, so the no-touching rule binds each material separately: each
    material is split on its own (split_polys_by_fog_boxes), and its pieces
    are then gathered into models BY AREA - a piece joins the nearest model
    of its kind (outside water / in the water / in a water box but dry) that
    is within AREA_JOIN_GAP of it, provided that if the model already holds
    that material, the two together are in the same water and still clear of
    every other one. Otherwise it starts a new model. (Before 2026-10-17 the
    k-th piece of every material went into model k, the fewest models
    possible but each one scattered across the map.) Returns buckets as
    split_polys_by_fog_boxes does, 'inside' being the union over the
    materials in that model; the first is the biggest one outside water."""
    by_name = OrderedDict()
    for name, poly in polys:
        by_name.setdefault(name, []).append((name, poly))
    entries = []        # (kind, material, inside, polys)
    for name, plist in by_name.items():
        for bk in split_polys_by_fog_boxes(plist, boxes):
            if not bk['inside']:
                parts = [(0, frozenset(), bk['polys'])]
            else:
                wet, dry = _split_wet_dry(bk['polys'], bk['inside'])
                parts = [(1, bk['inside'], wet), (2, bk['inside'], dry)]
            for kind, inside, pl in parts:
                for cl in (_spatial_clusters(pl) if pl else []):
                    entries.append((kind, name, inside, cl))
    entries.sort(key=lambda e: -len(e[3]))
    models = []
    for kind, name, inside, plist in entries:
        mn, mx = _polys_bounds(plist)
        best = None
        for m in models:
            if m['kind'] != kind:
                continue
            g = _box_gap(m['mn'], m['mx'], mn, mx)
            if g > AREA_JOIN_GAP or (best is not None and g >= best[0]):
                continue
            mat = m['mats'].get(name)
            if mat is not None:
                m_inside, m_mn, m_mx = mat
                if m_inside != inside:
                    continue
                umn = [min(m_mn[a], mn[a]) for a in range(3)]
                umx = [max(m_mx[a], mx[a]) for a in range(3)]
                if any(_box_overlap(umn, umx, bmn, bmx) for k, (bmn, bmx) in enumerate(boxes)
                       if k not in inside):
                    continue
            best = (g, m)
        if best is None:
            m = {'kind': kind, 'inside': frozenset(), 'polys': [], 'mats': {},
                 'mn': list(mn), 'mx': list(mx), 'nofog': kind == 2}
            models.append(m)
        else:
            m = best[1]
        m['inside'] = m['inside'] | inside
        m['polys'].extend(plist)
        old = m['mats'].get(name)
        if old is None:
            m['mats'][name] = (inside, list(mn), list(mx))
        else:
            m['mats'][name] = (inside, [min(old[1][a], mn[a]) for a in range(3)],
                               [max(old[2][a], mx[a]) for a in range(3)])
        for a in range(3):
            m['mn'][a] = min(m['mn'][a], mn[a])
            m['mx'][a] = max(m['mx'][a], mx[a])
    models.sort(key=lambda m: (m['kind'], -len(m['polys'])))
    return [{'inside': m['inside'], 'polys': m['polys'], 'nofog': m['nofog']} for m in models]


def _clip_poly_plane(poly, n, d):
    """Split a convex polygon by the plane n . p = d (vertex tuples, xyz
    first, the rest interpolated). Returns (part with n . p <= d, the rest)."""
    keep, other = [], []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da = d - (n[0] * a[0] + n[1] * a[1] + n[2] * a[2])
        db = d - (n[0] * b[0] + n[1] * b[1] + n[2] * b[2])
        if da >= 0:
            keep.append(a)
        if da <= 0:
            other.append(a)
        if (da > 0 > db) or (da < 0 < db):
            t = da / (da - db)
            v = tuple(a[k] + (b[k] - a[k]) * t for k in range(len(a)))
            keep.append(v)
            other.append(v)
    return (keep if len(keep) >= 3 else []), (other if len(other) >= 3 else [])


def _split_wet_dry(polys, inside):
    """Polygons of an in-a-fog-box bucket -> (wet, dry): the parts inside one
    of those water brushes, and the parts in the box but outside the brush
    (the corners of the box round an octagonal fountain basin). The dry ones
    still touch the box, so q3map2 would fog them; their model gets _remap to
    q3map_noFog shader copies instead (TERRAIN_NOFOG_TARGETS)."""
    wet, dry = [], []
    for name, poly in polys:
        rest = [poly]
        for k in sorted(inside):
            if k >= len(FOG_BRUSH_PLANES):
                continue
            nxt = []
            for pc in rest:
                cur = pc
                outside_bits = []
                for n, d in FOG_BRUSH_PLANES[k]:
                    inn, out = _clip_poly_plane(cur, n, d + FOG_SPLIT_MARGIN)
                    if out:
                        outside_bits.append(out)
                    cur = inn
                    if not cur:
                        break
                if cur:
                    wet.append((name, cur))
                nxt.extend(outside_bits)
            rest = nxt
        dry.extend((name, pc) for pc in rest)
    if not FOG_BRUSH_PLANES:
        return [(nm, pc) for nm, pc in polys], []
    return wet, dry


def _nofog_remap_lines(tag_map):
    """misc_model _remap keys sending each LWO tag (surface) of a model to its
    q3map_noFog copy; records the targets for vmt2shader.py."""
    lines = []
    for k, tag in enumerate(sorted(tag_map), 1):
        target = short_qpath(tag + '_nf')
        TERRAIN_NOFOG_TARGETS[tag] = target
        lines.append('"_remap{}" "{};{}"'.format(k, tag, target))
    return lines


def _fan(poly):
    return [(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1)]


def split_terrain_by_fog(points, point_alphas, shader_triangles, boxes):
    """Welded terrain (see split_mesh_by_region for the shape) -> list of
    (points, alphas, shader_triangles, inside_set) parts; one part, unchanged,
    when nothing touches a fog box. Clipped corners interpolate position,
    alpha and the corner's own u/v."""
    def touches(tri):
        mn = [min(points[i][a] for i, _u, _v in tri) for a in range(3)]
        mx = [max(points[i][a] for i, _u, _v in tri) for a in range(3)]
        return any(_box_overlap(mn, mx, bmn, bmx) for bmn, bmx in boxes)

    if not boxes or not any(touches(t) for tris in shader_triangles.values() for t in tris):
        return [(points, point_alphas, shader_triangles, frozenset(), False)]

    # vertex tuple: x, y, z, alpha, u, v, original point index (or -1 once
    # interpolated - an interpolated index is meaningless, so it is reset)
    def polys():
        for name, tris in shader_triangles.items():
            for tri in tris:
                yield name, [tuple(points[i]) + (float(point_alphas[i]), u, v, float(i))
                             for i, u, v in tri]

    parts = []
    for bk in split_polys_by_fog_boxes_per_group(polys(), boxes):
        pts, alphas, s_tris, index = [], [], {}, {}
        for name, poly in bk['polys']:
            ids = []
            for vtx in poly:
                orig = vtx[6]
                exact = abs(orig - round(orig)) < 1e-9 and all(
                    abs(vtx[a] - points[int(round(orig))][a]) < 1e-6 for a in range(3))
                # a clip-made corner welds with the same corner made from the
                # neighbouring triangle - per material, as the welder does
                # (lwo_weld_key), since alpha differs between materials
                key = ('p', int(round(orig))) if exact else (
                    'n', name, round(vtx[0], 3), round(vtx[1], 3), round(vtx[2], 3))
                if key not in index:
                    index[key] = len(pts)
                    pts.append(tuple(vtx[:3]) if not exact else points[int(round(orig))])
                    alphas.append(vtx[3] if not exact else point_alphas[int(round(orig))])
                ids.append((index[key], vtx[4], vtx[5]))
            for tri in _fan(ids):
                s_tris.setdefault(name, []).append(list(tri))
        parts.append((pts, alphas, s_tris, bk['inside'], bool(bk.get('nofog'))))
    return parts


def split_mesh_surfaces_by_fog(surfaces, boxes):
    """Overlay mesh {name: (points, uvs, triangles)} -> list of
    ({name: (points, uvs, triangles)}, inside_set), same rule as
    split_terrain_by_fog."""
    def touches(pts, tri):
        mn = [min(pts[i][a] for i in tri) for a in range(3)]
        mx = [max(pts[i][a] for i in tri) for a in range(3)]
        return any(_box_overlap(mn, mx, bmn, bmx) for bmn, bmx in boxes)

    if not boxes or not any(touches(p, t) for p, _uv, tris in surfaces.values() for t in tris):
        return [(surfaces, frozenset(), False)]

    def polys():
        for name, (pts, uvs, tris) in surfaces.items():
            for tri in tris:
                yield name, [tuple(pts[i]) + (uvs[i][0], uvs[i][1], float(i)) for i in tri]

    parts = []
    for bk in split_polys_by_fog_boxes_per_group(polys(), boxes):
        out, index = {}, {}
        for name, poly in bk['polys']:
            src_pts = surfaces[name][0]
            p_list, uv_list, t_list = out.setdefault(name, ([], [], []))
            ids = []
            for v in poly:
                orig = int(round(v[5]))
                exact = abs(v[5] - orig) < 1e-9 and all(
                    abs(v[a] - src_pts[orig][a]) < 1e-6 for a in range(3))
                key = (name, 'p', orig) if exact else (
                    name, 'n', round(v[0], 3), round(v[1], 3), round(v[2], 3),
                    round(v[3], 5), round(v[4], 5))
                if key not in index:
                    index[key] = len(p_list)
                    p_list.append(tuple(v[:3]))
                    uv_list.append((v[3], v[4]))
                ids.append(index[key])
            t_list.extend(_fan(ids))
        parts.append((out, bk['inside'], bool(bk.get('nofog'))))
    return parts


def split_oversized_shader_groups(shader_triangles, max_triangles=LWO_MAX_TRIANGLES_PER_GROUP):
    """{name: [triangles...]} -> ({new_name: [triangles...]}, {new_name: original_name})."""
    out, origin = {}, {}
    for name, tris in shader_triangles.items():
        if len(tris) <= max_triangles:
            out[name] = tris
            origin[name] = name
            continue
        for i in range(0, len(tris), max_triangles):
            chunk = tris[i:i + max_triangles]
            new_name = name if i == 0 else '{}_{}'.format(name, i // max_triangles)
            out[new_name] = chunk
            origin[new_name] = name
    return out, origin


def split_shader_groups_by_vert_count(shader_triangles, max_verts):
    """Same purpose as split_oversized_shader_groups, but caps by UNIQUE
    point count per group rather than triangle count - needed specifically
    for model_static, whose own runtime vertex limit (confirmed via a real
    "RB_CheckOverflow: verts > MAX" test) is far stricter relative to
    triangle count than misc_model's index-based one: a welded mesh shares
    roughly 2 triangles per vertex, so triangle-count capping alone (as
    split_oversized_shader_groups does, tuned for misc_model) leaves groups
    with close to double the vertices model_static can actually handle.

    Each triangle is a list of 3 (point_index, u, v) tuples, matching
    write_lwo2_terrain's own format. Walks triangles in order, starting a
    new chunk whenever adding a triangle's own (up to 3 new) point indices
    would exceed the cap - mirrors split_oversized_md3_surface's algorithm
    exactly, just against this dict-of-triangles shape instead of MD3's
    separate verts/tris lists.
    """
    out, origin = {}, {}
    for name, tris in shader_triangles.items():
        cur_tris, seen = [], set()
        chunk_i = 0
        for tri in tris:
            new_ids = {t[0] for t in tri if t[0] not in seen}
            if len(seen) + len(new_ids) > max_verts and cur_tris:
                new_name = name if chunk_i == 0 else '{}_{}'.format(name, chunk_i)
                out[new_name] = cur_tris
                origin[new_name] = name
                chunk_i += 1
                cur_tris, seen = [], set()
            seen |= {t[0] for t in tri}
            cur_tris.append(tri)
        if cur_tris:
            new_name = name if chunk_i == 0 else '{}_{}'.format(name, chunk_i)
            out[new_name] = cur_tris
            origin[new_name] = name
    return out, origin


def write_lwo2_terrain(points, point_alphas, shader_triangles, for_model_static=False):
    """Serializes a welded terrain mesh to LWO2 bytes. shader_triangles:
    dict base_shader_name -> list of triangles, each a list of 3
    (point_index, u, v) tuples. Structure verified (round-trip parsed,
    weld/alpha-average/VMAD-UV-divergence all confirmed correct) before
    this was wired into the real conversion pipeline; the VMAP/RGBA vertex
    alpha mechanism itself is additionally confirmed working live in SOF2.

    points are in idTech3 convention (right-handed, Z-up) - the SAME
    convention every other coordinate in this whole pipeline uses. LWO2
    itself is left-handed and Y-up (confirmed directly from the LightWave
    SDK format spec: "+Y direction is upward... +Z forward or north"), and
    the actual loader q3map2-family tools use (picomodel's pm_lwo.c,
    confirmed by reading its source) converts LWO's own Y-up storage back
    to Z-up by swapping components on load: xyz[1]=pos[2]; xyz[2]=pos[1].
    So writing idTech3-space points straight into PNTS unswapped comes out
    with Y and Z scrambled once the loader applies ITS swap - CONFIRMED
    the real, concrete cause of a real "the model is oriented wrong"
    report on actual terrain (not caught by an earlier flat, Y=0 test
    quad, which has no height variation to reveal an up/down swap).
    Pre-swapping Y and Z here exactly cancels the loader's own swap (two
    swaps = identity, verified by direct simulation), so this is the ONLY
    change needed - no triangle winding/normal adjustment, since swapping
    the same two axes twice is orientation-preserving, not a mirror.

    Splits any shader group over LWO_MAX_TRIANGLES_PER_GROUP first (see
    that function's own comment - a real, confirmed RB_CheckOverflow crash
    otherwise on a popular material). Returns (bytes, origin_mapping) -
    origin_mapping is {actual tag name written : original shader name},
    identity for anything that didn't need splitting; the caller uses it to
    generate a shader for every ACTUAL tag, not just the original names.

    for_model_static=True switches to a vertex-count cap instead
    (LWO_MAX_VERTS_PER_GROUP_STATIC) - see split_shader_groups_by_vert_count's
    own comment for why misc_model's triangle-based cap isn't tight enough
    for model_static's own, much stricter per-surface vertex limit.
    """
    if for_model_static:
        shader_triangles, origin_mapping = split_shader_groups_by_vert_count(
            shader_triangles, LWO_MAX_VERTS_PER_GROUP_STATIC)
    else:
        shader_triangles, origin_mapping = split_oversized_shader_groups(shader_triangles)
    shader_names = sorted(shader_triangles.keys())
    shader_tag_index = {name: i for i, name in enumerate(shader_names)}

    layr_data = (struct.pack('>HH', 0, 0) + struct.pack('>fff', 0.0, 0.0, 0.0) +
                 _lwo_s0("terrain"))
    layr_chunk = _lwo_chunk('LAYR', layr_data)

    # (x, y, z) idTech3 -> (x, z, y) for LWO2 storage - see this function's
    # own docstring above for exactly why.
    pnts_data = b''.join(struct.pack('>fff', p[0], p[2], p[1]) for p in points)
    pnts_chunk = _lwo_chunk('PNTS', pnts_data)

    vmap_data = b'RGBA' + struct.pack('>H', 4) + _lwo_s0("Alpha")
    for i, a in enumerate(point_alphas):
        vmap_data += _lwo_vx(i) + struct.pack('>ffff', 1.0, 1.0, 1.0, a)
    vmap_chunk = _lwo_chunk('VMAP', vmap_data)

    pols_data = b'FACE'
    ptag_entries = []
    vmad_entries = []
    poly_i = 0
    for shader_name in shader_names:
        tag_idx = shader_tag_index[shader_name]
        for tri in shader_triangles[shader_name]:
            pols_data += struct.pack('>H', 3)
            for (pidx, u, v) in tri:
                pols_data += _lwo_vx(pidx)
            for (pidx, u, v) in tri:
                vmad_entries.append((pidx, poly_i, u, v))
            ptag_entries.append((poly_i, tag_idx))
            poly_i += 1
    pols_chunk = _lwo_chunk('POLS', pols_data)

    # short_qpath: the tag is the shader name (+ -rename's _BSP), 64 bytes in idTech3
    tags_data = b''.join(_lwo_s0(short_qpath(name)) for name in shader_names)
    tags_chunk = _lwo_chunk('TAGS', tags_data)

    ptag_data = b'SURF'
    for pidx, tidx in ptag_entries:
        ptag_data += _lwo_vx(pidx) + struct.pack('>H', tidx)
    ptag_chunk = _lwo_chunk('PTAG', ptag_data)

    # V inverted, same reason as the overlay mesh's VMAP: q3map2 imports every
    # model with Assimp's aiProcess_FlipUVs, turning V into 1 - V, and this
    # terrain is a misc_model that q3map2 bakes. Writing 1 - V cancels it out.
    # See write_lwo2_mesh for the measurement and for the open MD3 case.
    vmad_data = b'TXUV' + struct.pack('>H', 2) + _lwo_s0("UVMap")
    for (pidx, polyi, u, v) in vmad_entries:
        vmad_data += _lwo_vx(pidx) + _lwo_vx(polyi) + struct.pack('>ff', u, 1.0 - v)
    vmad_chunk = _lwo_chunk('VMAD', vmad_data)

    surf_chunks = b''
    for shader_name in shader_names:
        colr_sub = _lwo_subchunk('COLR', struct.pack('>fff', 1.0, 1.0, 1.0) + struct.pack('>H', 0))
        side_sub = _lwo_subchunk('SIDE', struct.pack('>H', 1))
        vcol_sub = _lwo_subchunk('VCOL', struct.pack('>f', 1.0) + struct.pack('>H', 0) +
                                  b'RGBA' + _lwo_s0("Alpha"))
        surf_data = _lwo_s0(short_qpath(shader_name)) + _lwo_s0("") + colr_sub + side_sub + vcol_sub
        surf_chunks += _lwo_chunk('SURF', surf_data)

    body = (layr_chunk + pnts_chunk + vmap_chunk + pols_chunk + tags_chunk + ptag_chunk +
            vmad_chunk + surf_chunks)
    form_data = b'LWO2' + body
    return b'FORM' + struct.pack('>I', len(form_data)) + form_data, origin_mapping


def build_displacement_patch(side, all_sides, texture_map, unmapped_counter,
                              scale, offset, warnings):
    dispinfo = get1(side, 'dispinfo')
    if not isinstance(dispinfo, dict):
        return None

    material = get1(side, 'material', 'tools/toolsnodraw')
    mapped_material = convert_texture_name(material, texture_map, unmapped_counter)

    poly = get_face_polygon(side, all_sides)
    if not poly or len(poly) != 4:
        warnings.append(
            "displacement on material '{}' didn't resolve to a clean quad "
            "(got {} corners) - skipped, base face left flat".format(
                material, len(poly) if poly else 0))
        return None

    power, n, startpos, normals, distances, offsets, alphas = parse_dispinfo(dispinfo)
    quad = order_quad_from_startpos(poly, startpos)
    if quad is None:
        warnings.append("displacement on material '{}' - couldn't match startposition to a "
                         "quad corner, skipped".format(material))
        return None
    c00, c01, c11, c10 = quad  # row0/col0 .. row0/colMax .. rowMax/colMax .. rowMax/col0
    elev_unscaled = disp_elevation_vector(side, dispinfo)
    elev_scaled = scale_vec(elev_unscaled, scale)

    uaxis_str = get1(side, 'uaxis')
    vaxis_str = get1(side, 'vaxis')
    u_dir, u_shift, u_scale = parse_axis(uaxis_str) if uaxis_str else ((1, 0, 0), 0.0, 0.25)
    v_dir, v_shift, v_scale = parse_axis(vaxis_str) if vaxis_str else ((0, 1, 0), 0.0, 0.25)
    tex_w, tex_h = TEXTURE_PIXEL_SIZES.get(material.lower(), DEFAULT_TEXTURE_PIXEL_SIZE)
    if material.lower() not in TEXTURE_PIXEL_SIZES:
        warnings.append(
            "no known pixel size for '{}' - assuming {}x{} for displacement UVs "
            "(pass --texture-sizes if this looks wrong)".format(material, tex_w, tex_h))

    grid_points = []
    for r in range(n):
        t = r / (n - 1)
        row_start = lerp3(c00, c10, t)
        row_end = lerp3(c01, c11, t)
        row_pts = []
        for c in range(n):
            s = c / (n - 1)
            base_unscaled = lerp3(row_start, row_end, s)
            base = (base_unscaled[0] * scale - offset[0], base_unscaled[1] * scale - offset[1],
                    base_unscaled[2] * scale - offset[2])
            disp_normal = normals[r][c]
            disp_dist = distances[r][c] * scale
            off = scale_vec(offsets[r][c], scale)
            final = add(add(add(base, elev_scaled), scale_vec(disp_normal, disp_dist)), off)
            # Texture U/V must come from the ORIGINAL, unscaled Source
            # position - same reasoning as regular brush faces (see
            # convert_side's pts_source): Source's own texture axis
            # shift/scale values are calibrated against the map's native
            # unit size, so feeding them a scaled (e.g. 1.27x) position
            # would distort every displacement's texture tiling density by
            # that same factor. Confirmed by comparing a real generated
            # patch against its own uploaded, scaled .map counterpart: the
            # U/V values there scaled 1:1 with the world-scale factor,
            # which only happens if they were computed from scaled input.
            final_unscaled = add(add(add(base_unscaled, elev_unscaled), scale_vec(disp_normal, distances[r][c])),
                                  offsets[r][c])
            u, v = build_patch_uv(base_unscaled, u_dir, u_shift, u_scale, v_dir, v_shift,
                                   v_scale, tex_w, tex_h)
            row_pts.append((final, u, v))
        grid_points.append(row_pts)

    # Quake3 patches, like brush faces, render/light based on winding
    # direction - the row traversal order above happened to produce the
    # opposite winding from what's expected, showing the surface "from
    # underneath". Reversing row order flips winding without touching any
    # point's actual position or UV (each vertex still carries its own
    # correct (x,y,z,u,v) regardless of what order rows are listed in).
    grid_points = grid_points[::-1]

    lines = ["{", "patchDef2", "{", mapped_material, "( {} {} 0 0 0 )".format(n, n), "("]
    for row in grid_points:
        row_str = " ".join(
            "( {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} )".format(p[0], p[1], p[2], u, v)
            for p, u, v in row
        )
        lines.append("( {} )".format(row_str))
    lines.append(")")
    lines.append("}")
    lines.append("}")
    return "\n".join(lines)


    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    l = length(a)
    if l < 1e-9:
        return (0.0, 0.0, 0.0)
    return (a[0] / l, a[1] / l, a[2] / l)


def plane_normal(p1, p2, p3):
    return normalize(cross(sub(p2, p1), sub(p3, p1)))


# Standard idTech3 base texture axes, indexed by dominant plane-normal axis.
# (normal_ref, tex_x_axis, tex_y_axis) - taken from id Software's
# TextureAxisFromPlane table (brush.c), used by every idTech3 map compiler.
BASE_AXIS = [
    ((0, 0, 1), (1, 0, 0), (0, -1, 0)),    # 0: floor
    ((0, 0, -1), (1, 0, 0), (0, -1, 0)),   # 1: ceiling
    ((1, 0, 0), (0, 1, 0), (0, 0, -1)),    # 2: west wall
    ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),   # 3: east wall
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),    # 4: south wall
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),   # 5: north wall
]


def texture_axis_from_plane(normal):
    nx, ny, nz = normal
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        idx = 0 if nz > 0 else 1
    elif ax >= ay and ax >= az:
        idx = 2 if nx > 0 else 3
    else:
        idx = 4 if ny > 0 else 5
    _, tx, ty = BASE_AXIS[idx]
    return tx, ty


# ==============================================================================
# VMF field parsing helpers
# ==============================================================================

_PLANE_RE = re.compile(r'\(([^)]+)\)')
_AXIS_RE = re.compile(r'\[\s*([^\]]+)\]\s*([-\d.eE]+)')


def parse_plane(s):
    pts = _PLANE_RE.findall(s)
    return [tuple(float(x) for x in p.split()) for p in pts]


def parse_axis(s):
    """Parse a VMF uaxis/vaxis string: '[x y z shift] scale'."""
    m = _AXIS_RE.match(s.strip())
    if not m:
        return (1.0, 0.0, 0.0), 0.0, 0.25
    nums = [float(x) for x in m.group(1).split()]
    scale = float(m.group(2))
    direction = normalize((nums[0], nums[1], nums[2]))
    shift = nums[3] if len(nums) > 3 else 0.0
    return direction, shift, (scale if abs(scale) > 1e-6 else 1.0)


# ==============================================================================
# Texture name handling
# ==============================================================================

SKIP_SOLID_MATERIAL = 'TOOLS/VMF2MAP_SKIP_SOLID'
# A .vmf decompiled by BSPSource loses the clip textures: a compiled .bsp keeps
# a clip brush's CONTENTS (player clip) but its sides have no texture, and the
# decompiler writes them all as tools/toolsskip. de_overpass_csgo's .vmf
# ("Decompiled by BSPSource") has 0 clip faces and 580 brushes made only of
# skip - among them 2x1-unit slabs at exactly the warehouse railings' rail
# heights (the railings are "solid" "0" props: those slabs ARE their collision)
# and boxes over tree canopies and roofs. In an authored .vmf a skip-only brush
# has no contents at all (VBSP, like qbsp3: "hints and skips ... have no
# content"), so this is only switched on when the .vmf has no clip material
# anywhere - a real map always has some.
SKIP_BRUSHES_ARE_CLIPS = [False]
_CLIP_MATERIALS = {'TOOLS/TOOLSCLIP', 'TOOLS/TOOLSPLAYERCLIP', 'TOOLS/TOOLSNPCCLIP',
                   'TOOLS/TOOLSMONSTERCLIP', 'TOOLS/TOOLSGRENADECLIP'}


def scan_skip_brushes_are_clips(text):
    mats = {m.upper().replace('\\', '/') for m in
            re.findall(r'"material"\s+"([^"]*)"', text, flags=re.I)}
    return 'TOOLS/TOOLSSKIP' in mats and not (mats & _CLIP_MATERIALS)



DEFAULT_TOOL_TEXTURES = {
    # IMPORTANT: .map source files (as read by Radiant/GtkRadiant-family
    # editors) reference textures/shaders WITHOUT a literal "textures/"
    # prefix - that path is relative to the textures/ folder already. It's
    # q3map2 (the compiler) that prepends "textures/" when building the
    # compiled .bsp, which is why the RUNNING GAME and .shader script names
    # need that prefix (confirmed separately - see vmt2shader.py) while the
    # .map file itself must NOT have it, or Radiant/the compiler double it
    # up into "textures/textures/...".
    # ALL of the below are now CONFIRMED against SOF2's own shipped
    # tools.shader (the real shader definitions, not guesses). Each target
    # already has proper surfaceparms/qer_editorimage/q3map_material in
    # SOF2's file, which is exactly why faces get remapped to these names
    # rather than having vmt2shader.py invent duplicate definitions under
    # the Source names.
    "TOOLS/TOOLSNODRAW": "tools/_caulk",
    # Source's nodraw is SOLID + invisible. SOF2's own _nodraw declares
    # "surfaceparm nonsolid", so it is NOT the right target - _caulk
    # (nodraw + nolightmap + nomarks, no nonsolid) is the solid one.
    "TOOLS/TOOLSINVISIBLE": "tools/_caulk",
    "TOOLS/TOOLSTRIGGER": "tools/_trigger",
    "TOOLS/TOOLSHINT": "tools/_hint",
    # NOT SOF2's tools/_skip: that shader has no "surfaceparm skip", so q3map2
    # (facebsp.cpp only drops sides flagged C_SKIP) splits the BSP on every
    # "skip" side of every hint brush. Source's VBSP ignores them - a hint
    # brush there is one hint face plus five faces that do nothing. On
    # de_dust2 that is 93 hint brushes, several of them map-sized boxes, all
    # slicing the tree on planes nobody asked for. vmt2shader.py emits
    # vmf2map/skip with the missing surfaceparm.
    "TOOLS/TOOLSSKIP": "vmf2map/skip",
    # Player clips in a DECOMPILED .vmf (see SKIP_BRUSHES_ARE_CLIPS): only
    # set on brushes whose every side is toolsskip, and only when the .vmf
    # has no clip material at all.
    SKIP_SOLID_MATERIAL: "tools/_blockplayer",
    "TOOLS/TOOLSCLIP": "tools/_clip",
    # _blockplayer / _blocknpc match Source's player-only and NPC-only clip
    # semantics exactly (playerclip alone vs monsterclip alone), which is a
    # closer fit than the combined _clip these used to point at.
    "TOOLS/TOOLSPLAYERCLIP": "tools/_blockplayer",
    "TOOLS/TOOLSNPCCLIP": "tools/_blocknpc",
    "TOOLS/TOOLSMONSTERCLIP": "tools/_blocknpc",
    "TOOLS/TOOLSSKYBOX": "tools/_sky",
    "TOOLS/TOOLSSKYBOX2D": "tools/_sky",
    "TOOLS/TOOLSAREAPORTAL": "tools/_areaportal",
    # block_los only blocks NPC line of sight in Source; as _hint every face
    # of it became a vis splitter. Nothing in SOF2 corresponds, so it compiles
    # to nothing.
    "TOOLS/TOOLSBLOCK_LOS": "vmf2map/skip",
    "TOOLS/TOOLSORIGIN": "tools/_origin",
    "TOOLS/TOOLSBLACK": "tools/_black",
    # _shoot carries playerclip+monsterclip+shotclip - the closest thing to
    # Source's "blocks bullets" brush.
    "TOOLS/TOOLSBLOCKBULLETS": "tools/_shoot",
    "TOOLS/TOOLSINVISIBLELADDER": "tools/_ladder",
    "TOOLS/TOOLSLADDER": "tools/_ladder",
    "TOOLS/TOOLSNODROP": "tools/_nodrop",
    "TOOLS/TOOLSCLUSTERPORTAL": "tools/_clusterportal",
    "TOOLS/TOOLSFOG": "tools/_fogblack",
    # No SOF2 equivalent for these two concepts (light-blocking and Source's
    # occluder system have no counterpart in idTech3's vis model), so they
    # fall back to plain solid-invisible rather than being left dangling.
    "TOOLS/TOOLSBLOCKLIGHT": "tools/_caulk",
    "TOOLS/TOOLSOCCLUDER": "tools/_caulk",
}


def convert_texture_name(material, texture_map, unmapped_counter):
    key = material.strip().upper()
    if key in texture_map:
        return texture_map[key]
    if key in DEFAULT_TOOL_TEXTURES:
        return DEFAULT_TOOL_TEXTURES[key]
    # .map source files reference textures without a "textures/" prefix
    # (see note on DEFAULT_TOOL_TEXTURES above) - strip one off if the
    # source material name happened to already carry it, so we don't
    # silently pass through a mismatched convention from an odd VMF export.
    v = re.sub(r'^(textures/)+', '', material.strip().replace('\\', '/').lower(), flags=re.IGNORECASE)
    unmapped_counter[material] += 1
    return v


# ==============================================================================
# Brush / side conversion
# ==============================================================================

def _project_onto_plane(v, normal):
    """Return the component of v tangent to the face plane."""
    return sub(v, scale_vec(normal, dot(v, normal)))


def _texdef_fit(tx, ty, normal, u_dir, v_dir):
    """Fit a classic idTech3 texdef to Source's two 3-D texture axes.

    Classic .map has only a world-axis base pair plus one rotation.  Instead
    of deriving rotation from U alone, fit BOTH axes in the actual face plane.
    This matters on sloped faces: the world projection axes must first be
    projected onto the face, otherwise their apparent direction and texel
    density are wrong.

    Returns rotation, signed scale multipliers for U/V, and alignment errors.
    """
    # For a classic texdef, after rotation:
    #   S = tx*cos - ty*sin
    #   T = tx*sin + ty*cos
    # Project both onto the real face plane, because only those components
    # affect texture coordinates on the face.
    ptx = _project_onto_plane(tx, normal)
    pty = _project_onto_plane(ty, normal)
    u = normalize(u_dir)
    v = normalize(v_dir)

    # 2-D orthogonal Procrustes.  We choose the rotation maximizing the sum
    # of U/V directional alignment.  The legacy format can independently
    # mirror V through a negative V scale, so evaluate both handednesses.
    best = None
    for mirror_v in (False, True):
        vv = scale_vec(v, -1.0) if mirror_v else v
        # Objective for S(theta)·u + T(theta)·vv:
        # cos(theta)*(tx·u + ty·vv) + sin(theta)*(-ty·u + tx·vv)
        a = dot(ptx, u) + dot(pty, vv)
        b = -dot(pty, u) + dot(ptx, vv)
        theta = math.atan2(b, a)
        c, sn = math.cos(theta), math.sin(theta)
        s_axis = add(scale_vec(tx, c), scale_vec(ty, -sn))
        t_axis = add(scale_vec(tx, sn), scale_vec(ty, c))
        ps = normalize(_project_onto_plane(s_axis, normal))
        pt = normalize(_project_onto_plane(t_axis, normal))
        ua = dot(ps, u) if length(ps) else -1.0
        va = dot(pt, vv) if length(pt) else -1.0
        score = ua + va
        if best is None or score > best[0]:
            best = (score, theta, s_axis, t_axis, ua, va, mirror_v)

    _, theta, s_axis, t_axis, ua, va, mirror_v = best
    ps_raw = _project_onto_plane(s_axis, normal)
    pt_raw = _project_onto_plane(t_axis, normal)
    ps_mag = length(ps_raw)
    pt_mag = length(pt_raw)

    # A world-axis projection is foreshortened on a sloped face.  Compensate
    # for that here so the texture doesn't get stretched simply because the
    # legacy format cannot use a face-aligned 3-D projection.
    return (math.degrees(theta), ps_mag, pt_mag, ua, va, mirror_v)


def convert_side(side, texture_map, unmapped_counter, scale, offset, warnings):
    plane_str = get1(side, 'plane')
    if not plane_str:
        warnings.append("side with no plane skipped")
        return None
    pts = parse_plane(plane_str)
    if len(pts) != 3:
        warnings.append("side with malformed plane skipped")
        return None
    if length(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0]))) < 1e-6:
        # Source itself tolerates a plane whose 3 points don't form a real
        # triangle (e.g. two literally-identical points) - VBSP silently
        # discards these degenerate/sliver faces during its own compile,
        # so they never cause a problem in the original map. q3map2 has no
        # such tolerance: a zero-normal plane crashes the whole BSP step
        # outright ("BaseWindingForPlaneAccu: no dominant axis found
        # because normal is too short"), confirmed against a real map
        # where 21 such planes made it through unnoticed until compile.
        warnings.append("side with degenerate plane (zero-area) skipped - "
                         "would have crashed the BSP compile")
        return None
    ox, oy, oz = offset
    pts_source = pts  # keep the ORIGINAL (untransformed) points for the shift solve below
    pts = [(p[0] * scale - ox, p[1] * scale - oy, p[2] * scale - oz) for p in pts]

    # format_side() later writes each coordinate at only 6 significant
    # figures. At the coordinate magnitudes this map actually uses
    # (10000+ units), that leaves as little as ~1 decimal digit of
    # precision - so two points can be genuinely distinct here (passing
    # the raw check above) and still round down to byte-identical text,
    # producing a plane that reads as degenerate to q3map2 even though
    # the pre-rounding math wasn't. Confirmed against a real map: the raw
    # check above didn't catch these, but this rounded one does. Check
    # against what will ACTUALLY be written, not just the raw math.
    rounded_pts = [tuple(float('{:.10g}'.format(c)) for c in p) for p in pts]
    if length(cross(sub(rounded_pts[1], rounded_pts[0]), sub(rounded_pts[2], rounded_pts[0]))) < 1e-6:
        warnings.append("side with degenerate plane after rounding to map-file "
                         "precision skipped - would have crashed the BSP compile")
        return None

    material = get1(side, 'material', 'tools/toolsnodraw')
    texture = convert_texture_name(material, texture_map, unmapped_counter)

    uaxis_str = get1(side, 'uaxis')
    vaxis_str = get1(side, 'vaxis')
    if uaxis_str and vaxis_str:
        u_dir, u_shift, u_scale = parse_axis(uaxis_str)
        v_dir, v_shift, v_scale = parse_axis(vaxis_str)
        normal = plane_normal(*pts)

        # Evaluate every standard idTech3 projection basis and keep the one
        # that best reproduces BOTH Source axes on this actual face.  The old
        # code always chose the dominant normal axis before fitting rotation;
        # that is good for ordinary walls but can be noticeably worse on
        # diagonal/sloped faces.
        candidates = []
        for base_normal, tx, ty in BASE_AXIS:
            rotation, proj_u, proj_v, u_alignment, v_alignment, mirror_v = \
                _texdef_fit(tx, ty, normal, u_dir, v_dir)
            # Prefer angular fidelity, with a small penalty for very short
            # projected axes (which are inherently poor texture projections).
            error = (1.0 - max(-1.0, min(1.0, u_alignment))) + \
                    (1.0 - max(-1.0, min(1.0, v_alignment)))
            projection_quality = min(proj_u, proj_v)
            score = error + (1.0 - projection_quality) * 0.15
            candidates.append((score, tx, ty, rotation, proj_u, proj_v,
                               u_alignment, v_alignment, mirror_v, base_normal))
        candidates.sort(key=lambda x: x[0])
        _, tx, ty, rotation, proj_u, proj_v, u_alignment, v_alignment, mirror_v, base_normal = candidates[0]

        # Convert Source texel density to the density of the projected legacy
        # axis.  For a perfect fit proj_* is 1.0 and this is unchanged.  On a
        # sloped face this removes the otherwise unavoidable-looking stretch
        # caused by projecting a world axis through the face.
        scale_x = scale * u_scale * (proj_u if proj_u > 1e-6 else 1.0)
        scale_y = scale * v_scale * (proj_v if proj_v > 1e-6 else 1.0)
        if mirror_v:
            scale_y = -scale_y

        # The selected legacy axis is not necessarily the original Source
        # axis.  Shift must therefore be solved using the actual output axis,
        # not the Source uaxis/vaxis.  This keeps the texture origin anchored
        # at the same world-space locations while still fitting the direction.
        rad = math.radians(rotation)
        out_u = (tx[0] * math.cos(rad) - ty[0] * math.sin(rad),
                 tx[1] * math.cos(rad) - ty[1] * math.sin(rad),
                 tx[2] * math.cos(rad) - ty[2] * math.sin(rad))
        out_v = (tx[0] * math.sin(rad) + ty[0] * math.cos(rad),
                 tx[1] * math.sin(rad) + ty[1] * math.cos(rad),
                 tx[2] * math.sin(rad) + ty[2] * math.cos(rad))
        # Solve shift by anchoring to an ACTUAL point on the face, matching
        # Source's true UV there to what our chosen (possibly best-fit, not
        # exact) axis produces there - NOT by reusing Source's raw shift
        # value with only an offset correction. That reuse is only valid
        # when out_u/out_v exactly reproduce u_dir/v_dir (true for simple
        # axis-aligned faces), which is exactly NOT the case for a
        # best-fit-but-imperfect axis on a tilted face - reusing u_shift
        # there silently assumes zero approximation error, which is wrong
        # by construction whenever u_alignment/v_alignment aren't both 1.0.
        anchor_source = pts_source[0]
        anchor_out = pts[0]
        true_u = dot(anchor_source, u_dir) / u_scale + u_shift
        true_v = dot(anchor_source, v_dir) / v_scale + v_shift
        new_u_no_shift = dot(anchor_out, out_u) / scale_x if scale_x else 0.0
        new_v_no_shift = dot(anchor_out, out_v) / scale_y if scale_y else 0.0
        shift_x = true_u - new_u_no_shift
        shift_y = true_v - new_v_no_shift

        # Report only meaningful approximation loss.  This replaces the old
        # dominant-axis warning with a useful measurement of how well the
        # chosen classic texdef reproduces the Source axes.
        u_err = max(0.0, min(1.0, 1.0 - u_alignment))
        v_err = max(0.0, min(1.0, 1.0 - v_alignment))
        # Tool faces are not visible textures, so large UV differences on
        # nodraw/clip/hint/skip/etc. are noise rather than a useful warning.
        is_tool = material.strip().lower().startswith('tools/')
        if (u_err > 0.002 or v_err > 0.002) and not is_tool:
            u_deg = math.degrees(math.acos(max(-1.0, min(1.0, u_alignment))))
            v_deg = math.degrees(math.acos(max(-1.0, min(1.0, v_alignment))))
            warnings.append(
                "side with material '{}' has unavoidable classic-.map UV approximation: "
                "U error {:.1f} deg, V error {:.1f} deg (best-fit projection); "
                "brush geometry remains exact".format(material, u_deg, v_deg))
    else:
        rotation, shift_x, shift_y, scale_x, scale_y = 0.0, 0.0, 0.0, 0.25, 0.25

    return {
        'points': pts,
        'texture': texture,
        'shift': (shift_x, shift_y),
        'rotation': rotation,
        'scale': (scale_x, scale_y),
    }

# Source stores a light's colour AND brightness in one key, "_light"
# "<r> <g> <b> <brightness>". q3map2 stores them separately: "light" for
# intensity, "_color" for colour. Passing Source's key through untouched is
# not a no-op, it is silently WRONG: q3map2's light.c reads intensity with
# FloatForKey( e, "_light" ), which parses only the FIRST number - the RED
# channel. Measured on de_dust2, every one of its 25 point/spot lights
# compiled at intensity 255 (red channel of 254/255) in pure white, whatever
# it was authored as: brightnesses of 10, 35, 50 and 150 and a warm
# 254 251 192 tint all collapsed to the same flat white light. q3map2 reads
# "_light" BEFORE "light", so the Source key has to be removed, not merely
# supplemented.
SOURCE_LIGHT_KEYS_TO_DROP = (
    '_light', '_lightHDR', '_lightscaleHDR', '_constant_attn', '_linear_attn',
    '_quadratic_attn', '_distance', '_zero_percent_distance', '_fifty_percent_distance',
    '_hardfalloff', '_exponent', '_inner_cone', '_cone',
    # Source light spawnflags 1 is "Initially dark"; in q3map2 bit 1 means
    # linear falloff. The falloff type is written from the _*_attn keys.
    'spawnflags',
)

# Multiplier and offset applied to Source's brightness on its way to q3map2's
# "light" key. Default 1.0 / 0.0 - a straight carry-over, because there is no
# derivable unit conversion between the two: both engines fall off with
# inverse square (q3map2: photons = intensity * pointScale, then 1/d^2), so a
# pure multiplier is the only physically meaningful relation, and its value
# depends on choices neither engine documents. The offset exists only so an
# empirical remap can be tried; see --point-light-offset for why it distorts.
POINT_LIGHT_SCALE = [1.0]
POINT_LIGHT_OFFSET = [0.0]


CONTENTS_DETAIL = 134217728  # 0x8000000 - Q3's detail-brush content flag


def fp(p):
    """One plane point in .map syntax, shared by both face formats."""
    return "( {:.10g} {:.10g} {:.10g} )".format(p[0], p[1], p[2])


def format_side(s, detail=False):
    (p1, p2, p3) = s['points']
    content_flags = CONTENTS_DETAIL if detail else 0
    return "{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {} 0 0".format(
        fp(p1), fp(p2), fp(p3),
        s['texture'],
        s['shift'][0], s['shift'][1],
        s['rotation'],
        s['scale'][0], s['scale'][1],
        content_flags,
    )


def _side_plane(s):
    n = normalize(cross(sub(s['points'][1], s['points'][0]), sub(s['points'][2], s['points'][0])))
    d = dot(n, s['points'][0])
    return n, d


def dedup_and_clean_sides(out_sides, warnings, brush_label, is_entity_brush=False):
    """Port of bobToolz's DBrush::RemoveRedundantPlanes (same algorithm
    already verified in phy2clip.py): merge near-duplicate planes at
    bobToolz's own MAX_ROUND_ERROR=0.05 tolerance, then drop any plane
    touching fewer than 3 of the brush's own solved vertices - it isn't
    defining a real face, just leftover redundancy. This is exactly what
    was needed by hand via bobToolz's "Brush Cleanup" after conversion;
    doing it here removes that manual step for ordinary world/entity
    brushes, not just the clip brushes phy2clip.py generates.
    """
    DUP_DOT = 1 - 0.05 ** 2 / 2  # == 0.99875, bobToolz's exact equivalent
    DUP_DIST = 0.05
    kept = []
    for s in out_sides:
        n, d = _side_plane(s)
        for ks in kept:
            kn, kd = _side_plane(ks)
            if dot(kn, n) > DUP_DOT and abs(kd - d) < DUP_DIST:
                break
        else:
            kept.append(s)
    if len(kept) < len(out_sides):
        warnings.append("brush{} had {} duplicate plane(s) removed (bobToolz-equivalent cleanup)"
                         .format(brush_label, len(out_sides) - len(kept)))

    if len(kept) < 4:
        return kept

    # Orient every plane so the brush's own centroid sits on its inside
    # (negative) side. Source's raw 3-point plane ordering doesn't
    # guarantee outward-facing winding the way phy2clip.py's own generated
    # hulls did there (fully controlled point order) - confirmed on a real
    # brush from this exact map, where the naive cross-product gave two
    # directly-opposing half-space constraints (z>=-36 AND z<=-38
    # simultaneously), making solve_brush_vertices find zero vertices and
    # silently skip the thin-brush check it exists to catch.
    all_pts = [p for s in kept for p in s['points']]
    centroid = tuple(sum(p[i] for p in all_pts) / len(all_pts) for i in range(3))
    planes = []
    for s in kept:
        n, d = _side_plane(s)
        if dot(centroid, n) > d:
            n, d = (-n[0], -n[1], -n[2]), -d
        planes.append((n, d))

    verts = solve_brush_vertices(planes)
    if verts:
        survivors = []
        for s, (n, d) in zip(kept, planes):
            touching = sum(1 for v in verts if abs(dot(v, n) - d) < 0.05)
            if touching >= 3:
                survivors.append(s)
        if len(survivors) >= 4 and len(survivors) < len(kept):
            warnings.append("brush{} had {} redundant plane(s) removed (touched < 3 real vertices)"
                             .format(brush_label, len(kept) - len(survivors)))
            kept = survivors

        # Thin-brush diagnostic. This does NOT auto-fix anything - altering
        # real world geometry automatically risks visibly shifting walls or
        # floors, unlike the isolated generated clip brushes in phy2clip.py.
        # Confirmed cause of a real compile-time "Entity leaked" warning on
        # this project: a func_static brush only 2 units thick in one axis.
        # q3map2's entity-flood-fill can fail to resolve which interior area
        # a dangerously thin brush belongs to even when the overall map is
        # sealed, producing exactly that warning on an otherwise-clean
        # compile.
        ext = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
        if is_entity_brush and 0 < min(ext) < 4.0:
            warnings.append(
                "brush{} is only {:.1f} units thick on its shortest axis - "
                "this thickness is known to cause q3map2 'Entity leaked' warnings "
                "even on an otherwise-sealed map; consider thickening it by hand if "
                "you see that warning on this brush".format(brush_label, min(ext)))
    return kept


def solve_brush_vertices(planes):
    """Intersect every triple of (normal, dist) planes, keeping points that
    satisfy every plane's half-space - the brush's real, solved vertex set."""
    verts = []
    n = len(planes)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                n1, d1 = planes[i]
                n2, d2 = planes[j]
                n3, d3 = planes[k]
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
                if all(dot(p, pn) <= pd + 0.05 for pn, pd in planes):
                    verts.append(p)
    return verts


# Texture for nodraw "slab" brushes that Source mappers build displacements
# ON TOP OF (see scan_terrain_base_slabs). vmt2shader.py defines it as
# nodraw + nonsolid (NOT trans): q3map2 seals the map with any opaque
# structural brush, solid or not, so these still seal exactly as before but
# no longer collide.
TERRAIN_BASE_TEXTURE = 'vmf2map/terrain_base'
TERRAIN_BASE_SOLIDS = set()
TERRAIN_BASE_MIN_COVERAGE = 0.95
_NODRAW_TOOL_MATERIALS = {'TOOLS/TOOLSNODRAW', 'TOOLS/TOOLSINVISIBLE'}


def scan_terrain_base_slabs(root):
    """Find all-nodraw brushes with a face lying exactly on a displacement's
    base plane (same plane, same facing), where that face is fully covered -
    by the displacement terrain itself, by some OTHER real (non-nodraw-only)
    solid brush's upward face at or above it, or a mix of both.

    Why: Source builds displacements on such slabs, so wherever a
    displacement is flat its surface lies ON the slab's face (measured on
    de_dust2: often within 0.01 units), and wherever it dips, the slab is
    on top. Quake-engine collision can't separate surfaces closer than its
    0.125-unit clip epsilon, so players and bullets randomly hit the
    invisible slab instead of the terrain - the reported caulk footsteps and
    impacts on sand. The "or some other real brush" half matters just as
    much: a slab only 66-85% covered by terrain looked partial at first, but
    every part NOT under terrain turned out to already have its own ordinary
    floor brush over it (checked directly against real de_dust2 data) -
    Source mappers routinely build displacement slabs a bit larger than the
    displacement itself, with a normal floor brush picking up the rest. Left
    fully solid, that whole slab caused the same mismatch as a fully-covered
    one, just less often. A slab only reported (left solid) if some of its
    area truly has no cover at all - real walkable floor might be missing
    otherwise.

    Returns (set of solid ids made non-solid, list of (id, coverage 0..1, a
    corner point) for any slab that still has uncovered area).
    """
    solids = []
    def walk(node):
        if isinstance(node, dict):
            for k, vals in node.items():
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    if k == 'solid':
                        solids.append(v)
                    walk(v)
    walk(root)

    bases = []
    for s in solids:
        sides = [x for x in s.get('side', []) if isinstance(x, dict)]
        for x in sides:
            if isinstance(get1(x, 'dispinfo'), dict):
                n, d = plane_from_points(*parse_plane(get1(x, 'plane')))
                poly = get_face_polygon(x, sides)
                if poly:
                    bases.append((n, d, poly))
    if not bases:
        return set(), []

    def inside(p, poly, n):
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            if dot(cross(sub(b, a), sub(p, a)), n) < -1e-3:
                return False
        return True

    # Candidate "real floor": any brush that ISN'T entirely nodraw-tool-
    # textured (so it's actual visible/structural geometry, not another
    # nodraw slab) and isn't a displacement, upward-facing side only.
    other_tops = []
    for s in solids:
        sides = [x for x in s.get('side', []) if isinstance(x, dict)]
        if not sides or any(isinstance(get1(x, 'dispinfo'), dict) for x in sides):
            continue
        if all(get1(x, 'material', '').upper() in _NODRAW_TOOL_MATERIALS for x in sides):
            continue
        for x in sides:
            n, d = plane_from_points(*parse_plane(get1(x, 'plane')))
            if n[2] < 0.3:
                continue
            poly = get_face_polygon(x, sides)
            if poly:
                other_tops.append((n, d, poly))

    full, partial = set(), []
    for s in solids:
        sides = [x for x in s.get('side', []) if isinstance(x, dict)]
        if not sides or any(isinstance(get1(x, 'dispinfo'), dict) for x in sides):
            continue
        if not all(get1(x, 'material', '').upper() in _NODRAW_TOOL_MATERIALS for x in sides):
            continue
        best = 0.0
        corner = None
        for x in sides:
            n, d = plane_from_points(*parse_plane(get1(x, 'plane')))
            co = [(bn, bp) for bn, bd, bp in bases if dot(bn, n) > 0.999 and abs(bd - d) < 0.5]
            if not co:
                continue
            poly = get_face_polygon(x, sides)
            if not poly:
                continue
            u = normalize(sub(poly[1], poly[0]))
            w = cross(n, u)
            us = [dot(p, u) for p in poly]
            ws = [dot(p, w) for p in poly]
            tot = hit = 0
            for a in range(41):
                for b in range(41):
                    p = add(add(scale_vec(n, d), scale_vec(u, min(us) + (max(us) - min(us)) * a / 40.0)),
                            scale_vec(w, min(ws) + (max(ws) - min(ws)) * b / 40.0))
                    if not inside(p, poly, n):
                        continue
                    tot += 1
                    if any(inside(p, bp, bn) for bn, bp in co):
                        hit += 1
                        continue
                    for on, od, op in other_tops:
                        if abs(on[2]) < 1e-6:
                            continue
                        oz = (od - on[0] * p[0] - on[1] * p[1]) / on[2]
                        if oz < p[2] - 1.0:
                            continue  # must be at/above the slab, not buried below it
                        if inside(p, op, on):
                            hit += 1
                            break
            if tot and hit / float(tot) > best:
                best = hit / float(tot)
                corner = poly[0]
        if best >= TERRAIN_BASE_MIN_COVERAGE:
            full.add(get1(s, 'id'))
        elif best > 0.0:
            partial.append((get1(s, 'id'), best, corner))
    return full, partial

def _wrap_patch_as_func_group(patch):
    """build_displacement_patch returns a SOLID-level block
    ("{ patchDef2 {...} }"), meant to sit alongside other brush blocks
    inside an entity - not a standalone, appendable-at-the-top-level
    entity itself. Wrapping it here is what makes it safe to add to
    patch_output_blocks (which, like terrain_outputs, gets appended
    directly to out_blocks - see convert_vmf), matching exactly how
    build_displacement_terrain_patch already wraps ITS OWN patchDef2 in a
    func_group for the same reason.
    """
    return '{\n"classname" "func_group"\n' + patch + '\n}'


def convert_solid(solid, texture_map, unmapped_counter, scale, offset, warnings, detail=False,
                   is_entity_brush=False, materials_dir=None, terrain_outputs=None,
                   terrain_mode='brush', drop_backing_brush=False, patch_output_blocks=None,
                   sky_region=None):
    sides = solid.get('side', [])
    out_sides = []
    if patch_output_blocks is None:
        patch_output_blocks = []
    dropped_for_terrain = False
    _mats = [(get1(sd, 'material', '') or '').strip().upper().replace('\\', '/')
             for sd in sides if isinstance(sd, dict)]
    all_skip = (SKIP_BRUSHES_ARE_CLIPS[0] and bool(_mats)
                and all(m == 'TOOLS/TOOLSSKIP' for m in _mats))
    for side in sides:
        if not isinstance(side, dict):
            continue  # a plain "side" keyvalue rather than a nested side block
        if all_skip:
            side = dict(side)
            side['material'] = [SKIP_SOLID_MATERIAL]       # see DEFAULT_TOOL_TEXTURES
        if dropped_for_terrain and not isinstance(get1(side, 'dispinfo'), dict):
            continue  # brush is being dropped; only its other displaced sides still matter

        if isinstance(get1(side, 'dispinfo'), dict):
            material = get1(side, 'material', 'tools/toolsnodraw')
            blend_info = read_vmt_blend_info(material, materials_dir)
            if ((blend_info is not None or terrain_mode == 'lwo')
                    and terrain_outputs is not None):
                # In lwo mode EVERY displacement goes into the welded mesh,
                # blended or not. Plain ones used to become patchDef2
                # Beziers (which don't pass through their own control
                # points - measured up to ~200 units off Source's surface on
                # de_dust2) sitting on a SOLID caulk base brush, which poked
                # through and was walked on (the reported 'caulk under pure
                # sand' + stone footsteps).
                # A genuine two-texture vertex-blended displacement (Source's
                # WorldVertexTransition) - build it as a real SOF2 terrain
                # entity (confirmed working: verified against an actual,
                # tested SOF2 map using this exact system) instead of a flat
                # single-texture patch, so both textures actually show up,
                # blended, in game.
                if terrain_mode == 'lwo':
                    # REVERTED (confirmed real, reproduced crash): a
                    # displacement inside the 3D skybox area used to be
                    # diverted into its own full-resolution patchDef2
                    # (build_displacement_terrain_patch) instead of feeding
                    # the welded LWO mesh, as a diagnostic for whether
                    # misc_model's forced-meta terrain hitting the map's
                    # leak was skybox-specific. It wasn't worth keeping: a
                    # native-power Source displacement is up to a 17x17
                    # control-point Bezier patch over genuinely bumpy
                    # terrain, and q3map2/SOF2's own per-patch collision
                    # facet generation (CM_PatchCollide) has a fixed
                    # MAX_PATCH_PLANES array - confirmed as the real cause
                    # of an in-game crash on .bsp load (compile itself
                    # succeeds, wrote a valid .bsp; the crash is the
                    # engine building that one patch's collision at map
                    # load), reproduced with exactly the 8 displacements
                    # that fell inside --skybox-radius on a real de_dust2
                    # export. This is also exactly the seam/Bezier-mismatch
                    # failure mode --terrain-mode lwo exists to eliminate
                    # in the first place (see --terrain-mode's own help
                    # text) - so diverting a handful of displacements back
                    # into that same patch representation just because
                    # they're near the sky_camera reintroduced the bug the
                    # lwo mode was built to avoid.
                    #
                    # Every displacement - in or out of the skybox area -
                    # now feeds the same shared LWOTerrainBuilder below.
                    # The skybox-only portion is still kept structurally
                    # separate from the main map: split_mesh_by_region (see
                    # the sky_split logic in convert_vmf's finalize step)
                    # divides the FINISHED welded mesh by distance from the
                    # sky_camera and writes the skybox piece as its own
                    # model_static (bypasses q3map2's meta/skybox-marking
                    # pipeline entirely, so it isn't caught by this map's
                    # leak the way misc_model was) - same mechanism already
                    # proven for the overlay conforming mesh's own sky
                    # split, just applied to terrain too.
                    #
                    # No per-side entity/pcx here - every displacement's
                    # grid feeds into ONE shared, map-wide LWOTerrainBuilder
                    # (welded across all of them), finalized once at the end
                    # of convert_vmf into a single misc_model. See
                    # compute_displacement_grid_lwo/LWOTerrainBuilder.
                    global LWO_TERRAIN_BUILDER
                    lwo_result = compute_displacement_grid_lwo(
                        side, sides, scale, offset, warnings, material,
                        is_blend=blend_info is not None, texture_map=texture_map,
                        unmapped_counter=unmapped_counter, materials_dir=materials_dir)
                    if lwo_result and LWO_TERRAIN_BUILDER is not None:
                        # which Source face this displacement is - an
                        # info_overlay's 'sides' key names these, and only
                        # these are what VBSP projects that overlay onto
                        lwo_result['side_id'] = str(get1(side, 'id'))
                        LWO_TERRAIN_BUILDER.add_displacement(lwo_result)
                        # Always, in lwo mode: VBSP deletes every brush that
                        # carries a displacement (map.cpp: "don't keep this
                        # brush"), textured sides included - so keeping it
                        # would add geometry Source never had.
                        if True:
                            # Same reasoning/caveat as the patch/brush modes'
                            # own drop_backing_brush handling below - the
                            # model (with spawnflags 6) supplies both the
                            # visible surface and collision, so the original
                            # flat brush face is redundant, not just hidden.
                            dropped_for_terrain = True
                            continue
                        side = dict(side)
                        side['material'] = ['tools/toolsnodraw']
                    else:
                        # compute_displacement_grid_lwo already warned why;
                        # fall back to the flat-patch path.
                        #
                        # BUGFIX: this branch used to never check
                        # drop_backing_brush at all, unlike every other path
                        # in this function - it always kept the original
                        # brush around (nodraw'd). CONFIRMED as a real,
                        # visible bug: nodraw gets translated to SOF2's
                        # "tools/_caulk" (deliberately SOLID, see that
                        # mapping's own comment), and since this flat brush
                        # sits at a constant, undisplaced height while the
                        # surrounding LWO mesh is genuinely displaced, it
                        # pokes through in scattered spots wherever the real
                        # terrain dips below that flat level - reported
                        # directly as smooth square patches "coming through"
                        # the jagged terrain, visible in the SOF2 editor but
                        # absent from the original Hammer/.vmf (since this
                        # brush is generated fresh by this fallback, not
                        # present in Source's own data). Now dropping the
                        # brush here too when requested, exactly like the
                        # successful case above - the flat patch (still
                        # added to patch_blocks) is what stands in for the
                        # missing collision/visual, not a lingering nodraw
                        # copy of the original brush.
                        patch = build_displacement_patch(side, sides, texture_map,
                                                          unmapped_counter, scale, offset, warnings)
                        if patch:
                            patch_output_blocks.append(_wrap_patch_as_func_group(patch))
                            if drop_backing_brush:
                                dropped_for_terrain = True
                                continue
                            side = dict(side)
                            side['material'] = ['tools/toolsnodraw']
                elif terrain_mode == 'patch':
                    terrain = build_displacement_terrain_patch(
                        side, sides, texture_map, unmapped_counter,
                        scale, offset, warnings, material)
                    if terrain:
                        terrain_outputs.append(terrain)
                        if drop_backing_brush:
                            dropped_for_terrain = True
                            continue
                        side = dict(side)
                        side['material'] = ['tools/toolsnodraw']
                    else:
                        # Same bugfix as the lwo branch's fallback above -
                        # this never checked drop_backing_brush either.
                        patch = build_displacement_patch(side, sides, texture_map,
                                                          unmapped_counter, scale, offset, warnings)
                        if patch:
                            patch_output_blocks.append(_wrap_patch_as_func_group(patch))
                            if drop_backing_brush:
                                dropped_for_terrain = True
                                continue
                            side = dict(side)
                            side['material'] = ['tools/toolsnodraw']
                    # toggle) that Source's own flat, undisplaced base
                    # brush is purely an editor/fallback visualization,
                    # never separately visible or walked-on in the
                    # actual running game - the displacement completely
                    # replaces it. So dropping this whole brush (not
                    # just nodrawing its displacement face) may be
                    # safe, PROVIDED the patch itself supplies working
                    # collision and doesn't leave a compiler leak where
                    # this brush used to help seal the world - both
                    # unverified assumptions, which is why this is
                    # opt-in and not the default.
                    terrain = build_displacement_terrain(side, sides, texture_map, unmapped_counter,
                                                          scale, offset, warnings, material)
                    if terrain:
                        terrain_outputs.append(terrain)
                        if drop_backing_brush:
                            dropped_for_terrain = True
                            continue
                        side = dict(side)
                        side['material'] = ['tools/toolsnodraw']
                    else:
                        # build_displacement_terrain already warned why; fall
                        # back to the normal flat-patch path so the surface at
                        # least isn't left completely blank. Same bugfix as
                        # the lwo/patch branches' own fallbacks above - this
                        # never checked drop_backing_brush either.
                        patch = build_displacement_patch(side, sides, texture_map, unmapped_counter,
                                                          scale, offset, warnings)
                        if patch:
                            patch_output_blocks.append(_wrap_patch_as_func_group(patch))
                            if drop_backing_brush:
                                dropped_for_terrain = True
                                continue
                            side = dict(side)
                            side['material'] = ['tools/toolsnodraw']
            else:
                patch = build_displacement_patch(side, sides, texture_map, unmapped_counter,
                                                  scale, offset, warnings)
                if patch:
                    patch_output_blocks.append(_wrap_patch_as_func_group(patch))
                    if drop_backing_brush:
                        # Same as every terrain path: Source's own
                        # displacement base brush is never solid or drawn,
                        # so with the flag it isn't kept here either (it
                        # used to stay as solid caulk under the patch).
                        dropped_for_terrain = True
                        continue
                    # The brush itself still needs this face to stay a normal
                    # flat quad (so the solid stays sealed/valid) - it's just
                    # hidden behind the real displaced surface now provided by
                    # the patch above, so give it nodraw instead of the visible
                    # material to avoid a double-textured/z-fighting look.
                    side = dict(side)
                    side['material'] = ['tools/toolsnodraw']

        conv = convert_side(side, texture_map, unmapped_counter, scale, offset, warnings)
        if conv:
            out_sides.append(conv)

    if dropped_for_terrain:
        # Whole brush dropped (see drop_backing_brush above) - the terrain
        # entity built above is this displacement's only remaining output,
        # already appended to terrain_outputs. No brush block to return.
        return None

    if not out_sides:
        return None

    brush_id = get1(solid, 'id')
    brush_label = " (id {})".format(brush_id) if brush_id else ""
    out_sides = dedup_and_clean_sides(out_sides, warnings, brush_label, is_entity_brush)
    if len(out_sides) < 4:
        warnings.append("brush{} has fewer than 4 valid planes after cleanup, skipped"
                         .format(brush_label))
        return None

    if get1(solid, 'id') in TERRAIN_BASE_SOLIDS:
        for s in out_sides:
            s['texture'] = TERRAIN_BASE_TEXTURE
    lines = ["{"]
    for s in out_sides:
        lines.append(format_side(s, detail))
    lines.append("}")
    return "\n".join(lines)


# ==============================================================================
# Entity conversion
# ==============================================================================

SKIP_KEYS_FOR_ENTITY = {'solid', 'editor', 'connections', 'group', 'id', 'hidden'}

# Source-engine-only keys with no SOF2 counterpart. Passing these through
# is harmless-looking but clutters every entity, and in the case of
# spawnflags is actively destructive (see SPAWNFLAG_UNSAFE_CLASSES below).
SOURCE_ONLY_ENTITY_KEYS = {
    'rendermode', 'renderfx', 'rendercolor', 'renderamt', 'disableshadows',
    'shadowcastdist', 'physdamagescale', 'performancemode', 'nodamageforces',
    'pressuredelay', 'fademindist', 'fademaxdist', 'fadescale',
    'screenspacefade', 'mindxlevel', 'maxdxlevel', 'minclientdxlevel',
    'maxclientdxlevel', 'disablereceiveshadows', 'disableflashlight',
    'ignorenormals', 'texturescale', 'lightmapscale_source',
    'explodedamage', 'exploderadius', 'damagetoenablemotion',
    'forcetoenablemotion', 'preferredcarryangles', 'massscale',
    'inertiascale', 'damagefilter', 'shadowdepthnocache', 'disablebonefollowers',
    'generatelightmaps', 'lightmapresolutionx', 'lightmapresolutiony',
}

# Private marker key set by worldize_skybox_models() on an entity it has
# moved out of the 3D skybox area into real map space. Listed in
# SOURCE_ONLY_ENTITY_KEYS above so it never reaches the .map file; it only
# exists so the later passes (--skip-3d-skybox's exclusion, and
# convert_entity's skybox-region handling) can tell "this model USED to be
# skybox content but is now ordinary main-map geometry" from "this model is
# still skybox content". Needed because a worldized model's new origin can
# still land inside --skybox-radius (measured on de_dust2: the 18 distant
# background palms transform to x 3268..10116 while the sky_camera sits at
# x 7423 with radius 4096), so position alone can't distinguish them.
WORLDIZED_MARKER_KEY = 'vmf2map_skybox_worldized'
SOURCE_ONLY_ENTITY_KEYS.add(WORLDIZED_MARKER_KEY)

# Classes where a passed-through Source "spawnflags" value is DANGEROUS,
# because the destination entity reads completely different bit meanings
# from the same number. Confirmed against SOF2's own mp_entities.def:
# misc_model's flags are RMG_Cast_Shadows(1) / SOLID(2) / Lightmapped(4) /
# Auto_clip_and_Lightmapped(8). Real converted content was carrying Source
# prop_static values like 1035 and 1039, which decode in SOF2 as
# RMG_Cast_Shadows + SOLID + AUTO_CLIP - and the .def warns auto clip
# "can hurt in-game performance (as well as produce erroneous clipping
# results) when used on small, dense models". That is exactly what
# produced "MAX_MAP_PLANES hit generating clip brushes for model ...md3"
# on a dense panel model. Dropping the key entirely leaves the model at
# flags 0 (no autoclip, not solid - clip brush it by hand if you need
# collision, which is the documented normal workflow anyway).
SPAWNFLAG_UNSAFE_CLASSES = {'misc_model', 'func_static', 'func_wall', 'func_group'}

# Entity keys that hold a model path, where a Source .mdl reference should
# become an SOF2 .md3 reference (same directory structure, extension only).
MODEL_VALUE_KEYS = {'model', 'model2'}

# Classname -> classname remap, verified against SOF2's own mp_entities.def /
# mp_entitiesd.def (QUAKED entity definitions shipped with the SDK/editor).
# Entries marked CONFIRMED matched an existing SOF2 entity with clearly
# equivalent purpose. Entries marked BEST-GUESS have no exact equivalent -
# the mapping preserves geometry/position but you should sanity check the
# behavior in-editor. Override any of these with --class-map.
DEFAULT_CLASS_MAP = {
    # --- CONFIRMED: models ---
    # misc_model: "turns into map triangles - not solid" (baked into BSP at
    # compile time) - closest match to a permanent, non-interactive static prop.
    'prop_static': 'misc_model',
    # misc_model_static: "loaded as a model in the renderer - does not take
    # up precious bsp space" (entity-based, not baked) - better fit than
    # model_static, whose own docs say "best to not use these... if possible".
    'prop_dynamic': 'misc_model',
    # prop_dynamic_override is prop_dynamic that may use a model not built
    # for it (de_mirage_cs2's two TVs). Left unmapped, SOF2 spawned nothing.
    'prop_dynamic_override': 'misc_model',
    # prop_physics_multiplayer/prop_physics have no SOF2 equivalent with
    # physics simulation (confirmed: SOF2 has no runtime physics-prop
    # system) - both fall back to a static placeholder, per your request.
    'prop_physics_multiplayer': 'misc_model',
    'prop_physics': 'misc_model',

    # func_illusionary (visible, non-solid brush - e.g. railings) has no
    # entity-level idTech3 equivalent with matching solidity semantics:
    # idTech3 solidity lives on the SHADER (surfaceparm nonsolid), not the
    # entity, unlike Source. func_static is the right STRUCTURAL match
    # (its own distinct brush model, not merged into worldspawn like
    # func_group would be) but stays solid by default - the applied
    # material also needs surfaceparm nonsolid added for this to actually
    # be walk-through, which isn't done automatically here (that's a
    # per-texture change, safe only if the texture isn't also used
    # elsewhere on brushes that should stay solid).
    'func_illusionary': 'func_static',

    # --- CONFIRMED: gameplay/utility entities with a direct SOF2 counterpart ---
    'func_ladder': 'trigger_ladder',
    'logic_relay': 'target_relay',
    'info_target': 'info_null',
    # path_track (Source's movement-path waypoint) -> path_corner, idTech3's
    # equivalent waypoint entity. Source's per-node speed/orientation keys
    # don't carry over 1:1 and any func_tracktrain-driving entity needs
    # manual rework, but the positional waypoints are preserved.
    'path_track': 'path_corner',
    # sky_camera has NO idTech3 equivalent, confirmed in-game ("misc_skyportal
    # doesn't have a spawn function" - that guess was wrong). This isn't
    # fixable by picking a different classname: id Tech 3's sky is a SHADER
    # property (skyParms on the skybox surface's material), not something
    # controlled by a placeable entity at all, so there's no entity-level
    # translation target regardless of naming. Handled (if at all) on the
    # material side by vmt2shader.py's sky-shader generation, not here -
    # this entity itself is skipped, see DEFAULT_SKIP_CLASSES below.

    # --- BEST-GUESS: no exact equivalent, mapped to the closest generic entity ---
    # func_brush does not exist in SOF2 (confirmed absent from mp_entities*.def).
    # func_wall is the closest generic "solid bmodel that just sits there"
    # entity, and supports a START_OFF flag if the brush needs to be toggled.
    'func_brush': 'func_wall',
    # func_clip_vphysics (physics-only collision brush) has no SOF2 entity
    # equivalent; folding it into a generic solid wall preserves collision
    # geometry, but ideally its faces should carry tools/_clip.
    'func_clip_vphysics': 'func_wall',
    # light_spot/light_environment both fold onto SOF2's single generic
    # "light" entity (which becomes a spotlight when it has a "target").
    # light_environment (Source's global sun light) has no idTech3 entity
    # equivalent at all - true sun lighting is set via a shader's q3map_sun
    # directive at compile time, not a placed entity - so this mapping only
    # preserves it as an ordinary point light, which is a real behavior loss.
    'light_spot': 'light',
    'light_environment': 'light',
    # CS team spawns -> SOF2's team-deathmatch red/blue spawns (confirmed to
    # exist: info_player_start_red/_blue). This is a more universal default
    # than a Siege-mode assumption; T/CT -> red/blue assignment is arbitrary,
    # swap via --class-map if your teams should go the other way, or point
    # these at info_player_siegeteam1/2 instead if you're building Siege.
    'info_player_terrorist': 'info_player_start_red',
    'info_player_counterterrorist': 'info_player_start_blue',
    # func_door_rotating is NOT remapped: SOF2 has it (mp_entities.def,
    # "func_door_rotating ... You need to have an origin brush"). Its hinge
    # is Source's origin key, which convert_entity turns into that origin
    # brush (PIVOT_BRUSH_CLASSES).
}

# Brush entities that move around a pivot. SOF2's .def requires an origin
# brush for these; Source expresses the same point as the "origin" key.
PIVOT_BRUSH_CLASSES = {'func_door_rotating', 'func_rotating', 'func_train',
                       'func_bobbing', 'func_pendulum'}

# Classnames that are purely Source-specific concepts (Source's entity I/O
# system, HDR/tonemap/fog controllers, rope rendering, CS-specific game
# modes, etc.) with no SOF2 equivalent at all - confirmed absent from
# mp_entities*.def. These are dropped entirely rather than left in the .map
# as unrecognized classnames. Override by giving an explicit --class-map
# entry for any of these (that takes priority over being skipped).
# TEMPORARY DIAGNOSTIC: when True (via --debug-terrain-corner-marker),
# every terrain displacement's alphamap is overridden to mark ONLY its
# (row=0, col=0) grid vertex - which is always Source's own
# "startposition" corner, a known, fixed physical point - as layer 1,
# with everything else forced to layer 0. Purpose: get one unambiguous,
# ground-truth data point for exactly how the compiler's alphamap
# sampling maps image rows/columns onto a displacement's real geometry,
# instead of continuing to guess transformations from real (multi-
# directional) blend data where more than one thing varies at once.
# Remove once that mapping is confirmed and the real fix is applied
# permanently instead of behind a flag.
DEBUG_TERRAIN_CORNER_MARKER = False

# Set once per conversion run (see scan_max_terrain_power / convert_vmf) to
# the highest displacement "power" found anywhere in the map. Every
# WorldVertexTransition displacement's own normals/distances/offsets/alphas
# grids then get upsampled to match this power in parse_dispinfo, before
# any geometry is built - confirmed necessary, not just a guess: two real,
# adjacent displacements of different power (8951 power 3, 9154 power 2)
# were shown, via actual Bezier-curve simulation matching how idTech3's
# patchDef2 really tessellates, to produce genuinely different edge shapes
# even from byte-identical shared control points - because a Bezier curve's
# shape between two shared endpoints depends on how many intermediate
# control points exist between them, which differs whenever adjacent
# patches have different power. Forcing every displacement in the map to
# the same power gives every edge the same segment structure as its
# neighbors, eliminating that divergence. None if not yet computed (or
# irrelevant, e.g. for --terrain-mode brush's flat triangles, which don't
# have this Bezier-curve concern at all, though upsampling is harmless
# there too and kept uniform for simplicity).
TERRAIN_TARGET_POWER = None

# Set once per conversion run (see scan_terrain_alpha_conflicts / convert_vmf)
# to a dict mapping (material_lowercase, rounded_x, rounded_y) -> reconciled
# alpha value (0-255, pre-threshold) for every position where two or more
# displacements of the SAME material genuinely disagree about the alpha
# value at what is otherwise the same physical, shared vertex. This is a
# real, confirmed situation in Source's own VMF data - not a bug in this
# converter - verified directly (e.g. displacement 8825's own dispinfo says
# alpha 255 at a point its neighbor 9035 independently stores as alpha 87,
# for the exact same world-space vertex). A single alphamap pixel can only
# hold one value, so at these positions every displacement sharing that
# vertex must be made to agree - "each side just uses its own data" produces
# a visible seam at genuinely conflicting boundaries no matter how correct
# the rest of the projection is. Reconciled via simple averaging of the raw
# (pre-threshold) alpha values at that position, then thresholded normally -
# not because averaging is necessarily the "right" answer for what the
# level designer intended, but because it's a neutral compromise that
# doesn't arbitrarily favor whichever displacement happens to be processed
# first, and confirmed to at least remove the stark, binary jump a
# thresholded disagreement produces. None if not yet computed.
TERRAIN_ALPHA_RECONCILIATION = None

# Set once per conversion run (see scan_terrain_alpha_feather / convert_vmf),
# only when --terrain-edge-feather is given (None/disabled otherwise), to a
# dict mapping (material_lowercase, rounded_x, rounded_y) -> blended alpha
# value for every displacement vertex that sits ONE STEP INWARD from an
# edge shared with another displacement of the same material. EXPERIMENTAL:
# real testing showed that even with the exact shared boundary vertex
# forced to agree (TERRAIN_ALPHA_RECONCILIATION above), a visible seam can
# remain, because each patch's own Bezier curve shape approaching that
# boundary also depends on its other, non-shared nearby control points -
# not something matching the boundary value alone fixes. This blends the
# one-step-inward value toward the neighbor's own equivalent, by however
# much --terrain-edge-feather asks for, to narrow that curve-shape
# divergence without merging patches into one entity. None if not
# computed (default) or explicitly disabled.
TERRAIN_ALPHA_FEATHER = None

# Set once per conversion run (convert_vmf) when --terrain-mode lwo is
# selected: a single LWOTerrainBuilder instance shared across every
# displacement in the map, so every displacement's grid welds into the same
# points/index space regardless of which brush/entity it came from. None
# when the lwo terrain mode isn't in use.
LWO_TERRAIN_BUILDER = None

# Set by main() to '<mapname>_terrain.lwo' so each map gets its own model file
# in models/mapobjects/terrain instead of all maps sharing one name.
TERRAIN_LWO_FILENAME = 'welded_terrain.lwo'

# Prefix for the bare shader names baked into the terrain and overlay LWOs
# ("<tag>terrain0", "<tag>overlay3"), set from the map name in main().
#
# These used to be just "terrain0", "overlay0"... in EVERY converted map. Two
# converted maps installed side by side then define the same shader names with
# different textures, and only one definition can win: q3map2 takes the first
# it parsed (ShaderInfoForShader -> find_first, in shaderlist order) and so
# does the engine. de_overpass_csgo compiled with de_dust2's and
# de_mirage_cs2's .shader files loaded ahead of its own - its terrain resolved
# to theirs. The release audit surfaced it: overpass's overlay shaders were
# resolving through another map's .shader file.
MAP_SHADER_TAG = ''


def map_shader_tag(map_name):
    """'<map>_' kept short enough that the longest name these become -
    models/mapobjects/terrain/<tag>terrain123_12_bsp, the form netradiant's
    q3map2 looks up - stays inside vmt2shader's 59-character limit, so it is
    never shortened (shortening would break the match with the LWO's own tag).
    Long map names keep 11 readable characters plus a 4-digit hash."""
    import hashlib
    tag = re.sub(r'[^a-z0-9_]+', '_', map_name.lower()).strip('_') or 'map'
    if len(tag) > 16:
        tag = tag[:11].rstrip('_') + '_' + hashlib.md5(tag.encode('utf-8')).hexdigest()[:4]
    return tag + '_'


def scan_max_terrain_power(node, best=None):
    """Recursively walk a parsed VDF node (dict of lists of dicts/strings,
    as returned by parse_vdf) looking for every 'dispinfo' block's own
    'power' value, returning the maximum found (or None if none exist
    anywhere in the map). Used once, early in convert_vmf, to decide the
    single target power every displacement gets upsampled to.
    """
    if isinstance(node, dict):
        if 'dispinfo' in node:
            for di in node['dispinfo']:
                if isinstance(di, dict):
                    try:
                        p = int(float(get1(di, 'power', '0')))
                        if best is None or p > best:
                            best = p
                    except (TypeError, ValueError):
                        pass
        for v in node.values():
            best = scan_max_terrain_power(v, best)
    elif isinstance(node, list):
        for item in node:
            best = scan_max_terrain_power(item, best)
    return best


def _find_all_displacement_sides(node, out):
    """Recursively collect every (side, sibling_sides) pair for a side that
    has its own dispinfo block - sibling_sides is that side's own solid's
    full side list, needed by get_face_polygon to reconstruct the quad."""
    if isinstance(node, dict):
        if 'side' in node:
            sides = node['side']
            for s in sides:
                if isinstance(s, dict) and get1(s, 'dispinfo'):
                    out.append((s, sides))
        for v in node.values():
            _find_all_displacement_sides(v, out)
    elif isinstance(node, list):
        for item in node:
            _find_all_displacement_sides(item, out)


def _collect_terrain_displacement_grids(root):
    """Shared helper for scan_terrain_alpha_conflicts and
    scan_terrain_alpha_feather: finds every displacement side in the map
    and returns {material: [(flat_pos_grid, alphas_grid, n), ...]} - each
    displacement's own flat (undisplaced), UNSCALED world-space vertex
    positions and raw (pre-threshold) alpha grid, grouped by material.
    """
    all_disps = []
    _find_all_displacement_sides(root, all_disps)

    by_material = {}
    for side, sides in all_disps:
        material = get1(side, 'material', '').strip().lower()
        dispinfo = get1(side, 'dispinfo')
        if not dispinfo:
            continue
        try:
            poly = get_face_polygon(side, sides)
            power, n, startpos, normals, distances, offsets, alphas = parse_dispinfo(dispinfo)
            quad = order_quad_from_startpos(poly, startpos)
            c00, c01, c11, c10 = quad
        except Exception:
            continue
        flat = []
        for r in range(n):
            t = r / (n - 1)
            row_start = lerp3(c00, c10, t)
            row_end = lerp3(c01, c11, t)
            row = []
            for c in range(n):
                s = c / (n - 1)
                row.append(lerp3(row_start, row_end, s))
            flat.append(row)
        by_material.setdefault(material, []).append((flat, alphas, n))
    return by_material


def scan_terrain_alpha_conflicts(root):
    """Build TERRAIN_ALPHA_RECONCILIATION's dict (see its own comment above)
    by finding every displacement side in the map, grouping by material,
    and detecting positions where two or more displacements of that same
    material store genuinely different alpha values for what amounts to
    the same physical vertex (matched by rounding each vertex's own FLAT,
    UNSCALED world position to 1 decimal place - well within floating-point
    noise between independently-computed but physically-identical points,
    confirmed against real data from displacements 8825/9035 sharing an
    edge). Returns the dict directly; does not mutate the module global
    itself (the caller, convert_vmf, does that) - kept a pure function so
    it stays independently testable.
    """
    by_material = _collect_terrain_displacement_grids(root)

    reconciliation = {}
    for material, disps in by_material.items():
        pos_to_values = {}
        for flat, alphas, n in disps:
            for r in range(n):
                for c in range(n):
                    key = (round(flat[r][c][0], 1), round(flat[r][c][1], 1))
                    pos_to_values.setdefault(key, []).append(alphas[r][c])
        for (x, y), vals in pos_to_values.items():
            if len(vals) < 2:
                continue
            layers = set(1 if a >= 128.0 else 0 for a in vals)
            if len(layers) > 1:
                reconciliation[(material, x, y)] = sum(vals) / len(vals)
    return reconciliation


def scan_terrain_alpha_feather(root, blend_weight):
    """Build TERRAIN_ALPHA_FEATHER's dict: for every displacement edge
    vertex that is shared with another displacement of the same material
    (matched the same way as scan_terrain_alpha_conflicts), also blend
    that vertex's own IMMEDIATE INWARD NEIGHBOR's raw alpha toward the
    average of every other displacement's own inward neighbor at that
    same shared boundary position - by blend_weight (0 = no change, 1 =
    fully replace with the group average).

    Why: confirmed via real in-game testing that reconciling only the
    EXACT shared boundary vertex (scan_terrain_alpha_conflicts) can still
    leave a visible seam, because each patch's own Bezier curve shape
    approaching that boundary is also shaped by its OTHER, non-shared
    nearby control points - the row/column just inside the edge. Blending
    that one-step-inward value toward its neighbor's own equivalent (deep
    enough to be well short of any far interior detail, and only ever
    partially, per blend_weight, so it never fully imitates the neighbor's
    own painted intent) narrows that divergence without a full merge.

    Corner vertices (on two edges at once - e.g. where three or more
    displacements meet, like a real T-junction found in this map) are
    skipped entirely: "inward" is ambiguous there, and forcing one
    direction risked making an already-complex junction worse rather
    than better. Returns the dict directly (material, x, y) -> feathered
    alpha value, keyed by the INWARD vertex's own position (which is NOT
    shared between displacements, unlike the boundary vertex itself) - so
    each displacement's own inward vertex gets its own, unique entry.
    """
    by_material = _collect_terrain_displacement_grids(root)

    feather = {}
    for material, disps in by_material.items():
        # First pass: group (boundary position) -> list of each
        # displacement's own inward-neighbor raw alpha value.
        boundary_to_inward_vals = {}
        # Also remember, per displacement, each of its own boundary
        # positions and where its inward neighbor actually is, so the
        # second pass doesn't need to redo this work.
        per_disp_edges = []
        for disp_idx, (flat, alphas, n) in enumerate(disps):
            edges_here = []
            for r in range(n):
                for c in range(n):
                    on_r_edge = r == 0 or r == n - 1
                    on_c_edge = c == 0 or c == n - 1
                    if on_r_edge and on_c_edge:
                        continue  # corner - skip, ambiguous inward direction
                    if not (on_r_edge or on_c_edge):
                        continue  # interior vertex - not a boundary at all
                    if on_r_edge:
                        ir = 1 if r == 0 else n - 2
                        ic = c
                    else:
                        ir = r
                        ic = 1 if c == 0 else n - 2
                    boundary_key = (round(flat[r][c][0], 1), round(flat[r][c][1], 1))
                    inward_alpha = alphas[ir][ic]
                    boundary_to_inward_vals.setdefault(boundary_key, []).append(inward_alpha)
                    edges_here.append((flat[ir][ic], inward_alpha, boundary_key))
            per_disp_edges.append(edges_here)

        # Second pass: for every boundary position shared by 2+
        # displacements, blend each displacement's own inward value
        # toward the group's average (excluding nothing - even an
        # already-matching pair still gets pulled toward the shared
        # average, since the goal is curve-shape agreement, not just
        # fixing outright conflicts).
        for edges_here in per_disp_edges:
            for (inward_x, inward_y, inward_z), own_alpha, boundary_key in edges_here:
                vals = boundary_to_inward_vals.get(boundary_key)
                if not vals or len(vals) < 2:
                    continue
                group_avg = sum(vals) / len(vals)
                blended = (1.0 - blend_weight) * own_alpha + blend_weight * group_avg
                inward_key = (material, round(inward_x, 1), round(inward_y, 1))
                feather[inward_key] = blended
    return feather

DEFAULT_SKIP_CLASSES = {
    # Source rendering/lighting concepts with zero SOF2 equivalent (SOF2 has
    # NO env_* entities at all - confirmed absent as a whole category):
    # (env_soundscape and ambient_generic are turned into target_speakers by
    # build_sound_speakers, not converted entity by entity)
    'ambient_generic',
    'env_fog_controller', 'env_lightglow', 'env_soundscape',
    'env_soundscape_proxy', 'env_sprite', 'env_tonemap_controller',
    'env_cubemap', 'color_correction', 'shadow_control',
    # Source's entity I/O system and its helpers - no idTech3 equivalent:
    'logic_auto', 'point_clientcommand', 'point_template',
    'filter_activator_class', 'filter_activator_team', 'filter_multi',
    # CS-specific game-mode entities (bomb defusal, buy zones, C4, weapon
    # spawn rules) - SOF2 has no bomb-defusal gametype:
    'func_bomb_target', 'func_buyzone', 'weapon_c4', 'game_weapon_manager',
    # CS:GO/Source rendering-optimization or bot-AI helpers with no SOF2
    # analog (idTech3 handles visibility/PVS itself; SOF2 has no CS-style
    # nav-mesh bot system):
    'func_nav_blocker', 'func_occluder', 'func_dustmotes',
    # water_lod_control: Source's water-shader LOD distances. Point-only,
    # nothing to draw or collide. func_areaportalwindow: a Source areaportal
    # that fades by distance; inside an entity q3map2 rejects its brush
    # ("areaportals only allowed in world") and SOF2 has no spawn function
    # for it, and moving it into the world would make a portal nothing ever
    # opens - so it goes.
    'water_lod_control', 'func_areaportalwindow',
    # Purely cosmetic/authoring helpers with no functional target:
    'info_lighting', 'info_map_parameters', 'info_overlay',
    'keyframe_rope', 'move_rope', 'point_viewcontrol',
    # idTech3 has NO decal entity at all, regardless of which game sits on
    # top of it - decals are done via specially-tagged shaders applied to
    # small brushes, not a placeable point/brush entity. This isn't a
    # "SOF2 happens to lack it" gap like the others above, it's a
    # structural difference between the two engines' decal systems, so
    # there's no sensible entity-level translation target to map this to
    # regardless of what any .def file says.
    'infodecal',
    # Confirmed in-game: "misc_skyportal doesn't have a spawn function" -
    # see the comment where this used to be in DEFAULT_CLASS_MAP for why
    # there's no entity-level equivalent to map this to instead.
    'sky_camera',
    # func_areaportal's brushes are NOT solid and NOT rendered in Source -
    # purely a vis-culling hint the engine never draws or collides with.
    # An earlier version of this pipeline folded it into worldspawn as
    # ordinary solid geometry (first as textures/tools/_areaportal, later
    # as solid caulk after that first attempt caused a permanently-closed-
    # looking void) - but EITHER way, giving it real collision is wrong:
    # confirmed by a real map where an areaportal brush spanning a doorway
    # overlapped a walkable floor area (completely normal in Source, since
    # areaportals never block movement there), which then became solid,
    # incorrectly-blocking geometry in SOF2. Since SOF2's own areaportal
    # entity requirements are unresolved and dropping the brush outright
    # exactly matches its real, non-solid, invisible behavior in the
    # original map, skipping it entirely (no geometry at all) is the
    # correct, safe choice - not a stopgap for something better.
    'func_areaportal',
}

# Prefix-based fallback for the "SOF2 has NO env_* entities at all" policy
# above: the enumerated set only lists the specific env_ classnames a VMF
# commonly contains, but ANY env_-prefixed entity is covered by the same
# confirmed absence - this catches ones not explicitly listed (e.g.
# env_smokestack) instead of letting them slip through unrecognized.
SKIP_CLASS_PREFIXES = ('env_',)


def should_skip_class(classname, skip_classes):
    if classname in skip_classes:
        return True
    lc = classname.lower()
    return any(lc.startswith(p) for p in SKIP_CLASS_PREFIXES)

# Brush entity classnames whose geometry should be folded straight into
# worldspawn (same treatment as func_detail) because SOF2 needs no wrapping
# entity for them. func_areaportal is NOT here - see DEFAULT_SKIP_CLASSES
# above for why it's dropped entirely instead.
# Source buttons work only through entity I/O ("OnPressed"/"OnDamaged" ->
# some other entity's input), which this converter does not carry over, so in
# SOF2 every one of them is a button wired to nothing. An invisible one is
# dropped; a drawn one keeps its brushes as world detail (see the entity loop
# in convert_vmf).
SOURCE_IO_BUTTON_CLASSES = {'func_button', 'func_rot_button', 'momentary_rot_button'}


def _has_drawn_side(entity):
    """True if any side of the entity's brushes has a non-tools material."""
    for solid in get_solid_blocks(entity):
        for sd in solid.get('side', []):
            if not isinstance(sd, dict):
                continue
            mat = (get1(sd, 'material', '') or '').strip().lower().replace('\\', '/')
            if mat and not mat.startswith('tools/'):
                return True
    return False


FOLD_INTO_WORLDSPAWN_CLASSES = {
    'func_detail',
    # SOF2 MP has no shatter-on-damage behavior for this anyway, so folding
    # it into permanent solid world geometry (same treatment as func_detail)
    # is a safe simplification - it just never breaks.
    'func_breakable',
}


def convert_model_path(value):
    if re.search(r'\.mdl$', value, re.IGNORECASE):
        return re.sub(r'\.mdl$', '.md3', value, flags=re.IGNORECASE).replace('\\', '/')
    return value


_ORIGIN_RE = re.compile(r'^\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*$')


def convert_origin_value(value, scale, offset):
    m = _ORIGIN_RE.match(value)
    if not m:
        return value
    ox, oy, oz = offset
    x = float(m.group(1)) * scale - ox
    y = float(m.group(2)) * scale - oy
    z = float(m.group(3)) * scale - oz
    return "{:.10g} {:.10g} {:.10g}".format(x, y, z)


# {model path suffix (lowercase, '/', .md3): (x, y, z)} from mdl_to_md3.py's
# md3_origin_offsets.json. See _apply_model_origin_offset.
MODEL_ORIGIN_OFFSETS = {}


def _source_angle_matrix(pitch, yaw, roll):
    """Source SDK mathlib AngleMatrix (degrees). Columns = forward, left, up;
    world = M * local. q3map2's misc_model rotation matches it, which is why
    props already rotate correctly when angles pass through unchanged."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    return ((cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
            (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
            (-sp, sr * cp, cr * cp))


def _apply_model_origin_offset(lines):
    """MD3 stores vertices as int16 in 1/64 units, so nothing can be more than
    512 units from the model origin; mdl_to_md3.py moves larger models' geometry
    back into range and records by how much. Put the entity's origin at
    origin + R(angles) * offset * modelscale so the geometry lands exactly
    where Source had it (e.g. palm_tree_trunk, 536 units tall, had its bottom
    24 units clamped flat and floated)."""
    if not MODEL_ORIGIN_OFFSETS:
        return lines
    kv = {}
    for i, ln in enumerate(lines):
        parts = ln.split('"')
        if len(parts) >= 5:
            kv[parts[1].lower()] = (i, parts[3])
    if 'model' not in kv or 'origin' not in kv:
        return lines
    model = kv['model'][1].strip().lower().replace('\\', '/')
    off = None
    for key, val in MODEL_ORIGIN_OFFSETS.items():
        if model == key or model.endswith('/' + key):
            off = val
            break
    if not off or not any(off):
        return lines
    try:
        ox, oy, oz = (float(c) for c in kv['origin'][1].split())
        pitch = yaw = roll = 0.0
        if 'angles' in kv:
            pitch, yaw, roll = (float(c) for c in kv['angles'][1].split())
        elif 'angle' in kv:
            yaw = float(kv['angle'][1])
        s = float(kv['modelscale'][1]) if 'modelscale' in kv else 1.0
    except ValueError:
        return lines
    m = _source_angle_matrix(pitch, yaw, roll)
    local = (off[0] * s, off[1] * s, off[2] * s)
    world = tuple(sum(m[r][c] * local[c] for c in range(3)) for r in range(3))
    i = kv['origin'][0]
    lines = list(lines)
    lines[i] = '"{}" "{:.10g} {:.10g} {:.10g}"'.format(
        lines[i].split('"')[1], ox + world[0], oy + world[1], oz + world[2])
    return lines


def format_entity_keys(node, scale, offset, model_scale=1.0, apply_model_scale=False,
                        out_classname=None):
    lines = []
    has_modelscale = False
    drop_spawnflags = out_classname in SPAWNFLAG_UNSAFE_CLASSES
    # model_static does NOT honor the ordinary "modelscale" entity key at
    # runtime the way misc_model does (confirmed via a real in-game test:
    # props rendered at the wrong, unscaled size) - so instead of emitting
    # that key, point it at the pre-scaled "<name>_static.md3" variant
    # mdl_to_md3.py writes when given --model-scale, which has the scale
    # baked directly into its geometry.
    is_static_model = out_classname == 'model_static'
    for key, values in node.items():
        if key in SKIP_KEYS_FOR_ENTITY:
            continue
        lk = key.lower()
        if lk in SOURCE_ONLY_ENTITY_KEYS:
            continue
        if lk == 'spawnflags' and drop_spawnflags:
            continue
        if lk == 'modelscale' and is_static_model:
            continue  # meaningless for model_static; scale is baked into the _static file
        for v in values:
            if isinstance(v, dict):
                continue  # nested blocks other than solid/editor/connections: ignore
            if key.lower() in MODEL_VALUE_KEYS:
                v = convert_model_path(v)
                if is_static_model:
                    v = re.sub(r'\.md3$', '_static.md3', v, flags=re.IGNORECASE)
            elif key.lower() == 'origin':
                v = convert_origin_value(v, scale, offset)
            elif key.lower() == 'modelscale':
                has_modelscale = True
                if apply_model_scale:
                    try:
                        v = '{:.6g}'.format(float(v) * model_scale)
                    except ValueError:
                        pass
            lines.append('"{}" "{}"'.format(key, v))
    if apply_model_scale and not has_modelscale and not is_static_model:
        lines.append('"modelscale" "{:.6g}"'.format(model_scale))
    return _apply_model_origin_offset(lines)


# Worldspawn keys idTech3/SOF2 actually reads. Everything else a decompiled
# Source map carries on worldspawn is dead weight: q3map2 never looks at it and
# the engine never looks at it, so it only makes the entity lump harder to read
# and invites the assumption that it is doing something.
#
# Checked against q3map2's own source rather than documentation, which has
# already been wrong once in this project. Confirmed inert rather than
# dangerous: "light" on worldspawn does NOT create a stray lamp at the origin,
# because CreateEntityLights guards on Q_strncasecmp( classname, "light", 5 )
# and worldspawn fails that test.
#
# Source's "MaxRange" is the one with a real counterpart - it is Source's far
# clip, and Raven's engine spells that "distancecull" - so it is converted
# rather than dropped (see convert_world).
# --- EXPOSURE: Source's lighting pipeline, reproduced in q3map2 --------------
#
# Every number below is taken from the code of the tools involved, not tuned.
#
# SOURCE (source-sdk-2013, src/utils/vrad + src/mathlib):
#   * lightmap.cpp LightForString:  linear = pow(c/255, 2.2) * brightness
#     "// convert to linear". VRAD lights in LINEAR space.
#   * color_conversion.cpp BuildGammaTable: the engine shows a lightmap as
#     pow(linear, 1/gamma), gamma = 2.2 by default. Linear light, gamma display.
#   * GatherSampleAmbientSkySSE: _ambient is a SKY DOME - hemisphere samples,
#     each traced for sky visibility and cosine-weighted, normalised so a fully
#     open floor receives exactly _ambient. An open wall gets ~half (its lower
#     hemisphere faces the ground), a covered ceiling gets none.
#   * ParseLightGeneric: a point light's stated brightness is its value at 100
#     units ("// scale intensity for unit 100 distance").
#
# Q3MAP2 (both GtkRadiant 1.6's and netradiant-custom's, checked in source):
#   * ColorToBytes: byte = 255 * pow(value/255, 1/lightmapGamma). With SOF2's
#     game default gamma 1.0 (game_sof2.h) q3map2 writes LINEAR light straight
#     to the lightmap and the engine displays it as-is - so compiling the same
#     physical light gives far darker shadows than Source shows. -gamma 2.2
#     reproduces Source's pow(linear, 1/2.2) exactly; both compilers implement
#     it the same way and both apply it to the light grid too.
#   * q3map_skylight <value> <iterations>: a dome of occluded directional suns
#     - the same estimator as Source's sky ambient. Its response on an open
#     floor is computed below from q3map2's own dome construction.
#   * point lights: light * 7500 / d^2 (pointScale).
#
# So the conversion is: take Source's light in linear units, scale every light
# by ONE exposure factor k, and let -gamma 2.2 do what Source's display does.
# k is the only choice, fixed by TARGET_SUNLIT_LEVEL - what a sun-facing floor
# in the open should read on screen. Everything else keeps the ratios Source
# authored.
#
# !! The map MUST be lit with -gamma 2.2. Without it every value here is too
# !! dark in the shadows, by design.
SOURCE_GAMMA = 2.2   # Source's display gamma (BuildGammaTable); the COMPILE gamma is derived below

# On screen, 0-255. 235 leaves headroom under the clamp. SOF2 has no HDR and
# (as far as can be measured) no overbright, so Source's sunlit floor - which
# its LDR overbright lets go past 1.0 - has to land here instead.
TARGET_SUNLIT_LEVEL = 235.0

# q3map_skylight iterations. Source's estimator gives an open wall exactly half
# an open floor; q3map2's dome gives 0.479 at 3, 0.450 at 4, 0.407 at 6 (its
# rings are spaced evenly in elevation, not solid angle). 4 is within 10% of
# Source at 37 suns; 3 is closer on paper but its 17 suns show as visible
# stepped shadow copies.
SKYLIGHT_ITERATIONS = 4

# Source's radiosity (VRAD bounces until converged) is what lights a room the
# sky cannot see; q3map2 can't reproduce that without -bounce, so a flat
# worldspawn ambient stands in for it. MEASURED, not estimated: in CS:S's own
# compiled de_dust2.bsp, downward-facing surfaces - which can only be reached by
# bounce - have a median on-screen level of 0.293, i.e. linear 0.0672, against
# an _ambient luminance of 1.539 linear. Bounce = 0.0437 x _ambient.
# (The first version of this estimated 0.175 - four times too bright - which is
# why test D's ceilings sat at 0.31 of sunlit where CS:S has 0.17.)
# Measured again with -bounce 1 (same CS:S reference, position-matched luxels
# between a bounced and an unbounced bake): with one real bounce doing part of
# the work, the fill that best matches CS:S's walls and ceilings is 0.0130 x
# _ambient - walls then land at 0.20/0.52/0.87/1.12/1.25 (p10..p90) against
# CS:S's 0.21/0.49/0.81/1.05/1.20. Ceilings stay a little dark (median 0.20 vs
# 0.29): one bounce barely reaches them, VRAD bounces until converged.
COMPILE_BOUNCES = 1
BOUNCE_FILL = 0.0130 if COMPILE_BOUNCES >= 1 else 0.0437

# How SOF2 displays a lightmap byte - MEASURED, from the game's own cvars:
# r_mapOverBrightBits 1, r_overBrightBits 0. SOF2's renderer is Quake 3's here:
# R_ColorShiftLightingBytes shifts every lightmap, vertex-colour and light-grid
# byte left by (mapOverBrightBits - overBrightBits) = 1 AT LOAD TIME and clamps
# to 255 with colour normalisation. So on screen:
#     multiplier = min(1.0, 2 * byte/255)
# Bytes above 128 are all full brightness. Test E stored its open ground at
# 0.6-0.9 and every one of those came out clamped at 1.0 - sunlit and shaded
# sand identical, which is why the courtyard read flat.
# (A player running fullscreen with r_overBrightBits 1 gets shift 0 plus a x2
# hardware gamma instead - the same factor of 2, so the same result as long as
# nothing is stored above 128, which this model guarantees.)
ENGINE_LIGHTMAP_OVERBRIGHT = 2

# What the SHADERS add on top. Clamping at load caps the lightmap at 1.0x the
# texture - but CS:S's dust2 needs 1.75x, and unlike SOF2, Source clamps the
# FINAL PIXEL, not the lightmap. Measured from its .bsp (texture reflectivity x
# lightmap): only 0.3% of CS:S's luxels clip to white, so its sunlit sand really
# is ~0.97 on screen, where a 1.0x-capped lightmap can never take that texture
# past 0.58. No compile setting can close that gap; the shader can. The lightmap
# stage blends with GL_DST_COLOR GL_SRC_COLOR - dst*src + src*dst = 2 * texture
# * lightmap, clamped on the final pixel exactly where Source clamps (the
# standard Quake 3 "2x modulate", as id's own detail-texture stages use).
# Vertex-lit surfaces get the same 2x by repeating their stage additively.
# Lightmap range on screen becomes 0-2.0x; dust2's 1.75 fits, so the
# conversion is exact 1:1 at Source's own gamma.
SHADER_LIGHTMAP_SCALE = 2

# q3map2 pointScale, and Source's 100-unit reference distance.
Q3_POINT_SCALE = 7500.0
SOURCE_POINT_REFERENCE = 100.0


def _source_linear(rgb, brightness):
    """VRAD's LightForString: per-channel linear light."""
    return [((max(c, 0.0) / 255.0) ** 2.2) * brightness for c in rgb]


def _luminance(v):
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def skylight_floor_response(iterations=SKYLIGHT_ITERATIONS):
    """What an open, up-facing surface receives from q3map_skylight <value>,
    per unit of value - built the way CreateSkyLights builds its dome (same in
    GtkRadiant and netradiant-custom for the default 0-90 horizon)."""
    step = 90.0 / iterations
    angle_steps = (iterations - 1) * 4
    total = 0.0
    count = 0
    for i in range(iterations - 1):
        e = math.radians(step * (0.5 + i))
        total += angle_steps * math.sin(e)
        count += angle_steps
    total += 1.0      # the zenith sun
    count += 1
    return total / count


def solve_exposure(sun_rgb, sun_brightness, amb_rgb, amb_brightness, elevation):
    """Return the q3map2 values - and the -gamma to compile with - that put this
    light_environment on SOF2's screen as close to CS:S as SOF2 can show.

    Source shows a lightmap as M = pow(linear, 1/2.2), and on de_dust2 M reaches
    1.67 on sunlit sand. SOF2 shows min(1.0, 2*byte/255): nothing above 1.0. So
    the range has to be compressed, and the curve is fixed by two measured
    anchors instead of by eye:

      1. the sunlit open floor lands on TARGET_SUNLIT_LEVEL - the top of what
         SOF2 can show, with a little headroom under the clamp;
      2. a room lit only by bounce light stays EXACTLY as bright as in CS:S
         (0.293 on screen, measured from its compiled .bsp).

    One power curve through both, screen = c * M^p, spreads everything else.
    A power curve on screen is simply a different compile gamma: with
    -gamma G, q3map2 stores 255 * pow(k*L/255, 1/G), so G = 2.2/p. If Source's
    own sunlit level already fits, p = 1, G = 2.2 and the match is exact.
    """
    S = _source_linear(sun_rgb, sun_brightness)
    A = _source_linear(amb_rgb, amb_brightness)
    incidence = max(math.sin(math.radians(max(0.0, min(90.0, elevation)))), 0.30)
    floor_src = (_luminance(S) * incidence + _luminance(A) * (1.0 + BOUNCE_FILL)) / 255.0
    bounce_src = max(_luminance(A) * BOUNCE_FILL / 255.0, 1e-6)
    m_sun = floor_src ** (1.0 / SOURCE_GAMMA)
    m_bounce = bounce_src ** (1.0 / SOURCE_GAMMA)
    # the lightmap is clamped at 1.0 after the engine's doubling, then scaled by
    # the shader: SHADER_LIGHTMAP_SCALE is the most a surface can show
    max_m = float(SHADER_LIGHTMAP_SCALE)
    target = TARGET_SUNLIT_LEVEL / 255.0 * max_m
    if m_sun <= target or m_sun <= m_bounce * 1.01 or target <= m_bounce:
        p = 1.0
        c = 1.0 if m_sun <= target else target / m_sun
    else:
        p = math.log(target / m_bounce) / math.log(m_sun / m_bounce)
        p = min(1.0, max(0.35, p))
        c = target / (m_sun ** p)
    gamma = SOURCE_GAMMA / p
    O = float(ENGINE_LIGHTMAP_OVERBRIGHT * SHADER_LIGHTMAP_SCALE)
    k = (c / O) ** gamma           # q3map2 value = k * Source linear (255 units)
    s_peak = max(S + [1e-9])
    a_peak = max(A + [1e-9])
    screen = lambda lin255: min(max_m, c * (max(lin255, 0.0) / 255.0) ** (p / SOURCE_GAMMA))
    return {
        'k': k,
        'compile_gamma': round(gamma, 2),
        # the light grid lights players and model_static props with SOF2's normal
        # 1x shaders, so it must not carry the shaders' 2x: scaling it up by
        # SHADER^gamma before gamma (q3map2 -gridscale) puts it back on 1x.
        'grid_scale': round(float(SHADER_LIGHTMAP_SCALE) ** gamma, 2),
        'curve_p': p,
        'exact': p >= 0.999 and c >= 0.999,
        'sun_intensity': k * s_peak,
        'sun_color': [x / s_peak for x in S],
        'skylight_value': k * a_peak / skylight_floor_response(),
        'sky_color': [x / a_peak for x in A],
        'bounce_fill': k * a_peak * BOUNCE_FILL,
        # on-screen multipliers of the texture - same scale as the CS:S measurement
        'predicted': {
            'sunlit floor': screen(_luminance(S) * incidence + _luminance(A) * (1 + BOUNCE_FILL)),
            'open floor in shadow': screen(_luminance(A) * (1 + BOUNCE_FILL)),
            'open wall in shadow': screen(_luminance(A) * (0.5 + BOUNCE_FILL)),
            'covered room': screen(_luminance(A) * BOUNCE_FILL),
        },
        'source': {
            'sunlit floor': m_sun,
            'open floor in shadow': (_luminance(A) * (1 + BOUNCE_FILL) / 255.0) ** (1 / SOURCE_GAMMA),
            'open wall in shadow': (_luminance(A) * (0.5 + BOUNCE_FILL) / 255.0) ** (1 / SOURCE_GAMMA),
            'covered room': m_bounce,
        },
    }


def read_light_environment(root):
    """(sun_rgb, sun_brightness, amb_rgb, amb_brightness, elevation) from the
    VMF's first light_environment, or None."""
    for ent in root.get('entity', []):
        if not isinstance(ent, dict):
            continue
        if (get1(ent, 'classname') or '').lower() != 'light_environment':
            continue
        try:
            light = [float(x) for x in (get1(ent, '_light') or '').split()]
            amb = [float(x) for x in (get1(ent, '_ambient') or '').split()]
            pitch = float(get1(ent, 'pitch') or 0.0)
        except ValueError:
            continue
        if len(light) < 4:
            continue
        if len(amb) < 4:
            # VRAD's own fallback when _ambient is missing: half the sun.
            amb = light[0:3] + [light[3] * 0.5]
        return light[0:3], light[3], amb[0:3], amb[3], -pitch
    return None


# Set once per conversion from the map's light_environment, then used for every
# point light so that lamps sit at the same exposure as the sun they were
# authored against. The fallback is de_dust2's own k, for maps with no sun.
EXPOSURE_K = [0.02]
COMPILE_GAMMA = [SOURCE_GAMMA]
MAP_SCALE = [1.0]


def derive_ambient_from_light_environment(root, warnings, scale=1.0):
    """Solve the exposure, remember k for the point lights, and return the
    worldspawn (ambient, _color) that stands in for Source's bounce light."""
    MAP_SCALE[0] = scale
    le = read_light_environment(root)
    if not le:
        return None, None
    sol = solve_exposure(*le)
    EXPOSURE_K[0] = sol['k']
    color_str = "{:.4g} {:.4g} {:.4g}".format(*sol['sky_color'])
    p = sol['predicted']
    src = sol['source']
    warnings.append(
        "Exposure solved from light_environment (_light {} brightness {:g}, _ambient {} "
        "brightness {:g}, sun elevation {:g}). SOF2 shows lightmaps at min(1.0, {:g} x byte/255) "
        "x the 2x lightmap blend in the shaders; CS:S goes up to {:.2f}. Where that fits the "
        "match is exact; otherwise the range is compressed on a curve anchored at sunlit "
        "floor -> {:.2f} and bounce-lit room -> {:.2f} (exactly CS:S). That curve IS the compile gamma. COMPILE: -bsp ... -rename, "
        "-light ... -gamma {:g} -gridscale {:g} -bounce 1 (-gridscale is netradiant-only). Predicted on screen (x texture), SOF2 vs CS:S: sunlit floor {:.2f} vs "
        "{:.2f}, open shade {:.2f} vs {:.2f}, shaded wall {:.2f} vs {:.2f}, roofed room {:.2f} "
        "vs {:.2f}. k = {:.5f}; worldspawn 'ambient' {:.3g} stands in for Source's bounce light, "
        "sky fill is q3map_skyLight on the sky shader.".format(
            " ".join("{:g}".format(c) for c in le[0]), le[1],
            " ".join("{:g}".format(c) for c in le[2]), le[3], le[4],
            ENGINE_LIGHTMAP_OVERBRIGHT, src['sunlit floor'], p['sunlit floor'], p['covered room'],
            sol['compile_gamma'], sol['grid_scale'],
            p['sunlit floor'], src['sunlit floor'], p['open floor in shadow'],
            src['open floor in shadow'], p['open wall in shadow'], src['open wall in shadow'],
            p['covered room'], src['covered room'], sol['k'], sol['bounce_fill']))
    COMPILE_GAMMA[0] = sol['compile_gamma']
    return "{:.3g}".format(sol['bounce_fill']), color_str


_SPOT_COUNTER = [0]


def _source_light_normal(entity):
    """VRAD SetupLightNormalFromProps: 'angle' -1/-2 = straight up/down,
    otherwise yaw from 'angle' or angles[1]; pitch from 'pitch' or angles[0],
    with Source's light convention that NEGATIVE pitch points DOWN
    (z = sin(pitch))."""
    def f(key):
        try:
            return float(get1(entity, key) or 0.0)
        except ValueError:
            return 0.0
    ang = [0.0, 0.0, 0.0]
    raw = (get1(entity, 'angles') or '').split()
    if len(raw) == 3:
        try:
            ang = [float(x) for x in raw]
        except ValueError:
            pass
    angle, pitch = f('angle'), f('pitch')
    if angle == -1:
        n = [0.0, 0.0, 1.0]
    elif angle == -2:
        n = [0.0, 0.0, -1.0]
    else:
        yaw = angle or ang[1]
        n = [math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0]
    pitch = pitch or ang[0]
    n[2] = math.sin(math.radians(pitch))
    n[0] *= math.cos(math.radians(pitch))
    n[1] *= math.cos(math.radians(pitch))
    return n


def _first_face_along(origin, direction, max_dist=4096.0):
    """Distance (Source units) from origin along direction to the first brush
    face that faces back at it, or None."""
    best = None
    for poly, _side, n in ALL_FACE_INDEX:
        den = dot(n, direction)
        if den >= -1e-6:
            continue
        t = (dot(n, poly[0]) - dot(n, origin)) / den
        if t <= 0.5 or t > max_dist or (best is not None and t >= best):
            continue
        hitp = add(origin, scale_vec(direction, t))
        if point_in_polygon_3d(hitp, poly, n):
            best = t
    return best


def source_light_falloff_and_spot(entity, quad_value, scale, offset):
    """The q3map2 keys for a Source light's distance falloff, plus (for a
    light_spot) the info_null its cone aims at. Returns (key lines, extra
    entity blocks).

    VRAD (lightmap.cpp ParseLightGeneric/SetLightFalloffParams) lights with
        I(d) = I100 * ratio / (c + l*d + q*d^2),  ratio = c + 100 l + 10000 q
    and if all three are zero it uses c = 1 - NO falloff at all. q3map2 has
    only 1/d^2 and "linear" (I = 0.9375*light - fade*d). quad_value is the
    1/d^2 'light' that matches Source at 100 Source units, so I100 in q3map2
    units is 7500 * quad_value / r^2 with r = 100 * scale. Then:
      - any quadratic term: plain q3map2 light (exact at 100 units, and for
        d >> 1 the c and l terms vanish anyway);
      - linear only (1/d): q3map2 linear light through I100 at 100 units and
        I100/4 at 400 - on 1/d at both, zero at 500;
      - constant only: q3map2 linear light with a fade so small it only
        loses ~12% over 1000 units. On de_dust2 these are the six dim
        (brightness 10) fill lights in the B tunnels, which Source lights
        everywhere they can see; as 1/d^2 lights they barely reached the
        floor.
    Linear q3map2 lights lose angle attenuation unless _anglescale is set;
    Source keeps it, so _anglescale 1 goes back on.

    light_spot: q3map2 makes a spotlight from a light with a "target" (the
    info_null placed along Source's direction) and a cone of
    atan((radius + 16) / distance), with a 32-unit soft rim. The cone is
    Source's half-intensity angle - halfway between _inner_cone and _cone in
    cosine, where VRAD's fringe ramp is at 0.5."""
    def fk(key, default=0.0):
        try:
            v = get1(entity, key)
            return float(v) if v not in (None, '') else default
        except ValueError:
            return default
    c, l, q = fk('_constant_attn'), fk('_linear_attn'), fk('_quadratic_attn')
    c, l, q = [0.0 if v < 1e-4 else v for v in (c, l, q)]
    if c == 0 and l == 0 and q == 0:
        c = 1.0
    r = SOURCE_POINT_REFERENCE * scale
    i100 = Q3_POINT_SCALE * quad_value / (r * r)
    out = []
    classname = (get1(entity, 'classname') or '').lower()
    is_spot = classname == 'light_spot'
    if q > 0 or is_spot:
        out.append('"light" "{:.4g}"'.format(max(0.1, quad_value)))
    elif l > 0:
        fade = i100 / (4.0 * r)
        out.append('"light" "{:.4g}"'.format(max(0.1, 1.25 * i100 / 0.9375)))
        out.append('"spawnflags" "1"')
        out.append('"fade" "{:.6g}"'.format(fade))
        out.append('"_anglescale" "1"')
    else:
        out.append('"light" "{:.4g}"'.format(max(0.1, i100 / 0.9375)))
        out.append('"spawnflags" "1"')
        out.append('"fade" "{:.6g}"'.format(i100 / 8000.0))
        out.append('"_anglescale" "1"')

    blocks = []
    if is_spot and not get1(entity, 'target'):
        inner = fk('_inner_cone', 10.0) or 10.0
        outer = fk('_cone', inner) or inner
        outer = max(outer, inner)
        if not (inner >= 180 and outer >= 180):
            outer = min(outer, 89.0)
            n = _source_light_normal(entity)
            try:
                o_src = tuple(float(v) for v in get1(entity, 'origin').split())
            except (AttributeError, ValueError):
                return out, blocks
            # The cone edge is Source's OUTER cone, so the pool ends where
            # Source's does. q3map2 then fades the last 32 units linearly
            # (light.cpp: radiusAtDist - 32), which on a narrow spot is wider
            # than the whole pool: dust2's 5-degree lamps land a ~29-unit pool
            # on the floor, so their centre got only 29/32 of the light - and
            # at the old half-intensity angle, 21/32. Where the beam actually
            # lands (first face along the axis) the intensity is scaled by
            # 32 / pool radius, so the centre of the pool is at Source's value.
            hit = _first_face_along(o_src, n)
            d_map = (hit * scale) if hit else 256.0
            pool = math.tan(math.radians(outer)) * d_map
            if pool < 32.0:
                boost = 32.0 / max(pool, 1.0)
                out[0] = '"light" "{:.4g}"'.format(max(0.1, quad_value) * boost)
            # Target 32 units down the axis - close enough that it never ends
            # up under the floor ("Entity in solid"). radiusByDist =
            # (radius + 16) / dist, so radius may go negative for a narrow
            # cone; only exactly 0 would make q3map2 fall back to 64.
            dist = 32.0
            radius = math.tan(math.radians(outer)) * dist - 16.0
            if abs(radius) < 1e-3:
                radius = 1e-3
            _SPOT_COUNTER[0] += 1
            tname = 'spot{}'.format(_SPOT_COUNTER[0])
            ox, oy, oz = o_src
            tx, ty, tz = (ox * scale - offset[0] + n[0] * dist,
                          oy * scale - offset[1] + n[1] * dist,
                          oz * scale - offset[2] + n[2] * dist)
            out.append('"target" "{}"'.format(tname))
            out.append('"radius" "{:.4g}"'.format(radius))
            blocks.append('{{\n"classname" "info_null"\n"targetname" "{}"\n'
                          '"origin" "{:.10g} {:.10g} {:.10g}"\n}}'.format(tname, tx, ty, tz))
    return out, blocks


def point_light_value(rgb, brightness):
    """Source point light -> (q3map2 'light', '_color').

    Source's brightness is its value at 100 Source units; q3map2 gives
    light * 7500 / d^2. Matching them at the same physical distance - 100
    Source units is 100 * scale map units - gives
        light = linear * (100 * scale)^2 / 7500
    and the exposure k scales it like every other light.
    """
    lin = _source_linear(rgb, brightness)
    peak = max(lin + [1e-9])
    ref = SOURCE_POINT_REFERENCE * MAP_SCALE[0]
    value = EXPOSURE_K[0] * peak * ref * ref / Q3_POINT_SCALE
    return value, [c / peak for c in lin]


# --- Light grid cell size, derived from the map's own bounding box -----------
#
# The light grid is a 3D lattice over the whole map box, so its cost is volume
# divided by cell volume - and a converted map's box is far bigger than a
# hand-built one, because the seal box has to enclose the worldized 3D-skybox
# props as well as the playable area. At q3map2's default 64 64 128, de_dust2
# comes to 4.6 MILLION grid points, well past MAX_MAP_LIGHTGRID, so q3map2
# silently coarsens the cells itself and reports "Storing adjusted grid size".
#
# Solved rather than guessed: hold Z at twice X and Y (vertical resolution
# matters least - the grid only lights dynamic things, players and weapons and
# model_static props, which move horizontally far more than vertically), then
#
#     sx*sy*sz / (2c^3) = budget   ->   c = cbrt( sx*sy*sz / (2*budget) )
#
# rounded up to a multiple of 32. On de_dust2 that lands on 192 192 384, which
# is the value that was arrived at by hand over several compiles - so the
# formula reproduces the tuned answer instead of replacing it with a guess.
GRID_POINT_BUDGET = 200000
GRID_CELL_ROUND = 32.0
GRID_CELL_MIN = 64.0
GRID_CELL_MAX = 1024.0


def derive_gridsize(map_blocks, warnings):
    """Pick a worldspawn 'gridsize' from the finished map's own bounding box, or
    return None if there is no geometry to measure.

    Measured off the GENERATED map text, for the same reason
    find_geometry_clusters is: the VMF's own brush bounds undershoot badly here.
    A 3D skybox's geometry is authored at 1/16 scale and the converter scales and
    reparks it, so the box q3map2 ends up reporting is not a transform of the
    VMF's. On de_dust2 the VMF brush bounds come to 20048 x 10010 against the
    22887 x 17487 q3map2 actually reports - the Y axis short by 75%, enough on
    its own to put the grid well over budget.

    Call this AFTER the seal boxes are appended, so the box includes them.
    """
    clusters = find_geometry_clusters(map_blocks)
    if not clusters:
        return None
    mn = [min(c[0][i] for c in clusters) for i in range(3)]
    mx = [max(c[1][i] for c in clusters) for i in range(3)]
    size = [max(1.0, mx[i] - mn[i]) for i in range(3)]
    c = (size[0] * size[1] * size[2] / (2.0 * GRID_POINT_BUDGET)) ** (1.0 / 3.0)
    c = math.ceil(c / GRID_CELL_ROUND) * GRID_CELL_ROUND
    c = min(max(c, GRID_CELL_MIN), GRID_CELL_MAX)
    cells = (c, c, min(c * 2.0, GRID_CELL_MAX))
    points = 1
    for i in range(3):
        points *= int(size[i] / cells[i]) + 1
    warnings.append(
        "worldspawn gridsize {:.0f} {:.0f} {:.0f} derived from the finished map's bounding box "
        "({:.0f} x {:.0f} x {:.0f} units), giving about {:,} light grid points. At "
        "q3map2's own 64 64 128 default this map would want {:,} - past MAX_MAP_LIGHTGRID, so "
        "q3map2 would coarsen the cells itself and say 'Storing adjusted grid size'. The grid only "
        "lights dynamic things (players, weapons, model_static props), so coarse cells cost almost "
        "nothing visually.".format(
            cells[0], cells[1], cells[2], size[0], size[1], size[2], points,
            (int(size[0] / 64) + 1) * (int(size[1] / 64) + 1) * (int(size[2] / 128) + 1)))
    return "{:.0f} {:.0f} {:.0f}".format(*cells)


# --- Plane points on q3map2's own 1/8-unit grid -----------------------------
#
# q3map2 snaps every emitted VERTEX to 1/8 of a unit, but keeps each BSP PLANE
# at whatever distance the .map gave it. When those disagree by more than 0.001
# units, surfaces stop matching their own planes - and one consequence of that
# silently killed the sun across almost all of de_dust2.
#
# The mechanism, from netradiant-custom's light_trace.cpp: a surface is filed
# into the trace tree by FilterTraceWindingIntoNodes_r, which only keeps a
# surface on the correct side of a coplanar split plane if
# fabs(plane.dist - surface.dist) < 0.001. Miss that and ClipTraceWinding
# classifies the surface as "on" the plane (TW_ON_EPSILON 0.25) and sends it to
# the FRONT child - and axial split planes always face positive
# (facebsp.cpp: planenum & ~1). So a sky face whose room side points along -x,
# -y or -z is filed INSIDE its own solid sky brush. The sun trace stops at the
# first solid leaf and only counts sky triangles collected before it, so it
# never sees that face: everything whose path to the sun leaves through it is
# in shadow.
#
# On de_dust2 at scale 1.27 the sky ceiling sits at 1056 * 1.27 = 1341.12.
# Measured in the compiled .bsp: vertices at 1341.125 (snapped), plane at
# 1341.12 - off by 0.005, five times the tolerance. All 10 ceiling sky triangles
# failed; all 285 upward/outward-facing wall sky triangles passed; only 10.5% of
# the open terrain received any sun, and only along the south-west edge where
# the sun ray exits through a wall instead of the ceiling. 97.2% of all plane
# coordinates in that map were off the 1/8 grid.
#
# 1/8 is exactly representable in float32 at map sizes, so a snapped plane and
# a snapped vertex agree to the bit. It moves any point by at most 1/16 of a
# unit. Axial faces - the overwhelming majority - snap identically on both
# brushes that share them, so no cracks open; a non-axial face whose three
# points would collapse onto a line is left exactly as it was.
PLANE_GRID = 0.125

# Source's default luxel size (a face's "lightmapscale"), in Source units.
SOURCE_LIGHTMAPSCALE = 16


def source_lightmapscale(vmf_text):
    """(dominant lightmapscale, Counter of all of them) for a .vmf.

    Every Source side carries its own "lightmapscale" - 16 unless the mapper
    changed it. idTech3 has no per-face equivalent this converter can write:
    q3map2 takes the luxel size from the ENTITY's _lightmapsamplesize (and a
    shader's own q3map_lightmapSampleSize), never from a single face. So the
    honest single value is the one most of the map's faces actually use, which
    is what Source's own lighting mostly looked like, rather than a constant.
    Measured: de_dust2 13315 faces at 16, 919 at 64, 46 at 19; de_cache_csgo
    9253 at 16, 2253 at 64, 120 at 8 - the dominant value is 16 in both, but
    nothing says a given map has to be built that way."""
    counts = Counter()
    for m in re.finditer(r'"lightmapscale"\s+"([0-9]+(?:\.[0-9]+)?)"', vmf_text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v > 0:
            counts[v] += 1
    if not counts:
        return float(SOURCE_LIGHTMAPSCALE), counts
    # Most faces wins; on a tie, the finer value, so lighting is never
    # coarsened on a guess.
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0], counts
#
# Whole units were tried next (2026-09-30), because q3map2 stores leaf boxes as
# truncated integers and -light then drops a wall lying on a fractional leaf
# boundary out of the leaf in front of it (every luxel occluded -> the black
# bands along wall bottoms). It fixed those, but snapping axial planes by up to
# half a unit while sloped planes stay put breaks every place a sloped face was
# built to meet an axial one: measured on de_dust2 by compiling the same map
# both ways, vertices lying 0.1-0.3 units off a neighbouring edge went from 109
# to 1,280 (the visible cracks at arches, window reveals and tunnel ceilings)
# and the map even leaked. So geometry stays on the 1/8 grid, where the crack
# count matches the untouched Source geometry, and the leaf-box problem is fixed
# where it actually is - in the .bsp - by bsp_fix_leafbounds.py between the
# -bsp and -light stages (it widens every stored leaf box by the one unit the
# truncation can lose). Same result on the band wall, 16-24 instead of a flat 9.
LIGHT_SAFE_PLANE_GRID = 1.0

_FACE_LINE_RE = re.compile(
    r'^(\s*)\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)'
    r'\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)(\s+\S.*)$')


def _snap_face_points(v, grid):
    """Snapped copy of a face's 9 plane-point coordinates, or None if the
    three points would collapse onto a line."""
    s = [round(x / grid) * grid for x in v]
    p0, p1, p2 = s[0:3], s[3:6], s[6:9]
    e1 = [p0[i] - p1[i] for i in range(3)]
    e2 = [p2[i] - p1[i] for i in range(3)]
    cx = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
          e1[0] * e2[1] - e1[1] * e2[0])
    if cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2] < 1e-6:
        return None
    return s


def _face_plane(s):
    """(unit normal, dist) of a Q3 face from its three points (clockwise
    winding seen from outside, so the normal points out of the brush)."""
    p0, p1, p2 = s[0:3], s[3:6], s[6:9]
    a = [p2[i] - p0[i] for i in range(3)]
    b = [p1[i] - p0[i] for i in range(3)]
    n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    ln = math.sqrt(sum(c * c for c in n)) or 1.0
    n = [c / ln for c in n]
    return n, sum(n[i] * p0[i] for i in range(3))


def _is_axial_face(v, tol=1e-3):
    """Do the face's three points share one coordinate (an axis-aligned plane)?"""
    return any(abs(v[a] - v[3 + a]) < tol and abs(v[a] - v[6 + a]) < tol for a in range(3))


def _brush_still_solid(faces):
    """True unless two opposite axial faces of the brush now touch or cross
    (the only way a snap can destroy a convex brush that was at least a unit
    thick before)."""
    lo = [-1e18] * 3
    hi = [1e18] * 3
    for s in faces:
        n, d = _face_plane(s)
        for ax in range(3):
            if abs(n[ax] - 1.0) < 1e-9:
                hi[ax] = min(hi[ax], d)
            elif abs(n[ax] + 1.0) < 1e-9:
                lo[ax] = max(lo[ax], -d)
    return all(hi[ax] - lo[ax] > 0.5 for ax in range(3))


def snap_brush_planes_to_grid(map_text, warnings, grid=PLANE_GRID, fallback=PLANE_GRID):
    """Snap the three plane points of every brush face line to `grid` (whole
    units), falling back to `fallback` (1/8) for any brush that the coarse grid
    would squash. Patch control points are 5-tuples and don't match, so they
    are untouched."""
    lines = map_text.split('\n')
    out = []
    counts = {'grid': 0, 'fallback': 0, 'kept': 0, 'brush_fallback': 0}
    total_off = [0]

    def flush(block):
        # block: list of (line, match_or_None)
        faces = [(ln, m, [float(m.group(i)) for i in range(2, 11)]) for ln, m in block if m]
        use = grid
        coarse = [_snap_face_points(v, grid) for _, _, v in faces]
        if faces and not _brush_still_solid([c if c is not None else v
                                             for c, (_, _, v) in zip(coarse, faces)]):
            use = fallback
            counts['brush_fallback'] += 1
        for ln, m in block:
            if not m:
                out.append(ln)
                continue
            v = [float(m.group(i)) for i in range(2, 11)]
            # Only AXIAL planes go to whole units: they are the leaf-box edges
            # the -light cluster test compares against. A sloped face snapped
            # that coarsely tilts visibly and can land on top of a neighbouring
            # bevel ("duplicate plane", measured: 20+ brushes on de_dust2), so
            # it stays on the 1/8 grid it was already safe on.
            if use != fallback and not _is_axial_face(v):
                s = _snap_face_points(v, fallback)
                if s is not None:
                    counts['fallback'] += 1
                    total_off[0] += sum(1 for a, b in zip(v, s) if a != b)
                    out.append("{}( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) "
                               "( {:.10g} {:.10g} {:.10g} ){}".format(m.group(1), *s, m.group(11)))
                    continue
            s = _snap_face_points(v, use)
            if s is None and use != fallback:
                s = _snap_face_points(v, fallback)
                if s is not None:
                    counts['fallback'] += 1
            elif s is not None:
                counts['grid' if use == grid else 'fallback'] += 1
            if s is None:
                out.append(ln)
                counts['kept'] += 1
                continue
            total_off[0] += sum(1 for a, b in zip(v, s) if a != b)
            out.append("{}( {:.10g} {:.10g} {:.10g} ) ( {:.10g} {:.10g} {:.10g} ) "
                       "( {:.10g} {:.10g} {:.10g} ){}".format(m.group(1), *s, m.group(11)))

    block = []
    for line in lines:
        m = _FACE_LINE_RE.match(line)
        if m:
            try:
                [float(m.group(i)) for i in range(2, 11)]
            except ValueError:
                m = None
        if m:
            block.append((line, m))
            continue
        if block:
            flush(block)
            block = []
        out.append(line)
    if block:
        flush(block)
    warnings.append(
        "Snapped {:,} brush faces' plane points to the {:g}-unit grid ({:,} coordinates moved, "
        "none by more than {:g}); {} left as-is because snapping would have made them "
        "degenerate. Keeps downward-facing sky faces out of their own brush (q3map2 snaps "
        "vertices to this grid but not planes). AFTER the -bsp stage and BEFORE -light, run "
        "bsp_fix_leafbounds.py on the .bsp - q3map2 stores leaf boxes as truncated integers, "
        "and at these fractional coordinates -light otherwise drops walls out of the leaf in "
        "front of them (the black bands along wall bottoms).".format(
            counts['grid'] + counts['fallback'], grid, total_off[0], grid / 2, counts['kept']))
    return '\n'.join(out)


WORLDSPAWN_KEYS_KEPT = {
    # q3map2 compile-time
    'gridsize', 'ambient', '_ambient', '_color', '_minlight', '_mingridlight',
    '_minvertexlight', '_lightmapscale', 'lightmapscale', '_blocksize', 'blocksize',
    'chopsize', '_celshader', '_ignoreleaks', '_noshadersun', '_castshadows', '_cs',
    '_receiveshadows', '_rs', '_anglescale', '_farplanedist', '_foghull', '_indexmap',
    '_layers', '_shader', '_offsets',
    # engine-side
    'classname', 'message', 'music', 'gravity', 'distancecull', 'spawnflags',
}


def convert_world(world, texture_map, unmapped_counter, scale, offset, warnings,
                   materials_dir=None, terrain_outputs=None, terrain_mode='brush',
                   drop_backing_brush=False, patch_output_blocks=None, sky_region=None,
                   distance_cull=0.0, gridsize=None, ambient=None, ambient_color=None):
    lines = ["{", '"classname" "worldspawn"']
    # 'distancecull' is the Raven idTech3 far-clip worldspawn key (the engine
    # family SOF2 shares with JK2/JA): geometry beyond it simply isn't drawn,
    # which shows up as terrain and distant brushwork popping in as you walk
    # toward it. There is no equivalent in Source - Source's own far clip is
    # the worldspawn 'MaxRange' key, which this converter passes through
    # untouched because the Raven engine ignores it, so without this the map
    # runs on the engine default no matter what the original intended.
    # Source's own far clip, converted rather than dropped. 'MaxRange' is in
    # Source units, so it scales with the map like everything else; Raven's
    # engine spells the same idea 'distancecull'. de_dust2 sets 8000, which at
    # scale 1.27 is 10160 map units. An explicit --distance-cull still wins.
    if not (distance_cull and distance_cull > 0):
        try:
            max_range = float(get1(world, 'MaxRange') or get1(world, 'maxrange') or 0)
        except (TypeError, ValueError):
            max_range = 0.0
        if max_range > 0:
            distance_cull = max_range * scale
            warnings.append(
                "worldspawn 'MaxRange' {:g} (Source's far clip) converted to 'distancecull' "
                "{:.0f} - the same idea under Raven's name, scaled with the map. SOF2 ignores "
                "MaxRange itself, so without this the map ran on the engine default no matter "
                "what the original intended.".format(max_range, distance_cull))
    if distance_cull and distance_cull > 0:
        lines.append('"distancecull" "{:.10g}"'.format(distance_cull))
    # The light grid is a 3D lattice over the whole map bounding box, so its
    # cost is volume / (cell x * cell y * cell z) - and a converted map's box is
    # far bigger than a hand-built one, because --seal-map-box has to enclose
    # the worldized skybox props as well as the playable area. Measured on
    # de_dust2: at the default 64 64 128 that is 949,416 grid points, enough
    # that q3map2 shrinks the cells further to fit MAX_MAP_LIGHTGRID and reports
    # "Storing adjusted grid size"; at 192 192 384 it is about 91,000. The grid
    # only lights dynamic things - players, weapons, model_static props - so
    # coarse cells cost almost nothing visually.
    #
    # Written here rather than left as a hand edit on purpose: a hand-added
    # worldspawn key is silently lost the next time the map is regenerated,
    # which is exactly what happened on the compile that prompted this.
    if gridsize:
        lines.append('"gridsize" "{}"'.format(gridsize))
    # The shadow fill. Blunt by nature - q3map2 adds it to every luxel with no
    # regard for occlusion - but that bluntness is the feature here: it is the
    # only thing in the toolchain that reaches a closed room's ceiling. It is in
    # the same units as the sun's intensity, so it has to be tens against a
    # several-hundred sun, not single digits.
    #
    # Derived from light_environment by default (see
    # derive_ambient_from_light_environment); --ambient overrides it.
    if ambient:
        lines.append('"ambient" "{}"'.format(ambient))
        # q3map2 ColorNormalizes this and then scales it by "ambient", so it
        # only ever carries the tint - all the magnitude lives in the number
        # above. Source's _ambient colour on a daylight map is the sky's own
        # blue, which is what keeps converted shadows from going flat grey.
        if ambient_color:
            lines.append('"_color" "{}"'.format(ambient_color))
    # Keys this function has already written itself. A Source worldspawn that
    # happens to carry one of them would otherwise be passed through as a second
    # copy, and q3map2 takes the FIRST occurrence of a key - so the pass-through
    # would silently win over the derived value on some maps and not others.
    already_written = set()
    if distance_cull and distance_cull > 0:
        already_written.add('distancecull')
    if gridsize:
        already_written.add('gridsize')
    if ambient:
        already_written.update(('ambient', '_ambient'))
        if ambient_color:
            already_written.add('_color')
    dropped = []
    for key, values in world.items():
        if key in ('solid', 'editor', 'group', 'id', 'hidden', 'mapversion'):
            continue
        for v in values:
            if isinstance(v, dict):
                continue
            if key == 'classname':
                continue
            if key.lower() in already_written:
                continue
            if key.lower() not in WORLDSPAWN_KEYS_KEPT:
                dropped.append(key)
                continue
            lines.append('"{}" "{}"'.format(key, v))
    if dropped:
        warnings.append(
            "worldspawn: dropped {} Source-only key(s) that neither q3map2 nor SOF2 reads ({}). "
            "They were inert, not harmful - notably 'light', which does NOT make a stray lamp at "
            "the origin because CreateEntityLights guards on the classname starting with 'light' "
            "and worldspawn fails that - but carrying them made the entity lump look like it was "
            "configuring something it wasn't.".format(
                len(dropped), ", ".join(sorted(set(dropped)))))

    for solid in get_solid_blocks(world):
        block = convert_solid(solid, texture_map, unmapped_counter, scale, offset, warnings,
                               materials_dir=materials_dir, terrain_outputs=terrain_outputs,
                               terrain_mode=terrain_mode, drop_backing_brush=drop_backing_brush,
                               patch_output_blocks=patch_output_blocks, sky_region=sky_region)
        if block:
            lines.append(block)

    lines.append("}")
    return "\n".join(lines)


# Some translations need more than a classname swap - an extra key has to
# be injected too. classname -> (new_classname, {extra keys to add}).
SPECIAL_ENTITY_REWRITES = {
    # SOF2 MP doesn't have separate per-team spawn classnames like Source's
    # info_player_start_red/blue - team spawns are ALL gametype_player,
    # distinguished by a spawnflags bit (1=red, 2=blue). Gametype-
    # restriction bits (inf/elim/dem/ctf/ctrl) are deliberately left unset
    # here rather than guessed - the exact bit values weren't confirmed, and
    # a wrong bit risks the spawn not working in ANY mode, which is worse
    # than just not restricting it.
    'info_player_start_red': ('gametype_player', {'spawnflags': '1'}),
    'info_player_start_blue': ('gametype_player', {'spawnflags': '2'}),
    # CS's own native team spawn classnames (this is what de_cache_csgo.vmf
    # actually uses, rather than the _red/_blue naming). Mapped BY ROLE, not
    # by any color similarity: in SOF2 blue is the attacking team and red is
    # the defending team, and in CS bomb-defusal the terrorists attack (they
    # plant) while counter-terrorists defend. So T -> blue (2) and CT -> red
    # (1). This is the reverse of an earlier guess here that paired them the
    # other way round.
    'info_player_terrorist': ('gametype_player', {'spawnflags': '2'}),
    'info_player_counterterrorist': ('gametype_player', {'spawnflags': '1'}),
    # Source's generic/FFA spawn point, in case a CS map has one alongside
    # its team spawns.
    'info_player_start': ('info_player_deathmatch', {}),
}


def convert_entity(entity, texture_map, unmapped_counter, scale, offset, warnings,
                    class_map, skip_classes, model_scale=1.0, materials_dir=None,
                    terrain_outputs=None, terrain_mode='brush', drop_backing_brush=False,
                    patch_output_blocks=None, sky_region=None, model_static_paths=None):
    classname = get1(entity, 'classname', '')
    if should_skip_class(classname, skip_classes):
        return None

    if classname.lower() == 'light_environment':
        # Source's SUN, not a lamp. It has an origin like any entity, but the
        # origin is meaningless - the entity describes a directional light for
        # the whole map. Emitting it as a q3map2 point light puts a bright
        # white lamp at an arbitrary spot and lights nothing correctly.
        # de_dust2 carries TEN of them, so ten stray lights. The entity's two
        # halves are carried over separately and properly: its _light becomes
        # q3map_sun on the sky shader (vmt2shader.py), and its _ambient becomes
        # worldspawn "ambient" + "_color" here.
        warnings.append(
            "light_environment dropped as a point light - it is Source's sun, not a lamp, and "
            "emitting it here would add a stray white light at a meaningless origin (de_dust2 has "
            "10 of them). Its two halves are converted instead: _light becomes q3map_sun on the "
            "sky shader, so pass --vmf to vmt2shader.py as well, and _ambient becomes this "
            "worldspawn's 'ambient' + '_color'.")
        return None

    extra_keys = {}
    if classname in SPECIAL_ENTITY_REWRITES:
        new_classname, extra_keys = SPECIAL_ENTITY_REWRITES[classname]
    else:
        new_classname = class_map.get(classname, classname)

    # model_static is for ORDINARY clutter props only (the small-model
    # heuristic from mdl_to_md3.py's convert()): it's spawned at RUNTIME
    # rather than baked into the BSP by q3map2's AddTriangleModels, so it
    # never enters the meta pipeline (sidesteps MAX_MAP_LEAFFACES on a
    # leaking map) and is never loaded by q3map2 at compile time (sidesteps
    # MAX_MODELS=512, q3map2's own cap on unique model FILES it loads to
    # bake misc_model geometry - confirmed in its source, model.c).
    #
    # A model in the 3D SKYBOX AREA must NEVER be model_static, no matter
    # what the small-model heuristic says. Confirmed in q3map2's own source
    # and then in game: ds->skybox is only ever set on mapDrawSurface_t
    # objects, which model_static entities never become - they are
    # structurally invisible to skybox marking, so a model_static in the
    # skybox area simply never renders as a distant backdrop at all (this
    # is why the skybox palm trees never appeared). Routing skybox models
    # to model_static was never a fix for the leak-driven
    # MAX_MAP_LEAFFACES, just a silent way of dropping them from the
    # backdrop - the leak itself is what has to be sealed.
    #
    # So skybox-area models stay misc_model and are forced back to it even
    # if they're in --model-static-list. They get spawnflags 4:
    # SOF2's own mp_entities.def calls bit 4 "Lightmapped", which is the
    # same bit q3map2 reads as its forced-meta flag (model.c:
    # `( si->forceMeta ) || ( spawnFlags & 4 )`) - so the model's triangles
    # go through the meta-surface pipeline (where skybox marking happens,
    # and where q3map2 enforces the engine's per-surface index cap and
    # splits anything oversized by itself). Deliberately WITHOUT bit 2
    # (SOLID) or bit 8 (auto clip): 3D skybox geometry is a distant visual
    # backdrop that is never reachable or walked on, so it needs no
    # collision - and auto clip on dense models is what produced
    # "MAX_MAP_PLANES hit generating clip brushes for model ...md3"
    # (see SPAWNFLAG_UNSAFE_CLASSES).
    if new_classname == 'misc_model':
        in_sky = False
        # A model --skybox-models-to-world already moved into real map space
        # is ordinary main-map geometry: it must not be skybox-marked (it
        # isn't a backdrop any more) and its new origin can still land inside
        # --skybox-radius, so the position test below would wrongly claim it.
        if get1(entity, WORLDIZED_MARKER_KEY):
            sky_region = None
        if sky_region is not None:
            o = _vec_key(entity, 'origin')
            if o and len(o) >= 3:
                in_sky = in_skybox_area(tuple(o[:3]))
        if in_sky:
            extra_keys = dict(extra_keys)
            extra_keys['spawnflags'] = '4'
        elif model_static_paths:
            model_val = get1(entity, 'model') or get1(entity, 'model2')
            if model_val and convert_model_path(model_val).strip().lower() in model_static_paths:
                # ...unless its origin is inside a wall. SOF2 draws a
                # model_static only while its ORIGIN is in the viewer's PVS,
                # and an origin in a solid leaf is in nobody's - windows and
                # battlement teeth set into walls (origin on or inside the
                # wall brush) flicked out of view from most places. Measured
                # on the de_dust2 build of 2026-10-01: 20 model_statics had
                # their origin in a solid leaf, every one of them inside or
                # on the face of a world brush. Those stay misc_model, baked
                # into the BSP, where visibility goes by their triangles.
                # A prop resting ON a surface has its origin on that brush's
                # face, where the 1/8 snap of the scaled coordinates decides
                # which side it lands on; those are nudged a quarter unit out
                # along the face (invisible) instead of being baked.
                o = _vec_key(entity, 'origin')
                pen = world_brush_penetration(o) if o and len(o) >= 3 else None
                if pen is None:
                    new_classname = 'model_static'
                elif pen[0] < 1.0:
                    new_classname = 'model_static'
                    push = (pen[0] + 0.25) / 1.0
                    no = add(o, scale_vec(pen[1], push))
                    entity = dict(entity)
                    entity['origin'] = ['{:.6g} {:.6g} {:.6g}'.format(*no)]

    if 'connections' in entity:
        warnings.append(
            "entity '{}' had Source I/O (connections) which has no idTech3 "
            "equivalent; dropped".format(classname))

    is_model_entity = (new_classname in ('misc_model', 'model_static') and
                        any(key.lower() in MODEL_VALUE_KEYS for key in entity.keys()))

    # Confirmed against mp_entitiesd.def: info_player_deathmatch takes a
    # single "angle" key exactly like gametype_player does, not Source's
    # 3-part "angles" - this was only suspected/unconfirmed last round.
    ANGLE_ONLY_SPAWN_CLASSES = ('gametype_player', 'info_player_deathmatch')

    extra_blocks = []      # extra entities emitted alongside (a spotlight's info_null)
    lines = ["{"]
    lines.append('"classname" "{}"'.format(new_classname))
    for line in format_entity_keys(entity, scale, offset, model_scale, is_model_entity,
                                    new_classname):
        if line.startswith('"classname"'):
            continue
        if new_classname in ANGLE_ONLY_SPAWN_CLASSES and line.startswith('"angles"'):
            # Drop Source's 3-component "angles" - the "angle" replacement
            # is added via extra_keys below instead.
            continue
        if new_classname == 'light' and any(
                line.startswith('"{}"'.format(k)) for k in SOURCE_LIGHT_KEYS_TO_DROP):
            # Must be REMOVED, not just supplemented: q3map2 reads "_light"
            # before "light", so leaving it in means the red channel keeps
            # winning as the intensity.
            continue
        if new_classname == 'gametype_player' and line.startswith('"origin"'):
            # Spawns were sitting inside the floor - lift by 32 units.
            m = re.match(r'"origin" "([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)"', line)
            if m:
                ox_, oy_, oz_ = (float(m.group(i)) for i in (1, 2, 3))
                line = '"origin" "{:.10g} {:.10g} {:.10g}"'.format(ox_, oy_, oz_ + 32)
        lines.append(line)
    if new_classname in ANGLE_ONLY_SPAWN_CLASSES:
        angles_raw = get1(entity, 'angles')
        if angles_raw:
            parts = angles_raw.split()
            if len(parts) == 3:
                extra_keys['angle'] = parts[1]  # pitch yaw roll -> yaw only
    if new_classname == 'light':
        raw = get1(entity, '_light')
        parts = (raw or '').split()
        try:
            rgb = [float(x) for x in parts[:3]] if len(parts) >= 3 else [255.0, 255.0, 255.0]
            brightness = float(parts[3]) if len(parts) > 3 else 300.0
        except ValueError:
            rgb, brightness = [255.0, 255.0, 255.0], 300.0
        # Linear, matched to Source at the same physical distance, at the same
        # exposure as the sun - see point_light_value. --point-light-scale and
        # --point-light-offset still apply on top for a hand run.
        value, color = point_light_value(rgb, brightness)
        value = value * POINT_LIGHT_SCALE[0] + POINT_LIGHT_OFFSET[0]
        falloff_lines, spot_blocks = source_light_falloff_and_spot(entity, value, scale, offset)
        lines.extend(falloff_lines)
        lines.append('"_color" "{:.4g} {:.4g} {:.4g}"'.format(*color))
        extra_blocks.extend(spot_blocks)

    if new_classname in ('misc_model', 'model_static') and classname.lower().startswith('prop_'):
        # Where the prop came from, for plan_prop_budget's drop order and for
        # phy2clip to leave out the collision of a prop that was dropped.
        # Removed from the map before it is written (strip_source_prop_keys).
        extra_keys = dict(extra_keys)
        extra_keys[SRC_CLASS_KEY] = classname.lower()
        if get1(entity, 'id'):
            extra_keys[SRC_ID_KEY] = str(get1(entity, 'id')).strip()
        extra_keys[SRC_SOLID_KEY] = (get1(entity, 'solid', '6') or '6').strip()

    for k, v in extra_keys.items():
        lines.append('"{}" "{}"'.format(k, v))

    solids = get_solid_blocks(entity)

    # A Source brush entity's "origin" key is metadata (VBSP moves the brushes
    # into local space around it); its brush planes are still world space in
    # the .vmf. q3map2 only does that move for an ORIGIN BRUSH (map.cpp:
    # AdjustBrushesForOrigin(-originbrush_origin)) - a bare "origin" key leaves
    # the geometry in world space AND keeps the key, and the game then places
    # the inline model at that origin, i.e. the offset is applied twice.
    # Confirmed on de_mirage_cs2: 52 func_brush->func_wall entities carried an
    # origin; the crate stack at -800 -2617.5 -142.5 was drawn at about
    # -1600 -5235 -285, outside the map. So: drop the key, and where the class
    # really rotates around it, give q3map2 an origin brush at that point
    # instead, which yields the correct local geometry plus origin key.
    if solids and new_classname not in ('worldspawn', 'func_group'):
        origin_line = next((ln for ln in lines if ln.startswith('"origin"')), None)
        if origin_line:
            lines.remove(origin_line)
            if new_classname in PIVOT_BRUSH_CLASSES:
                o = [float(x) for x in origin_line.split('"')[3].split()]
                lines.append(_box_brush([c - 4 for c in o], [c + 4 for c in o],
                                        texture="tools/_origin"))

    # Source brush entities nothing collides with: func_brush "Solidity" "1"
    # (never solid) and func_illusionary. 15 of de_mirage_cs2's 52 func_brush
    # are Solidity 1 (window backings, trim) and all of them came out as
    # solid func_walls. idTech3 takes solidity from the shader alone, so
    # their faces get a non-solid copy of their shader instead.
    never_solid = (classname == 'func_illusionary' or
                   (classname == 'func_brush' and (get1(entity, 'Solidity') or '').strip() == '1'))

    for solid in solids:
        block = convert_solid(solid, texture_map, unmapped_counter, scale, offset, warnings,
                               is_entity_brush=True, materials_dir=materials_dir,
                               terrain_outputs=terrain_outputs, terrain_mode=terrain_mode,
                               drop_backing_brush=drop_backing_brush,
                               patch_output_blocks=patch_output_blocks)
        if block:
            if never_solid:
                block = _retexture_nonsolid(block)
            lines.append(block)

    lines.append("}")
    return "\n".join([("\n".join(lines))] + extra_blocks), False


NONSOLID_SUFFIX = '_nonsolid'
NONSOLID_MATERIALS = set()       # 'textures/<material>' whose _nonsolid copy is used
# Tool shaders that are solid in SOF2, and the non-solid stand-in vmt2shader.py
# always emits for each. Every other tool/helper shader is already non-solid
# (or a clip, which a never-solid entity doesn't carry) and is left alone.
_NONSOLID_TOOL_SWAP = {'tools/_caulk': 'vmf2map/nodraw_nonsolid',
                       'tools/_black': 'vmf2map/black_nonsolid'}
_P3 = r'\(\s*[-+\d.eE]+\s+[-+\d.eE]+\s+[-+\d.eE]+\s*\)'
_BRUSH_FACE_TEX_RE = re.compile(r'^(\s*' + _P3 + r'\s*' + _P3 + r'\s*' + _P3 + r'\s+)(\S+)(.*)$')


def _retexture_nonsolid(block):
    out = []
    lines = block.split('\n')
    for i, ln in enumerate(lines):
        m = _BRUSH_FACE_TEX_RE.match(ln)
        is_patch_tex = i > 0 and lines[i - 1].strip().startswith('patchDef')
        if not m and not is_patch_tex:
            out.append(ln)
            continue
        tex = m.group(2) if m else ln.strip()
        low = tex.lower()
        if low in _NONSOLID_TOOL_SWAP:
            new = _NONSOLID_TOOL_SWAP[low]
        elif low.startswith(('tools/', 'vmf2map/')) or low.endswith(NONSOLID_SUFFIX):
            new = tex
        else:
            new = tex + NONSOLID_SUFFIX
            NONSOLID_MATERIALS.add('textures/' + low)
        out.append(m.group(1) + new + m.group(3) if m else ln.replace(tex, new))
    return '\n'.join(out)


# ==============================================================================
# Top-level conversion
# ==============================================================================

def compute_bounds(root, scale):
    """Scan every brush plane point in the VMF (world + all entities) to
    find the overall bounding box, used for --recenter."""
    min_pt = [math.inf, math.inf, math.inf]
    max_pt = [-math.inf, -math.inf, -math.inf]
    found = False

    def scan_solid(solid):
        nonlocal found
        for side in solid.get('side', []):
            if not isinstance(side, dict):
                continue
            plane_str = get1(side, 'plane')
            if not plane_str:
                continue
            for p in parse_plane(plane_str):
                found = True
                for i in range(3):
                    v = p[i] * scale
                    if v < min_pt[i]:
                        min_pt[i] = v
                    if v > max_pt[i]:
                        max_pt[i] = v

    worlds = root.get('world', [])
    if worlds:
        for solid in get_solid_blocks(worlds[0]):
            scan_solid(solid)
    for entity in root.get('entity', []):
        for solid in get_solid_blocks(entity):
            scan_solid(solid)

    if not found:
        return (0.0, 0.0, 0.0), None
    center = tuple((min_pt[i] + max_pt[i]) / 2.0 for i in range(3))
    bounds = (tuple(min_pt), tuple(max_pt))
    return center, bounds


def find_sky_shader_name(shader_file):
    """Scan a generated .shader file for the actual sky shader vmt2shader.py
    wrote (identified by a 'surfaceparm sky' or 'skyParms' line inside its
    body), and return its exact name - or None if the file doesn't exist,
    can't be read, or has no sky shader in it.

    Using this instead of blindly constructing 'skybox/<skyname>' fixes a
    real, confirmed mismatch: a map's own skyname worldspawn key doesn't
    always match the basename its skybox VMTs actually use on disk (e.g. a
    map whose skyname points at an HDR-era sky name while the map's own
    extracted VMTs only ever had the LDR-named files) - the shader block
    vmt2shader.py generated from whatever material actually exists is the
    only sky shader name guaranteed to be real."""
    if not shader_file or not os.path.isfile(shader_file):
        return None
    try:
        with open(shader_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return None

    current_name = None
    in_block = False
    block_lines = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not in_block:
            if stripped and not stripped.startswith(('{', '}', '/', '\t')) and ' ' not in stripped:
                current_name = stripped
            elif stripped == '{' and current_name:
                in_block = True
                block_lines = []
        else:
            if stripped == '}':
                is_sky = any(
                    re.match(r'surfaceparm\s+sky\s*$', bl, re.IGNORECASE) or
                    re.match(r'skyparms\b', bl, re.IGNORECASE)
                    for bl in block_lines)
                if is_sky:
                    return current_name
                in_block = False
                current_name = None
            else:
                block_lines.append(stripped)
    return None


OVERLAY_MATERIALS = set()

# Decal and overlay patches draw a material through its own shader copy,
# "<material>_decal" (polygonOffset, cull none, nonsolid - vmt2shader.py
# --overlay-materials), never through the material's plain shader. Shader
# names are global across every converted map and the first .shader loaded
# wins, so a CS material must come out the same in every map's file: before,
# de_overpass_csgo added polygonOffset + nonsolid to a foliage material it
# used as an overlay, while a map using it on a tree did not.
DECAL_SHADER_SUFFIX = '_decal'


def decal_texture_name(material, mapped):
    """The .map texture name a decal of `material` uses. A material the
    texture map sends somewhere else (a stock SOF2 texture) keeps that."""
    plain = re.sub(r'^(textures/)+', '', material.strip().replace('\\', '/').lower(), flags=re.I)
    if mapped != plain:
        return mapped
    return plain + DECAL_SHADER_SUFFIX

# Overlays on terrain get a dense conforming mesh instead of a coarse
# patchDef2 (see build_overlay_conforming_mesh) - collected here across the
# whole map, same idea as LWO_TERRAIN_BUILDER but far simpler (no welding
# needed between separate overlays, each is its own independent geometry).
OVERLAY_MESH_SURFACES = {}   # flat shader name -> (points, uvs, triangles)
OVERLAY_MESH_SKY_SURFACES = {}  # same, for overlays inside the 3D skybox area
OVERLAY_MESH_MATERIAL_TO_FLAT = {}  # material (lowercased) -> flat shader name, dedup across overlays

# Grid spacing (post-scale units) for terrain-conforming overlay meshes.
# Measured on real de_dust2: the terrain's OWN triangles average ~88 units
# across, so sampling much finer than this adds triangles without adding
# detail - the surface is flat between its own vertices. Sampling at 24
# produced ~15000 triangles and contributed to a MAX_MAP_LEAFFACES compile
# failure (every surface is referenced by each BSP leaf it touches, and a
# map-spanning decal touches many). Lower = follows cliffs more tightly but
# costs compile budget; --overlay-mesh-spacing overrides it.
# How far a ground overlay is lifted off the terrain, along the terrain
# triangle's own normal. Since the overlay now inherits the terrain's exact
# triangulation, this constant IS the whole separation between the two
# surfaces - there is no sampling mismatch left for it to have to cover, so it
# only has to beat depth-buffer precision.
OVERLAY_MESH_LIFT = 0.5

# Minimum |BasisNormal.z| for an info_overlay to count as lying on the
# GROUND, and therefore to be conformed to the terrain surface at all.
# Measured across de_dust2's 55 overlays: ground ones are 0.970..1.000,
# wall ones are exactly 0.000 - so anything in between is a shape this data
# never produces, and 0.5 sits safely in that empty gap (~60 degrees off
# horizontal). See build_overlay_conforming_mesh for what going wrong here
# actually looked like.
OVERLAY_MESH_MIN_GROUND_NZ = 0.5
OVERLAY_MESH_MATERIALS = {}  # flat shader name -> material

# --- Vertex-lit decal density -------------------------------------------------
#
# Every decal and overlay is VERTEX lit (q3map_nolightmap +
# q3map_onlyvertexlighting, see vmt2shader.py _ensure_vertex_compile - an
# alpha-blended decal cannot multiply its own lightmap in SOF2), so its
# lighting is only as fine as its vertices. The surface underneath is
# lightmapped at ~20 units. Measured on de_overpass_csgo's 2026-10-12 build:
# the overlay mesh had edges up to 982 units (median 48), so a dirt or sand
# overlay whose corners all sat in a shadow came out as one dark rectangle on
# lit ground, and shadow edges crossing an overlay became long straight
# wedges between its few vertices. Brush-face overlays were a fixed 5x5
# patchDef2 (3x3 drawn vertices after the engine drops the flat mid rows,
# R_SubdividePatchToGrid) at any size.
#
# So no decal edge may be longer than this, post-scale. Vertex colours are
# interpolated linearly, like a bilinear-filtered lightmap, so this is close
# to the ground's ~20-unit luxels without matching them one for one. Measured
# on de_overpass_csgo's overlay mesh (2,980 triangles before): 24 gave 79,617
# triangles, 32 gave 47,121, 48 gave 21,849. 32 was used first and added
# 4.6 MiB to a .bsp that has to load as one block (see BSP_SIZE_TARGET); 48
# halves that. bsp_decal_relight.py copies the ground's lightmap into every
# decal vertex after -light, so the vertex colours are already smooth and the
# coarser spacing only loses a little shadow-edge sharpness on decals.
DECAL_VERTEX_LIGHT_SPACING = 48.0
# q3map2's MAX_PATCH_SIZE is 32 control points per side; patchDef2 sides must
# be odd. 31 caps a brush overlay at 16 drawn vertices per side.
PATCH_MAX_CONTROL_POINTS = 31


def decal_patch_side(length, minimum=3):
    """Control points along one side of a flat decal patchDef2 so that its
    DRAWN vertices (the even control points - the engine drops the flat odd
    rows) are at most DECAL_VERTEX_LIGHT_SPACING apart."""
    spans = max(1, int(math.ceil(length / DECAL_VERTEX_LIGHT_SPACING - 1e-6)))
    return max(minimum, min(PATCH_MAX_CONTROL_POINTS, 2 * spans + 1))


def subdivide_mesh_surfaces(surfaces, max_edge=DECAL_VERTEX_LIGHT_SPACING):
    """Split the triangles of {name: (points, uvs, triangles)} until no edge
    is longer than max_edge; returns (new_surfaces, points_added).

    Crack-free: each round marks every edge over the limit by its two end
    POSITIONS and splits it at its midpoint in every triangle that has it, so
    two triangles sharing an edge always get the same new vertex - also
    across a UV seam, where the shared edge has two pairs of point indices
    (each side interpolates its own uv). A triangle with one marked edge
    becomes 2, with two 3, with three 4. Midpoints of a flat triangle stay
    in its plane, and the overlay mesh is built on the terrain's own flat
    triangles, so nothing moves off the surface."""
    def key(p):
        return (round(p[0], 3), round(p[1], 3), round(p[2], 3))

    out = {}
    added = 0
    lim2 = max_edge * max_edge
    for name, (pts, uvs, tris) in surfaces.items():
        pts = list(pts)
        uvs = list(uvs)
        tris = [tuple(t) for t in tris]
        start = len(pts)
        for _round in range(32):
            long_edges = set()
            for t in tris:
                for i in range(3):
                    a, b = pts[t[i]], pts[t[(i + 1) % 3]]
                    if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) > lim2:
                        ka, kb = key(a), key(b)
                        long_edges.add((ka, kb) if ka <= kb else (kb, ka))
            if not long_edges:
                break
            mid_cache = {}

            def mid(i, j):
                k = (i, j) if i < j else (j, i)
                m = mid_cache.get(k)
                if m is None:
                    a, b = pts[i], pts[j]
                    pts.append(((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5))
                    ua, ub = uvs[i], uvs[j]
                    uvs.append(tuple((ua[c] + ub[c]) * 0.5 for c in range(len(ua))))
                    m = mid_cache[k] = len(pts) - 1
                return m

            def marked(i, j):
                ka, kb = key(pts[i]), key(pts[j])
                return ((ka, kb) if ka <= kb else (kb, ka)) in long_edges

            new_tris = []
            for t in tris:
                flags = [marked(t[i], t[(i + 1) % 3]) for i in range(3)]
                n = sum(flags)
                if n == 0:
                    new_tris.append(t)
                    continue
                if n == 3:
                    a, b, c = t
                    ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                    new_tris += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
                    continue
                # Rotate so the marked edge is a->b (one marked), or the two
                # marked ones are a->b and b->c. Winding is kept throughout.
                if n == 1:
                    r = flags.index(True)
                else:
                    r = [i for i in range(3) if flags[i] and flags[(i + 1) % 3]][0]
                a, b, c = t[r], t[(r + 1) % 3], t[(r + 2) % 3]
                ab = mid(a, b)
                if n == 1:
                    new_tris += [(a, ab, c), (ab, b, c)]
                else:
                    bc = mid(b, c)
                    new_tris += [(ab, b, bc), (a, ab, bc), (a, bc, c)]
            tris = new_tris
        added += len(pts) - start
        out[name] = (pts, uvs, tris)
    return out, added


OVERLAY_MESH_MODEL_PATH = 'models/mapobjects/terrain'
OVERLAY_MESH_FILENAME = 'welded_terrain_overlays.lwo'


def _vec_key(node, key):
    try:
        return tuple(float(c) for c in (get1(node, key) or '').split()[:3])
    except ValueError:
        return None


def sample_terrain_height(records, x, y, hint_z, max_dz=256.0):
    """Height (world Z) of the welded LWO terrain at (x, y), closest to
    hint_z among any records whose surface passes near there, or None if
    none do. Used to make overlay geometry follow the actual terrain
    surface (see convert_overlay) instead of sitting on a flat plane.

    Works against the RAW per-displacement records in
    LWOTerrainBuilder.records (each one's own 'pos' grid, Source's own
    triangulation) rather than the final welded mesh, since overlays are
    converted before finalize_points() runs - the raw grids already have
    the real surface shape, welding only affects which points are shared.
    """
    best = None
    for rec in records:
        n = rec['n']
        pos = rec['pos']
        border = [pos[i][j] for i in range(n) for j in (0, n - 1)] + \
                 [pos[i][j] for i in (0, n - 1) for j in range(n)]
        xs = [p[0] for p in border]
        ys = [p[1] for p in border]
        if x < min(xs) - 1 or x > max(xs) + 1 or y < min(ys) - 1 or y > max(ys) + 1:
            continue
        for i in range(n - 1):
            for j in range(n - 1):
                for tri in (((i, j), (i + 1, j), (i, j + 1)), ((i, j + 1), (i + 1, j), (i + 1, j + 1))):
                    A, B, C = (pos[a][b] for a, b in tri)
                    den = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
                    if abs(den) < 1e-9:
                        continue
                    l1 = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / den
                    l2 = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / den
                    l3 = 1 - l1 - l2
                    if min(l1, l2, l3) < -1e-6:
                        continue
                    z = l1 * A[2] + l2 * B[2] + l3 * C[2]
                    if abs(z - hint_z) > max_dz:
                        continue
                    if best is None or abs(z - hint_z) < abs(best - hint_z):
                        best = z
    return best



def split_oversized_mesh_surfaces(surfaces, max_triangles=LWO_MAX_TRIANGLES_PER_GROUP,
                                  max_verts=None):
    """Same purpose as split_oversized_shader_groups, for write_lwo2_mesh's
    different shape: {name: (points, uvs, triangles)}, each with its OWN
    local points/uvs (not a shared global list). Splitting here means
    re-indexing each chunk's own subset of points down to a fresh local
    0..n range, not just slicing the triangle list.

    max_verts, if given, caps by re-indexed vertex count per chunk instead
    of triangle count - see split_shader_groups_by_vert_count's own comment
    for why this matters specifically for model_static.

    Returns ({new_name: (points, uvs, triangles)}, {new_name: original_name}).
    """
    out, origin = {}, {}
    for name, (points, uvs, triangles) in surfaces.items():
        if max_verts is None and len(triangles) <= max_triangles:
            out[name] = (points, uvs, triangles)
            origin[name] = name
            continue
        if max_verts is not None and len(points) <= max_verts:
            out[name] = (points, uvs, triangles)
            origin[name] = name
            continue
        chunk_i, remap, new_points, new_uvs, new_tris = 0, {}, [], [], []
        for tri in triangles:
            if max_verts is not None:
                new_ids = [idx for idx in tri if idx not in remap]
                if len(new_points) + len(new_ids) > max_verts and new_tris:
                    new_name = name if chunk_i == 0 else '{}_{}'.format(name, chunk_i)
                    out[new_name] = (new_points, new_uvs, new_tris)
                    origin[new_name] = name
                    chunk_i += 1
                    remap, new_points, new_uvs, new_tris = {}, [], [], []
            elif len(new_tris) >= max_triangles:
                new_name = name if chunk_i == 0 else '{}_{}'.format(name, chunk_i)
                out[new_name] = (new_points, new_uvs, new_tris)
                origin[new_name] = name
                chunk_i += 1
                remap, new_points, new_uvs, new_tris = {}, [], [], []
            new_tri = []
            for idx in tri:
                if idx not in remap:
                    remap[idx] = len(new_points)
                    new_points.append(points[idx])
                    new_uvs.append(uvs[idx])
                new_tri.append(remap[idx])
            new_tris.append(tuple(new_tri))
        if new_tris:
            new_name = name if chunk_i == 0 else '{}_{}'.format(name, chunk_i)
            out[new_name] = (new_points, new_uvs, new_tris)
            origin[new_name] = name
    return out, origin


def write_lwo2_mesh(surfaces, for_model_static=False):
    """Plain textured mesh (no vertex-alpha welding) - each named surface
    is (points, uvs, triangles): points/uvs are parallel per-point lists,
    triangles is a list of 3-point-index tuples. Used for terrain-
    conforming overlay meshes (see build_overlay_conforming_mesh) - unlike
    write_lwo2_terrain, UV is a plain point-level VMAP (no VMAD needed:
    each mesh is a single overlay's own geometry, nothing is shared/welded
    across different overlays, so there's no seam to keep distinct UV
    across).

    Splits any surface over LWO_MAX_TRIANGLES_PER_GROUP first (see
    split_oversized_mesh_surfaces - a real, confirmed RB_CheckOverflow
    crash otherwise on a popular overlay material). Returns
    (bytes, origin_mapping), same meaning as write_lwo2_terrain's own.

    for_model_static=True switches to LWO_MAX_VERTS_PER_GROUP_STATIC
    (vertex-count cap) instead - see write_lwo2_terrain's own comment.
    """
    if for_model_static:
        surfaces, origin_mapping = split_oversized_mesh_surfaces(
            surfaces, max_verts=LWO_MAX_VERTS_PER_GROUP_STATIC)
    else:
        surfaces, origin_mapping = split_oversized_mesh_surfaces(surfaces)
    shader_names = sorted(surfaces.keys())

    all_points = []
    point_uv = []
    surf_tris = {}
    offset = 0
    for name in shader_names:
        pts, uvs, tris = surfaces[name]
        all_points.extend(pts)
        point_uv.extend(uvs)
        surf_tris[name] = [[i + offset for i in tri] for tri in tris]
        offset += len(pts)

    layr_data = (struct.pack('>HH', 0, 0) + struct.pack('>fff', 0.0, 0.0, 0.0) +
                 _lwo_s0("overlays"))
    layr_chunk = _lwo_chunk('LAYR', layr_data)

    pnts_data = b''.join(struct.pack('>fff', p[0], p[2], p[1]) for p in all_points)
    pnts_chunk = _lwo_chunk('PNTS', pnts_data)

    # V is written INVERTED on purpose. This mesh is emitted as a misc_model,
    # so q3map2 - not the engine - is what reads it, and q3map2 loads every
    # model through Assimp with aiProcess_FlipUVs in its post-process flags
    # (tools/quake3/q3map2/model.cpp, the flag list around line 320), which
    # rewrites every texture coordinate's V to 1 - V on import. Writing 1 - V
    # here cancels that out exactly.
    #
    # Measured before the source was consulted, on the toolchain that actually
    # compiles these maps (q3map2 2.5.17n-git, the version string the test
    # .bsp's worldspawn carries): matching 144 overlay vertices between a
    # shipped de_overpass .lwo and the .bsp q3map2 produced from it, 129 came
    # out with t = 1 - v and 2 unchanged, the rest sitting on v = 0.5 where the
    # flip is invisible. That is what put decals, signage and painted markings
    # the right way up.
    #
    # NOTE, still open: aiProcess_FlipUVs is not LWO-specific - it applies to
    # every format Assimp loads, .md3 included. So a prop .md3 that q3map2
    # BAKES (referenced by a misc_model) is flipped too, while the same file
    # loaded at runtime as a model_static is not. A real de_overpass .bsp does
    # contain 140 baked prop materials, so this is not hypothetical. It is not
    # fixed here because a baked prop and a model_static prop can share one
    # .md3, so the fix is a separate pre-flipped copy for the baked reference,
    # not a change to the file itself - and the measurement that would confirm
    # it is inconclusive on props (they are instanced and vertex-welded, so
    # their UVs do not correspond 1:1 to the source file the way these unique
    # overlay meshes do). Do not "fix" it by flipping the .md3 writer: that
    # would invert every runtime prop.
    vmap_data = b'TXUV' + struct.pack('>H', 2) + _lwo_s0("UVMap")
    for i, uv in enumerate(point_uv):
        vmap_data += _lwo_vx(i) + struct.pack('>ff', uv[0], 1.0 - uv[1])
    vmap_chunk = _lwo_chunk('VMAP', vmap_data)

    shader_tag_index = {name: i for i, name in enumerate(shader_names)}
    pols_data = b'FACE'
    ptag_entries = []
    poly_i = 0
    for name in shader_names:
        tag_idx = shader_tag_index[name]
        for tri in surf_tris[name]:
            pols_data += struct.pack('>H', 3)
            for idx in tri:
                pols_data += _lwo_vx(idx)
            ptag_entries.append((poly_i, tag_idx))
            poly_i += 1
    pols_chunk = _lwo_chunk('POLS', pols_data)

    # short_qpath: the tag is the shader name (+ -rename's _BSP), 64 bytes in idTech3
    tags_data = b''.join(_lwo_s0(short_qpath(name)) for name in shader_names)
    tags_chunk = _lwo_chunk('TAGS', tags_data)

    ptag_data = b'SURF'
    for pidx, tidx in ptag_entries:
        ptag_data += _lwo_vx(pidx) + struct.pack('>H', tidx)
    ptag_chunk = _lwo_chunk('PTAG', ptag_data)

    surf_chunks = b''
    for name in shader_names:
        colr_sub = _lwo_subchunk('COLR', struct.pack('>fff', 1.0, 1.0, 1.0) + struct.pack('>H', 0))
        side_sub = _lwo_subchunk('SIDE', struct.pack('>H', 3))
        surf_data = _lwo_s0(short_qpath(name)) + _lwo_s0("") + colr_sub + side_sub
        surf_chunks += _lwo_chunk('SURF', surf_data)

    body = layr_chunk + pnts_chunk + vmap_chunk + pols_chunk + tags_chunk + ptag_chunk + surf_chunks
    form_data = b'LWO2' + body
    return b'FORM' + struct.pack('>I', len(form_data)) + form_data, origin_mapping


# side id -> that face's polygon in SOURCE units, for clipping info_overlays to
# the faces they actually belong to (see clip_overlay_uv_to_sides). Rebuilt per
# conversion run in convert_vmf.
OVERLAY_SIDE_POLYS = {}
# Overlays built round several non-coplanar brush faces this run (reported).
WRAPPED_OVERLAYS = [0]
# Brush side ids that carry a dispinfo - i.e. faces that are really a
# displacement surface, not the flat face their plane describes.
DISP_SIDE_IDS = set()


def build_side_polygon_index(root):
    """Map every brush side's id to its face polygon (source units).

    Needed because an info_overlay in Source is not a free-floating quad: it
    carries a 'sides' key naming the exact faces it is projected onto, and
    VBSP CLIPS the decal to those faces. Without that step a decal whose uv
    rectangle is bigger than its wall keeps going past the edge, and one whose
    target face no longer exists floats in mid-air with nothing behind it -
    both reported on de_dust2 ("longer than the building it's sitting
    against", "floating ... with some wire texture on it")."""
    index = {}
    global ALL_FACE_INDEX
    ALL_FACE_INDEX = []

    def scan(solid):
        sides = solid.get('side', [])
        for side in sides:
            if not isinstance(side, dict):
                continue
            poly = get_face_polygon(side, sides)
            if not poly or len(poly) < 3:
                continue
            sid = get1(side, 'id')
            if sid is not None and str(sid) not in index:
                index[str(sid)] = poly
            if sid is not None and side.get('dispinfo'):
                DISP_SIDE_IDS.add(str(sid))
            # Outward normal from the face's own plane definition, used to
            # project infodecals onto the surface they rest against.
            plane_str = get1(side, 'plane')
            if not plane_str:
                continue
            tri = parse_plane(plane_str)
            if len(tri) != 3:
                continue
            nrm = cross(sub(tri[0], tri[1]), sub(tri[2], tri[1]))
            if length(nrm) < 1e-9:
                continue
            ALL_FACE_INDEX.append((poly, side, normalize(nrm)))

    worlds = root.get('world', [])
    global WORLD_BRUSH_PLANES
    WORLD_BRUSH_PLANES = []
    if worlds:
        for solid in get_solid_blocks(worlds[0]):
            scan(solid)
            # structural world brushes as (outward normal, dist) lists, for
            # origin_in_world_brush()
            planes, mats = [], set()
            for side in solid.get('side', []):
                if not isinstance(side, dict):
                    continue
                tri = parse_plane(get1(side, 'plane') or '')
                if len(tri) != 3:
                    continue
                nrm = cross(sub(tri[0], tri[1]), sub(tri[2], tri[1]))
                if length(nrm) < 1e-9:
                    continue
                nrm = normalize(nrm)
                planes.append((nrm, dot(nrm, tri[0])))
                mats.add((get1(side, 'material', '') or '').upper())
            if planes and not mats <= _NONSOLID_TOOL_MATERIALS:
                WORLD_BRUSH_PLANES.append(planes)
    for entity in root.get('entity', []):
        for solid in get_solid_blocks(entity):
            scan(solid)
    return index


WORLD_BRUSH_PLANES = []
# q3map2 can load at most 512 distinct model files for baking (MAX_MODELS);
# below this many unique models every prop is baked as misc_model.
MODEL_STATIC_ONLY_ABOVE = 400
_NONSOLID_TOOL_MATERIALS = {
    'TOOLS/TOOLSHINT', 'TOOLS/TOOLSSKIP', 'TOOLS/TOOLSTRIGGER', 'TOOLS/TOOLSCLIP',
    'TOOLS/TOOLSPLAYERCLIP', 'TOOLS/TOOLSNPCCLIP', 'TOOLS/TOOLSMONSTERCLIP',
    'TOOLS/TOOLSAREAPORTAL', 'TOOLS/TOOLSBLOCK_LOS', 'TOOLS/TOOLSOCCLUDER',
    'TOOLS/TOOLSBLOCKBULLETS', 'TOOLS/TOOLSINVISIBLELADDER', 'TOOLS/TOOLSFOG',
}


def world_brush_penetration(o, eps=0.25):
    """If the point (Source units) is inside - or within eps of the surface
    of - a structural world brush, return (depth, outward normal of the
    nearest face), depth = how far inside (negative if just outside). Such a
    point can compile into a solid leaf. None when it's clear of every brush."""
    best = None
    for planes in WORLD_BRUSH_PLANES:
        vals = [(dot(n, o) - d, n) for n, d in planes]
        top = max(vals, key=lambda v: v[0])
        if top[0] < eps:
            depth = -top[0]
            if best is None or depth > best[0]:
                best = (depth, top[1])
    return best


def _clip_overlay_tapered(uvs, us, vs, origin, bu, bv, sides_key):
    """Clip a TAPERED overlay quad (exactly two parallel edges) to its target
    faces along its uniform axis, keeping the taper.

    Returns (uvs, tex_fractions) like clip_overlay_uv_to_sides, or that same
    input unchanged when the shape isn't one this can handle."""
    if len(vs) == 2 and len(us) != 2:
        rail = 1          # v is uniform: clip along v
    elif len(us) == 2 and len(vs) != 2:
        rail = 0          # u is uniform: clip along u
    else:
        return uvs, None  # neither axis uniform - not a shape we can clip safely

    rail_vals = sorted({round(uv[rail], 4) for uv in uvs})
    lo_val, hi_val = rail_vals[0], rail_vals[-1]
    hi_idx = [i for i in range(4) if abs(uvs[i][rail] - hi_val) < 0.01]
    lo_idx = [i for i in range(4) if abs(uvs[i][rail] - lo_val) < 0.01]
    if len(hi_idx) != 2 or len(lo_idx) != 2:
        return uvs, None

    # Which texture parameter runs along the rail? Corners 0,1 share StartU and
    # 2,3 share EndU; corners 0,3 share StartV and 1,2 share EndV.
    if set(hi_idx) in ({0, 1}, {2, 3}):
        tex_axis = 0      # rail maps to the texture's U range
        partner = {0: 3, 1: 2, 2: 1, 3: 0}
    elif set(hi_idx) in ({0, 3}, {1, 2}):
        tex_axis = 1      # rail maps to the texture's V range
        partner = {0: 1, 1: 0, 2: 3, 3: 2}
    else:
        return uvs, None

    t_lo, t_hi = float('inf'), float('-inf')
    found = 0
    for raw in str(sides_key).split():
        poly = OVERLAY_SIDE_POLYS.get(raw.strip())
        if not poly:
            continue
        found += 1
        for pt in poly:
            rel = sub(pt, origin)
            val = dot(rel, bu) if rail == 0 else dot(rel, bv)
            t_lo = min(t_lo, val)
            t_hi = max(t_hi, val)
    if not found:
        return uvs, None

    c_lo, c_hi = max(lo_val, t_lo), min(hi_val, t_hi)
    if c_hi - c_lo <= 0.01:
        return None
    if abs(c_lo - lo_val) < 0.01 and abs(c_hi - hi_val) < 0.01:
        return uvs, None

    span = (hi_val - lo_val) or 1.0
    new_uvs = list(uvs)
    for i in hi_idx + lo_idx:
        target = c_hi if i in hi_idx else c_lo
        f = (target - lo_val) / span
        j = partner[i]
        other = 1 - rail
        # interpolate the tapering axis along this corner's own edge
        lo_pt = uvs[i] if i in lo_idx else uvs[j]
        hi_pt = uvs[i] if i in hi_idx else uvs[j]
        val_other = lo_pt[other] + (hi_pt[other] - lo_pt[other]) * f
        new_uvs[i] = (target, val_other) if rail == 0 else (val_other, target)

    f_lo = (c_lo - lo_val) / span
    f_hi = (c_hi - lo_val) / span
    # tex fractions are ordered (u_lo, u_hi, v_lo, v_hi); the rail maps onto
    # whichever texture axis tex_axis names, the other keeps its full range.
    lo_is_start = set(hi_idx) in ({0, 1}, {0, 3})
    a, b = (1.0 - f_hi, 1.0 - f_lo) if lo_is_start else (f_lo, f_hi)
    if tex_axis == 0:
        frac = (a, b, 0.0, 1.0)
    else:
        frac = (0.0, 1.0, a, b)
    return new_uvs, frac


def clip_overlay_uv_to_sides(entity, origin, bu, bv, uvs, warnings, material=''):
    """Clip an info_overlay's uv rectangle to the faces its 'sides' key names.

    Returns (uvs, (u_lo, u_hi, v_lo, v_hi) texture fractions) where the
    fractions say which sub-rectangle of the ORIGINAL decal image survives, so
    the caller can crop the texture coordinates instead of squashing the whole
    image into a smaller quad. Returns (uvs, None) when no clipping applies,
    or None when the decal does not overlap its target faces at all and should
    be dropped.

    Only axis-aligned uv rectangles are handled - which is what Source
    actually writes (verified across de_dust2's 55 overlays: every one is
    (-a,-b), (-a,b), (a,b), (a,-b)). Anything else is left alone rather than
    guessed at."""
    if not OVERLAY_SIDE_POLYS:
        return uvs, None
    sides_key = get1(entity, 'sides')
    if not sides_key or not str(sides_key).split():
        # An info_overlay with no target faces projects onto NOTHING in
        # Source, so it renders nothing there - VBSP has no surface to clip
        # it against. Emitting it anyway leaves a decal hanging in open air.
        # Measured on de_dust2: exactly one overlay is like this, a
        # 'decals/wires02' at source z=746 with an empty 'sides' key, and it
        # is the floating wire visible high in the sky in game. Every other
        # overlay on that map names its faces and sits 0.0 units from them.
        return None
    us = sorted({round(uv[0], 4) for uv in uvs})
    vs = sorted({round(uv[1], 4) for uv in uvs})
    if len(us) != 2 or len(vs) != 2:
        # Not a plain rectangle. A TAPERED quad (two parallel edges, the other
        # two converging) is common and must still be clipped: de_dust2's
        # hanging cable decals are trapezoids - uv corners like
        # (8,128) (-4,128) (-8,-128) (4,-128), 16 wide by 256 long - and one
        # was reported in game as far too big for the wall it hangs on.
        # Bailing out here left them at full length. Clip along the axis that
        # IS uniform (the long "rail"), interpolating the other axis so the
        # taper is preserved exactly.
        return _clip_overlay_tapered(uvs, us, vs, origin, bu, bv, sides_key)
    u_min, u_max = us
    v_min, v_max = vs

    t_u_min = t_v_min = float('inf')
    t_u_max = t_v_max = float('-inf')
    found = 0
    for raw in str(sides_key).split():
        poly = OVERLAY_SIDE_POLYS.get(raw.strip())
        if not poly:
            continue
        found += 1
        for pt in poly:
            rel = sub(pt, origin)
            u = dot(rel, bu)
            v = dot(rel, bv)
            t_u_min = min(t_u_min, u)
            t_u_max = max(t_u_max, u)
            t_v_min = min(t_v_min, v)
            t_v_max = max(t_v_max, v)
    if not found:
        warnings.append(
            "info_overlay ('{}') names side(s) '{}' that no longer exist in the converted map, so "
            "it couldn't be clipped to them - it may hang in the air where its target surface used "
            "to be.".format(material, sides_key))
        return uvs, None

    c_u_min, c_u_max = max(u_min, t_u_min), min(u_max, t_u_max)
    c_v_min, c_v_max = max(v_min, t_v_min), min(v_max, t_v_max)
    if c_u_max - c_u_min <= 0.01 or c_v_max - c_v_min <= 0.01:
        return None  # no overlap with its own target faces at all

    if (abs(c_u_min - u_min) < 0.01 and abs(c_u_max - u_max) < 0.01 and
            abs(c_v_min - v_min) < 0.01 and abs(c_v_max - v_max) < 0.01):
        return uvs, None  # already inside its faces; nothing to do

    du = (u_max - u_min) or 1.0
    dv = (v_max - v_min) or 1.0
    frac = ((c_u_min - u_min) / du, (c_u_max - u_min) / du,
            (c_v_min - v_min) / dv, (c_v_max - v_min) / dv)
    new_uvs = [(c_u_min, c_v_min), (c_u_min, c_v_max), (c_u_max, c_v_max), (c_u_max, c_v_min)]
    return new_uvs, frac


# Every brush face as (polygon, side dict, outward normal) in SOURCE units,
# for projecting infodecals onto the surface they belong to.
ALL_FACE_INDEX = []

_DECAL_SCALE_CACHE = {}


def read_vmt_decal_scale(material, materials_dir, default=1.0):
    """'$decalscale' from a decal's .vmt: world units per texel, which is what
    turns a decal texture's pixel size into its size in the map. Returns
    `default` when there's no materials folder, no .vmt, or no such key."""
    key = material.strip().lower().replace('\\', '/')
    if key in _DECAL_SCALE_CACHE:
        return _DECAL_SCALE_CACHE[key]
    value = default
    if materials_dir:
        vmt_path = os.path.join(materials_dir, *key.split('/')) + '.vmt'
        if os.path.isfile(vmt_path):
            try:
                with open(vmt_path, 'r', encoding='utf-8', errors='replace') as f:
                    root = parse_vdf(tokenize(f.read()))
                if root:
                    body = root[list(root.keys())[0]][0]
                    for k in list(body.keys()):
                        if k.strip().lower().lstrip('$') == 'decalscale':
                            try:
                                value = float(get1(body, k))
                            except (TypeError, ValueError):
                                pass
                            break
            except (OSError, IndexError, KeyError):
                pass
    _DECAL_SCALE_CACHE[key] = value
    return value


def convert_infodecal(entity, scale, offset, texture_map, unmapped_counter, warnings,
                      materials_dir=None, max_dist=24.0):
    """Source 'infodecal' -> a small patchDef2 laid onto the face it sits on.

    An infodecal is NOT an info_overlay: it carries only an origin and a
    texture, no basis and no uv rectangle. VBSP finds the face it is touching,
    orients the decal along that face's own texture axes, and stamps it at its
    texture's natural size (pixels x $decalscale). That's reproduced here -
    nearest face within `max_dist`, oriented by the face's uaxis/vaxis, sized
    from --texture-sizes and the material's own $decalscale, then clipped to
    the face so it can't hang over an edge.

    These were previously dropped wholesale (infodecal is in
    DEFAULT_SKIP_CLASSES), which on de_dust2 silently lost 135 decals - 33
    'win_square' and 16 'win_rectang' among them, reported missing in game.

    Returns [patch entity text] or [] if it can't be placed."""
    origin = _vec_key(entity, 'origin')
    material = get1(entity, 'texture') or get1(entity, 'material')
    if not origin or not material:
        return []
    if not ALL_FACE_INDEX:
        return []

    best = None
    for poly, side, n in ALL_FACE_INDEX:
        dist = dot(sub(origin, poly[0]), n)
        if abs(dist) > max_dist:
            continue
        # project onto the plane and require the hit to be inside the face,
        # so a decal isn't attached to a distant parallel wall
        proj = sub(origin, scale_vec(n, dist))
        if not point_in_polygon_3d(proj, poly, n):
            continue
        if best is None or abs(dist) < abs(best[0]):
            best = (dist, proj, poly, side, n)
    if best is None:
        warnings.append(
            "infodecal '{}' at {} isn't touching any brush face (searched {:.0f} units), so it was "
            "dropped - in Source it would be projected onto whatever it rests against.".format(
                material, tuple(round(c, 1) for c in origin), max_dist))
        return []
    _dist, centre, poly, side, n = best

    # Orientation: Source's static-decal basis (engine R_DecalComputeBasis with
    # no s-axis, which is how map infodecals are shot), NOT the face's texture
    # axes. Hammer's world-aligned texture axes are identical on opposite-
    # facing walls, so building the decal from them mirrored it on every wall
    # facing the "wrong" way - the backwards B's on B site. Source instead
    # uses: wall -> T (down) = -Z, S = N x T; floor/ceiling (|N.z| > sin 45)
    # -> S = +X, T = S x N, S = N x T. S runs along the texture's s (right),
    # T along t (down), so the decal reads the right way round from the front.
    if abs(n[2]) > 0.70710678:
        t_dir = cross((1.0, 0.0, 0.0), n)
        u_dir = cross(n, t_dir)
    else:
        t_dir = (0.0, 0.0, -1.0)
        u_dir = cross(n, t_dir)
        t_dir = cross(u_dir, n)
    u_dir, v_dir = normalize(u_dir), normalize(t_dir)
    if length(u_dir) < 1e-6 or length(v_dir) < 1e-6:
        return []

    px = TEXTURE_PIXEL_SIZES.get(material.strip().lower().replace('\\', '/'))
    guessed = px is None
    if guessed:
        px = (64, 64)
    dscale = read_vmt_decal_scale(material, materials_dir)
    hw = px[0] * dscale * 0.5
    hh = px[1] * dscale * 0.5

    # Clip the stamp to the surface, in the decal's own axes. "The surface" is
    # every visible face in the same plane that the stamp overlaps, not just
    # the one face under its centre: Source projects a static decal onto all
    # of them, and walls are routinely split into several brushes, so clipping
    # to the one face cut decals off at every brush seam (reported: "some
    # decals are cut off").
    plane_d = dot(poly[0], n)
    us, vs = [], []
    for poly2, side2, n2 in ALL_FACE_INDEX:
        if dot(n2, n) < 0.999 or abs(dot(poly2[0], n) - plane_d) > 0.5:
            continue
        if poly2 is not poly and (get1(side2, 'material', '') or '').upper().startswith('TOOLS/'):
            continue
        u2 = [dot(sub(pt, centre), u_dir) for pt in poly2]
        v2 = [dot(sub(pt, centre), v_dir) for pt in poly2]
        if max(u2) < -hw or min(u2) > hw or max(v2) < -hh or min(v2) > hh:
            continue
        us.extend(u2)
        vs.extend(v2)
    u_lo, u_hi = max(-hw, min(us)), min(hw, max(us))
    v_lo, v_hi = max(-hh, min(vs)), min(hh, max(vs))
    if u_hi - u_lo <= 0.01 or v_hi - v_lo <= 0.01:
        return []
    # Texture coordinates crop with the geometry rather than squashing.
    t_u0, t_u1 = (u_lo + hw) / (2 * hw), (u_hi + hw) / (2 * hw)
    t_v0, t_v1 = (v_lo + hh) / (2 * hh), (v_hi + hh) / (2 * hh)

    def world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])

    lifted = add(centre, scale_vec(n, 0.4))
    # Sized like convert_overlay's patches (DECAL_VERTEX_LIGHT_SPACING): an
    # infodecal is vertex lit too. u/v here are source units, hence * scale.
    GRID_A = decal_patch_side((u_hi - u_lo) * scale)
    GRID_B = decal_patch_side((v_hi - v_lo) * scale)
    # Same facing rule as _face_patch_along: patch normal = cross(column step,
    # row step) = cross(u_dir, v_dir) here, and it has to point away from the
    # wall (n) or q3map2 lights the decal from inside the brush.
    b_order = list(range(GRID_B))
    if dot(cross(u_dir, v_dir), n) < 0:
        b_order.reverse()
    rows = []
    for a in range(GRID_A):
        fa = a / float(GRID_A - 1)
        u = u_lo + (u_hi - u_lo) * fa
        tu = t_u0 + (t_u1 - t_u0) * fa
        cols = []
        for b in b_order:
            fb = b / float(GRID_B - 1)
            v = v_lo + (v_hi - v_lo) * fb
            tv = t_v0 + (t_v1 - t_v0) * fb
            pt = world(add(lifted, add(scale_vec(u_dir, u), scale_vec(v_dir, v))))
            cols.append("( {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} )".format(
                pt[0], pt[1], pt[2], tu, tv))
        rows.append("( " + " ".join(cols) + " )")

    mapped = decal_texture_name(material, convert_texture_name(material, texture_map,
                                                                unmapped_counter))
    OVERLAY_MATERIALS.add(material.strip().lower().replace('\\', '/'))
    if guessed:
        warnings.append(
            "infodecal '{}' has no entry in --texture-sizes, so its size was guessed at 64x64 "
            "pixels - pass the sizes JSON from build_texture_sizes.py for the real dimensions."
            .format(material))
    patch = "\n".join(['{', 'patchDef2', '{', mapped,
                       '( %d %d 0 0 0 )' % (GRID_A, GRID_B), '('] + rows + [')', '}', '}'])
    return [_wrap_patch_as_func_group(patch)]


def point_in_polygon_3d(p, poly, n):
    """Is p (already on the polygon's plane) inside the polygon? Projects onto
    whichever axis pair the face is most square to, then does a standard
    crossing test."""
    ax = max(range(3), key=lambda i: abs(n[i]))
    i1, i2 = [i for i in range(3) if i != ax]
    x, y = p[i1], p[i2]
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i][i1], poly[i][i2]
        xj, yj = poly[j][i1], poly[j][i2]
        if (yi > y) != (yj > y):
            xint = xi + (y - yi) / (yj - yi) * (xj - xi)
            if x < xint:
                inside = not inside
        j = i
    return inside


def _clip_poly_2d(subject, clip):
    """Sutherland-Hodgman: clip a polygon against a CONVEX polygon, both as
    lists of (x, y) wound counter-clockwise. Returns the clipped polygon
    (possibly empty)."""
    out = list(subject)
    for i in range(len(clip)):
        if not out:
            return []
        a, b = clip[i], clip[(i + 1) % len(clip)]
        ex, ey = b[0] - a[0], b[1] - a[1]

        def inside(p):
            return (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) >= -1e-9

        src = out
        out = []
        prev = src[-1]
        prev_in = inside(prev)
        for cur in src:
            cur_in = inside(cur)
            if cur_in != prev_in:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                den = ex * dy - ey * dx
                if abs(den) > 1e-12:
                    t = (ex * (prev[1] - a[1]) - ey * (prev[0] - a[0])) / -den
                    out.append((prev[0] + dx * t, prev[1] + dy * t))
            if cur_in:
                out.append(cur)
            prev, prev_in = cur, cur_in
    return out


def _inverse_bilinear(q, a, b, c, d):
    """Parameters (s, t) such that
        q = a(1-s)(1-t) + b*s(1-t) + c(1-s)t + d*s*t
    for 2D points - i.e. the inverse of the bilinear corner interpolation the
    overlay uses to place its texture. Needed because the overlay quad may be
    a TRAPEZOID (19 of de_dust2's 55 overlays are), so a point's texture
    coordinate can't be recovered by a simple axis ratio. Returns (s, t),
    clamped, or None if degenerate."""
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    e = (b[0] - a[0], b[1] - a[1])
    f = (c[0] - a[0], c[1] - a[1])
    g = (a[0] - b[0] - c[0] + d[0], a[1] - b[1] - c[1] + d[1])
    h = (q[0] - a[0], q[1] - a[1])

    k2 = cross(g, f)
    k1 = cross(e, f) + cross(h, g)
    k0 = cross(h, e)

    # k2 t^2 + k1 t + k0 = 0, solved in the cancellation-free form. k2 is
    # exactly 0 only for an exact parallelogram; a rotated parallelogram (a
    # Source overlay whose basis is turned, like de_overpass_csgo's
    # infwalldetail09 at 9.5 degrees) leaves k2 at float noise, and the
    # textbook (-k1 +- root) / 2k2 then divides two near-equal numbers by
    # almost nothing: t came out wild, was clamped to 0, and every point of
    # the decal got the texture's first row - the stretched streaks.
    disc = k1 * k1 - 4.0 * k0 * k2
    if disc < 0:
        if disc > -1e-9 * k1 * k1:
            disc = 0.0
        else:
            return None
    root = math.sqrt(disc)
    qq = -0.5 * (k1 + (root if k1 >= 0 else -root))
    cands = []
    if abs(qq) > 1e-300:
        cands.append(k0 / qq)
        if abs(k2) > 1e-12 * abs(qq):
            cands.append(qq / k2)
    elif abs(k2) > 1e-300:
        cands.append(0.0)
    if not cands:
        return None
    inside = [x for x in cands if -0.01 <= x <= 1.01]
    t = inside[0] if inside else min(cands, key=lambda x: min(abs(x), abs(x - 1.0)))
    den_x = e[0] + g[0] * t
    den_y = e[1] + g[1] * t
    if abs(den_x) >= abs(den_y):
        if abs(den_x) < 1e-12:
            return None
        sp = (h[0] - f[0] * t) / den_x
    else:
        if abs(den_y) < 1e-12:
            return None
        sp = (h[1] - f[1] * t) / den_y
    return max(0.0, min(1.0, sp)), max(0.0, min(1.0, t))


def iter_terrain_triangles(records, side_ids=None):
    """Every terrain triangle in world space, using the SAME triangulation the
    terrain model itself is built from (LWOTerrainBuilder._source_triangles,
    which follows Source's own alternating diagonal). Yields (p0, p1, p2).
    side_ids: only the displacements made from these Source side ids."""
    for rec in records:
        if side_ids is not None and rec.get('side_id') not in side_ids:
            continue
        n = rec['n']
        pos = rec['pos']
        for tri in LWOTerrainBuilder._source_triangles(n):
            yield tuple(pos[i][j] for i, j in tri)


# How far (post-scale units, along the overlay's normal) a displacement
# triangle may lie from the overlay's own plane and still be "the surface it
# was painted on". Source's displacement offsets on mirage's walls stay well
# inside this; a wall a room away is far outside it.
OVERLAY_WALL_MAX_DEPTH = 48.0


def _build_overlay_wall_mesh(entity, scale, offset, records, max_points, disp_sides=None,
                             tris=None, depth_filter=True, face_lift=False, min_facing=0.25):
    """build_overlay_conforming_mesh for a wall decal on a displacement wall.

    Same method as the ground case - clip the displacement's own triangles to
    the overlay quad, so the decal is an exact copy of that surface lifted
    clear of it - but done in the overlay's OWN frame (BasisU, BasisV,
    BasisNormal) instead of plan view, since plan view has no solution for a
    vertical surface. The lift is along BasisNormal (one constant vector, so
    shared vertices stay identical and the mesh welds)."""
    origin = _vec_key(entity, 'BasisOrigin') or _vec_key(entity, 'origin')
    bu, bv, bn = _vec_key(entity, 'BasisU'), _vec_key(entity, 'BasisV'), _vec_key(entity, 'BasisNormal')
    uvs_key = [_vec_key(entity, 'uv%d' % i) for i in range(4)]
    material = get1(entity, 'material')
    if not (origin and bu and bv and bn and material and all(uvs_key)):
        return None
    try:
        u0, u1 = float(get1(entity, 'StartU', '0')), float(get1(entity, 'EndU', '1'))
        v0, v1 = float(get1(entity, 'StartV', '0')), float(get1(entity, 'EndV', '1'))
    except ValueError:
        return None
    n = normalize(bn)
    _clipped = clip_overlay_uv_to_sides(entity, origin, bu, bv, uvs_key, [], material)
    if _clipped is None:
        return None
    uvs_key, _frac = _clipped
    if _frac:
        _fu0, _fu1, _fv0, _fv1 = _frac
        _du, _dv = (u1 - u0), (v1 - v0)
        u0, u1 = u0 + _fu0 * _du, u0 + _fu1 * _du
        v0, v1 = v0 + _fv0 * _dv, v0 + _fv1 * _dv

    # Orthonormal frame with n as "up". The uniform scale and the recentre
    # offset don't change directions, so the Source basis is valid as is.
    e1 = sub(bu, scale_vec(n, dot(bu, n)))
    if length(e1) < 1e-9:
        return None
    e1 = normalize(e1)
    e2 = cross(n, e1)

    def world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])

    def local(p):
        return (dot(p, e1), dot(p, e2), dot(p, n))

    corners = [world(add(origin, add(scale_vec(bu, uv[0]), scale_vec(bv, uv[1])))) for uv in uvs_key]
    tex = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    lc = [local(c) for c in corners]
    depth0 = lc[0][2]
    quad2d = [(c[0], c[1]) for c in lc]
    area2 = sum(quad2d[i][0] * quad2d[(i + 1) % 4][1] - quad2d[(i + 1) % 4][0] * quad2d[i][1]
                for i in range(4))
    if abs(area2) < 1e-6:
        return None
    clip_poly = quad2d[::-1] if area2 < 0 else quad2d
    qx0 = min(p[0] for p in quad2d) - 1.0
    qx1 = max(p[0] for p in quad2d) + 1.0
    qy0 = min(p[1] for p in quad2d) - 1.0
    qy1 = max(p[1] for p in quad2d) + 1.0
    # corner order for _inverse_bilinear: a=uv0, b=uv3, c=uv1, d=uv2
    ca, cb, cc, cd = quad2d[0], quad2d[3], quad2d[1], quad2d[2]
    lift = scale_vec(n, OVERLAY_MESH_LIFT)

    points, uv_list, triangles = [], [], []
    index_of = {}

    def add_point(p3, uv):
        key = (round(p3[0], 3), round(p3[1], 3), round(p3[2], 3))
        idx = index_of.get(key)
        if idx is None:
            idx = len(points)
            index_of[key] = idx
            points.append(p3)
            uv_list.append(uv)
        return idx

    for tri in (tris if tris is not None else iter_terrain_triangles(records, disp_sides)):
        lt = [local(p) for p in tri]
        if (max(p[0] for p in lt) < qx0 or min(p[0] for p in lt) > qx1 or
                max(p[1] for p in lt) < qy0 or min(p[1] for p in lt) > qy1):
            continue
        if depth_filter and min(abs(p[2] - depth0) for p in lt) > OVERLAY_WALL_MAX_DEPTH:
            continue
        nrm = cross(sub(lt[1], lt[0]), sub(lt[2], lt[0]))
        nl = length(nrm)
        if nl < 1e-9:
            continue
        nrm = scale_vec(nrm, 1.0 / nl)
        if nrm[2] < 0:
            nrm = scale_vec(nrm, -1.0)
        if nrm[2] < min_facing:
            continue                    # edge-on to the decal: not the painted face
        if face_lift:
            # Off THIS face, along its own normal (turned to the decal's
            # side). A pillar face at 70 degrees to the decal would get only
            # a third of the clearance from a lift along BasisNormal.
            lift = scale_vec(add(add(scale_vec(e1, nrm[0]), scale_vec(e2, nrm[1])),
                                 scale_vec(n, nrm[2])), OVERLAY_MESH_LIFT)
        plane_d = dot(nrm, lt[0])
        clipped = _clip_poly_2d([(p[0], p[1]) for p in lt], clip_poly)
        if len(clipped) < 3:
            continue
        ring = []
        for (x, y) in clipped:
            z = (plane_d - nrm[0] * x - nrm[1] * y) / nrm[2]
            st = _inverse_bilinear((x, y), ca, cb, cc, cd)
            if st is None:
                ring = []
                break
            sp, tp = st
            uv = (tex[0][0] * (1 - sp) * (1 - tp) + tex[3][0] * sp * (1 - tp) +
                  tex[1][0] * (1 - sp) * tp + tex[2][0] * sp * tp,
                  tex[0][1] * (1 - sp) * (1 - tp) + tex[3][1] * sp * (1 - tp) +
                  tex[1][1] * (1 - sp) * tp + tex[2][1] * sp * tp)
            p3 = add(add(scale_vec(e1, x), scale_vec(e2, y)), scale_vec(n, z))
            ring.append(add_point(add(p3, lift), uv))
        if len(ring) < 3:
            continue
        for k in range(1, len(ring) - 1):
            a_i, b_i, c_i = ring[0], ring[k], ring[k + 1]
            if a_i == b_i or b_i == c_i or a_i == c_i:
                continue
            A, B, C = points[a_i], points[b_i], points[c_i]
            up = cross(sub(B, A), sub(C, A))
            if length(up) < 1e-6:
                continue
            # same winding rule as the ground case, with n in place of +Z
            triangles.append((a_i, c_i, b_i) if dot(up, n) > 0 else (a_i, b_i, c_i))

    if not triangles or len(points) > max_points:
        return None
    return points, uv_list, triangles


def overlay_sides_wrap(side_ids, basis_normal=None, tolerance=0.9998):
    """True when an overlay cannot be drawn as one flat patch in its own plane:
    the brush faces it names do not all share one direction (round a pillar,
    over a corner), OR they do but the overlay's own plane is tilted against
    them. Source projects an overlay along BasisNormal ONTO its faces; a flat
    patch in the overlay's plane only matches that when the two planes are
    parallel. Measured on de_overpass_csgo: 408 brush overlays are parallel
    (the patch is exact), 52 span non-coplanar faces, and 40 sit on one flat
    wall with their plane tilted up to 53 degrees off it - its
    graffiti_germany_face_02 decals, 26.6 degrees off, drew as a square
    board sticking out of the wall."""
    normals = []
    planes = []
    if basis_normal is not None and length(basis_normal) > 1e-9:
        normals.append(normalize(basis_normal))
    for sid in side_ids:
        poly = OVERLAY_SIDE_POLYS.get(sid)
        if not poly or len(poly) < 3:
            continue
        nrm = cross(sub(poly[1], poly[0]), sub(poly[2], poly[0]))
        for k in range(3, len(poly)):        # first non-degenerate corner
            if length(nrm) > 1e-6:
                break
            nrm = cross(sub(poly[k - 1], poly[0]), sub(poly[k], poly[0]))
        if length(nrm) < 1e-6:
            continue
        nrm = normalize(nrm)
        if any(abs(dot(nrm, m)) < tolerance for m in normals):
            return True
        normals.append(nrm)
        # Parallel is not enough: the faces must also be ONE plane. Stair
        # treads all face up, but each sits at its own height, and a flat
        # patch can only lie on one of them - de_overpass_csgo's
        # portgravel001 across seven treads floated above the stairs.
        d = dot(nrm, poly[0])
        for pn, pd in planes:
            same = dot(pn, nrm) > 0
            if abs(pd - (d if same else -d)) > 0.5:
                return True
        planes.append((nrm, d))
    return False


def overlay_covered_by_sides(entity, side_ids, n=8, tol=0.5):
    """True when the overlay's quad lies wholly on the faces it names. A flat
    patch can only be clipped to a rectangle, so a decal whose faces don't
    fill its quad - past the end of a wall, across stair treads, over a gap -
    left its overhang hanging in the air; VBSP cuts it to each face exactly.
    Measured on de_overpass_csgo: 50 of 359 flat decals had more than 5% of
    their area off their faces (up to 80%). Those are built the way VBSP does
    instead (build_overlay_brush_mesh). Tested on an n x n grid of points
    inside the quad, each against every named face polygon."""
    origin = _vec_key(entity, 'BasisOrigin') or _vec_key(entity, 'origin')
    bu, bv, bn = _vec_key(entity, 'BasisU'), _vec_key(entity, 'BasisV'), _vec_key(entity, 'BasisNormal')
    uvs = [_vec_key(entity, 'uv%d' % i) for i in range(4)]
    if not (origin and bu and bv and bn and all(uvs)):
        return True
    polys = [OVERLAY_SIDE_POLYS.get(sd) for sd in side_ids]
    polys = [pl for pl in polys if pl and len(pl) >= 3]
    if not polys:
        return True
    bn = normalize(bn)
    clipped = clip_overlay_uv_to_sides(entity, origin, bu, bv, uvs, [], '')
    if clipped is None:
        return True
    uvs = clipped[0]
    corners = [add(origin, add(scale_vec(bu, uv[0]), scale_vec(bv, uv[1]))) for uv in uvs]
    # each face: its plane and its edges, for an "inside, along bn" test
    faces = []
    for pl in polys:
        fn = normalize(cross(sub(pl[1], pl[0]), sub(pl[2], pl[0])))
        if abs(dot(fn, bn)) < 0.2:
            continue
        faces.append((pl, fn, dot(fn, pl[0])))
    if not faces:
        return True

    def on_some_face(p):
        for pl, fn, fd in faces:
            den = dot(fn, bn)
            q = add(p, scale_vec(bn, (fd - dot(fn, p)) / den))     # along bn onto the face
            m = len(pl)
            sign = 0
            ok = True
            for i in range(m):
                a_, b_ = pl[i], pl[(i + 1) % m]
                c = dot(cross(sub(b_, a_), sub(q, a_)), fn)
                edge = length(sub(b_, a_)) or 1.0
                if abs(c) / edge <= tol:
                    continue                                       # on the edge
                sg = 1 if c > 0 else -1
                if sign == 0:
                    sign = sg
                elif sg != sign:
                    ok = False
                    break
            if ok:
                return True
        return False

    for i in range(n):
        for j in range(n):
            s_, t_ = (i + 0.5) / n, (j + 0.5) / n
            p = add(add(scale_vec(corners[0], (1 - s_) * (1 - t_)), scale_vec(corners[3], s_ * (1 - t_))),
                    add(scale_vec(corners[1], (1 - s_) * t_), scale_vec(corners[2], s_ * t_)))
            if not on_some_face(p):
                return False
    return True


def build_overlay_brush_mesh(entity, scale, offset, side_ids, max_points=20000):
    """An info_overlay that cannot be one flat patch, built the way VBSP
    builds it (overlay.cpp): each named face is projected into the overlay's
    frame, clipped to the overlay's quad, and the surviving piece is laid back
    onto THAT face's plane with texture coordinates from the quad. The result
    wraps round a pillar or folds over a corner.

    It used to become one flat patch in the overlay's own plane, which on
    de_overpass_csgo's round concrete pillars showed as a square board
    through the column instead of graffiti_germany_face_02 painted on it -
    two of those overlays span 10 and 8 faces, and 71 overlays in the map
    span three or more. A single flat face keeps the flat patch.

    Returns (points, uvs, triangles) in map units, or None."""
    def world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])
    tris = []
    for sid in side_ids:
        poly = OVERLAY_SIDE_POLYS.get(sid)
        if not poly or len(poly) < 3:
            continue
        wp = [world(p) for p in poly]
        tris.extend((wp[0], wp[k], wp[k + 1]) for k in range(1, len(wp) - 1))
    if not tris:
        return None
    # No depth filter (the 'sides' list says which faces, however far round
    # the pillar they are); lift along each face; and only skip faces truly
    # edge-on (84 degrees) - Source itself paints those, just stretched.
    return _build_overlay_wall_mesh(entity, scale, offset, [], max_points, tris=tris,
                                    depth_filter=False, face_lift=True, min_facing=0.1)


def build_overlay_conforming_mesh(entity, scale, offset, records, max_points=20000):
    """A dense, terrain-following triangle mesh for one info_overlay,
    instead of convert_overlay's coarse 5x5 patchDef2 - built the same way
    from Source's own BasisOrigin/BasisU/BasisV/uv0..uv3/StartU../EndV data,
    but sampled finely enough to actually match the underlying LWO
    terrain's own triangulated shape (see sample_terrain_height), not just
    approximate it with a handful of control points. A patchDef2's Bezier
    curves also smoothly interpolate between control points, which can't
    match terrain built from flat triangles no matter how the control
    points are placed - a plain triangle mesh sidesteps that entirely.

    grid spacing is chosen finely enough that, if the actual terrain has no
    vertex between two adjacent samples, linear interpolation between the
    two sampled heights already equals what the terrain's own flat-
    triangle surface would give there - i.e. this reconstructs the real
    surface, not just an approximation of it, wherever sampling is fine
    enough relative to the terrain's own vertex spacing.

    Returns (points, uvs, triangles) if any sample point found real terrain
    nearby, else None (caller should fall back to convert_overlay's flat
    patch - this overlay isn't on terrain at all).
    """
    origin = _vec_key(entity, 'BasisOrigin') or _vec_key(entity, 'origin')
    bu, bv, bn = _vec_key(entity, 'BasisU'), _vec_key(entity, 'BasisV'), _vec_key(entity, 'BasisNormal')
    uvs_key = [_vec_key(entity, 'uv%d' % i) for i in range(4)]
    material = get1(entity, 'material')
    if not (origin and bu and bv and bn and material and all(uvs_key)):
        return None
    try:
        u0, u1 = float(get1(entity, 'StartU', '0')), float(get1(entity, 'EndU', '1'))
        v0, v1 = float(get1(entity, 'StartV', '0')), float(get1(entity, 'EndV', '1'))
    except ValueError:
        return None
    bn = normalize(bn)

    # A WALL decal must never be terrain-conformed. Source's own
    # BasisNormal says which way the overlay faces, and the split is
    # absolute on real data - measured over de_dust2's 55 info_overlays:
    # every ground overlay (brickroad, trashdecal, target, decalrug) has
    # |Nz| 0.970..1.000, and every wall overlay (graffiti, posters, bills,
    # wires, siteb, dustflag, arrows) has |Nz| exactly 0.000. Nothing sits
    # anywhere near the middle, so this test can't misfire on this class of
    # data.
    #
    # Without it, a vertical decal on a wall standing over sloped ground
    # still finds terrain under its footprint (sample_terrain_height looks
    # up to max_dz=256 away) and gets its vertices yanked down onto that
    # ground - turning a flat wall decal into a mangled, near-vertical
    # smear. CONFIRMED as the reported "bad parts" in the overlay mesh: 21
    # of de_dust2's wall decals were being routed here, 6 of them emitting
    # 32 garbage near-vertical triangles between them (graffiti03 alone was
    # 6 triangles, 100% near-vertical, right where the artifact was
    # reported) and the other 15 emitting NO triangles at all while still
    # contributing their orphan points to the model.
    #
    # A wall decal doesn't need conforming in the first place: the surface
    # it sits on is a flat brush face, so convert_overlay's flat patch is
    # already the correct shape for it.
    # Conform only to the displacements this overlay is actually projected
    # onto: VBSP projects an info_overlay onto the faces its 'sides' key names
    # and nothing else. Without this, every terrain triangle inside the
    # decal's footprint was used, at any height: de_mirage_cs2's
    # decals/sanddecal01, a FLOOR decal at map z -210, was built onto a
    # displacement wall 470-790 units above it (reported at -449 -3038 418,
    # "the decal looks messed up"). An overlay that names no displacement
    # lies on flat brush faces, which convert_overlay's flat patch already
    # matches exactly.
    disp_sides = {s for s in str(get1(entity, 'sides') or '').split() if s in DISP_SIDE_IDS}
    if not disp_sides:
        return None

    if abs(bn[2]) < OVERLAY_MESH_MIN_GROUND_NZ:
        # ...unless Source says the wall IS a displacement. The 'sides' key
        # names the faces VBSP projects the overlay onto; when one of them
        # carries a dispinfo, the surface is the displaced mesh, not the flat
        # base face, and the flat patch ends up behind it (hidden) or in
        # front of it (floating). Measured on de_mirage_cs2:
        # 43 of 102 overlays are wall decals on displacement sides, among
        # them the bombsite A letter and arrows reported hidden/offset.
        return _build_overlay_wall_mesh(entity, scale, offset, records, max_points, disp_sides)

    # Same clipping as the patch path - a road decal shouldn't run past the
    # edge of the surface it was painted on either.
    _clipped = clip_overlay_uv_to_sides(entity, origin, bu, bv, uvs_key, [], material)
    if _clipped is None:
        return None
    uvs_key, _frac = _clipped
    if _frac:
        _fu0, _fu1, _fv0, _fv1 = _frac
        _du, _dv = (u1 - u0), (v1 - v0)
        u0, u1 = u0 + _fu0 * _du, u0 + _fu1 * _du
        v0, v1 = v0 + _fv0 * _dv, v0 + _fv1 * _dv

    def world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])

    corners = [world(add(origin, add(scale_vec(bu, uv[0]), scale_vec(bv, uv[1])))) for uv in uvs_key]
    tex = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]

    # Build this overlay from the TERRAIN'S OWN TRIANGLES, by clipping them to
    # the overlay quad - not by sampling terrain heights on a regular grid.
    #
    # The grid approach this replaces could not work reliably, and the failure
    # was measured on real output: sampling one surface with a topology that
    # doesn't match the other means that between two samples the overlay's flat
    # triangle cuts straight through the terrain's actual surface wherever that
    # surface bulges upward. On de_dust2's own de_dust2_terrain_overlays.lwo,
    # 205 of 1878 overlay triangles (10.9%) had their centre BELOW the terrain,
    # by as much as 31.5 units - each one a hole where the ground draws over
    # the road. No nudge value fixes that; a 0.5-unit lift cannot rescue a
    # triangle sunk 31 units.
    #
    # Inheriting the terrain's triangulation makes the two surfaces coplanar by
    # construction: every output vertex lies exactly on a terrain triangle's
    # own plane, so the ONLY separation is the constant nudge along that
    # triangle's normal. Clipping is done in plan view, which is valid here
    # because this path only ever runs for ground-facing overlays (see the
    # |Nz| test above).
    quad2d = [(c[0], c[1]) for c in corners]
    area2 = sum(quad2d[i][0] * quad2d[(i + 1) % 4][1] - quad2d[(i + 1) % 4][0] * quad2d[i][1]
                for i in range(4))
    if area2 < 0:
        quad2d = quad2d[::-1]          # Sutherland-Hodgman needs a CCW clip polygon
    if abs(area2) < 1e-6:
        return None

    qx0 = min(p[0] for p in quad2d) - 1.0
    qx1 = max(p[0] for p in quad2d) + 1.0
    qy0 = min(p[1] for p in quad2d) - 1.0
    qy1 = max(p[1] for p in quad2d) + 1.0

    # uv-space corners, in the same order the texture corners use, for
    # recovering each point's texture coordinate through the quad's own
    # bilinear mapping (corner0 -> s,t = 0,0 ; corner3 -> s=1 ; corner1 -> t=1).
    a_uv, b_uv = uvs_key[0], uvs_key[3]
    c_uv, d_uv = uvs_key[1], uvs_key[2]

    points, uv_list, triangles = [], [], []
    index_of = {}

    def add_point(p3, uv):
        key = (round(p3[0], 3), round(p3[1], 3), round(p3[2], 3))
        idx = index_of.get(key)
        if idx is None:
            idx = len(points)
            index_of[key] = idx
            points.append(p3)
            uv_list.append(uv)
        return idx

    for tri in iter_terrain_triangles(records, disp_sides):
        txs = [p[0] for p in tri]
        tys = [p[1] for p in tri]
        if max(txs) < qx0 or min(txs) > qx1 or max(tys) < qy0 or min(tys) > qy1:
            continue

        nrm = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]))
        nl = length(nrm)
        if nl < 1e-9:
            continue
        nrm = scale_vec(nrm, 1.0 / nl)
        if nrm[2] < 0:
            nrm = scale_vec(nrm, -1.0)     # lift away from the ground, not into it
        if abs(nrm[2]) < 1e-6:
            continue                        # vertical terrain face: no plan-view solution
        plane_d = dot(nrm, tri[0])
        # Lift straight up, NOT along each triangle's own normal. Neighbouring
        # terrain triangles have slightly different normals, so per-normal
        # lifting moves a shared vertex to two slightly different places - the
        # mesh then fails to weld (measured: 7606 points for 3043 triangles)
        # and every shared edge becomes a hairline crack. A constant +Z lift
        # keeps shared vertices identical, so the overlay welds properly and is
        # an exact copy of the terrain surface raised clear of it.
        lift = (0.0, 0.0, OVERLAY_MESH_LIFT)

        clipped = _clip_poly_2d([(p[0], p[1]) for p in tri], quad2d)
        if len(clipped) < 3:
            continue

        ring = []
        for (x, y) in clipped:
            z = (plane_d - nrm[0] * x - nrm[1] * y) / nrm[2]
            st = _inverse_bilinear((( x + offset[0]) / scale, (y + offset[1]) / scale),
                                   (origin[0] + a_uv[0] * bu[0] + a_uv[1] * bv[0],
                                    origin[1] + a_uv[0] * bu[1] + a_uv[1] * bv[1]),
                                   (origin[0] + b_uv[0] * bu[0] + b_uv[1] * bv[0],
                                    origin[1] + b_uv[0] * bu[1] + b_uv[1] * bv[1]),
                                   (origin[0] + c_uv[0] * bu[0] + c_uv[1] * bv[0],
                                    origin[1] + c_uv[0] * bu[1] + c_uv[1] * bv[1]),
                                   (origin[0] + d_uv[0] * bu[0] + d_uv[1] * bv[0],
                                    origin[1] + d_uv[0] * bu[1] + d_uv[1] * bv[1]))
            if st is None:
                ring = []
                break
            sp, tp = st
            uv = (tex[0][0] * (1 - sp) * (1 - tp) + tex[3][0] * sp * (1 - tp) +
                  tex[1][0] * (1 - sp) * tp + tex[2][0] * sp * tp,
                  tex[0][1] * (1 - sp) * (1 - tp) + tex[3][1] * sp * (1 - tp) +
                  tex[1][1] * (1 - sp) * tp + tex[2][1] * sp * tp)
            ring.append(add_point(add((x, y, z), lift), uv))
        if len(ring) < 3:
            continue

        for k in range(1, len(ring) - 1):
            a_i, b_i, c_i = ring[0], ring[k], ring[k + 1]
            if a_i == b_i or b_i == c_i or a_i == c_i:
                continue
            A, B, C = points[a_i], points[b_i], points[c_i]
            if length(cross(sub(B, A), sub(C, A))) < 1e-6:
                continue
            # Match the terrain mesh's own winding, which is what renders
            # correctly in game (see the note on the old code's winding).
            up = cross(sub(B, A), sub(C, A))
            triangles.append((a_i, c_i, b_i) if up[2] > 0 else (a_i, b_i, c_i))

    if not triangles:
        return None
    if len(points) > max_points:
        return None
    return points, uv_list, triangles


def _overlay_target_plane(entity):
    """(normal, dist) in Source units of the single plane shared by every
    brush face an overlay's 'sides' key names, or None - no faces found,
    faces on different planes (those overlays are built as the wrapped mesh
    instead), or displacement faces (the terrain sampling handles those)."""
    sides = [x for x in str(get1(entity, 'sides') or '').split()]
    if not sides or any(x in DISP_SIDE_IDS for x in sides):
        return None
    plane = None
    for sid in sides:
        poly = OVERLAY_SIDE_POLYS.get(sid)
        if not poly or len(poly) < 3:
            continue
        # Newell normal: robust for any convex polygon
        nx = ny = nz = 0.0
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        if length((nx, ny, nz)) < 1e-9:
            continue
        n = normalize((nx, ny, nz))
        cen = scale_vec(reduce_add(poly), 1.0 / len(poly))
        d = dot(n, cen)
        if plane is None:
            plane = (n, d)
            continue
        pn, pd = plane
        c = dot(pn, n)
        # 0.5: VMF plane points on angled walls are rounded, so the faces of
        # one wall disagree by up to ~0.3 units (measured on de_overpass_csgo)
        if abs(c) < 0.999 or abs(pd - (d if c > 0 else -d)) > 0.5:
            return None
    return plane


def reduce_add(points):
    out = (0.0, 0.0, 0.0)
    for p in points:
        out = add(out, p)
    return out


def _face_patch_along(grid, want_normal):
    """Make a patchDef2 control grid face `want_normal` (the direction away
    from the wall/floor it sits on). grid[col][row] = (point, uv).

    Facing doesn't matter for drawing - the decal shaders are cull none - but
    it decides where q3map2 LIGHTS the patch from: every luxel and vertex
    sample is pushed one unit along the patch's own normal, and a decal whose
    normal points into its wall gets all of its samples inside the brush, so
    every one comes back occluded and the decal is black (or, before it was
    vertex-lit, fullbright). q3map2's patch normal is
    cross(next column - first, next row - first); measured on the de_dust2
    build of 2026-09-30, where 23 of 197 decals faced into their walls.
    Reversing the rows flips the normal and keeps every uv on its point, so the
    image is unchanged."""
    if len(grid) < 2 or len(grid[0]) < 2:
        return grid
    p00, p10, p01 = grid[0][0][0], grid[1][0][0], grid[0][1][0]
    n = cross(sub(p10, p00), sub(p01, p00))
    if dot(n, want_normal) < 0:
        return [list(reversed(col)) for col in grid]
    return grid


def convert_overlay(entity, scale, offset, texture_map, unmapped_counter, index, warnings=None):
    """Source info_overlay -> a patchDef2 built directly from Source's own
    overlay data (vbsp overlay.cpp): corner i = BasisOrigin + uv_i.x*BasisU +
    uv_i.y*BasisV; texture runs StartU->EndU along BasisU and StartV->EndV
    along BasisV (corner 0 = (StartU,StartV), 1 = (StartU,EndV),
    2 = (EndU,EndV), 3 = (EndU,StartV)).

    CHANGED from the original q3map2-decal-projection approach: that relied
    on q3map2 finding the real surface within a fixed-depth box below the
    overlay, which silently produces NOTHING wherever bumpy displacement
    terrain dips further than the box is deep under part of an overlay -
    confirmed as the likely reason overlays weren't appearing in-game at
    all. This instead samples the ACTUAL terrain surface directly (see
    sample_terrain_height) and builds the overlay's own geometry to follow
    it, exactly like Source's own overlays conform to the surface they're
    painted on - not just a decal projected at it. Grid is 5x5 (vs Source's
    plain quad) so the patch has enough control points to actually bend
    across bumpy ground; corners without nearby terrain (an overlay on an
    ordinary flat wall) fall back to the flat bilinear position, unchanged
    from before. A small nudge along the overlay's own normal keeps it from
    exactly coinciding with the surface underneath.

    Returns [the patch's func_group entity text] or [] if the entity is
    missing required data.
    """
    origin = _vec_key(entity, 'BasisOrigin') or _vec_key(entity, 'origin')
    bu, bv, bn = _vec_key(entity, 'BasisU'), _vec_key(entity, 'BasisV'), _vec_key(entity, 'BasisNormal')
    uvs = [_vec_key(entity, 'uv%d' % i) for i in range(4)]
    material = get1(entity, 'material')
    if not (origin and bu and bv and bn and material and all(uvs)):
        return []
    try:
        u0, u1 = float(get1(entity, 'StartU', '0')), float(get1(entity, 'EndU', '1'))
        v0, v1 = float(get1(entity, 'StartV', '0')), float(get1(entity, 'EndV', '1'))
    except ValueError:
        return []
    bn = normalize(bn)
    if warnings is None:
        warnings = []

    # Clip to the faces this overlay actually belongs to (its 'sides' key)
    # before building anything - otherwise it overhangs the wall it was
    # painted on, or floats where a face we dropped used to be. See
    # clip_overlay_uv_to_sides.
    _clipped = clip_overlay_uv_to_sides(entity, origin, bu, bv, uvs, warnings, material)
    if _clipped is None:
        warnings.append(
            "info_overlay ('{}') doesn't overlap any of the face(s) its 'sides' key names, so it "
            "was dropped instead of left floating in mid-air.".format(material))
        return []
    uvs, _frac = _clipped
    if _frac:
        _fu0, _fu1, _fv0, _fv1 = _frac
        _du, _dv = (u1 - u0), (v1 - v0)
        u0, u1 = u0 + _fu0 * _du, u0 + _fu1 * _du
        v0, v1 = v0 + _fv0 * _dv, v0 + _fv1 * _dv
        warnings.append(
            "info_overlay ('{}') was clipped to the face(s) its 'sides' key names (Source does the "
            "same in VBSP); its texture coordinates were cropped to match rather than squashed."
            .format(material))

    def world(p):
        return (p[0] * scale - offset[0], p[1] * scale - offset[1], p[2] * scale - offset[2])

    src_corners = [add(origin, add(scale_vec(bu, uv[0]), scale_vec(bv, uv[1]))) for uv in uvs]
    # Onto the face it is painted on. VBSP builds an overlay's fragments on
    # the faces its 'sides' key names, so where the entity itself sits does
    # not matter in Source - mappers leave BasisOrigin off the wall all the
    # time. Built from BasisOrigin as-is, those decals stood off their walls:
    # measured on de_overpass_csgo, the three graffiti_germany_headshot
    # decals on one 26.6-degree wall at 10.4, 66.6 and 85.9 units, the
    # zapems graffiti at 57, a bombsite B at 16.5 (axis-aligned walls happen
    # to have their overlays placed on the plane, hence only angled walls).
    target = _overlay_target_plane(entity)
    if target is not None:
        tn, td = target
        den = dot(tn, bn)
        if abs(den) > 0.2:
            src_corners = [add(c, scale_vec(bn, (td - dot(tn, c)) / den)) for c in src_corners]
    corners = [world(c) for c in src_corners]
    tex = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    nudge = scale_vec(bn, 0.5)

    # Same ground-facing test as build_overlay_conforming_mesh: only an
    # overlay lying on the GROUND should have its control points pulled onto
    # the terrain surface. For a wall decal standing over sloped ground,
    # sample_terrain_height still finds that ground (up to max_dz away) and
    # snapping to it would drag the decal off its wall - the flat bilinear
    # position IS the correct shape there, since the wall is a flat brush
    # face. (Latent rather than reported here: de_dust2's wall decals happen
    # to sit where no terrain is within range, so they came out correct by
    # luck, not by rule.)
    records = LWO_TERRAIN_BUILDER.records if LWO_TERRAIN_BUILDER is not None else []
    if abs(bn[2]) < OVERLAY_MESH_MIN_GROUND_NZ:
        records = []
    # Same rule as build_overlay_conforming_mesh: only the displacements the
    # overlay's 'sides' key names. A decal on a brush floor must not be pulled
    # onto some other terrain that happens to be within reach.
    _ds = {s for s in str(get1(entity, 'sides') or '').split() if s in DISP_SIDE_IDS}
    records = [r for r in records if r.get('side_id') in _ds]

    # Control points per side from the decal's size, so its vertex lighting
    # is as fine as the lightmap around it (DECAL_VERTEX_LIGHT_SPACING; the
    # old fixed 5x5 drew 3x3 vertices however big the decal). Never fewer
    # than the old 5, which also keeps terrain-following overlays as close
    # to the ground as before.
    def _side(i, j, k, l):
        return max(math.sqrt(sum((corners[j][c] - corners[i][c]) ** 2 for c in range(3))),
                   math.sqrt(sum((corners[l][c] - corners[k][c]) ** 2 for c in range(3))))
    GRID_A = decal_patch_side(_side(0, 3, 1, 2), minimum=5)
    GRID_B = decal_patch_side(_side(0, 1, 3, 2), minimum=5)

    def cp(a, b):   # a: 0..GRID_A-1 along corner0->corner3, b: 0..GRID_B-1 along corner0->corner1
        s, t = a / float(GRID_A - 1), b / float(GRID_B - 1)
        p = add(add(scale_vec(corners[0], (1 - s) * (1 - t)), scale_vec(corners[3], s * (1 - t))),
                add(scale_vec(corners[1], (1 - s) * t), scale_vec(corners[2], s * t)))
        uv = tuple(tex[0][k] * (1 - s) * (1 - t) + tex[3][k] * s * (1 - t) +
                   tex[1][k] * (1 - s) * t + tex[2][k] * s * t for k in range(2))
        z = sample_terrain_height(records, p[0], p[1], p[2])
        if z is not None:
            p = (p[0], p[1], z)
        return add(p, nudge), uv

    # outer list = patch columns (q3map2 ParsePatch: verts[i*width + j]),
    # in Source's own uv0..uv3 corner order, unmodified. (A previous
    # version tried to "correct" winding here for consistency with the old
    # q3map2 decal-projector approach this replaced - that projector had
    # its own front-plane test for an unrelated purpose, the sign
    # convention was never independently verified for patchDef2's own
    # front-facing rule, and it's the likely cause of a real "overlay
    # flipped/skewed" report. cull none in the shader, added by
    # vmt2shader.py --overlay-materials, makes the correct-facing question
    # moot instead of guessing at it.)
    grid = [[cp(a, b) for b in range(GRID_B)] for a in range(GRID_A)]
    grid = _face_patch_along(grid, bn)

    mapped = decal_texture_name(material, convert_texture_name(material, texture_map,
                                                                unmapped_counter))
    OVERLAY_MATERIALS.add(material.strip().lower().replace('\\', '/'))
    rows = []
    for col in grid:
        rows.append("( " + " ".join("( {:.10g} {:.10g} {:.10g} {:.10g} {:.10g} )".format(
            p[0], p[1], p[2], uv[0], uv[1]) for p, uv in col) + " )")
    # q3map2 ParsePatch reads "( width height )" then width groups of height
    # points - grid is [column][row], so width = len(grid).
    patch = "\n".join(['{', 'patchDef2', '{', mapped,
                       '( %d %d 0 0 0 )' % (len(grid), len(grid[0])), '('] +
                       rows + [')', '}', '}'])
    return [_wrap_patch_as_func_group(patch)]


def convert_vmf(text, texture_map, class_map, skip_classes, scale, recenter=False, model_scale=1.0,
                 shader_file=None, materials_dir=None, terrain_mode='brush',
                 skip_3d_skybox=False, skybox_radius=4096.0, drop_backing_brush=False,
                 terrain_min_power=None, terrain_edge_feather=None,
                 terrain_lwo_model_path='models/mapobjects/terrain',
                 keep_terrain_base_solid=False, keep_overlays_skipped=False,
                 emit_skybox_entity=False, model_static_paths=None,
                 skybox_models_to_world=False, skybox_landscape_brushes=0,
                 seal_leak_from_lin=None, seal_output_path='out.map',
                 seal_terrain_under=0, distance_cull=0.0, skip_infodecals=False,
                 seal_map_box=None, gridsize=None, ambient=None):
    tokens = tokenize(text)
    root = parse_vdf(tokens)
    SKIP_BRUSHES_ARE_CLIPS[0] = scan_skip_brushes_are_clips(text)

    # Every displacement in the map gets upsampled to the single highest
    # power found anywhere, so no two adjacent displacements ever have
    # mismatched Bezier segment structure along a shared edge - see
    # TERRAIN_TARGET_POWER's own comment for why this is necessary, not
    # just precautionary. terrain_min_power (--terrain-min-power), when
    # given, raises this further still - real per-displacement testing
    # showed that even with matching boundary values, adjacent patches'
    # Bezier curves can still visibly diverge approaching a shared edge,
    # since each patch's own curve shape depends on its OTHER nearby
    # control points too, not just the shared one. Higher resolution
    # doesn't eliminate that divergence, but shrinks each individual
    # Bezier segment's own footprint, so any mismatch becomes visually
    # smaller - a real, testable mitigation, not a full fix.
    global TERRAIN_TARGET_POWER
    TERRAIN_TARGET_POWER = scan_max_terrain_power(root)
    if terrain_min_power is not None and (TERRAIN_TARGET_POWER is None or terrain_min_power > TERRAIN_TARGET_POWER):
        TERRAIN_TARGET_POWER = terrain_min_power

    # See TERRAIN_ALPHA_RECONCILIATION's own comment: some displacements'
    # own Source alpha data genuinely disagrees with a neighbor's at what is
    # otherwise the same shared vertex - confirmed real, not a conversion
    # bug. Computed once here so every displacement's own grid-building
    # loop can look up and use the reconciled value at these positions.
    global TERRAIN_ALPHA_RECONCILIATION
    TERRAIN_ALPHA_RECONCILIATION = scan_terrain_alpha_conflicts(root)

    # EXPERIMENTAL, opt-in - see TERRAIN_ALPHA_FEATHER's own comment.
    global TERRAIN_ALPHA_FEATHER
    TERRAIN_ALPHA_FEATHER = None
    if terrain_edge_feather:
        TERRAIN_ALPHA_FEATHER = scan_terrain_alpha_feather(root, terrain_edge_feather)

    # Fresh builder per run when terrain_mode is lwo - every displacement in
    # the map welds into this ONE shared mesh regardless of which brush/
    # entity it came from (see LWOTerrainBuilder's own comment for why).
    global LWO_TERRAIN_BUILDER
    LWO_TERRAIN_BUILDER = LWOTerrainBuilder() if terrain_mode == 'lwo' else None


    warnings = []
    if SKIP_BRUSHES_ARE_CLIPS[0]:
        warnings.append(
            "This .vmf has no clip material at all but has brushes made only of toolsskip - the "
            "signature of a decompiled map (BSPSource writes a clip brush's untextured sides as "
            "toolsskip). Those brushes were turned into player clips (tools/_blockplayer); see "
            "SKIP_BRUSHES_ARE_CLIPS.")
    overlay_count = 0
    OVERLAY_MATERIALS.clear()
    OVERLAY_MESH_SURFACES.clear()
    OVERLAY_MESH_SKY_SURFACES.clear()
    OVERLAY_MESH_MATERIAL_TO_FLAT.clear()
    WRAPPED_OVERLAYS[0] = 0
    OVERLAY_MESH_MATERIALS.clear()
    global TERRAIN_BASE_SOLIDS
    TERRAIN_BASE_SOLIDS = set()
    terrain_base_partial = []
    if materials_dir and not keep_terrain_base_solid:
        TERRAIN_BASE_SOLIDS, terrain_base_partial = scan_terrain_base_slabs(root)
    if TERRAIN_BASE_SOLIDS:
        warnings.append(
            "{} nodraw slab brush(es) under displacement terrain made non-solid ('{}'): the "
            "terrain is built on their top face, so collision couldn't tell them apart (stone "
            "footsteps/impacts on terrain). They still seal the map. Needs the '{}' shader "
            "from vmt2shader.py (re-run it).".format(
                len(TERRAIN_BASE_SOLIDS), TERRAIN_BASE_TEXTURE, 'textures/' + TERRAIN_BASE_TEXTURE))
    for sid, cov, corner in terrain_base_partial:
        warnings.append(
            "nodraw slab brush (VMF id {}) only {:.0f}% covered by displacement terrain - left "
            "solid, since its uncovered part may be real floor. If you get stone footsteps near "
            "({:.0f} {:.0f} {:.0f}) Source units, check it by hand.".format(
                sid, 100 * cov, corner[0], corner[1], corner[2]))
    unmapped_counter = Counter()

    if TERRAIN_ALPHA_RECONCILIATION:
        conflict_counts = Counter(material for material, x, y in TERRAIN_ALPHA_RECONCILIATION)
        for material, count in conflict_counts.items():
            warnings.append(
                "Terrain alphamap conflicts: '{}' has {} shared-vertex position(s) where two or "
                "more displacements independently store different alpha values for what is "
                "otherwise the same physical vertex (a real inconsistency in the source VMF's own "
                "displacement data, not a bug in this converter) - reconciled to one averaged "
                "value per position so every affected displacement agrees.".format(material, count))

    if TERRAIN_ALPHA_FEATHER:
        feather_counts = Counter(material for material, x, y in TERRAIN_ALPHA_FEATHER)
        for material, count in feather_counts.items():
            warnings.append(
                "Terrain edge feathering (EXPERIMENTAL): '{}' has {} vertex position(s) one step "
                "inward from a shared boundary blended toward their neighbor's own equivalent "
                "(weight {:.2g}), to reduce Bezier curve-shape divergence near patch "
                "boundaries.".format(material, count, terrain_edge_feather))

    worlds = root.get('world', [])
    if not worlds:
        raise ValueError("No 'world' block found - is this a valid VMF file?")
    world = worlds[0]

    # Where the 3D skybox actually is (the sealed room around the sky_camera),
    # for every skybox-related decision below - see find_skybox_area.
    setup_skybox_area(root, warnings, skybox_radius)

    # BEFORE the --skip-3d-skybox exclusion pass and before any skybox-region
    # handling: a worldized model is ordinary main-map geometry from here on,
    # and is tagged so neither of those passes claims it back.
    if skybox_models_to_world:
        _wb_offset, _wb_bounds = compute_bounds(root, 1.0)
        worldize_skybox_models(root, warnings, main_bounds=_wb_bounds, map_scale=scale)
        world = root.get('world', [world])[0] if root.get('world') else world

    skybox_excluded_brushes = 0
    skybox_excluded_entities = 0
    if skip_3d_skybox:
        sky_origin = find_sky_camera_origin(root)
        if sky_origin is None:
            warnings.append("--skip-3d-skybox was given but no sky_camera entity was found in "
                             "this VMF - nothing excluded (this map may not use a 3D skybox at all).")
        else:
            def near_skybox(pos):
                return in_skybox_area(pos)

            kept_world_solids = []
            for solid in world.get('solid', []):
                if isinstance(solid, dict) and near_skybox(brush_approx_position(solid)):
                    skybox_excluded_brushes += 1
                    continue
                kept_world_solids.append(solid)
            world = dict(world)
            world['solid'] = kept_world_solids

            kept_entities = []
            for entity in root.get('entity', []):
                origin_str = get1(entity, 'origin')
                pos = None
                if origin_str:
                    parts = origin_str.split()
                    if len(parts) == 3:
                        try:
                            pos = tuple(float(p) for p in parts)
                        except ValueError:
                            pos = None
                if pos is None:
                    for solid in get_solid_blocks(entity):
                        pos = brush_approx_position(solid)
                        if pos:
                            break
                if near_skybox(pos) and not get1(entity, WORLDIZED_MARKER_KEY):
                    # ...and not something --skybox-models-to-world already
                    # moved into real map space: those are main-map models
                    # now, even when their new origin still happens to fall
                    # inside this radius.
                    skybox_excluded_entities += 1
                    continue
                kept_entities.append(entity)
            root = dict(root)
            root['entity'] = kept_entities

            warnings.append(
                "--skip-3d-skybox: excluded {} world brush(es) and {} entity/entities in the 3D "
                "skybox area ({}) around the sky_camera at {}.".format(
                    skybox_excluded_brushes, skybox_excluded_entities, SKYBOX_AREA_INFO[0],
                    sky_origin))

    # Confirmed bug: DEFAULT_TOOL_TEXTURES maps Source's sky material
    # straight to SOF2's tools/_sky, which is just a generic placeholder
    # (its own definition is literally "skyParms - 1 -" - no real skybox
    # images at all). That silently discarded whatever custom sky
    # vmt2shader.py actually generated from this map's own skybox VMTs -
    # the custom shader existed in the .shader file, nothing ever
    # referenced it. If this map declares a skyname, route sky faces to the
    # SAME shader name vmt2shader.py computes for it instead (a plain
    # "textures/skybox/<skyname>" - skybox/ is never a model-owned root, so
    # it always gets the textures/ prefix, unlike models/-rooted materials).
    skyname = get1(world, 'skyname')
    if skyname:
        # BARE path, no "textures/" prefix - matches every other texture in
        # this .map (see the file-top note on --texture-map convention).
        # q3map2 adds "textures/" itself at compile time, so this resolves
        # to the same "textures/skybox/<skyname>" shader name vmt2shader.py
        # writes in the .shader file - which DOES carry the prefix, since
        # .shader names use the different, already-compiled-form
        # convention. Writing the prefix here too (an earlier version of
        # this fix did) doubles it into "textures/textures/skybox/...",
        # which is exactly the bug this note exists to prevent.
        #
        # Prefer the ACTUAL sky shader name found in the .shader file over
        # blindly constructing one from skyname: confirmed real-world case
        # where a map's skyname ("vertigo_hdr") didn't match the basename
        # its own skybox VMTs actually shipped under ("vertigo_bk" etc,
        # no "hdr") - vmt2shader.py correctly generated "skybox/vertigo"
        # from the real files, but this code was still constructing
        # "skybox/vertigo_hdr" from the mismatched skyname, so the sky
        # faces pointed at a shader that never existed. Falls back to the
        # skyname-based guess when no shader file is given or it has no
        # sky shader in it (e.g. this step run standalone, before step 6).
        found_sky = find_sky_shader_name(shader_file)
        if found_sky:
            custom_sky = re.sub(r'^textures/', '', found_sky, flags=re.IGNORECASE)
        else:
            custom_sky = 'skybox/{}'.format(skyname.strip().lower())
        texture_map = dict(texture_map)
        texture_map.setdefault('TOOLS/TOOLSSKYBOX', custom_sky)
        texture_map.setdefault('TOOLS/TOOLSSKYBOX2D', custom_sky)

    offset = (0.0, 0.0, 0.0)
    recenter_info = None
    if recenter:
        offset, bounds = compute_bounds(root, scale)
        recenter_info = (offset, bounds)

    # Face polygons by side id, so info_overlays can be clipped to the faces
    # they name (see build_side_polygon_index).
    global OVERLAY_SIDE_POLYS
    OVERLAY_SIDE_POLYS = build_side_polygon_index(root)

    out_blocks = []
    infodecal_count = 0
    dropped_source_buttons = [0]
    terrain_outputs = []
    patch_output_blocks = []
    overlay_sky_centre = None
    overlay_sky_radius = skybox_radius * scale
    if emit_skybox_entity:
        _sky_o = find_sky_camera_origin(root)
        if _sky_o is not None:
            overlay_sky_centre = (_sky_o[0] * scale - offset[0], _sky_o[1] * scale - offset[1],
                                  _sky_o[2] * scale - offset[2])
    sky_region = (overlay_sky_centre, overlay_sky_radius) if overlay_sky_centre is not None else None

    # Report the map's real extent and what distancecull it implies, so the
    # value isn't a guess. Source's own MaxRange is reported alongside for
    # reference: the Raven engine ignores it, but it's what the original map
    # considered far enough.
    _dc_off, _dc_bounds = compute_bounds(root, scale)
    if _dc_bounds:
        (_mn, _mx) = _dc_bounds
        _diag = math.sqrt(sum((_mx[i] - _mn[i]) ** 2 for i in range(3)))
        _src_max = get1(world, 'MaxRange') or get1(world, 'maxrange')
        _src_note = ''
        if _src_max:
            try:
                _src_note = (" Source's own MaxRange on this map is {:g}, i.e. {:.0f} units at "
                             "this scale.".format(float(_src_max), float(_src_max) * scale))
            except ValueError:
                pass
        if distance_cull and distance_cull > 0:
            if distance_cull >= _diag:
                verdict = ("covers the whole map (bounding box diagonal {:.0f} units), so nothing "
                           "should pop in".format(_diag))
            else:
                verdict = ("is SHORTER than the map's bounding box diagonal ({:.0f} units), so the "
                           "furthest geometry can still pop in. Note that diagonal includes the 3D "
                           "skybox area if this map has one, which sits far from the playable part "
                           "- so a value below it may still be fine in practice"
                           .format(_diag))
            warnings.append("distancecull set to {:.0f}; this {}.{}".format(
                distance_cull, verdict, _src_note))
        else:
            # Nothing set it: neither --distance-cull nor a Source MaxRange. Derive
            # it from the map's own bounding box rather than leave it unset.
            #
            # What is verified about this key in RETAIL SOF2 (not OpenSoF2MP,
            # whose tr_bsp.cpp reads it with a 6000 default): retail SoF2MP.exe
            # does not contain the string at all - only "RMG_distancecull" - but
            # retail cgamex86.dll reads "distancecull", squares it, and uses it
            # as a sphere-distance cull (|pos - view|^2 - r^2 against it) on
            # things it adds to the scene. Set to the map's full diagonal it can
            # never hide anything a player could see. It was NOT the cause of
            # the missing props: those were the 256-entity snapshot limit (see
            # networked_prop_cap).
            #
            # Rounding UP to the next 1024 is deliberate: over-covering costs
            # draw distance, while under-covering hides geometry. The diagonal
            # includes the 3D skybox area on maps that have one, which sits far
            # from the playable part, so this can be generous - still the safe
            # direction. An explicit --distance-cull overrides it.
            distance_cull = math.ceil(_diag / 1024.0) * 1024
            warnings.append(
                "distancecull {:.0f} derived from this map's bounding box (diagonal {:.0f} units), "
                "because the map set no far clip of its own. SOF2's cgame uses it as a distance "
                "cull; at the full diagonal it can never hide anything a player could see. Pass "
                "--distance-cull to override.{}".format(
                    distance_cull, _diag, _src_note))

    # The shadow fill, from the map's own light_environment unless --ambient
    # overrode it. Derived here rather than left to the user because a sun with
    # no ambient bakes every ceiling in the map to black, and nothing in the
    # compile output says so - see derive_ambient_from_light_environment.
    # Always solved, even when --ambient overrides the result: the point lights
    # take their exposure from it.
    derived_ambient, derived_color = derive_ambient_from_light_environment(root, warnings, scale)
    ambient_color = None
    if not ambient:
        ambient, ambient_color = derived_ambient, derived_color
        if not ambient:
            warnings.append(
                "No worldspawn 'ambient' written: this map has no light_environment to solve an "
                "exposure from, so rooms the lights cannot reach will bake black and point lights "
                "use de_dust2's exposure (k = {:.2f}). Pass --ambient if it needs a floor."
                .format(EXPOSURE_K[0]))

    out_blocks.append(convert_world(world, texture_map, unmapped_counter, scale, offset, warnings,
                                     materials_dir=materials_dir, terrain_outputs=terrain_outputs,
                                     terrain_mode=terrain_mode, drop_backing_brush=drop_backing_brush,
                                     patch_output_blocks=patch_output_blocks, sky_region=sky_region,
                                     distance_cull=distance_cull, gridsize=gridsize,
                                     ambient=ambient, ambient_color=ambient_color))

    # Brushes from FOLD_INTO_WORLDSPAWN_CLASSES entities (func_detail,
    # func_breakable) get folded straight into worldspawn; everything else
    # becomes its own entity block. Both get the CONTENTS_DETAIL flag
    # (Hammer's own equivalent of GTK Radiant's "Make Detail" - func_detail
    # IS Source's detail-brush mechanism, there's no separate flag beyond
    # assigning a brush to it). func_areaportal is NOT handled here at all
    # any more - see DEFAULT_SKIP_CLASSES for why it's dropped entirely.
    DETAIL_CLASSES = {'func_detail', 'func_breakable'}

    # --seal-leak-from-lin: promote the detail brushes the leak flood walks
    # through to STRUCTURAL, so they seal. See
    # find_detail_solids_on_leak_path for why detail brushes can be
    # load-bearing in Source yet seal nothing here.
    seal_promote_ids = set()
    if seal_leak_from_lin:
        leak_pts = read_leak_pointfile(seal_leak_from_lin, scale, offset, warnings)
        if leak_pts:
            found_ids = find_detail_solids_on_leak_path(root, leak_pts, DETAIL_CLASSES)
            # ACCUMULATE across passes. Sealing is iterative: each compile's
            # pointfile shows ONE escape route, and once that route is shut
            # the flood finds the next one - so the next .lin describes a
            # different path and names different brushes (measured on
            # de_dust2: pass 1 named 5 detail brushes, pass 2 named none of
            # them). Promoting only what the CURRENT .lin names would quietly
            # un-promote the previous pass's brushes and reopen the route
            # already sealed, so the loop could never converge. The running
            # set lives next to the .map and is merged every run.
            seal_state_path = os.path.splitext(seal_output_path)[0] + '_sealed_detail_ids.json'
            previous = set()
            try:
                with open(seal_state_path, encoding='utf-8') as f:
                    previous = {str(i) for i in json.load(f)}
            except (OSError, ValueError):
                pass
            seal_promote_ids = previous | found_ids
            try:
                with open(seal_state_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted(seal_promote_ids), f, indent=2)
            except OSError as e:
                warnings.append(
                    "--seal-leak-from-lin: couldn't write the running sealed-brush list to '{}' "
                    "({}), so the NEXT pass won't remember this one - previously sealed routes "
                    "may reopen.".format(seal_state_path, e))
            if found_ids and previous:
                warnings.append(
                    "--seal-leak-from-lin: this pass found {} detail brush(es) on the leak path "
                    "({}), carried forward together with {} already sealed in earlier passes - {} "
                    "promoted in total. The running list is '{}'; delete it to start over.".format(
                        len(found_ids), ", ".join(sorted(found_ids)), len(previous),
                        len(seal_promote_ids), os.path.basename(seal_state_path)))
            elif previous and not found_ids:
                warnings.append(
                    "--seal-leak-from-lin: this pass's leak path ran through NO detail brush, so "
                    "nothing new was promoted - the {} brush(es) sealed in earlier passes are "
                    "still promoted (from '{}'). The remaining route is empty in Source too, so a "
                    "detail promotion can't close it: see --seal-terrain-under, which covers the "
                    "usual cause (the flood getting underneath displacement terrain that seals in "
                    "Source but is only a model here).".format(
                        len(previous), os.path.basename(seal_state_path)))
            if found_ids and not previous:
                warnings.append(
                    "--seal-leak-from-lin: the leak path runs through {} detail brush(es) "
                    "(source solid id(s) {}) - emitted as STRUCTURAL instead of detail so they "
                    "actually seal. q3map2 runs FloodEntities before detail brushes enter the "
                    "tree, so brushwork that is solid in Source seals nothing here while it "
                    "carries CONTENTS_DETAIL. Recompile: if it still leaks, the new .lin will "
                    "point at the next gap - feed it back and repeat (the list accumulates).".format(
                        len(found_ids), ", ".join(sorted(found_ids))))
            elif not seal_promote_ids:
                warnings.append(
                    "--seal-leak-from-lin: the leak path didn't pass through ANY detail brush, so "
                    "nothing was promoted. That means the gap is genuinely empty in Source too "
                    "(or is bounded only by geometry this converter drops, e.g. a model-only "
                    "structure) - it needs a sealing brush built by hand, not a detail promotion.")

    detail_solid_blocks = []
    for entity in root.get('entity', []):
        classname = get1(entity, 'classname', '')
        if classname.lower() == 'sky_camera' and not skip_3d_skybox and emit_skybox_entity:
            # Source's 3D skybox -> q3map2's own _skybox entity. This is a
            # COMPILER-only entity (q3map2 2.5.11+; this project's q3map2
            # reports 2.5.17), which is exactly why it works where the
            # earlier misc_skyportal attempt failed with "doesn't have a
            # spawn function" - _skybox never exists at runtime, so it
            # needs no game-side support at all. Same concept as
            # sky_camera: it marks the origin of a separate, wholly
            # contained skybox area that becomes visible wherever sky is.
            # Source's 'scale' and q3map2's '_scale' mean the same thing
            # (1 unit inside the skybox reads as <scale> units of world),
            # so it carries straight over; Source defaults to 16 when the
            # key is absent. The skybox geometry itself must NOT be
            # dropped for this to work, hence 'not skip_3d_skybox'.
            sky_o = _vec_key(entity, 'origin') or (0.0, 0.0, 0.0)
            try:
                sky_scale = float(get1(entity, 'scale', '16') or 16)
            except ValueError:
                sky_scale = 16.0
            sky_lines = [
                "{",
                '"classname" "_skybox"',
                '"origin" "{:.10g} {:.10g} {:.10g}"'.format(
                    sky_o[0] * scale - offset[0], sky_o[1] * scale - offset[1],
                    sky_o[2] * scale - offset[2]),
                '"_scale" "{:.10g}"'.format(sky_scale),
            ]
            sky_angles = get1(entity, 'angles')
            if sky_angles:
                sky_lines.append('"angles" "{}"'.format(sky_angles))
            sky_lines.append("}")
            out_blocks.append("\n".join(sky_lines))
            warnings.append(
                "sky_camera -> q3map2 '_skybox' entity at scale {:g} (Source's own 'scale' key; "
                "same meaning in both). This is a compiler-only entity, so it needs no game-side "
                "support - but it DOES require the 3D skybox geometry to be kept and fully sealed "
                "from the main map. If the compile takes very long or the result looks wrong, "
                "check for a leak between the two areas.".format(sky_scale))
            continue

        if classname.lower() == 'info_overlay' and not keep_overlays_skipped:
            records = LWO_TERRAIN_BUILDER.records if LWO_TERRAIN_BUILDER is not None else []
            mesh = build_overlay_conforming_mesh(entity, scale, offset, records) if records else None
            brush_sides = [s_ for s_ in str(get1(entity, 'sides') or '').split()
                           if s_ not in DISP_SIDE_IDS]
            wrap_mesh = None
            if brush_sides and (overlay_sides_wrap(brush_sides, _vec_key(entity, 'BasisNormal'))
                                or not overlay_covered_by_sides(entity, brush_sides)):
                part = dict(entity)
                part['sides'] = [' '.join(brush_sides)]
                wrap_mesh = build_overlay_brush_mesh(part, scale, offset, brush_sides)
                if wrap_mesh:
                    WRAPPED_OVERLAYS[0] += 1
            if wrap_mesh and not mesh:
                mesh, wrap_mesh, brush_sides = wrap_mesh, None, []
            if mesh:
                points, uv_list, triangles = mesh
                if wrap_mesh:          # displacement part + wrapped brush part
                    wp, wu, wt = wrap_mesh
                    base_ = len(points)
                    points, uv_list = list(points) + list(wp), list(uv_list) + list(wu)
                    triangles = list(triangles) + [(a + base_, b + base_, c + base_)
                                                   for a, b, c in wt]
                    brush_sides = []
                material = get1(entity, 'material')
                # Same material -> same flat shader (like terrain's own
                # _flat_shader_name), not a fresh one per overlay INSTANCE.
                # Confirmed as a real, direct cause of a MAX_SHADER_INFO
                # (8192, a fixed engine array - checked in q3map2's own
                # source) compile failure: this map alone had 11 separate,
                # identical shader definitions for one repeated decal
                # material, 37 total for just 18 unique ones - and the base
                # game's own shaders already use nearly the whole budget on
                # their own, leaving very little room to waste. Multiple
                # overlays sharing a material now merge their geometry
                # under ONE shared SURF tag, exactly like several
                # displacements sharing a shader already do in the terrain
                # mesh.
                mat_key = material.strip().lower().replace('\\', '/')
                flat_name = OVERLAY_MESH_MATERIAL_TO_FLAT.get(mat_key)
                if flat_name is None:
                    flat_name = '{}overlay{}'.format(MAP_SHADER_TAG, len(OVERLAY_MESH_MATERIAL_TO_FLAT))
                    OVERLAY_MESH_MATERIAL_TO_FLAT[mat_key] = flat_name
                    OVERLAY_MESH_MATERIALS[flat_name] = material
                # Overlays inside the 3D skybox area go in their own model,
                # for the same reason the terrain does (see
                # split_mesh_by_region): one model spanning both regions
                # gives q3map2's _skybox no way to tell them apart. It also
                # stops Radiant drawing one huge entity bounding box
                # stretching from the main map out to the skybox.
                dest = OVERLAY_MESH_SURFACES
                if overlay_sky_centre is not None:
                    cx = sum(p[0] for p in points) / float(len(points))
                    cy = sum(p[1] for p in points) / float(len(points))
                    cz = sum(p[2] for p in points) / float(len(points))
                    if in_skybox_area_mapspace((cx, cy, cz), scale, offset):
                        dest = OVERLAY_MESH_SKY_SURFACES
                if flat_name not in dest:
                    dest[flat_name] = ([], [], [])
                ex_points, ex_uvs, ex_tris = dest[flat_name]
                base = len(ex_points)
                ex_points.extend(points)
                ex_uvs.extend(uv_list)
                ex_tris.extend((a + base, b + base, c + base) for a, b, c in triangles)
                OVERLAY_MATERIALS.add(mat_key)
                overlay_count += 1
                # An overlay spanning a displacement AND ordinary brush faces
                # (3 on de_mirage_cs2, plaster-decay decals running down onto
                # a displaced base) needs both: the mesh above covers only
                # the displacement, so the brush faces get the usual flat
                # patch, clipped to just those faces.
                flat_sides = brush_sides
                if flat_sides:
                    part = dict(entity)
                    part['sides'] = [' '.join(flat_sides)]
                    blocks = convert_overlay(part, scale, offset, texture_map, unmapped_counter,
                                             overlay_count, warnings)
                    if blocks:
                        out_blocks.extend(blocks)
                continue
            blocks = convert_overlay(entity, scale, offset, texture_map, unmapped_counter,
                                     overlay_count, warnings)
            if blocks:
                overlay_count += 1
                out_blocks.extend(blocks)
            continue
        if classname.lower() == 'infodecal' and not skip_infodecals:
            blocks = convert_infodecal(entity, scale, offset, texture_map, unmapped_counter,
                                       warnings, materials_dir)
            if blocks:
                out_blocks.extend(blocks)
                infodecal_count += 1
            continue
        if should_skip_class(classname, skip_classes):
            continue
        source_button = classname.lower() in SOURCE_IO_BUTTON_CLASSES
        if source_button and not _has_drawn_side(entity):
            # Invisible (tools-only) Source button: nothing to see, and its
            # behaviour lives in Source I/O outputs this converter never
            # carries over. Kept, SOF2 still plays its default button sound
            # when shot - the "door unlock" click on de_overpass_csgo's
            # balloons (6 nodraw func_buttons, health 1, OnDamaged only).
            dropped_source_buttons[0] += 1
            continue
        if source_button or classname.lower() in FOLD_INTO_WORLDSPAWN_CLASSES:
            # A drawn button keeps its geometry as static world detail.
            is_detail = source_button or classname.lower() in DETAIL_CLASSES
            for solid in get_solid_blocks(entity):
                # A brush on the leak path loses its detail flag (see
                # seal_promote_ids above) - structural is what seals.
                solid_detail = is_detail and get1(solid, 'id') not in seal_promote_ids
                block = convert_solid(solid, texture_map, unmapped_counter, scale, offset,
                                       warnings, solid_detail, materials_dir=materials_dir,
                                       terrain_outputs=terrain_outputs, terrain_mode=terrain_mode,
                                       drop_backing_brush=drop_backing_brush,
                                       patch_output_blocks=patch_output_blocks)
                if block:
                    detail_solid_blocks.append(block)
            if get_solid_blocks(entity):
                warnings.append(
                    "'{}' brush entity flattened into worldspawn".format(classname))
            continue
        result = convert_entity(entity, texture_map, unmapped_counter, scale, offset,
                                 warnings, class_map, skip_classes, model_scale,
                                 materials_dir=materials_dir, terrain_outputs=terrain_outputs,
                                 terrain_mode=terrain_mode, drop_backing_brush=drop_backing_brush,
                                 patch_output_blocks=patch_output_blocks,
                                 sky_region=sky_region,
                                 model_static_paths=model_static_paths)
        if result:
            block, _ = result
            out_blocks.append(block)

    if detail_solid_blocks:
        # splice detail brushes into the worldspawn block we already built
        worldspawn_block = out_blocks[0]
        insert_at = worldspawn_block.rfind("\n}")
        out_blocks[0] = (worldspawn_block[:insert_at] + "\n" +
                          "\n".join(detail_solid_blocks) +
                          worldspawn_block[insert_at:])

    # Each WorldVertexTransition displacement built a real, separate
    # func_group terrain entity (see build_displacement_terrain) - these
    # are whole top-level entities, not brushes nested inside worldspawn,
    # so they're appended here rather than spliced into anything above.
    # patch_output_blocks (the flat-patch fallback, now also wrapped as
    # its own func_group entity - see _wrap_patch_as_func_group) is
    # appended the same way, for the same reason: it has to survive even
    # when the displacement's own containing solid gets dropped (see that
    # function's own comment for the real bug this fixes).
    out_blocks.extend(patch_output_blocks)
    terrain_pcx_files = []
    seen_base_shaders = []  # preserves first-seen order, deduplicated
    terrain_scales = {}  # base_shader_name -> {material, u_scale, v_scale}
    for entity_text, pcx_bytes, pcx_filename, base_shader_name, disp_material, u_scale, v_scale \
            in terrain_outputs:
        out_blocks.append(entity_text)
        terrain_pcx_files.append((pcx_bytes, pcx_filename))
        if base_shader_name not in seen_base_shaders:
            seen_base_shaders.append(base_shader_name)
        terrain_scales[base_shader_name] = {
            'material': disp_material, 'u_scale': u_scale, 'v_scale': v_scale,
        }

    if seen_base_shaders:
        # Confirmed required by comparing against a real, working SOF2 map:
        # its worldspawn carries a "vertexremapshaderN" key for every
        # generated per-layer/blend terrain shader, each pointing back at
        # one shared "<base>.vertex" reference shader (also generated, see
        # vmt2shader.py's convert_worldvertextransition_to_terrain_shaders).
        # Without these, the compiler leaves the raw tools/_terrain shader
        # name in place instead of ever substituting the real per-layer
        # ones - confirmed against a real compile log showing zero terrain
        # processing at all without them.
        remap_lines = []
        idx = 0
        for base in seen_base_shaders:
            textures_base = 'textures/' + base
            vertex_ref = textures_base + '.vertex'
            for suffix in ('_0', '_1', '_0to1'):
                remap_lines.append('"vertexremapshader{}" "{};{}"'.format(
                    idx, textures_base + suffix, vertex_ref))
                idx += 1
        worldspawn_block = out_blocks[0]
        brace_end = worldspawn_block.index('\n', worldspawn_block.index('{')) + 1
        out_blocks[0] = (worldspawn_block[:brace_end] +
                          '\n'.join(remap_lines) + '\n' +
                          worldspawn_block[brace_end:])

    # Finalize the welded LWO2 terrain mesh (terrain_mode == 'lwo') into ONE
    # misc_model entity, if any displacements were collected into it. Kept
    # separate from the patch/brush terrain_outputs loop above since this
    # produces exactly one output regardless of how many displacements fed
    # it, not one per displacement.
    terrain_lwo = None
    terrain_lwo_scales = {}
    terrain_sky_lwo = None
    FOG_SPLIT_LWO_FILES[:] = []
    FOG_BRUSH_PLANES[:] = []
    TERRAIN_NOFOG_TARGETS.clear()
    FOG_BOXES[:] = find_fog_volume_boxes(root, materials_dir, scale, offset)
    if LWO_TERRAIN_BUILDER is not None and not LWO_TERRAIN_BUILDER.is_empty():
        points, point_alphas = LWO_TERRAIN_BUILDER.finalize_points()
        shader_tris = LWO_TERRAIN_BUILDER.shader_triangles
        # With _skybox in play, the skybox area's terrain has to be its own
        # model - see split_mesh_by_region for why one model spanning both
        # regions breaks q3map2's skybox handling outright.
        sky_split = None
        if emit_skybox_entity:
            sky_o = find_sky_camera_origin(root)
            if sky_o is not None:
                sky_scaled = (sky_o[0] * scale - offset[0], sky_o[1] * scale - offset[1],
                              sky_o[2] * scale - offset[2])
                main_part, sky_part = split_mesh_by_region(
                    points, point_alphas, shader_tris,
                    lambda p: in_skybox_area_mapspace(p, scale, offset))
                if sky_part[2] and main_part[2]:
                    sky_split = (main_part, sky_part)

        if sky_split and skybox_landscape_brushes:
            # The skybox area's terrain becomes a COARSE SOLID BRUSH
            # landscape instead of a model, so it stays inside q3map2's
            # skybox surface budget (see build_skybox_landscape_brushes for
            # the measured numbers and for the two approaches this replaces).
            # The sky part of the welded mesh is simply dropped - the brushes
            # stand in for it.
            (points, point_alphas, shader_tris), _sky_unused = sky_split
            sky_o2 = find_sky_camera_origin(root)
            sky_c = (sky_o2[0] * scale - offset[0], sky_o2[1] * scale - offset[1],
                     sky_o2[2] * scale - offset[2])
            r2 = (skybox_radius * scale) ** 2
            sky_recs = []
            for rec in LWO_TERRAIN_BUILDER.records:
                nn = rec['n']
                pts_ = [rec['pos'][i][j] for i in range(nn) for j in range(nn)]
                cx = sum(p[0] for p in pts_) / len(pts_)
                cy = sum(p[1] for p in pts_) / len(pts_)
                cz = sum(p[2] for p in pts_) / len(pts_)
                if in_skybox_area_mapspace((cx, cy, cz), scale, offset):
                    sky_recs.append(rec)
            land = build_skybox_landscape_brushes(sky_recs, skybox_landscape_brushes,
                                                  texture_map, unmapped_counter, warnings)
            if land:
                land_text, n_brushes, n_surfaces = land
                out_blocks.append(land_text)
                warnings.append(
                    "--skybox-landscape-brushes {}: the 3D skybox area's {} displacement(s) were "
                    "rebuilt as {} solid prism brush(es) approximating their shape, instead of a "
                    "{}-triangle model. Only the top face of each carries the real material (sides "
                    "and bottom are caulk). MEASURED on a real compile, each prism costs about SIX skybox "
                    "drawsurface(s), not one - so expect roughly {} against "
                    "the ~827 the whole 3D skybox can afford, where q3map2 "
                    "replicates each skybox surface into ~1241 sky-visible leaves against the "
                    "1048576 MAX_MAP_LEAFFACES limit. Raise the value for a closer-fitting "
                    "landscape, lower it if the budget gets tight.".format(
                        skybox_landscape_brushes, len(sky_recs), n_brushes,
                        sum(len(v) for v in _sky_unused[2].values()), n_surfaces * 6))
                if not skybox_models_to_world:
                    warnings.append(
                        "--skybox-landscape-brushes is on but --skybox-models-to-world is NOT: the "
                        "skybox area's PROPS are still skybox-marked models, and on de_dust2 those "
                        "measured ~7262 surfaces on their own - far past the ~827 budget, so the "
                        "compile will still hit MAX_MAP_LEAFFACES. Turn --skybox-models-to-world on "
                        "as well so the props become ordinary real-scale geometry.")
        elif sky_split:
            (points, point_alphas, shader_tris), (s_pts, s_alphas, s_tris) = sky_split
            # Default (triangle-count) splitting, not the stricter
            # model_static vertex cap: this model is a forced-meta
            # misc_model, so q3map2 does the per-surface splitting itself.
            sky_bytes, _sky_map = write_lwo2_terrain(s_pts, s_alphas, s_tris)
            sky_name = os.path.splitext(TERRAIN_LWO_FILENAME)[0] + '_sky.lwo'
            terrain_sky_lwo = (sky_bytes, sky_name)
            warnings.append(
                "Terrain (lwo mode): the 3D skybox area's displacements were split into their own "
                "model ('{}', {} point(s)) so q3map2's _skybox has the separate, self-contained "
                "skybox area it requires - they were previously welded into the SAME model as the "
                "main map, which is what made _skybox generate thousands of skybox surfaces and "
                "exceed MAX_MAP_LEAFFACES.".format(sky_name, len(s_pts)))

        # Cut around fog volumes first (see find_fog_volume_boxes' block
        # comment): part 0 keeps the usual file name, the rest are written
        # as <map>_terrain_partN.lwo next to it.
        fog_parts = split_terrain_by_fog(points, point_alphas, shader_tris, FOG_BOXES)
        points, point_alphas, shader_tris, _inside0, _nf0 = fog_parts[0]
        lwo_bytes, origin_mapping = write_lwo2_terrain(points, point_alphas, shader_tris)
        lwo_filename = TERRAIN_LWO_FILENAME
        model_ref = terrain_lwo_model_path.rstrip('/') + '/' + lwo_filename
        terrain_lwo = (lwo_bytes, lwo_filename)
        for k, (p_pts, p_alphas, p_tris, p_inside, p_nofog) in enumerate(fog_parts[1:], 1):
            p_bytes, p_map = write_lwo2_terrain(p_pts, p_alphas, p_tris)
            p_name = '{}_part{}.lwo'.format(os.path.splitext(TERRAIN_LWO_FILENAME)[0], k)
            FOG_SPLIT_LWO_FILES.append((p_bytes, p_name))
            for actual, original in p_map.items():
                origin_mapping.setdefault(actual, original)
            out_blocks.append("\n".join([
                "{",
                '"classname" "misc_model"',
                '"model" "{}"'.format(terrain_lwo_model_path.rstrip('/') + '/' + p_name),
                '"origin" "0 0 0"',
                '"spawnflags" "6"',
            ] + _nofog_remap_lines(p_map if p_nofog else {}) + ["}"]))
        if len(fog_parts) > 1:
            warnings.append(
                "Terrain (lwo mode): cut into {} models around {} water volume(s) so their fog "
                "stays inside the water - q3map2 fogs a model surface whole if its bounding box "
                "touches a fog brush ({} part(s) sit inside a water volume). Copy the "
                "_partN.lwo files along with the main one.".format(
                    len(fog_parts), len(FOG_BOXES),
                    sum(1 for fp in fog_parts if fp[3] and not fp[4])))
        # every ACTUAL tag written (including split ones) needs its own
        # shader - see split_oversized_shader_groups's own comment. Only
        # blend (flat-named terrainN) shaders live in shader_scales - a
        # non-blend material's tag IS already its ordinary "textures/..."
        # name (see LWOTerrainBuilder.add_displacement), which
        # vmt2shader.py generates the normal way, not from this JSON.
        terrain_lwo_scales = {actual: LWO_TERRAIN_BUILDER.shader_scales[original]
                              for actual, original in origin_mapping.items()
                              if original in LWO_TERRAIN_BUILDER.shader_scales}
        # Non-blended displacements keep their ordinary "textures/<material>"
        # name inside the .lwo - but q3map2 -rename (needed for the props)
        # renames EVERY misc_model surface to <name>_BSP, the terrain included.
        # Without a <name>_bsp shader that surface compiles against nothing and
        # SOF2 draws its grey-and-white "no texture" grid (de_dust2: the cliff
        # displacements using nature/rockwall009b). Listed here with
        # "plain": true so vmt2shader.py writes the _bsp copy.
        for actual, original in origin_mapping.items():
            if original in LWO_TERRAIN_BUILDER.shader_scales or actual in terrain_lwo_scales:
                continue
            terrain_lwo_scales[actual] = {'material': re.sub(r'^textures/', '', original),
                                          'plain': True}
        if len(origin_mapping) != len(set(origin_mapping.values())):
            warnings.append(
                "Terrain (lwo mode): one or more shader groups exceeded {} triangles (a real, "
                "hard idTech3 renderer limit per shader-batch - RB_CheckOverflow otherwise) and "
                "were split into extra tags sharing the same material, e.g. 'terrainN_1' next to "
                "'terrainN'.".format(LWO_MAX_TRIANGLES_PER_GROUP))

        n_points = len(points)
        n_tris = sum(len(v) for v in shader_tris.values())
        n_shaders = len(shader_tris)
        warnings.append(
            "Terrain (lwo mode): welded {} displacement vertex position(s) into {} point(s), "
            "{} triangle(s) across {} shader(s), into one misc_model ('{}'). Requires "
            "spawnflags 6 for collision (confirmed necessary in SOF2) - copy {} into your "
            "models folder under this same relative path, or change "
            "--terrain-lwo-model-path to match wherever you place it.".format(
                LWO_TERRAIN_BUILDER.total_vertex_adds, n_points, n_tris,
                n_shaders, model_ref, lwo_filename))
        warnings.append(
            "Terrain (lwo mode): {} displacement(s) built at native power with Source's own "
            "triangulation; {} edge vertex(es) snapped onto lower-power neighbours and {} "
            "boundary triangle(s) split to match (Source's mixed-power edge handling, no "
            "T-junctions).".format(len(LWO_TERRAIN_BUILDER.records),
                                   LWO_TERRAIN_BUILDER.snapped_edge_verts,
                                   LWO_TERRAIN_BUILDER.split_triangles))

        misc_model_lines = [
            "{",
            '"classname" "misc_model"',
            '"model" "{}"'.format(model_ref),
            '"origin" "0 0 0"',
            '"spawnflags" "6"',
            "}",
        ]
        out_blocks.append("\n".join(misc_model_lines))

        if seal_terrain_under:
            seal = build_terrain_seal_brushes(LWO_TERRAIN_BUILDER.records, seal_terrain_under)
            if seal:
                seal_text, n_seal = seal
                out_blocks.append(seal_text)
                warnings.append(
                    "--seal-terrain-under {}: added {} structural nodraw/nonsolid prism brush(es) "
                    "as a thin sealing shell under the terrain ({} displacement(s), {} cell(s) "
                    "each per axis, offset 16 units down, each 1024 deep). A displacement SEALS the map "
                    "in Source, but here the terrain is a model and a model seals nothing - "
                    "confirmed on de_dust2, where 207 of its 1732 leak-path sample points sat "
                    "directly beneath displacement footprints. These carry no detail flag (detail "
                    "brushes are filtered in after FloodEntities and would seal nothing) and use "
                    "'vmf2map/terrain_base', which is nonsolid but opaque - q3map2 decides sealing "
                    "from C_TRANSLUCENT rather than solidity, so it seals without adding collision "
                    "on top of the model's own. Needs that shader from vmt2shader.py.".format(
                        seal_terrain_under, n_seal, len(LWO_TERRAIN_BUILDER.records),
                        seal_terrain_under))

    if terrain_sky_lwo is not None:
        out_blocks.append("\n".join([
            "{",
            # misc_model, NOT model_static. model_static was tried here as
            # a way around this map's leak-driven MAX_MAP_LEAFFACES and it
            # is not a fix: q3map2 only ever sets ds->skybox on
            # mapDrawSurface_t objects, and model_static never becomes one
            # (it's spawned at runtime, outside the meta pipeline), so a
            # model_static skybox terrain is structurally invisible to
            # skybox marking and never renders as a backdrop at all -
            # confirmed in game. The leak is the thing that has to be
            # sealed; hiding the geometry from the compiler just hides the
            # backdrop too.
            #
            # spawnflags 4 = forced meta (q3map2 model.c:
            # `( si->forceMeta ) || ( spawnFlags & 4 )`; SOF2's own
            # mp_entities.def calls the same bit "Lightmapped"), so these
            # triangles enter the meta pipeline where skybox marking
            # happens and where q3map2 splits anything past the engine's
            # per-surface index cap by itself. Without bit 2 (SOLID) or bit
            # 8 (auto clip): the 3D skybox is a distant visual backdrop,
            # never walked on, so it needs no collision.
            '"classname" "misc_model"',
            '"model" "{}"'.format(terrain_lwo_model_path.rstrip('/') + '/' + terrain_sky_lwo[1]),
            '"origin" "0 0 0"',
            '"spawnflags" "4"',
            "}",
        ]))

    overlay_mesh_sky_lwo = None
    if OVERLAY_MESH_SKY_SURFACES:
        # Default (triangle-count) splitting - forced-meta misc_model, same
        # as the skybox terrain model.
        sky_ov_bytes, sky_ov_map = write_lwo2_mesh(OVERLAY_MESH_SKY_SURFACES)
        sky_ov_name = os.path.splitext(OVERLAY_MESH_FILENAME)[0] + '_sky.lwo'
        overlay_mesh_sky_lwo = (sky_ov_bytes, sky_ov_name)
        for actual, original in sky_ov_map.items():
            OVERLAY_MESH_MATERIALS.setdefault(actual, OVERLAY_MESH_MATERIALS.get(original, ''))
        out_blocks.append("\n".join([
            "{",
            # misc_model + spawnflags 4 (forced meta, not solid) - same
            # reasoning as the skybox terrain model right above:
            # model_static is invisible to q3map2's skybox marking, so it
            # can never render as a backdrop.
            '"classname" "misc_model"',
            '"model" "{}"'.format(OVERLAY_MESH_MODEL_PATH.rstrip('/') + '/' + sky_ov_name),
            '"origin" "0 0 0"',
            '"spawnflags" "4"',
            "}",
        ]))
        warnings.append(
            "Overlay mesh: {} surface(s) of it lie in the 3D skybox area and were written to "
            "their own model ('{}') - same reason as the terrain split.".format(
                len(OVERLAY_MESH_SKY_SURFACES), sky_ov_name))

    overlay_mesh_lwo = None
    overlay_mesh_materials = {}
    DECAL_MESH_POINTS[0] = 0
    if OVERLAY_MESH_SURFACES:
        # Vertex-lit, so dense enough to carry the shadows the lightmapped
        # ground under it shows (see DECAL_VERTEX_LIGHT_SPACING). Not the
        # 3D-skybox copy above: it is only ever seen from far away.
        _n_before = sum(len(t) for _, _, t in OVERLAY_MESH_SURFACES.values())
        _dense, _added = subdivide_mesh_surfaces(OVERLAY_MESH_SURFACES)
        OVERLAY_MESH_SURFACES.clear()
        OVERLAY_MESH_SURFACES.update(_dense)
        DECAL_MESH_POINTS[0] = sum(len(p_) for p_, _u, _t in OVERLAY_MESH_SURFACES.values())
        if _added:
            warnings.append(
                "Overlay mesh: triangles split until no edge is longer than {:g} units "
                "({:,} -> {:,} triangles, {:,} point(s) added). Overlays are vertex lit, so "
                "without this a shadow crossing one shows as a dark rectangle or long "
                "straight wedges instead of following the lightmapped ground.".format(
                    DECAL_VERTEX_LIGHT_SPACING, _n_before,
                    sum(len(t) for _, _, t in OVERLAY_MESH_SURFACES.values()), _added))
        ov_parts = split_mesh_surfaces_by_fog(OVERLAY_MESH_SURFACES, FOG_BOXES)
        if len(ov_parts) > 1:
            OVERLAY_MESH_SURFACES.clear()
            OVERLAY_MESH_SURFACES.update(ov_parts[0][0])
        overlay_mesh_lwo_bytes, overlay_origin_mapping = write_lwo2_mesh(OVERLAY_MESH_SURFACES)
        for k, (p_surfs, _p_inside, p_nofog) in enumerate(ov_parts[1:], 1):
            p_bytes, p_map = write_lwo2_mesh(p_surfs)
            p_name = '{}_part{}.lwo'.format(os.path.splitext(OVERLAY_MESH_FILENAME)[0], k)
            FOG_SPLIT_LWO_FILES.append((p_bytes, p_name))
            for actual, original in p_map.items():
                overlay_origin_mapping.setdefault(actual, original)
            out_blocks.append("\n".join([
                "{",
                '"classname" "misc_model"',
                '"model" "{}"'.format(OVERLAY_MESH_MODEL_PATH.rstrip('/') + '/' + p_name),
                '"origin" "0 0 0"',
                '"spawnflags" "4"',
            ] + _nofog_remap_lines(p_map if p_nofog else {}) + ["}"]))
        if len(ov_parts) > 1:
            warnings.append(
                "Overlay mesh: cut into {} models around water volumes, same reason as the "
                "terrain.".format(len(ov_parts)))
        overlay_mesh_lwo = (overlay_mesh_lwo_bytes, OVERLAY_MESH_FILENAME)
        # every ACTUAL tag written (including split ones) needs its own
        # shader - see split_oversized_mesh_surfaces's own comment.
        overlay_mesh_materials = {actual: OVERLAY_MESH_MATERIALS[original]
                                  for actual, original in overlay_origin_mapping.items()}
        overlay_model_ref = OVERLAY_MESH_MODEL_PATH.rstrip('/') + '/' + OVERLAY_MESH_FILENAME
        n_pts = sum(len(p) for p, _, _ in OVERLAY_MESH_SURFACES.values())
        n_tris = sum(len(t) for _, _, t in OVERLAY_MESH_SURFACES.values())
        warnings.append(
            "{} material group(s) of ground info_overlays built as a dense, terrain-following "
            "mesh instead of coarse patches ({} point(s), {} triangle(s)) into one misc_model "
            "('{}') with spawnflags 4 (q3map2 forced-meta, so it gets split to respect the "
            "engine's per-surface index cap; NOT solid - overlays are visual only). Needs the "
            "'{}' file next to it and its own shaders from vmt2shader.py "
            "--overlay-mesh-materials.".format(
                len(OVERLAY_MESH_SURFACES), n_pts, n_tris, overlay_model_ref,
                OVERLAY_MESH_FILENAME))
        if len(overlay_origin_mapping) != len(set(overlay_origin_mapping.values())):
            warnings.append(
                "Overlay mesh: one or more materials exceeded {} triangles once merged (a real, "
                "hard idTech3 renderer limit per shader-batch - RB_CheckOverflow otherwise) and "
                "were split into extra tags sharing the same material.".format(
                    LWO_MAX_TRIANGLES_PER_GROUP))
        out_blocks.append("\n".join([
            "{",
            '"classname" "misc_model"',
            '"model" "{}"'.format(overlay_model_ref),
            '"origin" "0 0 0"',
            # spawnflags 4 = q3map2's "forced meta" bit (confirmed in its
            # model.c: `( si->forceMeta ) || ( spawnFlags & 4 )`), WITHOUT
            # bit 2 (clip/solid) - overlays are visual only and must not
            # collide. Forced meta is what puts the model's triangles
            # through q3map2's meta-surface pipeline, which enforces the
            # engine's per-surface index cap (game_sof2.h: 6000 max surface
            # indexes) and splits anything oversized by itself. Without it,
            # a misc_model's triangles go in as raw SURFACE_TRIANGLES with
            # NO cap at all, and the whole model lands as one surface -
            # confirmed as the real cause of a runtime
            # "RB_CheckOverflow: indices > MAX (45714 > 6000)" on this
            # mesh, matching its total triangle count rather than any
            # single group. That's also why the terrain model (spawnflags
            # 6 = solid + forced meta) never hit this despite being larger:
            # q3map2 was already splitting it.
            '"spawnflags" "4"',
            "}",
        ]))

    if dropped_source_buttons[0]:
        warnings.append(
            "{} invisible Source button(s) (func_button / func_rot_button / "
            "momentary_rot_button with only tool textures) dropped: they work only through "
            "Source entity I/O, which is not converted, and SOF2 would still play its default "
            "button sound when one is shot or used.".format(dropped_source_buttons[0]))
    if infodecal_count:
        warnings.append(
            "{} infodecal(s) converted to patches on the face each one rests against (Source's own "
            "behaviour for this entity - see --skip-infodecals). Their materials are in the "
            "overlay-materials JSON, so vmt2shader.py --overlay-materials gives them the "
            "polygonOffset/cull they need.".format(infodecal_count))

    out_blocks.extend(build_sound_speakers(root, scale, offset, materials_dir, warnings))

    if seal_map_box is not None:
        out_blocks.extend(build_seal_boxes(out_blocks, seal_map_box, warnings))

    # gridsize last, and patched into the worldspawn block that is already
    # written: the cell size depends on the finished map's bounding box, and that
    # box is only final once the seal slabs are in. Written rather than left as a
    # hand edit on purpose - a worldspawn key added by hand in Radiant is
    # silently lost the next time the map is regenerated, which is exactly what
    # happened on the compile that prompted this.
    if not gridsize:
        gridsize = derive_gridsize(out_blocks, warnings)
    if gridsize and out_blocks and '"gridsize"' not in out_blocks[0]:
        out_blocks[0] = out_blocks[0].replace(
            '"classname" "worldspawn"',
            '"classname" "worldspawn"\n"gridsize" "{}"'.format(gridsize), 1)

    map_text = snap_brush_planes_to_grid("\n".join(out_blocks) + "\n", warnings)
    return (map_text, warnings, unmapped_counter, recenter_info, terrain_pcx_files, terrain_scales,
            terrain_lwo, terrain_lwo_scales, overlay_count, overlay_mesh_lwo, overlay_mesh_materials,
            terrain_sky_lwo, overlay_mesh_sky_lwo)


# ==============================================================================
# CLI
# ==============================================================================

def load_json_map(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {k.upper(): v for k, v in data.items()}


def main():
    ap = argparse.ArgumentParser(
        description="Convert a Source engine .vmf map into an idTech3 .map "
                     "file compatible with SOF2 (and other Q3-derived editors).")
    ap.add_argument("input", help="input .vmf file")
    ap.add_argument("output", help="output .map file")
    ap.add_argument("--texture-map", help="JSON file mapping VMF material -> SOF2 texture path")
    ap.add_argument("--class-map", help="JSON file mapping VMF classname -> SOF2 classname")
    ap.add_argument("--skip-classes", default="",
                     help="comma-separated list of classnames to drop entirely")
    ap.add_argument("--scale", type=float, default=1.0,
                     help="uniform scale multiplier applied to all coordinates (default 1.0)")
    ap.add_argument("--recenter", action="store_true",
                     help="translate the whole map (brushes + entity origins) so its "
                          "bounding-box center sits at the origin. Strongly recommended "
                          "for maps built far from (0,0,0) in Hammer - idTech3's BSP/CSG "
                          "compiler uses single-precision floats and can produce degenerate "
                          "'brush plane with no normal' errors on geometry far from origin.")
    ap.add_argument("--texture-sizes", help="JSON file mapping material -> [width, height] in "
                                             "pixels, used for displacement (patchDef2) UV "
                                             "scaling. Materials not listed default to 512x512 "
                                             "(flagged in the warnings so you know which ones "
                                             "are guesses).")
    ap.add_argument("--shader-file", help="the .shader file generated by vmt2shader.py. When given, "
                                           "the end-of-run texture report cross-checks against it and "
                                           "lists ONLY materials that have no shader definition (i.e. the "
                                           "ones that will genuinely show up missing in-game), instead of "
                                           "listing every passed-through material.")
    ap.add_argument("--model-scale", type=float, default=1.1,
                     help="additional uniform scale applied only to converted misc_model "
                          "entities (default 1.1), via the modelscale key (existing Source "
                          "modelscale values are multiplied by this, not replaced). This "
                          "compensates for Source and SOF2 not sharing an identical world-unit "
                          "scale - NOTE: 1.1 is a tuned-by-testing value, not a verified "
                          "physical constant (neither engine's exact units-per-inch figure is "
                          "authoritatively documented anywhere we could find) - adjust freely "
                          "if your own comparison suggests a different factor.")
    ap.add_argument("--materials-dir", help="path to the extracted 'materials' folder (as produced "
                                             "by vmf_asset_extractor.py). When given, any displacement "
                                             "whose material is a genuine two-texture Source "
                                             "WorldVertexTransition blend gets converted into a real "
                                             "SOF2 terrain entity (func_group + tools/_terrain prism "
                                             "brushes + a generated .pcx alphamap) instead of a flat "
                                             "single-texture patch, so both blended textures actually "
                                             "show up in-game. Without this, displacements fall back to "
                                             "the flat-patch behavior regardless of their VMT type.")
    ap.add_argument("--terrain-mode", choices=["brush", "patch", "lwo"], default="brush",
                     help="How a terrain entity's actual displaced surface is represented (only "
                          "matters when --materials-dir is also given). 'brush' (default): one "
                          "watertight triangular-prism brush per displacement triangle - matches a "
                          "confirmed-working real SOF2 map's own structure exactly, but is currently "
                          "not being recognized as valid func_group entities in one user's GtkRadiant "
                          "build (unresolved as of this flag's introduction). 'patch': EXPERIMENTAL "
                          "alternative - wraps a single patchDef2 (the same proven displacement-patch "
                          "geometry used for non-blend displacements) inside the same terrain "
                          "func_group instead of prism brushes. Untested against q3map2's actual "
                          "terrain-compile behavior either way; try both and compare. 'lwo': EVERY "
                          "displacement in the map is welded (by shared world-space position, "
                          "regardless of shader) into ONE triangle mesh, exported as a single LWO2 "
                          "misc_model with per-vertex alpha (confirmed working live in SOF2 via a "
                          "separate isolated test) instead of a rasterized alphamap PCX - this "
                          "structurally eliminates patchDef2's seam/Bezier-mismatch/3-way-junction "
                          "problems, since a shared boundary vertex literally IS one shared point "
                          "rather than two coincident-but-separate ones. Needs spawnflags 6 on the "
                          "generated misc_model for collision (confirmed required in SOF2). See "
                          "--terrain-lwo-model-path for where the model key points.")
    ap.add_argument("--model-offsets", help="md3_origin_offsets.json written by mdl_to_md3.py. "
                     "Models taller/wider than MD3's 512-unit limit get their geometry moved back "
                     "into range there; this moves those entities' origins to compensate, so they "
                     "sit exactly where Source had them.")
    ap.add_argument("--keep-model-static", action="store_true",
                    help="Use --model-static-list even when the map has few enough unique models "
                         "to bake them all (by default every prop is baked as a misc_model when "
                         "there are at most {} unique models - see MODEL_STATIC_ONLY_ABOVE).".format(400))
    ap.add_argument("--lightmap-samplesize", type=int, default=0,
                     help="world units per lightmap pixel (0 = from Source: each face's own "
                          "lightmapscale, 16 by default, x --scale). Raising it makes lighting "
                          "blockier but costs fewer lightmap pages, and the engine creates one "
                          "shader per material per page - measured on de_mirage_cs2: 20 -> 27 "
                          "pages/293 combinations, 24 -> 23/263, 32 -> 15/223, 40 -> 11/206. "
                          "Raise it only when a map hits MAX_SHADERS.")
    ap.add_argument("--prop-vertex-budget", type=int, default=PROP_VERTEX_BUDGET, metavar="N",
                    help="most md3 vertices of baked prop geometry the .bsp may hold (needs "
                         "--md3-dir). Over it: Source's own lower LODs, then the heaviest props "
                         "networked as model_static up to what one snapshot can carry, then "
                         "reduced-detail copies, then the smallest props dropped - every step "
                         "listed in the log. Default {:,} = a {:.0f} MB .bsp target (see "
                         "PROP_VERTEX_BUDGET); 0 = no limit.".format(
                             PROP_VERTEX_BUDGET, BSP_SIZE_TARGET / 1048576.0))
    ap.add_argument("--md3-dir", help="the converted models folder (models_fixed/, step 3's output). "
                    "Read to plan the prop budget; reduced-detail copies of models "
                    "(<name>_simp1.md3 / _simp2.md3) are written here when the plan needs them.")
    ap.add_argument("--model-static-list", help="the 'md3_small_models.json' mdl_to_md3.py "
                    "writes listing small-clutter model paths (bottles, cans, tires - see its "
                    "own --small-model-max-extent). Any prop using one of these models spawns as "
                    "model_static instead of misc_model, by model identity rather than placement - "
                    "EXCEPT inside the 3D skybox area (with --skybox-entity), where a model is "
                    "always kept as misc_model: q3map2 only marks mapDrawSurface_t objects as "
                    "skybox surfaces and model_static never becomes one, so a model_static there "
                    "would never render as a distant backdrop at all.")
    ap.add_argument("--skip-infodecals", action="store_true",
                    help="Drop Source 'infodecal' entities instead of converting them. They are "
                         "converted by default: an infodecal is a bare origin plus a texture (no "
                         "basis, no uv rectangle, unlike info_overlay), which VBSP stamps onto "
                         "whatever face it rests against at the texture's natural size. This "
                         "converter now reproduces that - nearest face, oriented by that face's own "
                         "uaxis/vaxis, sized from --texture-sizes times the material's $decalscale, "
                         "and clipped to the face. They used to be dropped wholesale, which on "
                         "de_dust2 silently lost 135 decals including 33 'win_square' and 16 "
                         "'win_rectang' (reported missing in game). Their materials are added to "
                         "the overlay-materials list, so pass that to vmt2shader.py "
                         "--overlay-materials as usual for the polygonOffset/cull they need.")
    ap.add_argument("--distance-cull", type=float, default=0.0, metavar="UNITS",
                    help="Write a 'distancecull' worldspawn key (in FINAL map units, i.e. after "
                         "--scale). This is the Raven idTech3 far-clip key - the engine family SOF2 "
                         "shares with JK2/JA - and geometry past it is simply not drawn, which "
                         "looks like terrain and distant brushwork popping in as you approach. "
                         "0 (default) = don't write the key, so the engine default applies. Source "
                         "has no equivalent the engine reads: its own far clip is worldspawn "
                         "'MaxRange' (de_dust2 sets 8000, i.e. 10160 units at scale 1.27), which "
                         "is passed through untouched and ignored. Pick a value that comfortably "
                         "covers the map's own diagonal; the conversion prints the map's size and a "
                         "suggested value in its warnings.")
    ap.add_argument("--gridsize", default=None, metavar="\"X Y Z\"",
                    help="Write a 'gridsize' worldspawn key, e.g. \"192 192 384\". Sets the light "
                         "grid's cell size; the grid lights dynamic things (players, weapons, "
                         "model_static props), not world surfaces. Cost is map volume divided by "
                         "cell volume, and a converted map's bounding box is far larger than a "
                         "hand-built one because --seal-map-box must enclose the worldized skybox "
                         "props too. Measured on de_dust2: the default 64 64 128 gives 949,416 "
                         "grid points - enough that q3map2 shrinks the cells further to fit "
                         "MAX_MAP_LIGHTGRID and reports \"Storing adjusted grid size\" - while "
                         "192 192 384 gives about 91,000. Written here rather than added by hand "
                         "because a hand-added worldspawn key is lost the next time the map is "
                         "regenerated.")
    ap.add_argument("--ambient", default=None, metavar="N",
                    help="Override the worldspawn 'ambient' key - the shadow fill, a flat amount "
                         "q3map2 adds to every luxel. Normally you do not need this: it is derived "
                         "from the map's own light_environment _ambient, alongside the matching "
                         "q3map_sun vmt2shader.py puts on the sky shader, so the two add up to a "
                         "sunlit surface near lightmap level 235. It is in the same units as the "
                         "sun's intensity, so useful values are tens (de_dust2 derives 44 against "
                         "a sun of 220); single digits against a several-hundred sun do nothing. "
                         "Setting this also suppresses the derived '_color' tint. Raise it for "
                         "flatter, more open shadow, lower it for contrast.")
    ap.add_argument("--point-light-scale", type=float, default=1.0, metavar="N",
                    help="Multiply every converted point/spot light's brightness by N on its way "
                         "to q3map2's 'light' key. Default 1.0 (straight carry-over). There is no "
                         "derivable unit conversion between the two engines: both fall off with "
                         "inverse square, so a pure multiplier is the only physically meaningful "
                         "relation, but its value depends on constants neither engine documents. "
                         "Calibrate instead of guessing - de_dust2's lamps are authored at 10, 35, "
                         "50 and 150 against q3map2's unspecified-light default of 300, but they "
                         "are meant to be subtle accents on top of the sun and sky fill, so 1.0 is "
                         "a reasonable starting point once the sun is working.")
    ap.add_argument("--point-light-offset", type=float, default=0.0, metavar="N",
                    help="Add N to every converted light's brightness AFTER --point-light-scale. "
                         "Default 0. Provided for testing empirical remaps such as the "
                         "'brightness * 2.5 + 150' rule that circulates for Source-to-idTech3 "
                         "conversion, but be aware of what an offset does: it is not a unit "
                         "conversion (both engines are inverse-square, so no additive term can be "
                         "physical) and it CRUSHES the relative brightness the map was authored "
                         "with. On de_dust2, lights of 10 and 150 - a 15x range - come out as 175 "
                         "and 525, barely 3x apart, flattening every lighting contrast Valve "
                         "placed. If lights are too dim, raise --point-light-scale instead.")
    ap.add_argument("--seal-map-box", type=float, nargs="?", const=64.0, default=None,
                    metavar="MARGIN",
                    help="Seal the map by wrapping it in a tight closed caulk box, MARGIN units "
                         "clear of the geometry (default 64 when the flag is given with no value). "
                         "This is the blunt fix for the leak that --seal-terrain-under could not "
                         "close. In Source a displacement SURFACE seals the world; with "
                         "--terrain-mode lwo the terrain is a misc_model and a model seals nothing, "
                         "so the flood walks under the ground and out. A terrain-following prism "
                         "shell was tried and MEASURED as a net loss on de_dust2: 552 thin sloped "
                         "structural brushes took leafs from 8127 to 15234, tiny portals from 67 to "
                         "1416, references from 1.62M to 1.79M, and STILL leaked because the flood "
                         "passed beneath it. Six large axis-aligned brushes cost the BSP almost "
                         "nothing, and a closed box cannot be gone around. On de_dust2 the leak ran "
                         "at z -512..-674, below the playable map's own lowest brush at -406, so "
                         "the floor alone swallows the whole escape route. Geometry separated by a "
                         "large vertical gap (the playable map vs the 3D skybox's miniature copy) "
                         "gets one box each, so no box encloses the empty space between them. The "
                         "brushes are structural on purpose: q3map2 filters detail brushes in after "
                         "FloodEntities, so a detail box would seal nothing.")
    ap.add_argument("--seal-terrain-under", type=int, default=0, metavar="CELLS",
                    help="Seal the map under displacement terrain: add a thin STRUCTURAL "
                         "nodraw/nonsolid prism shell just below the ground, CELLS x CELLS cells "
                         "per displacement. 0 (default) = off. Try 2. This fixes a structural gap "
                         "rather than a mistake: in Source a displacement SURFACE seals the world, "
                         "but with --terrain-mode lwo the terrain is a misc_model and a model "
                         "seals nothing, while --drop-terrain-backing-brush removes the last brush "
                         "standing in for it - so the flood walks underneath the ground and out. "
                         "VERIFIED on de_dust2: 207 of the 1732 points sampled along its leak path "
                         "sit directly beneath displacement footprints, under 5 displacements, and "
                         "the whole path is empty in Source too. Tops come from the displacement's "
                         "own welded grid so neighbouring cells and neighbouring displacements "
                         "meet exactly (a continuous shell, no seams), offset 16 units down so it "
                         "cannot poke through the visible terrain. Raise CELLS if the shell is too "
                         "coarse to follow a steep slope; each step costs 2 brushes per cell per "
                         "displacement, and these are structural, so they do add BSP splits.")
    ap.add_argument("--seal-leak-from-lin", metavar="FILE",
                    help="Seal a compiler leak automatically, using the .lin pointfile q3map2 "
                         "wrote for the failing compile (next to the .bsp, same name). Source's "
                         "func_detail becomes q3map2 DETAIL, and q3map2 runs FloodEntities BEFORE "
                         "detail brushes enter the tree - so brushwork that seals the map in "
                         "Source can seal nothing here. This reads the leak path, finds every "
                         "detail brush it passes through (sampling along each segment, since the "
                         "pointfile's vertices are portal crossings and can straddle a brush), and "
                         "emits those as STRUCTURAL instead. VERIFIED on de_dust2: exactly 2 of "
                         "its 50 leak points lie inside Source solids, both func_detail - so this "
                         "promotes 2 brushes rather than making all 8354 detail brushes "
                         "structural. Iterate: recompile, and if it still leaks feed the NEW .lin "
                         "back in (each pass seals one route). If a pass reports that the path hit "
                         "no detail brush at all, the gap is empty in Source too and needs a "
                         "hand-built brush instead.")
    ap.add_argument("--skybox-landscape-brushes", type=int, default=0, metavar="CELLS",
                    help="Only with --skybox-entity. Rebuild the 3D skybox area's displacement "
                         "terrain as a COARSE SOLID BRUSH landscape (CELLS x CELLS quads per "
                         "displacement, 2 prism brushes each) instead of a full-resolution model, "
                         "so the distant landscape is actually visible as a backdrop without "
                         "blowing q3map2's skybox surface budget. 0 (default) = off. Try 3. "
                         "MEASURED on de_dust2: each skybox surface is replicated into ~1241 "
                         "sky-visible leaves against the 1048576 MAX_MAP_LEAFFACES limit, so the "
                         "whole 3D skybox can afford only ~827 surfaces - the terrain model alone "
                         "is 1024 triangles, while CELLS=3 over 8 displacements is 144. Only each "
                         "prism's TOP face carries the real material (sides and bottom are caulk), "
                         "so one prism costs one drawsurface, and the prisms are solid and "
                         "watertight so the landscape seals as well. Use together with "
                         "--skybox-models-to-world, or the skybox PROPS will still exceed the "
                         "budget by themselves (~7262 surfaces on de_dust2).")
    ap.add_argument("--skybox-models-to-world", action="store_true",
                    help="Move every prop in the 3D skybox area into REAL map space, at real "
                         "scale, instead of relying on q3map2's _skybox to draw it as a backdrop. "
                         "Uses Source's own relation P_world = (P_sky - sky_camera_origin) * "
                         "sky_camera_scale, with modelscale multiplied to match. VERIFIED against "
                         "de_dust2's own data: that map keeps the CROWNS of its 8 tall palms in "
                         "the 3D skybox (so they read as very tall) while the trunks are "
                         "full-size props in the playable map - this transform lands all 8 crowns "
                         "on their trunks with an identical ~3-unit residual (Valve's own grid "
                         "snapping at 1/4 scale), which is what confirms the mapping. The payoff "
                         "is that these become ORDINARY models: they cost nothing against "
                         "MAX_MAP_LEAFFACES, unlike a skybox surface (measured on de_dust2: every "
                         "skybox surface is replicated into ~1241 sky-visible leaves, so the whole "
                         "3D skybox has a budget of only ~827 surfaces against the 1048576 limit - "
                         "26 full-geometry palms alone are ~7262). Pair with --skip-3d-skybox to "
                         "drop the leftover miniature copy, and note distant horizon scenery will "
                         "land far outside the map at real scale, which is what it was pretending "
                         "to be.")
    ap.add_argument("--skybox-entity", action="store_true",
                    help="convert Source's sky_camera into q3map2's compiler-only '_skybox' "
                         "entity, so the 3D skybox area renders as a distant backdrop the way it "
                         "does in Source. OFF by default, and not because it doesn't work: "
                         "q3map2's _skybox makes every skybox surface visible from every leaf "
                         "that can see sky, so a detailed 3D skybox explodes the leaf-face count. "
                         "Measured on de_dust2, whose 3D skybox is a whole miniature city: 2749 "
                         "skybox surfaces took the compile from 31317 to 3438492 leaf references "
                         "against a 1048576 limit - MAX_MAP_LEAFFACES, compile failed. Worth "
                         "enabling only with a SIMPLE skybox (a few big brushes); for a detailed "
                         "one, simplify the skybox area first.")
    ap.add_argument("--skip-overlays", action="store_true",
                     help="Drop info_overlay entities (the old behaviour) instead of converting "
                          "them to q3map2 _decal projections.")
    ap.add_argument("--keep-terrain-base-solid", action="store_true",
                     help="Don't make nodraw slabs under displacement terrain non-solid (see "
                          "scan_terrain_base_slabs). Default: they become 'vmf2map/terrain_base', "
                          "which still seals but doesn't collide.")
    ap.add_argument("--terrain-lwo-steep-reproject", action="store_true",
                     help="Only with --terrain-mode lwo. Re-project UVs of steep terrain "
                          "triangles by dominant axis to reduce stretching. Off by default: "
                          "Source itself stretches textures on steep displacement faces (its "
                          "UVs follow the flat base face), and the default reproduces that.")
    ap.add_argument("--terrain-lwo-model-path", default="models/mapobjects/terrain",
                     help="Only used with --terrain-mode lwo: the VFS-style folder (no leading/"
                          "trailing slash needed) the generated misc_model's 'model' key will "
                          "reference, e.g. 'models/mapobjects/terrain' (default) produces "
                          "\"models/mapobjects/terrain/welded_terrain.lwo\". The actual .lwo file "
                          "is written next to your .map output regardless - copy it into your SOF2 "
                          "base folder at this same relative path (or pass the path you actually "
                          "intend to use here).")
    ap.add_argument("--skip-3d-skybox", action="store_true",
                     help="Exclude Source's 3D skybox area (a separate, far-away, scaled-down copy "
                          "of the map rendered as a distant backdrop, anchored by a 'sky_camera' "
                          "entity) - idTech3 has no equivalent rendering concept, so this content "
                          "just becomes stray, usually-misplaced geometry and props in the converted "
                          "map (confirmed real case: skybox-only palm tree props, name containing "
                          "'_skybx', ending up ~10000+ units from the main map, appearing to float "
                          "or otherwise misplaced). Off by default since this excludes content based "
                          "on distance from sky_camera, a heuristic that could in principle drop "
                          "legitimate far-flung main-map content on an unusual map - check the "
                          "printed warning showing how much was excluded.")
    ap.add_argument("--skybox-radius", type=float, default=4096.0,
                     help="with --skip-3d-skybox: exclude any world brush or entity within this "
                          "many units of the sky_camera entity's own origin. Default 4096 (a 3D "
                          "skybox is typically a compact, heavily scaled-down copy of the map, so "
                          "this should comfortably cover one without reaching into the real map, "
                          "given they're normally placed thousands of units apart specifically to "
                          "avoid any overlap) - increase if some skybox content still shows up, "
                          "decrease if real map content gets excluded by mistake.")
    ap.add_argument("--drop-terrain-backing-brush", action="store_true",
                     help="EXPERIMENTAL, opt-in: for a WorldVertexTransition displacement that "
                          "successfully became a real terrain entity (func_group), drop its "
                          "original flat base brush entirely instead of keeping it (with its "
                          "displacement face set to nodraw/caulk, all its other faces kept as "
                          "normal solid geometry). Confirmed in Hammer (the 'Disp Draw 3D' toggle) "
                          "that this flat base brush is purely an editor/fallback visualization in "
                          "Source - never separately visible or walked-on in the actual running "
                          "game, since the displacement fully replaces it. Dropping it may resolve "
                          "cases where this flat brush ends up visibly or physically overlapping "
                          "the real, bumpy terrain surface above/around it - but this is UNVERIFIED "
                          "against two real risks: whether the resulting patch alone provides "
                          "correct collision, and whether removing this brush leaves a compiler "
                          "leak where it used to help seal the world. Off by default - test a "
                          "compile both ways and compare before committing to either.")
    ap.add_argument("--terrain-min-power", type=int, default=None,
                     help="CONFIRMED PROBLEMATIC via real testing at value 4 - do not use without "
                          "re-verifying first. Was meant to raise every displacement's upsample "
                          "target power (e.g. 4 = 17x17 control points) to shrink each individual "
                          "Bezier segment's own footprint, reducing the visible size of a curve-"
                          "shape mismatch between adjacent patches. Real in-game testing instead "
                          "hit the compiler's own MAX_PATCH_PLANES limit and, worse, broke the "
                          "already-confirmed-working per-displacement edge alignment fix - so this "
                          "made real, working geometry worse, not just \"not enough of an "
                          "improvement\". Left here for reference/future re-investigation, not as a "
                          "usable option - see --terrain-edge-feather for the direction that "
                          "replaced this one.")
    ap.add_argument("--terrain-edge-feather", type=float, default=None,
                     help="EXPERIMENTAL, opt-in, off by default. Blends each displacement's own "
                          "vertex ONE STEP INWARD from a boundary shared with another displacement "
                          "of the same material toward that neighbor's own equivalent vertex, by "
                          "this weight (0 = no change, 1 = fully replace with the neighbor's "
                          "average - try something like 0.3-0.5 first). This targets the same "
                          "curve-shape divergence --terrain-min-power tried and failed to fix: "
                          "reconciling only the EXACT shared boundary vertex (always on, see "
                          "TERRAIN_ALPHA_RECONCILIATION) can still leave a visible seam, since each "
                          "patch's own Bezier curve approaching that boundary is also shaped by its "
                          "other, non-shared nearby control points. This adjusts alpha data only, "
                          "not geometry or resolution, so it doesn't carry --terrain-min-power's "
                          "same failure mode - but it is still unverified in-game as of writing. "
                          "Corner vertices, where 3+ displacements meet, are left unchanged (the "
                          "inward direction is ambiguous there).")
    ap.add_argument("--debug-terrain-corner-marker", action="store_true",
                     help="TEMPORARY DIAGNOSTIC. Overrides every terrain displacement's alphamap "
                          "to mark ONLY its (row 0, col 0) grid vertex - always Source's own "
                          "'startposition' corner, a known, fixed physical point at the "
                          "displacement's own base - as layer 1, forcing everything else to layer "
                          "0. Real alpha data is completely ignored while this is on - the goal is "
                          "a single unambiguous data point for how the compiler's alphamap "
                          "sampling maps image rows/columns onto real geometry, not a usable map. "
                          "After compiling with this on, report exactly where the single marked "
                          "spot appears on one specific, known displacement (relative to its "
                          "corners/edges) so the real row/column mapping can be confirmed instead "
                          "of guessed.")
    args = ap.parse_args()
    POINT_LIGHT_SCALE[0] = args.point_light_scale
    POINT_LIGHT_OFFSET[0] = args.point_light_offset

    if args.debug_terrain_corner_marker:
        global DEBUG_TERRAIN_CORNER_MARKER
        DEBUG_TERRAIN_CORNER_MARKER = True

    with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    texture_map = load_json_map(args.texture_map)
    if args.texture_sizes:
        raw_sizes = json.load(open(args.texture_sizes))
        TEXTURE_PIXEL_SIZES.update({k.lower(): tuple(v) for k, v in raw_sizes.items()})
    user_class_map = json.load(open(args.class_map)) if args.class_map else {}
    class_map = dict(DEFAULT_CLASS_MAP)
    class_map.update(user_class_map)  # user-supplied overrides win
    user_skip_classes = set(c.strip() for c in args.skip_classes.split(',') if c.strip())
    # An explicit --class-map entry for a normally-skipped classname means
    # "please convert this instead", so it takes priority over the default skip.
    skip_classes = (DEFAULT_SKIP_CLASSES | user_skip_classes) - set(user_class_map.keys())

    global LWO_STEEP_REPROJECT
    global TERRAIN_LWO_FILENAME
    global OVERLAY_MESH_FILENAME
    global MAP_SHADER_TAG
    MAP_SHADER_TAG = map_shader_tag(os.path.splitext(os.path.basename(args.output))[0])
    TERRAIN_LWO_FILENAME = os.path.splitext(os.path.basename(args.output))[0] + '_terrain.lwo'
    OVERLAY_MESH_FILENAME = os.path.splitext(os.path.basename(args.output))[0] + '_terrain_overlays.lwo'
    MODEL_ORIGIN_OFFSETS.clear()
    if args.model_offsets:
        with open(args.model_offsets, encoding='utf-8') as f:
            for k, v in json.load(f).items():
                MODEL_ORIGIN_OFFSETS[k.strip().lower().replace('\\', '/')] = tuple(float(c) for c in v)
    LWO_STEEP_REPROJECT = bool(args.terrain_lwo_steep_reproject)
    SKYBOX_WORLDIZE_RADIUS[0] = args.skybox_radius
    model_static_paths = None
    if args.model_static_list:
        with open(args.model_static_list, encoding='utf-8') as f:
            model_static_paths = {p.strip().lower() for p in json.load(f)}
        # Bake them anyway unless the map actually needs model_static.
        # A model_static is lit at runtime from the light grid with idTech3's
        # entity model - 0.6 x ambient + one directed light, clamped at 1.0 -
        # while a baked misc_model gets q3map2's per-vertex lighting from the
        # same sun and sky as the walls (plus the 2x stage in its _bsp shader),
        # which is also what CS:S does for prop_static. Simulated on dust2's
        # own sun and sky: a crate side in shade came out at 0.52 against the
        # 0.91 of the wall next to it, and nothing grid-lit can pass 1.0 where
        # a sunlit wall reaches 1.8 - the reported "crates and barrels look
        # dark". model_static only exists to stay under q3map2's MAX_MODELS
        # (512 model FILES loaded for baking) and to dodge MAX_MAP_LEAFFACES on
        # a leaking map; with few unique models there is nothing to dodge.
        n_models = len(set(re.findall(r'"model"\s+"(models/[^"]+)"', text, re.I)))
        if n_models <= MODEL_STATIC_ONLY_ABOVE and not args.keep_model_static:
            print("NOTE: {} unique models (< {}): baking every prop as misc_model instead of "
                  "using --model-static-list, so they are lit like the walls. Pass "
                  "--keep-model-static to use the list anyway.".format(
                      n_models, MODEL_STATIC_ONLY_ABOVE), file=sys.stderr)
            model_static_paths = None
    try:
        SOUND_OUT_DIR[0] = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'sound_sof2')
        (map_text, warnings, unmapped_counter, recenter_info, terrain_pcx_files, terrain_scales,
         terrain_lwo, terrain_lwo_scales, overlay_count, overlay_mesh_lwo, overlay_mesh_materials,
         terrain_sky_lwo, overlay_mesh_sky_lwo) = \
            convert_vmf(text, texture_map, class_map, skip_classes, args.scale, args.recenter,
                        args.model_scale, args.shader_file, args.materials_dir, args.terrain_mode,
                        args.skip_3d_skybox, args.skybox_radius, args.drop_terrain_backing_brush,
                        args.terrain_min_power, args.terrain_edge_feather,
                        args.terrain_lwo_model_path, args.keep_terrain_base_solid,
                        args.skip_overlays, args.skybox_entity, model_static_paths,
                        args.skybox_models_to_world, args.skybox_landscape_brushes,
                        args.seal_leak_from_lin, args.output, args.seal_terrain_under,
                        args.distance_cull, args.skip_infodecals,
                        args.seal_map_box, args.gridsize, args.ambient)
    except Exception as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        sys.exit(1)

    if terrain_pcx_files:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        for pcx_bytes, pcx_filename in terrain_pcx_files:
            pcx_path = os.path.join(out_dir, pcx_filename)
            with open(pcx_path, 'wb') as f:
                f.write(pcx_bytes)
        print("\nWrote {} terrain alphamap(s) alongside {}:".format(
            len(terrain_pcx_files), args.output), file=sys.stderr)
        for _, pcx_filename in terrain_pcx_files:
            print("  {}".format(pcx_filename), file=sys.stderr)

    if terrain_scales:
        scales_path = os.path.splitext(args.output)[0] + '_terrain_scales.json'
        with open(scales_path, 'w', encoding='utf-8') as f:
            json.dump(terrain_scales, f, indent=2)
        print("\nWrote {} (pass this to vmt2shader.py's --terrain-scales so each "
              "displacement's own uaxis/vaxis scale drives its own terrain shader's "
              "tcmod, instead of one flat value assumed for the whole map)".format(
                  scales_path), file=sys.stderr)

    if NONSOLID_MATERIALS:
        ns_path = os.path.splitext(args.output)[0] + '_nonsolid_materials.json'
        with open(ns_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(NONSOLID_MATERIALS), f, indent=2)
        print("\nNever-solid brush entities (func_brush Solidity 1, func_illusionary) use "
              "non-solid copies of {} shader(s). Wrote {} - pass it to vmt2shader.py "
              "--nonsolid-materials.".format(len(NONSOLID_MATERIALS), ns_path), file=sys.stderr)

    if OVERLAY_MATERIALS:
        ov_path = os.path.splitext(args.output)[0] + '_overlay_materials.json'
        with open(ov_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(OVERLAY_MATERIALS), f, indent=2)
        print("\nBuilt {} info_overlay(s) using {} material(s) (flat ones as patches, terrain-"
              "sitting ones as a dense conforming mesh - see below if any). Wrote {} - pass it "
              "to vmt2shader.py --overlay-materials so the PATCH-based ones get the "
              "polygonOffset/cull they need.".format(overlay_count, len(OVERLAY_MATERIALS),
                                                       ov_path), file=sys.stderr)
        if WRAPPED_OVERLAYS[0]:
            print("{} of them could not be one flat patch - their faces are not coplanar, or "
                  "the overlay's plane is tilted against its wall, or its faces don't fill it (past a "
                  "wall's end, across stair treads) - and were projected onto each "
                  "face the way VBSP builds them, as part of the overlay mesh.".format(
                      WRAPPED_OVERLAYS[0]), file=sys.stderr)

    if terrain_sky_lwo:
        sky_bytes, sky_filename = terrain_sky_lwo
        sky_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), sky_filename)
        with open(sky_path, 'wb') as f:
            f.write(sky_bytes)
        print("\nWrote {} ({} bytes) - the 3D SKYBOX area's terrain, split into its own model so "
              "q3map2's _skybox works. Copy it alongside the main terrain model.".format(
                  sky_path, len(sky_bytes)), file=sys.stderr)

    if overlay_mesh_sky_lwo:
        b, fn = overlay_mesh_sky_lwo
        pth = os.path.join(os.path.dirname(os.path.abspath(args.output)), fn)
        with open(pth, 'wb') as f:
            f.write(b)
        print("\nWrote {} ({} bytes) - overlays inside the 3D skybox area, as their own "
              "model.".format(pth, len(b)), file=sys.stderr)

    if overlay_mesh_lwo:
        lwo_bytes, lwo_filename = overlay_mesh_lwo
        out_dir = os.path.dirname(os.path.abspath(args.output))
        lwo_path = os.path.join(out_dir, lwo_filename)
        with open(lwo_path, 'wb') as f:
            f.write(lwo_bytes)
        print("\nWrote {} ({} bytes) alongside {} - copy this into your SOF2 base folder "
              "under {}/".format(lwo_path, len(lwo_bytes), args.output,
                                  OVERLAY_MESH_MODEL_PATH), file=sys.stderr)

        ov_mesh_path = os.path.splitext(args.output)[0] + '_overlay_mesh_materials.json'
        with open(ov_mesh_path, 'w', encoding='utf-8') as f:
            json.dump(overlay_mesh_materials, f, indent=2)
        print("Wrote {} - the flat shader names baked into the mesh's SURF/TAGS (overlayN), "
              "with each one's material. Pass it to vmt2shader.py "
              "--overlay-mesh-materials.".format(ov_mesh_path), file=sys.stderr)

    if terrain_lwo:
        lwo_bytes, lwo_filename = terrain_lwo
        out_dir = os.path.dirname(os.path.abspath(args.output))
        lwo_path = os.path.join(out_dir, lwo_filename)
        with open(lwo_path, 'wb') as f:
            f.write(lwo_bytes)
        print("\nWrote {} ({} bytes) alongside {} - copy this into your SOF2 base folder "
              "under {}/".format(lwo_path, len(lwo_bytes), args.output,
                                  args.terrain_lwo_model_path), file=sys.stderr)

        lwo_scales_path = os.path.splitext(args.output)[0] + '_terrain_lwo_shaders.json'
        with open(lwo_scales_path, 'w', encoding='utf-8') as f:
            json.dump(terrain_lwo_scales, f, indent=2)
        print("Wrote {} - the blended terrain shaders baked into the .lwo (terrainN), with each one's material and u/v scale. Pass it to vmt2shader.py --terrain-lwo-shaders. Non-blended displacements in the .lwo use their material's ordinary shader (textures/<material>), which vmt2shader.py already generates.".format(lwo_scales_path), file=sys.stderr)

    # Extra terrain/overlay models cut around water (find_fog_volume_boxes).
    # Parts an earlier run left behind are removed first, so a re-run with
    # fewer parts leaves nothing stale for the install step to pick up.
    _out_dir = os.path.dirname(os.path.abspath(args.output))
    for _stem in (os.path.splitext(TERRAIN_LWO_FILENAME)[0],
                  os.path.splitext(OVERLAY_MESH_FILENAME)[0]):
        for _old in (glob.glob(os.path.join(_out_dir, _stem + '_part*.lwo')) +
                     glob.glob(os.path.join(_out_dir, *(args.terrain_lwo_model_path.strip('/').split('/')
                                                       + [_stem + '_part*.lwo'])))):
            if re.match(r'^' + re.escape(_stem) + r'_part\d+\.lwo$', os.path.basename(_old)):
                os.remove(_old)
    for _b, _fn in FOG_SPLIT_LWO_FILES:
        with open(os.path.join(_out_dir, _fn), 'wb') as f:
            f.write(_b)
    if FOG_SPLIT_LWO_FILES:
        print("\nWrote {} extra model(s) cut around water volumes ({}) - they go in the same "
              "folder as the main terrain model.".format(
                  len(FOG_SPLIT_LWO_FILES), ', '.join(fn for _b, fn in FOG_SPLIT_LWO_FILES)),
              file=sys.stderr)

    map_text, short_names = shorten_map_texture_names(map_text)
    if getattr(args, 'md3_dir', None):
        _off = recenter_info[0] if recenter_info else (0.0, 0.0, 0.0)
        map_text, n_wet = unnetwork_props_in_fog(map_text, args.md3_dir, args.model_scale)
        prop_budget, decal_note = args.prop_vertex_budget, None
        if prop_budget:
            n_decal = decal_vertex_count(map_text)
            prop_budget = max(0, prop_budget - n_decal)
            decal_note = ("Decals: {:,} vertices of overlay mesh and decal patches go into the "
                          ".bsp too, so the prop budget was lowered by the same number to "
                          "{:,}.".format(n_decal, prop_budget))
        map_text, budget_report = plan_prop_budget(map_text, args.md3_dir, prop_budget,
                                                   args.model_scale, args.scale, _off)
        if decal_note:
            budget_report = [decal_note] + list(budget_report or [])
        map_text, n_dry = keep_baked_props_out_of_fog(map_text, args.md3_dir, args.model_scale)
        if n_wet or n_dry:
            budget_report = list(budget_report or []) + [
                "Props at water: {} networked prop(s) baked instead and {} baked prop(s) kept "
                "out of the water's fog - idTech3 fogs a whole model whose bounds touch the "
                "fog volume, Source only what is seen through the water (shaders: "
                "<map>_nofog_shaders.json -> vmt2shader.py --nofog-shaders).".format(n_wet, n_dry)]
        report_baked_prop_geometry(map_text, args.md3_dir)
        if budget_report:
            print("\n" + "\n".join(budget_report), file=sys.stderr)
    if short_names:
        print("\nShortened {} texture name(s) longer than {} characters (idTech3 shader names "
              "are 64 bytes; vmt2shader.py names the shaders the same way):".format(
                  len(short_names), QPATH_NAME_MAX), file=sys.stderr)
        for old_n, new_n in sorted(short_names.items()):
            print("  {} -> {}".format(old_n, new_n), file=sys.stderr)
    # A model_static (whether from plan_prop_budget's budget fallback or
    # --model-static-list) is loaded by the ENGINE at runtime, so its "model"
    # path is bound by idTech3's 63-char qpath limit. A baked misc_model's
    # path isn't: q3map2 reads that file straight off disk at compile time,
    # with no such limit, which is why mdl_to_md3.py never shortens the .md3
    # files' own paths - only their internal shader-name strings. Both
    # model_static sources above build their path by appending "_static.md3"
    # to that same long, unshortened original path, so a prop whose base
    # path is already near the limit (common in CS2-style deep model trees)
    # overflows once "_static" is added. Fix it here, once, after both
    # sources have run: shorten the overflowing path the same way texture
    # and shader names are shortened, and copy the .md3 mdl_to_md3.py already
    # wrote at the long name over to the short one so the short path on disk
    # actually exists.
    long_models = {}
    for blk in re.finditer(r'\{([^{}]*)\}', map_text):
        cls = re.search(r'"classname"\s+"([^"]+)"', blk.group(1))
        mdl = re.search(r'"model"\s+"(models/[^"]+)"', blk.group(1))
        if cls and mdl and cls.group(1).lower() != 'misc_model' and len(mdl.group(1)) > 63:
            long_models[mdl.group(1)] = None
    if long_models and getattr(args, 'md3_dir', None):
        for old in list(long_models):
            base, ext = os.path.splitext(old)
            short = short_qpath(base) + ext
            old_disk = os.path.join(args.md3_dir, *re.sub(r'^/?models/', '', old, flags=re.I).split('/'))
            short_disk = os.path.join(args.md3_dir,
                                       *re.sub(r'^/?models/', '', short, flags=re.I).split('/'))
            if os.path.isfile(old_disk):
                os.makedirs(os.path.dirname(short_disk), exist_ok=True)
                if not os.path.isfile(short_disk):
                    shutil.copyfile(old_disk, short_disk)
                map_text = map_text.replace('"{}"'.format(old), '"{}"'.format(short))
                long_models[old] = short
    fixed = {old: new for old, new in long_models.items() if new}
    still_long = sorted(old for old, new in long_models.items() if not new)
    if fixed:
        print("\nShortened {} runtime model path(s) longer than 63 characters (model_static is "
              "loaded by the engine at runtime, unlike a baked misc_model, which q3map2 reads "
              "straight off disk with no such limit) and copied the .md3 to match:".format(
                  len(fixed)), file=sys.stderr)
        for old, new in sorted(fixed.items()):
            print("  {} -> {}".format(old, new), file=sys.stderr)
    if still_long:
        print("\nWARNING: {} runtime model path(s) are over 63 characters; SOF2 cannot load "
              "them (baked misc_models are fine): {}".format(
                  len(still_long), ", ".join(still_long)), file=sys.stderr)
    # Lightmap density as Source has it. A Source face's luxel is its
    # "lightmapscale" in Source units, 16 by default (96% of de_mirage_cs2's
    # and de_dust2's faces) - i.e. 16 * scale map units here. The compile used
    # a flat 16, 1.56x Source's density at scale 1.25: more lightmap pages, and
    # idTech3 makes one runtime shader per (shader, lightmap page), so more
    # shaders - de_overpass_csgo hit SOF2's MAX_SHADERS. Measured on
    # de_mirage_cs2: 39 -> 27 pages, 341 -> 293 lightmap/shader combinations.
    # Set per entity because -samplesize on the -light command line is
    # ignored: -bsp writes the default into the .srf and -light uses that. A
    # shader's own q3map_lightmapSampleSize (terrain) still takes precedence.
    src_lm, lm_counts = source_lightmapscale(text)
    lm_ss = args.lightmap_samplesize or max(1, int(round(src_lm * args.scale)))
    map_text = re.sub(r'(\{\n"classname" "(?:worldspawn|func_[A-Za-z_]+)"\n)',
                      lambda m: m.group(1) + '"_lightmapsamplesize" "{}"\n'.format(lm_ss), map_text)
    if args.lightmap_samplesize:
        how = "--lightmap-samplesize"
    else:
        total = sum(lm_counts.values()) or 1
        how = "this map's own lightmapscale {:g} on {:.0f}% of its {} faces, x scale {:g}".format(
            src_lm, 100.0 * lm_counts.get(src_lm, 0) / total, total, args.scale)
    print("\nLightmap sample size {} ({}) set on worldspawn and brush entities. The engine "
          "makes one shader per material PER LIGHTMAP PAGE, so this is also what decides how "
          "close the map runs to MAX_SHADERS - run bsp_shader_count.py on the finished .bsp to "
          "see the total. To change it later there is no need to convert again: it is the "
          "\"_lightmapsamplesize\" key on worldspawn, editable in Radiant.".format(lm_ss, how),
          file=sys.stderr)
    # Faces Source itself lit coarsely are worth knowing about: idTech3 cannot
    # follow them per face, so they are being lit FINER here than CS:S lit
    # them, for no visual gain and at a cost in lightmap pages.
    coarse = sum(n for v, n in lm_counts.items() if v > src_lm * 1.9)
    if coarse and sum(lm_counts.values()):
        print("  ({} of its faces had a lightmapscale over {:g} in Source - up to {:g} - and "
              "are now lit at {} like everything else. q3map2 has no per-face luxel size, so "
              "that density cannot be followed face by face.)".format(
                  coarse, src_lm * 1.9, max(lm_counts), lm_ss), file=sys.stderr)

    map_text = strip_source_prop_keys(map_text)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(map_text)

    print("Wrote {}".format(args.output), file=sys.stderr)

    nofog_path = os.path.splitext(args.output)[0] + '_nofog_shaders.json'
    with open(nofog_path, 'w', encoding='utf-8') as f:
        json.dump(dict(NOFOG_SHADER_TARGETS, **TERRAIN_NOFOG_TARGETS), f, indent=1, sort_keys=True)

    # Written on every run, empty when nothing was dropped, so a stale list
    # from an earlier run can never take a prop's collision away.
    dropped_path = os.path.splitext(args.output)[0] + '_dropped_props.json'
    with open(dropped_path, 'w', encoding='utf-8') as f:
        json.dump(list(DROPPED_PROP_IDS), f)
    if DROPPED_PROP_IDS:
        print("Wrote {} - the {} dropped prop(s); pass it to phy2clip.py --skip-ids so their "
              "collision is left out too.".format(dropped_path, len(DROPPED_PROP_IDS)),
              file=sys.stderr)

    if recenter_info:
        offset, bounds = recenter_info
        print("\n--- Recentered ---", file=sys.stderr)
        print("  original bounds: min {} max {}".format(bounds[0], bounds[1]), file=sys.stderr)
        print("  shifted by: (-{:.6g}, -{:.6g}, -{:.6g})".format(*offset), file=sys.stderr)
        print("  NOTE: if this map connects to/reuses coordinates from other sources", file=sys.stderr)
        print("  (docs, trigger targets defined elsewhere, etc.) remember everything", file=sys.stderr)
        print("  moved together by this same offset.", file=sys.stderr)

    if warnings:
        print("\n--- Warnings ({}) ---".format(len(warnings)), file=sys.stderr)
        seen = Counter(warnings)
        for w, count in seen.most_common():
            suffix = " (x{})".format(count) if count > 1 else ""
            print("  - {}{}".format(w, suffix), file=sys.stderr)

    if unmapped_counter:
        shader_names = set()
        if args.shader_file and os.path.isfile(args.shader_file):
            with open(args.shader_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    # shader block names are bare lines (no braces/keywords)
                    if line and not line.startswith(('{', '}', '/', '\t')) and ' ' not in line:
                        shader_names.add(line.lower().lstrip('/'))

        if shader_names:
            # A brush face material reference is ALWAYS resolved by the
            # engine through the implicit 'textures/' prefix - never the
            # bare name, even if a shader happens to exist under the bare
            # name too (that form only ever satisfies a MODEL's own,
            # separate lookup). Accepting the bare name here as if it were
            # equivalent was confirmed to hide a real bug: a material that
            # lives in a folder some .mdl's cdmaterials also uses got
            # written to the .shader file bare-only, this check called it
            # "resolved" because the bare name existed, and GtkRadiant then
            # failed to activate it in-game with "Activate shader failed
            # for textures/<path>" - the one name that's actually needed
            # for a world brush was never there.
            missing = {}
            for tex, count in unmapped_counter.items():
                key = tex.lower().lstrip('/')
                if ('textures/' + key) not in shader_names:
                    missing[tex] = count
            resolved = len(unmapped_counter) - len(missing)
            print("\n--- Texture cross-check against {} ---".format(args.shader_file), file=sys.stderr)
            print("{} of {} material(s) have a matching shader definition (fine, nothing to do)."
                  .format(resolved, len(unmapped_counter)), file=sys.stderr)
            if missing:
                print("\n{} material(s) have NO shader and NO --texture-map entry - these are"
                      .format(len(missing)), file=sys.stderr)
                print("the ones that will actually show up missing in-game:", file=sys.stderr)
                for tex, count in sorted(missing.items(), key=lambda kv: -kv[1]):
                    print("  - {}  (used on {} side(s))".format(tex, count), file=sys.stderr)
            else:
                print("Nothing missing - every material resolves to a generated shader.",
                      file=sys.stderr)
        else:
            print("\n--- Textures passed through without a --texture-map entry ({} unique) ---"
                  .format(len(unmapped_counter)), file=sys.stderr)
            print("This is EXPECTED for a normal conversion: these keep their Source", file=sys.stderr)
            print("paths and are meant to resolve to the shaders vmt2shader.py generates", file=sys.stderr)
            print("from the same material names. Pass --shader-file <your.shader> to have", file=sys.stderr)
            print("this cross-checked automatically and list only genuinely missing ones.", file=sys.stderr)
            print("(--texture-map is only for repointing a material at a DIFFERENT texture.)",
                  file=sys.stderr)
            for tex, count in unmapped_counter.most_common(20):
                print("  - {}  (used on {} side(s))".format(tex, count), file=sys.stderr)
            if len(unmapped_counter) > 20:
                print("  ... and {} more".format(len(unmapped_counter) - 20), file=sys.stderr)


if __name__ == "__main__":
    main()
