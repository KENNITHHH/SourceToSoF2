#!/usr/bin/env python3
"""Count the shaders SOF2 will create for a compiled .bsp (MAX_SHADERS).

The renderer does NOT make one shader per material. It makes one per
(material name, lightmap page) pair: tr_shader.cpp's IsShader() compares the
name AND every lightmapIndex/style before reusing an existing shader, and
tr_bsp.c asks for each surface's material with that surface's own lightmap
number. So a wall texture spread over 12 lightmap pages is 12 shaders, and a
map can run out long before it runs out of textures ("WARNING:
GeneratePermanentShader - MAX_SHADERS hit", after which surfaces draw with
the default shader).

Counted here:
  world    distinct (shader, lightmapNum[]) pairs over the BSP's own surfaces
  models   shader names inside every .md3 the entity lump references
           (model_static and friends are loaded at runtime, unlit by a
           lightmap, so each name is one more shader)
The base game adds its own on top - HUD, weapons, players, effects - which
this cannot see. Measured once on a real install: de_mirage_cs2 loaded with
`shaderlist` reporting 1237 total while this script counted 923 for the map,
so the base game cost 314 (but see BASE_GAME_SHADERS below - that 923 was
miscounted) and the table holds 2048 (SOF2 MP prints
"GeneratePermanentShader - MAX_SHADERS hit" and draws the rest with the
default shader once it is full). Hence the budget below. Check it on your own
install the same way: load a map that works, run `shaderlist`, subtract what
this prints for that map.

Usage:
  bsp_shader_count.py <map.bsp> [--base <sof2/base folder>] [--top N]
"""

import argparse
import os
import re
import struct
import sys
import zipfile
from collections import Counter

MAX_SHADERS = 2048
# What the base game registers on top of the map (HUD, weapons, player models,
# effects). MEASURED on a real install: de_mirage_cs2 reported 1233 with
# `shaderlist` in game while this script counted 815 for the map, so 418.
#
# It is a floor, not a fixed cost: the game registers more as a session runs -
# each player model and skin, each weapon drawn, gore and explosion effects.
# de_overpass_csgo counts 1562 here, and 1562 + 418 = 1980 is inside the 2048
# table, yet it still hit MAX_SHADERS in play - about 70 shaders' worth of
# runtime registration took it over. That is what SAFETY below is for: treat
# anything within it of the budget as over, not as fitting.
#
# Measure it on your own install: load a map that works, run `shaderlist`,
# then run this on THAT map's .bsp with --calibrate <the total it printed>.
BASE_GAME_SHADERS = 418
SAFETY = 100

RBSP_SURFACE_SIZE = 148
RBSP_SHADER_SIZE = 72
MAXLIGHTMAPS = 4


def lumps(data):
    if data[:4] not in (b'RBSP', b'IBSP'):
        raise ValueError("not a .bsp (ident {!r})".format(data[:4]))
    return [struct.unpack_from('<ii', data, 8 + i * 8) for i in range(18)]


def md3_shader_names(path=None, data=None):
    """Every surface shader name in an .md3, from a file or from its bytes."""
    d = data
    if d is None:
        try:
            with open(path, 'rb') as f:
                d = f.read()
        except OSError:
            return None
    if len(d) < 108 or d[:4] != b'IDP3':
        return None
    names = []
    nsurf = struct.unpack_from('<i', d, 84)[0]
    off = struct.unpack_from('<i', d, 100)[0]
    for _ in range(nsurf):
        if off + 108 > len(d) or d[off:off + 4] != b'IDP3':
            break
        nshader = struct.unpack_from('<i', d, off + 76)[0]
        # md3Surface_t: ... numShaders @76, numVerts @80, numTriangles @84,
        # ofsTriangles @88, ofsShaders @92, ofsSt @96, ofsXyzNormals @100.
        # This read @96 until fix AM - the texture coordinates, so every
        # surface produced one unique garbage "name" and the model count was
        # really a surface count (measured on de_cache's 151 models: 170
        # counted against 121 real shader names).
        sh_off = struct.unpack_from('<i', d, off + 92)[0]
        end = struct.unpack_from('<i', d, off + 104)[0]
        for s in range(nshader):
            p = off + sh_off + s * 68
            if p + 64 <= len(d):
                names.append(d[p:p + 64].split(b'\0')[0].decode('latin-1').lower())
        off += end
    return names


def base_archives(base, map_name):
    """Every .pk3 in base, this map's own first: since step 10 packs the
    models into base/<map>.pk3 they are no longer loose files."""
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for fn in sorted(names, key=lambda n: n.lower() != map_name + '.pk3'):
        if fn.lower().endswith('.pk3'):
            try:
                z = zipfile.ZipFile(os.path.join(base, fn))
                out.append((z, {n.lower(): n for n in z.namelist()}))
            except (OSError, zipfile.BadZipFile):
                pass
    return out


def read_from_archives(archives, rel):
    key = rel.replace('\\', '/').lower()
    for z, names in archives:
        if key in names:
            return z.read(names[key])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bsp")
    ap.add_argument("--base", help="SOF2 base folder, to resolve the .md3 models the "
                                   "entity lump names (default: two levels up from the .bsp)")
    ap.add_argument("--top", type=int, default=15,
                    help="how many of the worst materials to list (default 15)")
    ap.add_argument("--calibrate", type=int, metavar="TOTAL",
                    help="the total `shaderlist` printed in game with THIS map loaded; "
                         "prints what the base game costs on your install")
    args = ap.parse_args()

    with open(args.bsp, 'rb') as f:
        data = f.read()
    L = lumps(data)

    sh_off, sh_len = L[1]
    shaders = [data[sh_off + i * RBSP_SHADER_SIZE:
                    sh_off + i * RBSP_SHADER_SIZE + 64].split(b'\0')[0].decode('latin-1')
               for i in range(sh_len // RBSP_SHADER_SIZE)]

    srf_off, srf_len = L[13]
    # Counted the way the engine's own loader does (rd-vanilla/tr_bsp.cpp):
    # ShaderForShaderNum passes the surface's lightmapNum[4] and styles to
    # R_FindShader, which reuses a shader only when the NAME and all four
    # lightmap indices and styles match (IsShader). Two details matter:
    # triangle-soup surfaces are looked up with lightmapsVertex and flares
    # with {LIGHTMAP_BY_VERTEX,0,0,0} whatever the BSP stores, and when
    # lightmapNum[0] is LIGHTMAP_BY_VERTEX the VERTEX styles are the key.
    VERTEX = (-3, -3, -3, -3)
    pairs = set()
    per_shader = Counter()
    lm_pages = set()
    for i in range(srf_len // RBSP_SURFACE_SIZE):
        o = srf_off + i * RBSP_SURFACE_SIZE
        shader_num = struct.unpack_from('<i', data, o)[0]
        surface_type = struct.unpack_from('<i', data, o + 8)[0]
        # 7 ints, then lightmapStyles[4], vertexStyles[4], lightmapNum[4]
        lm_styles, vert_styles = data[o + 28:o + 32], data[o + 32:o + 36]
        lm = struct.unpack_from('<4i', data, o + 36)
        if surface_type == 3:       # MST_TRIANGLE_SOUP
            key_lm = VERTEX
        elif surface_type == 4:     # MST_FLARE
            key_lm = (-3, 0, 0, 0)
        else:
            key_lm = lm
        styles = vert_styles if key_lm[0] == -3 else lm_styles
        name = shaders[shader_num] if 0 <= shader_num < len(shaders) else '?'
        head, _, tail = name.rpartition('.')
        if head and '/' not in tail:
            name = head                      # COM_StripExtension
        key = (name.lower(), key_lm, bytes(styles))
        if key not in pairs:
            pairs.add(key)
            per_shader[name] += 1
        for n in lm:
            if n >= 0:
                lm_pages.add(n)

    ent = data[L[0][0]:L[0][0] + L[0][1]].decode('latin-1', 'replace')
    base = args.base or os.path.dirname(os.path.dirname(os.path.abspath(args.bsp)))
    map_name = os.path.splitext(os.path.basename(args.bsp))[0].lower()
    archives = base_archives(base, map_name)
    model_names, missing = set(), set()
    for rel in sorted(set(re.findall(r'"model"\s+"(models/[^"]+\.md3)"', ent, re.I))):
        names = md3_shader_names(os.path.join(base, *rel.split('/')))
        if names is None:
            blob = read_from_archives(archives, rel)
            names = md3_shader_names(data=blob) if blob is not None else None
        if names is None:
            missing.add(rel)
        else:
            model_names.update(n.lower() for n in names)

    world = len(pairs)
    print("{}:".format(os.path.basename(args.bsp)))
    print("  {:6d} world shaders   ({} materials over {} lightmap pages)".format(
        world, len(per_shader), len(lm_pages)))
    print("  {:6d} model shaders   (from {} .md3 file(s) the entities name{})".format(
        len(model_names),
        len(set(re.findall(r'"model"\s+"(models/[^"]+\.md3)"', ent, re.I))) - len(missing),
        "; {} not found under {}".format(len(missing), base) if missing else ""))
    total = world + len(model_names)
    print("  {:6d} TOTAL from this map, before the base game's own".format(total))
    budget = MAX_SHADERS - BASE_GAME_SHADERS
    if args.calibrate:
        print("\n  calibration: {} in game - {} for this map = {} for the base game on your "
              "install.\n  Put that in BASE_GAME_SHADERS at the top of this script.".format(
                  args.calibrate, total, args.calibrate - total))
    elif total <= budget - SAFETY:
        print("  budget {} = {} (MAX_SHADERS) - {} (base game, measured): fits, {} to spare".format(
            budget, MAX_SHADERS, BASE_GAME_SHADERS, budget - total))
    else:
        print("  budget {} = {} (MAX_SHADERS) - {} (base game, measured): {}".format(
            budget, MAX_SHADERS, BASE_GAME_SHADERS,
            "CLOSE, only {} to spare".format(budget - total) if total <= budget
            else "OVER by {}".format(total - budget)))
        print("  The base game also registers more shaders as a session runs (player models,\n"
              "  weapons, gore and explosion effects), so a map that only just fits at load\n"
              "  can still hit MAX_SHADERS in play - de_overpass_csgo did, 68 under the\n"
              "  budget. Leave at least {} spare.".format(SAFETY))
    if total > budget - SAFETY:
        print("\n  To cut the world count: make sure the .shader carries the per-material\n"
              "  q3map_lightmapBrightness lines vmt2shader.py writes (they stop one texture\n"
              "  inheriting a lightmap page from every other texture on the same wall -\n"
              "  measured on de_overpass_csgo: 1383 world shaders -> 517), and that -light runs\n"
              "  with -lightmapsearchblocksize 2. After that the remaining lever is fewer\n"
              "  distinct materials; raising worldspawn's _lightmapsamplesize barely helps once\n"
              "  the two above are in place (62 -> 33 pages moved it only 1383 -> 1324).")

    if per_shader:
        print("\n  materials costing the most (copies = lightmap pages they land on):")
        for name, n in per_shader.most_common(args.top):
            if n < 2:
                break
            print("    {:4d} x  {}".format(n, name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
