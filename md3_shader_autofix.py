#!/usr/bin/env python3
"""
md3_shader_autofix.py - Batch-fix .md3 shader names, correctly, automatically.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
md3_shader_tool.py + md3_shader_map_builder.py fix shader names by matching
a bare name (e.g. "truck001c_01") against .vmt FILENAMES across your whole
materials tree. That's inherently unreliable, not just buggy: Source
content regularly reuses the same bare texture name for two completely
different props in different maps (this is confirmed, not theoretical -
e.g. "truck001c_01" legitimately exists as both
materials/models/props/de_train/truck001c_01.vmt AND
materials/models/props_vehicles/truck001c_01.vmt, for two different
vehicles). A bare-name lookup can flag that as ambiguous, but it can never
correctly RESOLVE it, because the bare name genuinely doesn't carry enough
information - the answer depends on which specific model you're asking
about, which the bare name has already thrown away.

fix_md3_shaders.py doesn't have this problem: it reads each .md3's OWN
matching .mdl file directly and uses THAT model's own cdmaterials +
texture table. Two different props with colliding bare texture names
resolve independently and correctly, because each is read from its own
source of truth, not a shared global name lookup.

So: this script tries the guaranteed-correct .mdl-based fix FIRST for
every .md3 (same logic as fix_md3_shaders.py), and only falls back to
bare-name/.vmt-basename matching (same logic as md3_shader_tool.py +
md3_shader_map_builder.py) for the surfaces where no matching .mdl was
found at all. Bare-name fallback matches are still checked for ambiguity
and reported for manual review rather than guessed - some fraction of
"no .mdl available" cases genuinely can't be resolved automatically, and
this tells you exactly which ones instead of silently picking one.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 md3_shader_autofix.py \\
        --md3-dir "E:\\CSS\\Extracted\\de_cache_files\\models" \\
        --mdl-dir "E:\\CSS\\Extracted\\de_cache_files\\models" \\
        --materials-dir "E:\\CSS\\Extracted\\de_cache_files\\materials" \\
        --output "E:\\CSS\\Extracted\\de_cache_files\\models_fixed" \\
        --manual-map manual_overrides.json

--md3-dir        : folder of .md3 files to fix (searched recursively)
--mdl-dir        : folder of the ORIGINAL .mdl files (searched recursively;
                   often the same folder as --md3-dir if your Noesis batch
                   export dropped .md3s alongside the source .mdls)
--materials-dir  : your .vmt root, for the bare-name fallback path
--output         : where fixed .md3 copies go (originals untouched)
--manual-map     : optional JSON {"bare_name": "full/path"} for cases the
                   fallback can't resolve on its own (same format as
                   md3_shader_tool.py's --map). Checked before falling
                   back to basename search, so it also lets you override
                   a specific bad case without editing this script.

Requires vmt2shader.py in the same folder (used for .vmt parsing/patch
resolution in the fallback path, same as md3_shader_map_builder.py).

At the end, prints a breakdown of how many shader references were fixed
via the guaranteed-correct .mdl path vs. the fallback path, and writes
every still-unresolved bare name (with its candidates, if any) to
<output>/_unresolved_shaders.txt so you know exactly what still needs a
manual --manual-map entry.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
from collections import defaultdict

try:
    import vmt2shader as vt
except ImportError:
    print("ERROR: vmt2shader.py must be in the same folder as this script.", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
# .mdl parsing (same as fix_md3_shaders.py)
# ==============================================================================

def cstr(data, off):
    end = data.index(b'\x00', off)
    return data[off:end].decode('latin1')


def join_material_path(cdmat, tex):
    """cdmaterials dir + texture name, as one forward-slash path. Either part
    can be empty or carry its own slashes: a model with cdmaterials "" and
    texture "models/props/de_dust/sign_street01" (de_mirage_cs2's signs and
    rugs) used to come out as "/models/...", which matched no .vmt and no
    shader."""
    parts = [p for p in re.split(r'[\\/]+', (cdmat or '') + '/' + (tex or '')) if p]
    return '/'.join(parts)


def _resolve_cdmat_for_texture(cdmaterials, texname, materials_dir):
    """Pick the cdmaterials directory that actually has a VMT for this
    texture, trying each one in the model's own listed order - rather
    than assuming the first cdmaterials entry is always right. Mirrors
    the identical fix in mdl_to_md3.py: a model with a per-map override
    folder listed first, and shared fallback folders after it, can have
    its ACTUAL vmt for a given texture living under one of the later
    folders. Without this, autofix would independently re-derive the
    WRONG (cdmaterials[0]-based) name and stamp it back into the .md3,
    even after mdl_to_md3.py had already embedded the correct one -
    confirmed against a real map where exactly this happened.

    Falls back to cdmaterials[0] (previous behavior) when materials_dir
    isn't given or none of the candidates can be found there."""
    if materials_dir and os.path.isdir(materials_dir):
        for cdmat in cdmaterials:
            cdmat_norm = cdmat.rstrip('\\/').replace('\\', '/')
            candidate_dir = os.path.join(materials_dir, *cdmat_norm.split('/')) if cdmat_norm else materials_dir
            if os.path.isfile(os.path.join(materials_dir, *join_material_path(cdmat_norm, texname).split('/')) + '.vmt'):
                return cdmat_norm
            if os.path.isdir(candidate_dir):
                target = (texname + '.vmt').lower()
                try:
                    if any(fn.lower() == target for fn in os.listdir(candidate_dir)):
                        return cdmat_norm
                except OSError:
                    pass
    return cdmaterials[0].rstrip('\\/').replace('\\', '/') if cdmaterials else ''


def parse_mdl_shader_paths(path, materials_dir=None):
    with open(path, 'rb') as f:
        data = f.read()

    if len(data) < 240 or data[0:4] != b'IDST':
        raise ValueError("not a Source studiomdl (bad magic)")

    numtextures, textureindex = struct.unpack_from('<ii', data, 204)
    numcdtextures, cdtextureindex = struct.unpack_from('<ii', data, 212)
    numskinref, numskinfamilies, skinindex = struct.unpack_from('<iii', data, 220)
    numbodyparts, bodypartindex = struct.unpack_from('<ii', data, 232)

    warnings = []
    cdmaterials = []
    for i in range(numcdtextures):
        off = struct.unpack_from('<i', data, cdtextureindex + i * 4)[0]
        cdmaterials.append(cstr(data, off))

    texnames = []
    for i in range(numtextures):
        tex_struct_off = textureindex + i * 64
        sznameindex = struct.unpack_from('<i', data, tex_struct_off)[0]
        texnames.append(cstr(data, tex_struct_off + sznameindex))

    skin_family0 = []
    for i in range(numskinref):
        idx = struct.unpack_from('<h', data, skinindex + i * 2)[0]
        skin_family0.append(texnames[idx])

    # Walk every mesh, across every bodypart, resolving each one's OWN
    # material index into skin_family0 - rather than assuming mesh position
    # in the surface list lines up 1:1 with skinref position. That
    # assumption holds for the common case (each mesh has its own distinct
    # skinref) but silently breaks whenever multiple meshes share a single
    # skinref (confirmed against a real model: two bodyparts, one shared
    # texture, numskinref=1 but 2 meshes/surfaces - the old code produced a
    # 1-entry list and left the second surface "unresolved" even though its
    # correct shader was fully determinable). Mirrors the per-mesh
    # resolution mdl_to_md3.py already does correctly, so both tools now
    # agree by construction rather than by coincidence.
    paths = []
    for bpi in range(numbodyparts):
        bp_off = bodypartindex + bpi * 16
        sznameindex, nummodels, base, modelindex = struct.unpack_from('<4i', data, bp_off)
        model_off = bp_off + modelindex  # model 0 only, matching mdl_to_md3.py's convention
        o = model_off + 64
        nummeshes, meshindex = struct.unpack_from('<ii', data, o + 8)
        for m in range(nummeshes):
            mesh_off = model_off + meshindex + m * 116
            material, _back = struct.unpack_from('<ii', data, mesh_off)
            tex = skin_family0[material] if material < len(skin_family0) else None
            if tex is not None and cdmaterials:
                cdmat_norm = _resolve_cdmat_for_texture(cdmaterials, tex, materials_dir)
                paths.append(join_material_path(cdmat_norm, tex).lower())
            else:
                paths.append(None)

    return paths, warnings


def build_mdl_index(mdl_root):
    index = {}
    for dirpath, _, filenames in os.walk(mdl_root):
        for fn in filenames:
            if fn.lower().endswith('.mdl'):
                key = os.path.splitext(fn)[0].lower()
                if key not in index:
                    index[key] = os.path.join(dirpath, fn)
    return index


# ==============================================================================
# .md3 binary I/O (same layout as md3_shader_tool.py / fix_md3_shaders.py)
# ==============================================================================

MAX_QPATH = 64
MD3_IDENT = b"IDP3"


def decode_qpath(raw):
    return raw.split(b'\x00', 1)[0].decode('ascii', errors='replace')


SHORTENED = {}   # full name -> short name written, for the summary


def encode_qpath(name):
    # Over-long names (CS:GO/CS2 content: models/props/de_dust/hr_dust/...) get
    # vmt2shader's deterministic short form, which leaves room for q3map2
    # -rename's _BSP. vmt2shader.py renames the shader block the same way, so
    # the two always agree. (This used to raise ValueError and stop the run.)
    short = vt.short_qpath(name)
    if short != name:
        SHORTENED[name] = short
    b = short.encode('ascii', errors='replace')
    return b + b'\x00' * (MAX_QPATH - len(b))


def read_surfaces(data):
    if data[0:4] != MD3_IDENT:
        raise ValueError("not a valid .md3 file (bad ident)")
    numSurfaces = struct.unpack_from('<i', data, 84)[0]
    ofsSurfaces = struct.unpack_from('<i', data, 100)[0]

    surf_off = ofsSurfaces
    for _ in range(numSurfaces):
        snumShaders = struct.unpack_from('<i', data, surf_off + 76)[0]
        ofsShaders = struct.unpack_from('<i', data, surf_off + 92)[0]
        ofsEndSurf = struct.unpack_from('<i', data, surf_off + 104)[0]

        shaders = []
        for i in range(snumShaders):
            sh_off = surf_off + ofsShaders + i * 68
            sh_name_raw = struct.unpack_from('<64s', data, sh_off)[0]
            shaders.append((sh_off, decode_qpath(sh_name_raw)))

        yield surf_off, shaders
        surf_off += ofsEndSurf


def iter_md3_files(root):
    for r, _dirs, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith('.md3'):
                yield os.path.join(r, fname)


# ==============================================================================
# fallback: bare-name -> .vmt basename matching (same as md3_shader_map_builder.py)
# ==============================================================================

def build_vmt_basename_index(materials_dir):
    index = defaultdict(list)
    for root, _dirs, files in os.walk(materials_dir):
        for fname in files:
            if not fname.lower().endswith('.vmt'):
                continue
            base = os.path.splitext(fname)[0].lower()
            index[base].append(os.path.join(root, fname))
    return index


def resolve_basetexture(vmt_path, vmt_include_index):
    shader_class, body = vt.resolve_vmt(vmt_path, vmt_include_index)
    basetexture = vt.get_param(body, 'basetexture')
    if not basetexture:
        return None
    return vt.with_textures_prefix(basetexture)


# ==============================================================================
# main pipeline
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md3-dir", required=True)
    ap.add_argument("--mdl-dir", required=True)
    ap.add_argument("--materials-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manual-map", help="JSON {bare_name: full/path} checked before basename fallback search")
    args = ap.parse_args()

    manual_map = {}
    if args.manual_map and os.path.exists(args.manual_map):
        with open(args.manual_map, 'r', encoding='utf-8') as f:
            manual_map = json.load(f)

    print("Indexing .mdl files under {}...".format(args.mdl_dir), file=sys.stderr)
    mdl_index = build_mdl_index(args.mdl_dir)
    print("  found {} mdl files".format(len(mdl_index)), file=sys.stderr)

    print("Indexing .vmt files under {} (fallback path)...".format(args.materials_dir), file=sys.stderr)
    vmt_basename_index = build_vmt_basename_index(args.materials_dir)
    vmt_include_index = vt.build_vmt_index(args.materials_dir)
    print("  found {} unique .vmt basenames".format(len(vmt_basename_index)), file=sys.stderr)

    fixed_via_mdl = 0
    fixed_via_manual = 0
    fixed_via_fallback = 0
    files_touched = 0
    unresolved = defaultdict(list)   # bare_name -> [(md3_rel, candidates_or_None)]
    stale_conflicts = []

    for md3_path in iter_md3_files(args.md3_dir):
        rel = os.path.relpath(md3_path, args.md3_dir)
        key = os.path.splitext(os.path.basename(md3_path))[0].lower()
        # mdl_to_md3's variants (<name>_lod1.md3, <name>_static.md3) come from <name>.mdl
        key = re.sub(r'_(lod\d+|static)$', '', key)

        with open(md3_path, 'rb') as f:
            data = bytearray(f.read())

        try:
            surfaces = list(read_surfaces(bytes(data)))
        except Exception as e:
            print("  ! {}: {}".format(rel, e), file=sys.stderr)
            continue

        changed_any = False
        mdl_path = mdl_index.get(key)

        shader_paths = []
        if mdl_path:
            try:
                shader_paths, _warnings = parse_mdl_shader_paths(mdl_path, args.materials_dir)
            except Exception as e:
                print("  ! {}: couldn't parse matching mdl {}: {}".format(rel, mdl_path, e), file=sys.stderr)

        # mdl_to_md3.py splits any mesh over 999 vertices into several MD3
        # surfaces (SOF2's SHADER_MAX_VERTEXES), so surface N is mesh N only
        # when nothing was split. Matching by position after a split put mesh
        # N+1's material on the second half of mesh N, and so on down the
        # list (de_mirage_cs2: 341 surfaces from 196 meshes). A surface that
        # already names one of this model's own materials is right as it is.
        own_paths = {}
        for p in shader_paths:
            if p:
                own_paths[p.lower()] = p
                own_paths[vt.short_qpath(p).lower()] = p
        by_position = len(surfaces) == len(shader_paths)

        for s_idx, (surf_off, shaders) in enumerate(surfaces):
            current = [n.replace('\\', '/').lower() for _o, n in shaders]
            if current and all(n in own_paths for n in current):
                for sh_off, sh_name in shaders:
                    encoded = encode_qpath(own_paths[sh_name.replace('\\', '/').lower()])
                    if bytes(data[sh_off:sh_off + MAX_QPATH]) != encoded:
                        data[sh_off:sh_off + MAX_QPATH] = encoded
                        changed_any = True
                    fixed_via_mdl += 1
                continue
            if by_position and shader_paths[s_idx] is not None:
                # GUARANTEED-CORRECT PATH: this model's own cdmaterials/
                # texture table, mapped to surfaces by order (same
                # convention as fix_md3_shaders.py).
                encoded = encode_qpath(shader_paths[s_idx])
                for sh_off, _sh_name in shaders:
                    data[sh_off:sh_off + MAX_QPATH] = encoded
                    changed_any = True
                    fixed_via_mdl += 1
                continue

            # FALLBACK PATH for this surface: either there's no matching
            # .mdl at all, or (just as commonly) there IS one but it simply
            # doesn't have an entry for this many surfaces - e.g. a model
            # with a separate alpha/overlay mesh whose texture isn't in the
            # main .mdl's own texture table. Falling through here instead
            # of skipping is exactly the fix for that case: bare-name/.vmt-
            # basename matching is the best information actually available
            # for THIS surface, resolved only when truly unambiguous.
            for sh_off, sh_name in shaders:
                if sh_name in manual_map:
                    encoded = encode_qpath(manual_map[sh_name])
                    data[sh_off:sh_off + MAX_QPATH] = encoded
                    changed_any = True
                    fixed_via_manual += 1
                    continue

                candidates = vmt_basename_index.get(sh_name.lower(), [])
                if len(candidates) == 1:
                    tex_path = resolve_basetexture(candidates[0], vmt_include_index)
                    if tex_path:
                        encoded = encode_qpath(tex_path)
                        data[sh_off:sh_off + MAX_QPATH] = encoded
                        changed_any = True
                        fixed_via_fallback += 1
                        continue

                # 0 candidates, >1 candidates, or unresolved $basetexture -
                # can't safely auto-fix, leave the existing shader name
                # untouched and report it.
                resolved_candidates = None
                if candidates:
                    resolved_candidates = [
                        (os.path.relpath(c, args.materials_dir), resolve_basetexture(c, vmt_include_index))
                        for c in candidates
                    ]
                unresolved[sh_name].append((rel, resolved_candidates))

        # Write the output unconditionally, not just when something changed.
        # An earlier version only wrote out_path when changed_any was True,
        # which meant models_fixed/ silently omitted every .md3 that was
        # ALREADY correct - an incomplete folder masquerading as "the fixed
        # models", exactly the kind of gap that produces a "missing model"
        # error far downstream with no clue why. Also copy this model's
        # sibling files (.phy collision, .vvd/.ani skeleton data, .vtx LOD
        # data) alongside it - they're never touched by this script (only
        # the .md3's own embedded shader names change), but they need to
        # exist next to the fixed .md3 for anything downstream (phy2clip.py
        # in particular) to find them without having to know to look in a
        # DIFFERENT folder than "the models folder" for collision data.
        out_path = os.path.join(args.output, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        if changed_any:
            files_touched += 1

        base_noext = md3_path[:-4]
        out_base_noext = out_path[:-4]
        for sib_ext in ('.phy', '.vvd', '.ani', '.dx90.vtx', '.dx80.vtx', '.sw.vtx'):
            sib_src = base_noext + sib_ext
            if os.path.isfile(sib_src):
                sib_dst = out_base_noext + sib_ext
                if not os.path.exists(sib_dst):
                    shutil.copy2(sib_src, sib_dst)

    total_fixed = fixed_via_mdl + fixed_via_manual + fixed_via_fallback
    print("\nDone. {} file(s) touched, {} shader reference(s) fixed:".format(files_touched, total_fixed))
    print("  via matching .mdl (guaranteed correct): {}  (includes names step 3 already had right)".format(fixed_via_mdl))
    print("  via --manual-map override:              {}".format(fixed_via_manual))
    print("  via unambiguous .vmt basename fallback:  {}".format(fixed_via_fallback))
    if SHORTENED:
        print("\n{} shader name(s) longer than {} characters were written in their short form "
              "(vmt2shader.py names the shader the same way):".format(len(SHORTENED), vt.QPATH_NAME_MAX))
        for full, short in sorted(SHORTENED.items()):
            print("  {} -> {}".format(full, short))

    if unresolved:
        report_path = os.path.join(args.output, "_unresolved_shaders.txt")
        os.makedirs(args.output, exist_ok=True)
        lines = []
        for name, occurrences in sorted(unresolved.items()):
            files = sorted(set(f for f, _c in occurrences))
            lines.append("{}  ({} file(s), no matching .mdl found)".format(name, len(files)))
            for f in files[:10]:
                lines.append("    {}".format(f))
            candidates = next((c for _f, c in occurrences if c), None)
            if candidates:
                lines.append("  candidate .vmt(s) found (ambiguous - pick one for --manual-map):")
                for rel_vmt, tex_path in candidates:
                    lines.append("    {}  ->  {}".format(rel_vmt, tex_path))
            else:
                lines.append("  no matching .vmt basename found anywhere in --materials-dir either")
            lines.append("")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print("\n{} shader name(s) left unresolved (no .mdl match, and fallback couldn't be sure).".format(len(unresolved)))
        print("See: {}".format(report_path))
        print("Add entries to a --manual-map JSON and re-run to resolve them.")

    print("\nOutput: {}".format(args.output))


if __name__ == "__main__":
    main()
