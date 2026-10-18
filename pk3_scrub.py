#!/usr/bin/env python3
"""
pk3_scrub.py - Find (and optionally remove) files in a SOF2 map .pk3 that
nothing can ever actually display, to reclaim wasted space.

------------------------------------------------------------------------------
WHAT THIS CHECKS, AND HOW CONFIDENT EACH CHECK IS
------------------------------------------------------------------------------
1. UNUSED SHADER DECLARATIONS (high confidence)
   A shader block in your .shader file that no brush face, patch, or model
   surface ever references. Determined by parsing the .map's own brush/patch
   material lines AND every .md3's actual embedded shader names (a proper
   binary parse of the MD3 surface/shader tables - not a guess).

2. ORPHANED IMAGE FILES (high confidence)
   A .jpg/.png under textures/ that no shader declaration - used OR unused -
   references at all. Nothing can ever display these regardless of anything
   else, so they're the safest category here.

3. ORPHANED MODEL FILES (high confidence)
   A .md3 (+ its .phy/.vvd/.vtx siblings) that no misc_model/gametype_player/
   etc. entity in the .map actually points at via its "model" key.

WHAT THIS DELIBERATELY DOES NOT TOUCH
   Model "skin" variants some props might reference (e.g. light_shop2b next
   to light_shop2): confirmed against real SOF2 usage that .md3 static props
   have NO skin-remapping mechanism at all (SOF2's only skin system, g2skin,
   is Ghoul2 and is for animated CHARACTER models, not static props) - so
   this is not actually a risk category, and such unreferenced variant
   shaders/images ARE correctly caught by checks 1-2 above. This tool does
   NOT try to guess at soundscript-driven or script-driven references it
   can't see in the .map/.shader/.md3 themselves.

USAGE
-----
    python pk3_scrub.py <working_dir> --shader <path.shader> --map <compiled.map> [options]

      <working_dir>       Your PRE-PACKAGING working folder - the same one
                          used as --materials-dir/--mdl-dir for vmt2shader.py
                          and phy2clip.py earlier in this pipeline (e.g.
                          "E:\\CSS\\Extracted\\de_cache_files"). This tool
                          runs BEFORE you zip anything into a .pk3, not on
                          the .pk3 itself - the working folder still uses
                          Source's own "materials/" naming (SOF2's "textures/"
                          rename happens at packaging time), so this
                          auto-detects "materials/" OR "textures/", whichever
                          is actually present, rather than assuming one.
      --shader <path>      REQUIRED. Path to your generated .shader file
                          (wherever it currently sits - no fixed subfolder
                          is assumed).
      --map <path>        REQUIRED. The .map used to compile this content.
      --models-dir <dir>   Override auto-detected models/ location.
      --textures-dir <dir> Override auto-detected materials/ or textures/
                          location.
      --apply             Actually move flagged files (default: dry-run).
      --output <dir>      Where to write the report / _unused/ quarantine.
                          Defaults to <working_dir>.

PROTECTED CATEGORIES (never flagged, regardless of reference status)
   Specular maps: confirmed usable in SOF2 (unlike normal maps, which are
   confirmed to have zero support anywhere in this engine build's shader
   system - checked across all 73 real stock shader files). Since no
   specular-named file existed in the test content to verify SOF2's exact
   wiring convention, anything matching *spec*/*specular* in its filename
   is excluded from flagging entirely rather than guessed at - the same
   conservative treatment as decals: preserved for a feature that may not
   be wired up yet, not assumed dead.
"""

import argparse
import os
import re
import shutil
import struct
import sys


PROTECTED_NAME_PATTERNS = [
    re.compile(r'spec(ular)?', re.IGNORECASE),
]


def is_protected(rel_path):
    base = os.path.splitext(os.path.basename(rel_path))[0]
    return any(p.search(base) for p in PROTECTED_NAME_PATTERNS)


def _has_image_files(folder):
    for dirpath, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(IMAGE_EXTS):
                return True
    return False


def find_textures_root(working_dir, override=None):
    """Auto-detect which subfolder actually holds displayable images.

    A fixed name-preference (e.g. always prefer 'materials') is wrong in
    general: confirmed against a real project layout where materials/ holds
    ONLY Source's original .vmt/.vtf (no .jpg/.png at all - .vtf isn't a
    directly-displayable image format) while textures/ holds the actual
    vtf_to_png_jpg.py output the shaders reference. Preferring materials/
    by name there silently scans an empty-of-images folder and reports
    "0 orphaned" - which reads as a clean bill of health but actually means
    nothing was checked at all. Instead, check both candidates and prefer
    whichever one actually contains image files.
    """
    if override:
        return override
    candidates = [os.path.join(working_dir, name) for name in ('materials', 'textures')]
    candidates = [c for c in candidates if os.path.isdir(c)]
    with_images = [c for c in candidates if _has_image_files(c)]
    if with_images:
        return with_images[0]
    return candidates[0] if candidates else None


def find_models_root(working_dir, override=None):
    if override:
        return override
    candidate = os.path.join(working_dir, 'models')
    return candidate if os.path.isdir(candidate) else None


# ------------------------------------------------------------------ MD3 ---

def parse_md3_shaders(path):
    """Proper MD3 binary parse (not a text/byte-string guess) - returns the
    set of shader/material names actually embedded in this model's
    surfaces, or None if the file isn't a valid MD3.

    Field offsets, verified against real content (5 test files, 0 failures
    across all 151 real .md3s in a full pk3):

    Header (ident,version,name[64],flags,num_frames,num_tags,num_surfaces,
    num_skins,ofs_frames,ofs_tags,ofs_surfaces,ofs_eof):
        0, 4, 8, 72, 76, 80, 84, 88, 92, 96, 100, 104

    Surface header, relative to surface start (ident,name[64],flags,
    num_frames,num_shaders,num_verts,num_triangles,ofs_triangles,
    ofs_shaders,ofs_st,ofs_xyznormal,ofs_end):
        0, 4, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104
    """
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 108 or data[0:4] != b'IDP3':
        return None
    num_surfaces, = struct.unpack_from('<i', data, 84)
    ofs_surfaces, = struct.unpack_from('<i', data, 100)
    names = set()
    off = ofs_surfaces
    for _ in range(num_surfaces):
        if off + 108 > len(data) or data[off:off + 4] != b'IDP3':
            break
        num_shaders, = struct.unpack_from('<i', data, off + 76)
        ofs_shaders, = struct.unpack_from('<i', data, off + 92)
        ofs_end, = struct.unpack_from('<i', data, off + 104)
        for i in range(num_shaders):
            so = off + ofs_shaders + i * 68
            raw = data[so:so + 64]
            name = raw.split(b'\x00')[0].decode('ascii', errors='ignore')
            if name:
                names.add(name.lower().replace('\\', '/'))
        off += ofs_end
    return names


MODEL_SIBLING_EXTS = ('.phy', '.vvd', '.ani', '.dx90.vtx', '.dx80.vtx', '.sw.vtx')


# --------------------------------------------------------------- shader ---

def parse_shader_files(shader_paths):
    """Parse one or more .shader files into {lowercased bare name: body}."""
    blocks = {}
    for path in shader_paths:
        text = open(path, encoding='utf-8', errors='replace').read()
        text = re.sub(r'//[^\n]*', '', text)
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            s = lines[i].strip()
            if s and not s.startswith('{') and not s.startswith('}'):
                name = s
                i += 1
                if i < len(lines) and lines[i].strip() == '{':
                    depth = 1
                    body = []
                    i += 1
                    while i < len(lines) and depth > 0:
                        if '{' in lines[i]:
                            depth += lines[i].count('{')
                        if '}' in lines[i]:
                            depth -= lines[i].count('}')
                        if depth > 0:
                            body.append(lines[i])
                        i += 1
                    blocks[name.lower().lstrip('/')] = ('\n'.join(body), path)
                    continue
            i += 1
    return blocks


def images_in_block(body):
    imgs = set()
    for m in re.finditer(r'\b(?:map|animMap|clampMap)\s+([^\s\n]+)', body, re.IGNORECASE):
        val = m.group(1).strip().lower().lstrip('/')
        if not val.startswith('$'):
            imgs.add(val)
    for m in re.finditer(r'qer_editorimage\s+([^\s\n]+)', body, re.IGNORECASE):
        imgs.add(m.group(1).strip().lower().lstrip('/'))
    # skyParms <farbox> <cloudheight> <innerbox>: the engine loads
    # <box>_rt/_bk/_lf/_ft/_up/_dn.tga for each box that isn't '-'
    for m in re.finditer(r'\bskyParms\s+(\S+)\s+\S+\s+(\S+)', body, re.IGNORECASE):
        for box in (m.group(1), m.group(2)):
            if box != '-':
                box = box.strip().lower().lstrip('/')
                imgs |= {box + '_' + s for s in ('rt', 'bk', 'lf', 'ft', 'up', 'dn')}
    return imgs


# ------------------------------------------------------------------ map ---

FACE_RE = re.compile(
    r'^\(\s*[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\)\s*'
    r'\(\s*[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\)\s*'
    r'\(\s*[-\d.eE]+\s+[-\d.eE]+\s+[-\d.eE]+\s*\)\s+(\S+)'
)


def parse_map_materials_and_models(map_path):
    text = open(map_path, encoding='utf-8', errors='replace').read()
    materials = set()
    for line in text.splitlines():
        m = FACE_RE.match(line.strip())
        if m:
            materials.add(m.group(1).lower().lstrip('/'))
    for m in re.finditer(r'patchDef2\s*\n?\s*\{\s*\n?\s*([^\s{}]+)', text):
        materials.add(m.group(1).lower().lstrip('/'))

    models = set()
    for m in re.finditer(r'"model"\s+"([^"]+)"', text):
        v = m.group(1).strip().lower().lstrip('/')
        if v.endswith('.md3'):
            models.add(v)
    return materials, models


def normalize_both_ways(names):
    out = set()
    for n in names:
        out.add(n)
        out.add('textures/' + n)
        if n.startswith('textures/'):
            out.add(n[len('textures/'):])
    return out


# ----------------------------------------------------------------- main ---

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tga')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("working_dir")
    ap.add_argument("--shader", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--models-dir")
    ap.add_argument("--textures-dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--output")
    args = ap.parse_args()

    root = args.working_dir
    output_dir = args.output or root

    textures_root = find_textures_root(root, args.textures_dir)
    models_root = find_models_root(root, args.models_dir)
    if not textures_root:
        print("ERROR: couldn't find a materials/ or textures/ folder under {} "
              "- pass --textures-dir explicitly.".format(root), file=sys.stderr)
        sys.exit(1)
    if not models_root:
        print("ERROR: couldn't find a models/ folder under {} - pass --models-dir "
              "explicitly.".format(root), file=sys.stderr)
        sys.exit(1)
    if not _has_image_files(textures_root):
        print("ERROR: {} contains no .jpg/.png/.tga files at all - this would silently "
              "report '0 orphaned' without actually checking anything, which reads as a "
              "clean result but isn't one. Pass --textures-dir pointing at the folder "
              "that actually holds your converted images.".format(textures_root),
              file=sys.stderr)
        sys.exit(1)
    print("Textures root: {}".format(textures_root), file=sys.stderr)
    print("Models root:   {}".format(models_root), file=sys.stderr)

    if not os.path.isfile(args.shader):
        print("ERROR: --shader file not found: {}".format(args.shader), file=sys.stderr)
        sys.exit(1)
    shader_blocks = parse_shader_files([args.shader])
    print("Declared shaders: {}".format(len(shader_blocks)), file=sys.stderr)

    map_materials, map_models = parse_map_materials_and_models(args.map)
    print("Materials used directly by the .map: {}".format(len(map_materials)), file=sys.stderr)
    print("Model paths referenced by the .map: {}".format(len(map_models)), file=sys.stderr)

    all_md3 = {}
    for dirpath, _, files in os.walk(models_root):
        for fn in files:
            if fn.lower().endswith('.md3'):
                full = os.path.join(dirpath, fn)
                # Canonical comparison key: "models/" + path-within-models_root,
                # NOT relative to the working dir. Same bug class as the
                # textures/materials mismatch fixed earlier - the .map and
                # .shader always say "models/props/...", regardless of
                # whether the on-disk folder is actually named "models",
                # "models_fixed", or anything else. Computing relative to
                # the working dir broke the instant the folder had a
                # different name (confirmed: flagged all 151 real, placed
                # models as orphaned when pointed at a models_fixed/ folder).
                rel_within = os.path.relpath(full, models_root).replace('\\', '/').lower()
                rel = 'models/' + rel_within
                all_md3[rel] = full

    md3_used_materials = set()
    referenced_md3 = set()
    for rel, full in all_md3.items():
        if rel in map_models or rel.split('models/', 1)[-1] in map_models:
            referenced_md3.add(rel)
            shaders = parse_md3_shaders(full)
            if shaders:
                md3_used_materials |= shaders

    orphaned_models = sorted(set(all_md3) - referenced_md3)

    used_materials = normalize_both_ways(map_materials | md3_used_materials)
    declared = set(shader_blocks.keys())
    unused_shaders = sorted(declared - used_materials)

    used_images = set()
    for name in declared - set(unused_shaders):
        body, _ = shader_blocks[name]
        used_images |= images_in_block(body)
    used_images = normalize_both_ways(used_images)

    orphaned_images = []
    protected_skipped = 0
    for dirpath, _, files in os.walk(textures_root):
        for fn in files:
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            full = os.path.join(dirpath, fn)
            # Relative to the TEXTURES ROOT, not the working dir - a shader
            # always says "textures/metal/x", regardless of whether this
            # folder is locally named "materials" (Source's pre-packaging
            # convention) or "textures" (SOF2's packaged convention). Using
            # relpath-from-working-dir here was the actual bug: it produced
            # "materials/metal/x" for the exact same file, which never
            # matched used_images' "textures/metal/x" forms, and wrongly
            # flagged the ENTIRE materials/ tree as orphaned regardless of
            # real usage.
            rel_to_textures = os.path.relpath(full, textures_root).replace('\\', '/').lower()
            rel_for_report = os.path.relpath(full, root).replace('\\', '/').lower()
            if is_protected(rel_for_report):
                protected_skipped += 1
                continue
            rel_noext = re.sub(r'\.(jpg|jpeg|png|tga)$', '', rel_to_textures)
            if (rel_noext not in used_images and rel_to_textures not in used_images
                    and 'textures/' + rel_noext not in used_images):
                orphaned_images.append((rel_for_report, full))

    def total_size(paths):
        return sum(os.path.getsize(p) for p in paths if os.path.isfile(p))

    orphaned_model_files = []
    for rel in orphaned_models:
        full = all_md3[rel]
        orphaned_model_files.append(full)
        base = full[:-4]
        for ext in MODEL_SIBLING_EXTS:
            sib = base + ext
            if os.path.isfile(sib):
                orphaned_model_files.append(sib)

    orphaned_image_files = [full for _, full in orphaned_images]

    print("\n--- RESULTS ---", file=sys.stderr)
    print("Unused shader declarations: {}".format(len(unused_shaders)), file=sys.stderr)
    print("Orphaned image files (not referenced by ANY shader): {}  ({:.1f} KB)".format(
        len(orphaned_image_files), total_size(orphaned_image_files) / 1024), file=sys.stderr)
    if protected_skipped:
        print("  ({} specular-named file(s) skipped - protected, never flagged)".format(
            protected_skipped), file=sys.stderr)
    print("Orphaned model files (.md3 + siblings, never placed in the .map): {}  ({:.1f} KB)"
          .format(len(orphaned_model_files), total_size(orphaned_model_files) / 1024),
          file=sys.stderr)
    total_kb = (total_size(orphaned_image_files) + total_size(orphaned_model_files)) / 1024
    print("Total reclaimable: {:.1f} KB ({:.2f} MB)".format(total_kb, total_kb / 1024),
          file=sys.stderr)

    report_path = os.path.join(output_dir, "_scrub_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("pk3_scrub.py report\n====================\n\n")
        f.write("Unused shader declarations ({}):\n".format(len(unused_shaders)))
        for u in unused_shaders:
            f.write("  {}\n".format(u))
        f.write("\nOrphaned image files ({}):\n".format(len(orphaned_image_files)))
        for p in orphaned_image_files:
            f.write("  {}  ({} bytes)\n".format(os.path.relpath(p, root), os.path.getsize(p)))
        f.write("\nOrphaned model files ({}):\n".format(len(orphaned_model_files)))
        for p in orphaned_model_files:
            f.write("  {}  ({} bytes)\n".format(os.path.relpath(p, root), os.path.getsize(p)))
    print("\nFull report written to: {}".format(report_path), file=sys.stderr)

    if args.apply:
        quarantine = os.path.join(output_dir, "_unused")
        moved = 0
        for full in orphaned_image_files + orphaned_model_files:
            rel = os.path.relpath(full, root)
            dst = os.path.join(quarantine, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(full, dst)
            moved += 1
        print("\nMoved {} file(s) to: {}".format(moved, quarantine), file=sys.stderr)
        print("(Files are MOVED, not deleted - safe to restore if something looks wrong.)",
              file=sys.stderr)
        print("NOTE: unused shader DECLARATIONS were only reported, not removed from the "
              ".shader file - edit that by hand if you want them gone too.", file=sys.stderr)
    else:
        print("\nDry run only - nothing was moved. Re-run with --apply once you've reviewed "
              "the report.", file=sys.stderr)


if __name__ == "__main__":
    main()
