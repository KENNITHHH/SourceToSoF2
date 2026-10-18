#!/usr/bin/env python3
"""
MD3 Shader Path Fixer
=======================

Noesis's mdl -> md3 batch export writes the bare texture name (e.g.
"forklift") into each surface's shader field, instead of the full
engine-relative path idTech3 needs (e.g. "models/props/cs_assault/forklift").
That's why Radiant/the game reports "Failed to load shader <name>".

This script reads the ORIGINAL Source .mdl for each model (which has the
correct cdmaterials directory + texture names in its texture table) and
patches the matching .md3's shader field(s) in place - same file size,
only the 64-byte shader name fields change. Geometry, UVs, frames and
tags are untouched.

USAGE
-----
    python fix_md3_shaders.py <mdl_folder> <md3_folder> <output_folder>

- <mdl_folder> : where your source .mdl files live (searched recursively)
- <md3_folder> : where your converted .md3 files live (searched recursively,
                 folder structure preserved into the output)
- <output_folder> : patched .md3 files are written here (originals untouched)

Matching is by filename (e.g. forklift.mdl <-> forklift.md3), case-insensitive.

MULTI-SURFACE / MULTI-SKIN MODELS
----------------------------------
If a model has multiple surfaces AND multiple textures, this maps them by
ORDER (surface 0 -> first skin-table texture, surface 1 -> second, etc.),
using the model's default skin family (family 0). That's correct for the
vast majority of static props. If a model's surface count doesn't match
its texture count, it's flagged in the summary so you can eyeball it -
the script still does its best-effort patch rather than skipping it.

Only the first cdmaterials directory is used per model. If a model has
more than one candidate cdmaterials directory, that's also flagged.
"""

import os
import sys
import struct


def cstr(data, off):
    end = data.index(b'\x00', off)
    return data[off:end].decode('latin1')


def parse_mdl_shader_paths(path):
    """Returns (paths, warnings) where paths[i] is the shader path to use
    for the i-th surface (by skin-table order, default skin family 0)."""
    with open(path, 'rb') as f:
        data = f.read()

    if data[0:4] != b'IDST':
        raise ValueError("not a Source studiomdl (bad magic)")

    numtextures, textureindex = struct.unpack_from('<ii', data, 204)
    numcdtextures, cdtextureindex = struct.unpack_from('<ii', data, 212)
    numskinref, numskinfamilies, skinindex = struct.unpack_from('<iii', data, 220)

    warnings = []

    cdmaterials = []
    for i in range(numcdtextures):
        off = struct.unpack_from('<i', data, cdtextureindex + i * 4)[0]
        cdmaterials.append(cstr(data, off))
    if len(cdmaterials) > 1:
        warnings.append(f"{len(cdmaterials)} cdmaterials dirs found, using the first: {cdmaterials[0]!r}")
    cdmat = cdmaterials[0] if cdmaterials else ''

    texnames = []
    for i in range(numtextures):
        tex_struct_off = textureindex + i * 64
        sznameindex = struct.unpack_from('<i', data, tex_struct_off)[0]
        texnames.append(cstr(data, tex_struct_off + sznameindex))

    skin_family0 = []
    for i in range(numskinref):
        idx = struct.unpack_from('<h', data, skinindex + i * 2)[0]
        skin_family0.append(texnames[idx])

    cdmat_norm = cdmat.rstrip('\\/').replace('\\', '/')
    paths = [f"{cdmat_norm}/{t}".lower() for t in skin_family0]
    return paths, warnings


def patch_md3(md3_path, out_path, shader_paths, warnings):
    with open(md3_path, 'rb') as f:
        data = bytearray(f.read())

    if data[0:4] != b'IDP3':
        raise ValueError("not an MD3 file (bad magic)")

    numSurfaces = struct.unpack_from('<i', data, 84)[0]
    ofsSurfaces = struct.unpack_from('<i', data, 100)[0]

    if numSurfaces != len(shader_paths):
        warnings.append(f"surface count ({numSurfaces}) != texture count ({len(shader_paths)}) - "
                         f"mapping by order, clamping to available")

    surf_off = ofsSurfaces
    for s in range(numSurfaces):
        snumShaders = struct.unpack_from('<i', data, surf_off + 76)[0]
        ofsShaders = struct.unpack_from('<i', data, surf_off + 92)[0]
        ofsEndSurf = struct.unpack_from('<i', data, surf_off + 104)[0]

        path = shader_paths[min(s, len(shader_paths) - 1)] if shader_paths else None
        if path is not None:
            encoded = path.encode('ascii', errors='replace')[:63]
            encoded = encoded + b'\x00' * (64 - len(encoded))
            for sh in range(snumShaders):
                shoff = surf_off + ofsShaders + sh * 68
                data[shoff:shoff + 64] = encoded

        surf_off += ofsEndSurf

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(data)


def build_mdl_index(mdl_root):
    index = {}
    for dirpath, _, filenames in os.walk(mdl_root):
        for fn in filenames:
            if fn.lower().endswith('.mdl'):
                key = os.path.splitext(fn)[0].lower()
                if key not in index:
                    index[key] = os.path.join(dirpath, fn)
    return index


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    mdl_root, md3_root, out_root = sys.argv[1], sys.argv[2], sys.argv[3]

    print("Indexing .mdl files...")
    mdl_index = build_mdl_index(mdl_root)
    print(f"  found {len(mdl_index)} mdl files")

    fixed = 0
    unmatched = []
    failed = []
    flagged = []

    for dirpath, _, filenames in os.walk(md3_root):
        for fn in filenames:
            if not fn.lower().endswith('.md3'):
                continue
            md3_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(md3_path, md3_root)
            key = os.path.splitext(fn)[0].lower()

            mdl_path = mdl_index.get(key)
            if not mdl_path:
                unmatched.append(rel)
                continue

            out_path = os.path.join(out_root, rel)
            try:
                shader_paths, warnings = parse_mdl_shader_paths(mdl_path)
                patch_md3(md3_path, out_path, shader_paths, warnings)
                fixed += 1
                if warnings:
                    flagged.append((rel, warnings))
            except Exception as e:
                failed.append((rel, str(e)))

    print(f"\nDone. Fixed: {fixed}  Unmatched (no .mdl found): {len(unmatched)}  Failed: {len(failed)}")

    if flagged:
        print(f"\n{len(flagged)} file(s) worth a manual glance:")
        for rel, warns in flagged[:20]:
            print(f"  {rel}:")
            for w in warns:
                print(f"      {w}")

    if unmatched:
        print(f"\nNo matching .mdl found for {len(unmatched)} file(s), e.g.:")
        for rel in unmatched[:10]:
            print(f"  {rel}")

    if failed:
        print(f"\n{len(failed)} file(s) failed to patch:")
        for rel, reason in failed[:10]:
            print(f"  {rel}: {reason}")

    print(f"\nOutput: {out_root}")


if __name__ == '__main__':
    main()
