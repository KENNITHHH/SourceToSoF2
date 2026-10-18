#!/usr/bin/env python3
"""
build_texture_sizes.py - Scan a content folder for real texture pixel
dimensions and write the JSON map vmf2map.py's --texture-sizes wants.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
vmf2map.py's displacement-to-patchDef2 conversion needs each material's
actual pixel width/height to scale patch UVs correctly (unlike brush
texdefs, patch control points store literal 0..1-per-repeat UV coordinates,
not texel offsets) - without real sizes it has to guess 512x512 for
anything you don't tell it about.

This reads pixel dimensions straight from source data, no full image
decode needed:
  - .vtf files: just the header's width/height fields (fast - the same
    parsing vtf_to_png_jpg.py uses, minus the actual pixel decompression)
  - .jpg/.jpeg/.png files: via Pillow, for an already-converted textures/
    folder

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 build_texture_sizes.py "E:\\CSS\\Extracted\\de_vertigo_files" sizes.json

Point it at the project root that CONTAINS your "materials" (.vmt/.vtf) and/or
"textures" (.jpg/.png) folders - it searches recursively for both kinds and
finds every "materials" or "textures" folder anywhere under the given root,
not just at the top level. If both exist for the same material, the .vtf
header value is used (checked first) and the image file is only consulted as
a fallback.

The output keys match how vmf2map.py looks materials up: lowercase, forward
slashes, no file extension, relative to whichever "materials"/"textures"
folder the file was found under - e.g. a file at
".../materials/models/props_wasteland/fence_alpha.vmt" (or the equivalent
"textures/models/props_wasteland/fence_alpha.jpg") becomes the key
"models/props_wasteland/fence_alpha".

Requires: pip install pillow   (only needed if you have a textures/ folder;
.vtf-only scanning needs nothing beyond the standard library)
"""

import os
import struct
import sys
import json

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tga')


def vtf_header_size(path):
    """Read just width/height from a VTF header - no pixel decoding needed."""
    try:
        with open(path, 'rb') as f:
            data = f.read(23)
        if len(data) < 20 or data[0:4] != b'VTF\x00':
            return None
        width, height = struct.unpack_from('<HH', data, 16)
        if width <= 0 or height <= 0:
            return None
        return (width, height)
    except OSError:
        return None


def image_file_size(path):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def find_root_dirs(content_root, folder_name):
    """Find every directory anywhere under content_root literally named
    folder_name (case-insensitive) - a project can have more than one
    (e.g. one per extracted content pack)."""
    matches = []
    for dirpath, dirnames, _filenames in os.walk(content_root):
        for d in dirnames:
            if d.lower() == folder_name:
                matches.append(os.path.join(dirpath, d))
    return matches


def relative_material_key(file_path, base_dir):
    rel = os.path.relpath(file_path, base_dir)
    rel = os.path.splitext(rel)[0].replace('\\', '/').lower()
    return rel


def scan_vtf_sizes(content_root):
    sizes = {}
    count = 0
    for materials_dir in find_root_dirs(content_root, 'materials'):
        for dirpath, _dirs, filenames in os.walk(materials_dir):
            for fn in filenames:
                if not fn.lower().endswith('.vtf'):
                    continue
                full = os.path.join(dirpath, fn)
                size = vtf_header_size(full)
                if size:
                    key = relative_material_key(full, materials_dir)
                    if key not in sizes:
                        sizes[key] = size
                        count += 1
    return sizes, count


def scan_image_sizes(content_root):
    sizes = {}
    count = 0
    for textures_dir in find_root_dirs(content_root, 'textures'):
        for dirpath, _dirs, filenames in os.walk(textures_dir):
            for fn in filenames:
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                full = os.path.join(dirpath, fn)
                size = image_file_size(full)
                if size:
                    key = relative_material_key(full, textures_dir)
                    if key not in sizes:
                        sizes[key] = size
                        count += 1
    return sizes, count


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    content_root, output_path = sys.argv[1], sys.argv[2]

    if not os.path.isdir(content_root):
        print("ERROR: '{}' is not a directory".format(content_root), file=sys.stderr)
        sys.exit(1)

    print("Scanning for .vtf files under any 'materials' folder...", file=sys.stderr)
    vtf_sizes, vtf_count = scan_vtf_sizes(content_root)
    print("  found sizes for {} .vtf file(s)".format(vtf_count), file=sys.stderr)

    print("Scanning for .jpg/.png files under any 'textures' folder...", file=sys.stderr)
    img_sizes, img_count = scan_image_sizes(content_root)
    print("  found sizes for {} image file(s)".format(img_count), file=sys.stderr)

    combined = dict(img_sizes)
    added_from_vtf = 0
    for key, size in vtf_sizes.items():
        if key not in combined:
            combined[key] = size
            added_from_vtf += 1
        # if both exist, keep the vtf value that was already there but
        # they should agree exactly (converted images preserve resolution)

    # store as [width, height] lists (vmf2map.py's json.load + tuple() works
    # the same either way, but plain JSON has no tuple type)
    out = {k: list(v) for k, v in combined.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("\nWrote {} material size(s) -> {}".format(len(out), output_path))
    print("({} from .vtf headers, {} from converted images, {} unique overall)".format(
        vtf_count, img_count, len(out)))


if __name__ == '__main__':
    main()
