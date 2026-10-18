#!/usr/bin/env python3
"""
VTF Batch Converter (alpha-aware)
==================================

Walks a folder of extracted .vtf textures and converts each one to:
  - PNG  : if it has a meaningfully-used alpha channel, OR it looks like a
           normal map (by filename), OR it uses a palette/indexed mode.
  - JPG  : everything else (opaque diffuse/spec/etc. textures), to keep
           file size down for a target that natively loads jpg/png.

Folder structure is preserved relative to the input root.

REQUIREMENTS
------------
    pip install vtf2img pillow

USAGE
-----
    python vtf_batch_convert.py <input_folder> <output_folder>

Example:
    python vtf_batch_convert.py "E:\\CSS\\VertigoAssets\\materials" "E:\\CSS\\VertigoAssets_converted\\materials"

TUNING
------
- JPEG_QUALITY: raise/lower for size vs. quality trade-off (92 is a safe,
  visually-lossless-ish default for diffuse maps).
- ALPHA_OPAQUE_THRESHOLD: alpha values >= this are treated as "fully
  opaque" so a VTF that technically has an alpha channel but never
  actually uses it (all 255) still gets routed to JPG.
- NORMAL_MAP_HINTS: filename substrings that force PNG regardless of
  alpha, to protect normal/bump maps from JPEG block artifacts.
"""

import os
import sys

try:
    from vtf2img import Parser
except ImportError:
    print("Missing dependency. Run:  pip install vtf2img pillow")
    sys.exit(1)

JPEG_QUALITY = 92
ALPHA_OPAQUE_THRESHOLD = 250
NORMAL_MAP_HINTS = ('_normal', '_n.', '_n_', 'normal_', '_local', '_bump', 'bump_', '_nrm')


def is_normal_map(path):
    name = os.path.basename(path).lower()
    return any(h in name for h in NORMAL_MAP_HINTS)


def has_meaningful_alpha(img):
    if img.mode not in ('RGBA', 'LA', 'PA'):
        return False
    alpha = img.getchannel('A')
    lo, hi = alpha.getextrema()
    return lo < ALPHA_OPAQUE_THRESHOLD


def convert_file(vtf_path, in_root, out_root):
    rel = os.path.relpath(vtf_path, in_root)
    rel_noext = os.path.splitext(rel)[0]

    try:
        img = Parser(vtf_path).get_image()
    except Exception as e:
        print(f"  ! failed to parse: {rel}  ({e})")
        return None

    use_png = is_normal_map(vtf_path) or has_meaningful_alpha(img) or img.mode == 'P'
    ext = '.png' if use_png else '.jpg'
    out_path = os.path.join(out_root, rel_noext + ext)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if use_png:
        img.save(out_path, 'PNG', optimize=True)
    else:
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.save(out_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)

    return ext


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    in_root, out_root = sys.argv[1], sys.argv[2]
    counts = {'.png': 0, '.jpg': 0, 'fail': 0}

    for dirpath, _, filenames in os.walk(in_root):
        for fn in filenames:
            if fn.lower().endswith('.vtf'):
                result = convert_file(os.path.join(dirpath, fn), in_root, out_root)
                if result:
                    counts[result] += 1
                else:
                    counts['fail'] += 1

    print(f"\nDone. PNG: {counts['.png']}  JPG: {counts['.jpg']}  Failed: {counts['fail']}")
    print(f"Output: {out_root}")


if __name__ == '__main__':
    main()
