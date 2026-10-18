#!/usr/bin/env python3
"""
Runs the REAL vmf2map.py convert_solid() function against EVERY single
brush in the .vmf, one at a time, and directly checks each one's actual
OUTPUT text for a degenerate plane. This bypasses all guessing about
which brush is responsible - it runs the exact real conversion logic and
finds the answer directly, brush by brush.

Must be run from the same folder as vmf2map.py (or with it on PYTHONPATH).

Usage:
    python scan_all_brushes.py path/to/de_vertigo_css.vmf
"""
import sys
import re
from collections import defaultdict

sys.path.insert(0, '.')
import vmf2map as v2m


def check_output_for_degenerate(text):
    """Check every '( ) ( ) ( )'-style line in a convert_solid() output
    string for a degenerate (near-zero-area) plane."""
    bad_lines = []
    for line in text.split('\n'):
        pts = re.findall(r'\(([^)]+)\)', line)
        if len(pts) < 3:
            continue
        try:
            p = [tuple(float(x) for x in pt.split()) for pt in pts[:3]]
        except ValueError:
            continue
        n = v2m.cross(v2m.sub(p[1], p[0]), v2m.sub(p[2], p[0]))
        if v2m.length(n) < 1e-3:
            bad_lines.append(line.strip())
    return bad_lines


def main():
    if len(sys.argv) != 2:
        print("Usage: python scan_all_brushes.py path/to/file.vmf")
        sys.exit(1)

    with open(sys.argv[1], 'r', errors='ignore') as f:
        text = f.read()

    tokens = v2m.tokenize(text)
    root = v2m.parse_vdf(tokens)

    unmapped = defaultdict(int)
    total_solids = 0
    bad_solids = []

    def scan_solids(node, label):
        nonlocal total_solids
        for solid in v2m.get_solid_blocks(node):
            total_solids += 1
            warnings = []
            try:
                out = v2m.convert_solid(solid, {}, unmapped, scale=1.0,
                                         offset=(0.0, 0.0, 0.0), warnings=warnings,
                                         detail=False, is_entity_brush=False)
            except Exception as e:
                print(f"  ! solid id={v2m.get1(solid, 'id')} in {label} raised an exception: {e}")
                continue
            bad_lines = check_output_for_degenerate(out) if out else []
            if bad_lines:
                bad_solids.append((v2m.get1(solid, 'id'), label, bad_lines, warnings))

    worlds = root.get('world', [])
    if worlds:
        scan_solids(worlds[0], 'world')
    for i, entity in enumerate(root.get('entity', [])):
        classname = v2m.get1(entity, 'classname', '?')
        scan_solids(entity, f'entity[{i}] ({classname})')

    print(f"Scanned {total_solids} solid(s) total.")
    print(f"Solids producing a degenerate output line: {len(bad_solids)}")
    print()
    for solid_id, label, bad_lines, warnings in bad_solids:
        print(f"=== solid id={solid_id} in {label} ===")
        for bl in bad_lines:
            print(f"  DEGENERATE LINE: {bl}")
        if warnings:
            print(f"  warnings from this conversion: {warnings}")
        print()


if __name__ == '__main__':
    main()
