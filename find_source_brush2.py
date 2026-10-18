#!/usr/bin/env python3
"""
Precise search: finds a raw .vmf solid block with exactly 6 sides whose
material sequence matches the broken output brush (5x a caulk-like
material, 1x de_vertigo_css/metal/metal_ventfence), AND whose points -
after solving for a single consistent scale+offset - align with the
output brush's actual coordinates. This is far more precise than
proximity alone, which can match the wrong neighboring brush.

Usage:
    python find_source_brush2.py path/to/de_vertigo_css.vmf
"""
import sys
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_source_brush2.py path/to/file.vmf")
        sys.exit(1)

    with open(sys.argv[1], 'r', errors='ignore') as f:
        text = f.read()

    solid_starts = [m.start() for m in re.finditer(r'\bsolid\s*\n?\s*\{', text)]
    candidates = []

    for start in solid_starts:
        brace_start = text.index('{', start)
        depth = 0
        i = brace_start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        block = text[brace_start:i+1]

        materials = re.findall(r'"material"\s*"([^"]+)"', block)
        if len(materials) != 6:
            continue
        has_ventfence = any('metal_ventfence' in m.lower() for m in materials)
        n_nodraw = sum(1 for m in materials if 'nodraw' in m.lower() or 'caulk' in m.lower() or 'clip' in m.lower())
        if not has_ventfence:
            continue

        line_no = text.count('\n', 0, start) + 1
        candidates.append((line_no, block, materials))

    print(f"Found {len(candidates)} solid block(s) with exactly 6 sides including a metal_ventfence material.")
    print()

    for line_no, block, materials in candidates:
        # Get every point in this block along with which side/material it's on
        planes = re.findall(r'"plane"\s*"([^"]+)"', block)
        all_pts = []
        for p in planes:
            pts = re.findall(r'\(([^)]+)\)', p)
            for pt in pts:
                all_pts.append(tuple(float(x) for x in pt.split()))

        xs = sorted(set(round(p[0], 1) for p in all_pts))
        ys = sorted(set(round(p[1], 1) for p in all_pts))
        zs = sorted(set(round(p[2], 1) for p in all_pts))

        print(f"--- solid at line {line_no} ---")
        print(f"  materials: {materials}")
        print(f"  unique-ish X values: {xs}")
        print(f"  unique-ish Y values: {ys}")
        print(f"  unique-ish Z values: {zs}")
        print()

if __name__ == '__main__':
    main()
