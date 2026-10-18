#!/usr/bin/env python3
"""
Finds the raw .vmf "solid" block(s) that contain a side near a given
coordinate, and dumps their FULL text (every side, every plane, any
dispinfo) so we can see exactly what Source itself defines there -
independent of vmf2map.py's own parsing/conversion entirely.

Usage:
    python find_source_brush.py path/to/de_vertigo_css.vmf
"""
import sys
import re

# The coordinates of the degenerate plane found in the OUTPUT .map -
# we're searching the RAW .vmf for whichever solid contains a point
# near this, to see what Source itself actually stored here.
TARGETS = [
    (-1823, 683, 11948),
    (-1823, 812, 11948),
]
TOLERANCE = 5.0  # units - generous, to allow for scale/offset differences

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_source_brush.py path/to/file.vmf")
        sys.exit(1)

    with open(sys.argv[1], 'r', errors='ignore') as f:
        text = f.read()

    # Find every top-level "solid { ... }" block by brace matching.
    solid_starts = [m.start() for m in re.finditer(r'\bsolid\s*\n?\s*\{', text)]
    found_any = False

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

        # Check every plane point in this block against our targets
        matches = False
        for m in re.finditer(r'\(([^)]+)\)', block):
            try:
                pt = tuple(float(x) for x in m.group(1).split())
            except ValueError:
                continue
            if len(pt) != 3:
                continue
            for tx, ty, tz in TARGETS:
                if abs(pt[0]-tx) < TOLERANCE and abs(pt[1]-ty) < TOLERANCE and abs(pt[2]-tz) < TOLERANCE:
                    matches = True
                    break
            if matches:
                break

        if matches:
            found_any = True
            line_no = text.count('\n', 0, start) + 1
            print(f"=== Found matching solid block at line {line_no} ===")
            print(block)
            print()
            print(f"Number of 'side' sub-blocks: {block.count('side')}")
            print(f"Has 'dispinfo': {'dispinfo' in block}")
            print("=" * 70)
            print()

    if not found_any:
        print("No solid block found containing a point near the target coordinates.")
        print("Try increasing TOLERANCE in this script, or the map may use a")
        print("different scale/offset than assumed.")

if __name__ == '__main__':
    main()
