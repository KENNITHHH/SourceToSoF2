#!/usr/bin/env python3
"""
Standalone diagnostic - finds degenerate brush-side planes directly in a
Source .vmf file, independent of vmf2map.py entirely. Run this against
your ORIGINAL .vmf (not the converted .map) to see whether these
degenerate planes genuinely exist in the raw source data.

Usage:
    python diagnose_degenerate.py path/to/de_vertigo_css.vmf
"""
import sys
import re

def parse_plane(s):
    pts = re.findall(r'\(([^)]+)\)', s)
    return [tuple(float(x) for x in p.split()) for p in pts]

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def length(v):
    return (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5

def main():
    if len(sys.argv) != 2:
        print("Usage: python diagnose_degenerate.py path/to/file.vmf")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    # Find every "plane" "..." keyvalue anywhere in the file - independent
    # of any block/nesting structure, exactly matching how vmf2map.py's own
    # parser reads it.
    plane_kv_re = re.compile(r'"plane"\s*"([^"]+)"')
    total = 0
    raw_degenerate = []
    rounded_degenerate = []

    for m in plane_kv_re.finditer(text):
        plane_str = m.group(1)
        pts = parse_plane(plane_str)
        if len(pts) != 3:
            continue
        total += 1
        p1, p2, p3 = pts

        raw_len = length(cross(sub(p2, p1), sub(p3, p1)))
        if raw_len < 1e-6:
            line_no = text.count('\n', 0, m.start()) + 1
            raw_degenerate.append((line_no, raw_len, plane_str))
            continue  # already caught by the raw check, no need to also check rounded

        # Simulate the SAME 6-sig-fig rounding vmf2map.py's format_side()
        # applies to the final output (no scale/offset here - checking
        # the raw VMF values directly, scale=1/offset=0 is the common case)
        rp1 = tuple(float('{:.6g}'.format(c)) for c in p1)
        rp2 = tuple(float('{:.6g}'.format(c)) for c in p2)
        rp3 = tuple(float('{:.6g}'.format(c)) for c in p3)
        rounded_len = length(cross(sub(rp2, rp1), sub(rp3, rp1)))
        if rounded_len < 1e-6:
            line_no = text.count('\n', 0, m.start()) + 1
            rounded_degenerate.append((line_no, raw_len, rounded_len, plane_str))

    print(f"Total 'plane' keyvalues found: {total}")
    print()
    print(f"=== Degenerate in RAW (unrounded) VMF data: {len(raw_degenerate)} ===")
    for line_no, raw_len, plane_str in raw_degenerate[:30]:
        print(f"  line {line_no}: raw normal length={raw_len:.10g}")
        print(f"    plane \"{plane_str}\"")
    print()
    print(f"=== Distinct raw, but collapse to degenerate after 6-sig-fig rounding: {len(rounded_degenerate)} ===")
    for line_no, raw_len, rounded_len, plane_str in rounded_degenerate[:30]:
        print(f"  line {line_no}: raw normal length={raw_len:.10g}  rounded normal length={rounded_len:.10g}")
        print(f"    plane \"{plane_str}\"")

    if not raw_degenerate and not rounded_degenerate:
        print()
        print("No degenerate planes found in the raw .vmf at all - if the")
        print("compiled .map still has them, they're being introduced somewhere")
        print("in the conversion process itself, not present in the source.")

if __name__ == '__main__':
    main()
