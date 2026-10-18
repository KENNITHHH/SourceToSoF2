#!/usr/bin/env python3
"""bsp_fix_leafbounds.py - widen every leaf's stored bounding box by one unit
in a SOF2 (RBSP) or Quake 3 (IBSP) .bsp, so q3map2 -light stops dropping walls
out of the leaf in front of them.

Why: q3map2 stores leaf boxes as integers and converts with a plain float->int
cast (writebsp.cpp EmitLeaf), which truncates toward zero - a leaf starting at
x = -2763.5 is written as starting at -2763. -light then builds each surface's
list of clusters by first rejecting every leaf whose stored box doesn't touch
the surface (lightmaps_ydnar.cpp SetupSurfaceLightmaps), so a wall lying
exactly on x = -2763.5 loses the very leaf in front of it, every one of its
luxels is marked occluded, and it bakes to the bare ambient (the black bands
along wall bottoms on converted maps, whose coordinates are Source x 1.27 and
almost never whole numbers).

Widening by one unit undoes the truncation (it is never more than one unit);
the real test that follows - is the surface actually in the leaf's surface
list - is untouched, so no surface gains a cluster it isn't in. In game the
leaf boxes are only used for frustum culling, where a unit of slack is
harmless.

Run it ONCE on the .bsp after the -bsp stage (or after -vis; -vis keeps the
boxes) and before -light:
    python bsp_fix_leafbounds.py maps/de_dust2.bsp
Running it twice just widens by two units - still harmless."""
import struct, sys

LUMP_LEAFS = 4
LEAF_SIZE = 48      # cluster, area, mins[3], maxs[3], firstsurf, numsurf, firstbrush, numbrush


def fix(path):
    data = bytearray(open(path, 'rb').read())
    ident = bytes(data[0:4])
    if ident not in (b'RBSP', b'IBSP'):
        raise SystemExit("%s: not an RBSP/IBSP file (ident %r)" % (path, ident))
    ofs, length = struct.unpack_from('<ii', data, 8 + LUMP_LEAFS * 8)
    if length % LEAF_SIZE:
        raise SystemExit("%s: leaf lump size %d isn't a multiple of %d" % (path, length, LEAF_SIZE))
    n = length // LEAF_SIZE
    done = 0
    for i in range(n):
        o = ofs + i * LEAF_SIZE + 8
        mins = struct.unpack_from('<3i', data, o)
        maxs = struct.unpack_from('<3i', data, o + 12)
        if any(a > b for a, b in zip(mins, maxs)) or any(abs(v) >= 2**31 - 2 for v in mins + maxs):
            continue        # empty leaf (leaf 0 is stored as +/- INT_MAX)
        done += 1
        struct.pack_into('<3i', data, o, *(v - 1 for v in mins))
        struct.pack_into('<3i', data, o + 12, *(v + 1 for v in maxs))
    open(path, 'wb').write(data)
    print("%s: widened %d leaf boxes by 1 unit" % (path, done))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        fix(p)
