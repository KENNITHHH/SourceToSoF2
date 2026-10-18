#!/usr/bin/env python3
"""
bsp_decal_relight.py - Give every decal in a compiled SOF2 .bsp the light of
the surface it is painted on.

WHY
---
Decals and overlays are VERTEX lit (vmt2shader.py _ensure_vertex_compile: an
alpha-blended decal cannot multiply its own lightmap in SOF2). q3map2 lights
each decal vertex on its own: one point sample, no -samples supersampling,
no -filter. The ground under it is lightmapped, supersampled and filtered.
So the same spot comes out differently lit on the decal and on the ground:
a decal whose corners all fall in shadow is one dark shape on lit ground,
and where a shadow edge crosses a decal every vertex is either fully in or
fully out, so the edge turns into a row of triangles.

WHAT IT DOES
------------
After -light: for every vertex of a vertex-lit surface whose shader has
polygonOffset (the decal shaders vmt2shader.py writes - nothing else in a
converted map uses it), find the lightmapped surface directly under it
(within MAX_DISTANCE along the vertex normal, facing the same way), read
that surface's lightmap at that point (bilinear, as the renderer does), and
write it as the vertex colour. A vertex with nothing lightmapped under it
keeps q3map2's own colour.

q3map2 writes lightmap texels and vertex colours through the same
ColorToBytes, and the engine shifts both the same way at load, so a texel
copied into a vertex colour is drawn as the same brightness.

USAGE
-----
    python bsp_decal_relight.py <map.bsp> <shader file or folder> [...]

The shader arguments are where the decal shaders are (the map's own .shader,
or the whole shaders/ folder). The .bsp is rewritten in place; run it after
-light (the compile .bat from install_to_sof2.py does). Running it twice is
harmless.
"""
import math
import os
import re
import struct
import sys

LUMP_SHADERS, LUMP_DRAWVERTS, LUMP_DRAWINDEXES, LUMP_SURFACES, LUMP_LIGHTMAPS = 1, 10, 11, 13, 14
SURF_SIZE = 148
VERT_SIZE = 80
LM_SIZE = 128
MAX_DISTANCE = 4.0      # decal lift is 0.4..0.5; generous for patch bulge
EDGE_REACH = 8.0        # a decal vertex just past the edge of the surface under it
MIN_FACING = 0.5        # cos(60 deg) between decal normal and ground normal
CELL = 64.0


def decal_shader_names(paths):
    """Lower-case names of every shader with polygonOffset, plus the _bsp
    names q3map2 -rename gives misc_model copies of them."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith('.shader')]
        elif os.path.isfile(p):
            files.append(p)
    names = set()
    for f in files:
        text = open(f, encoding='utf-8', errors='ignore').read()
        text = re.sub(r'//[^\n]*', '', text)
        depth, name, start = 0, None, 0
        tok = re.finditer(r'\{|\}|[^\s{}]+', text)
        for m in tok:
            t = m.group(0)
            if t == '{':
                if depth == 0:
                    start = m.end()
                depth += 1
            elif t == '}':
                depth -= 1
                if depth == 0 and name:
                    if re.search(r'^\s*polygonoffset\b', text[start:m.start()], re.I | re.M):
                        names.add(name.lower())
                        names.add(name.lower() + '_bsp')
                    name = None
            elif depth == 0:
                name = t
    return names


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    bsp_path = argv[1]
    decal_names = decal_shader_names(argv[2:])
    data = bytearray(open(bsp_path, 'rb').read())
    if data[:4] != b'RBSP':
        print("{}: not an SOF2 (RBSP) .bsp - left alone".format(bsp_path))
        return 1
    lumps = [struct.unpack_from('<ii', data, 8 + i * 8) for i in range(18)]
    so, sl = lumps[LUMP_SHADERS]
    shaders = [data[so + i * 72:so + i * 72 + 64].split(b'\0')[0].decode('latin-1').lower()
               for i in range(sl // 72)]
    vo, vl = lumps[LUMP_DRAWVERTS]
    io_, il = lumps[LUMP_DRAWINDEXES]
    lo, ll = lumps[LUMP_LIGHTMAPS]
    n_pages = ll // (LM_SIZE * LM_SIZE * 3)
    sfo, sfl = lumps[LUMP_SURFACES]

    def vert_pos(i):
        return struct.unpack_from('<3f', data, vo + i * VERT_SIZE)

    def vert_lm(i):
        return struct.unpack_from('<2f', data, vo + i * VERT_SIZE + 20)   # lightmap[0]

    def vert_normal(i):
        return struct.unpack_from('<3f', data, vo + i * VERT_SIZE + 52)

    def index(i):
        return struct.unpack_from('<i', data, io_ + i * 4)[0]

    receivers = []     # (page, (p0, p1, p2), (st0, st1, st2), normal, d)
    grid = {}
    decal_surfs = []
    for s in range(sfl // SURF_SIZE):
        f = struct.unpack_from('<7i4B4B4i', data, sfo + s * SURF_SIZE)
        shader, stype, fv, nv, fi, ni = f[0], f[2], f[3], f[4], f[5], f[6]
        lm0 = f[15]
        name = shaders[shader] if 0 <= shader < len(shaders) else ''
        if lm0 < 0:
            if name in decal_names and lm0 == -3:
                decal_surfs.append((fv, nv))
            continue
        if stype not in (1, 3) or lm0 >= n_pages or name in decal_names:
            continue
        # The surface's own block on its lightmap page, from its vertices'
        # lightmap coordinates. Sampling is kept inside it: past its edge
        # lie other surfaces' luxels (or unused, black ones), and bilinear
        # filtering at the border would mix those in.
        sts = [vert_lm(i) for i in range(fv, fv + nv)]
        rect = (max(0, int(math.floor(min(s_[0] for s_ in sts) * LM_SIZE))),
                max(0, int(math.floor(min(s_[1] for s_ in sts) * LM_SIZE))),
                min(LM_SIZE - 1, int(math.ceil(max(s_[0] for s_ in sts) * LM_SIZE)) - 1),
                min(LM_SIZE - 1, int(math.ceil(max(s_[1] for s_ in sts) * LM_SIZE)) - 1))
        for t in range(ni // 3):
            ids = [fv + index(fi + t * 3 + k) for k in range(3)]
            p = [vert_pos(i) for i in ids]
            e1 = [p[1][c] - p[0][c] for c in range(3)]
            e2 = [p[2][c] - p[0][c] for c in range(3)]
            n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0])
            ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            if ln < 1e-6:
                continue
            n = (n[0] / ln, n[1] / ln, n[2] / ln)
            tri = (lm0, p, [vert_lm(i) for i in ids], n, rect)
            k = len(receivers)
            receivers.append(tri)
            mn = [min(q[c] for q in p) - EDGE_REACH for c in range(3)]
            mx = [max(q[c] for q in p) + EDGE_REACH for c in range(3)]
            for gx in range(int(math.floor(mn[0] / CELL)), int(math.floor(mx[0] / CELL)) + 1):
                for gy in range(int(math.floor(mn[1] / CELL)), int(math.floor(mx[1] / CELL)) + 1):
                    for gz in range(int(math.floor(mn[2] / CELL)), int(math.floor(mx[2] / CELL)) + 1):
                        grid.setdefault((gx, gy, gz), []).append(k)

    def sample(page, s, t, rect):
        rx0, ry0, rx1, ry1 = rect
        x = s * LM_SIZE - 0.5
        y = t * LM_SIZE - 0.5
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        out = [0.0, 0.0, 0.0]
        base = lo + page * LM_SIZE * LM_SIZE * 3
        for dy, wy in ((0, 1 - fy), (1, fy)):
            yy = min(max(ry0, ry1), max(ry0, y0 + dy))
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xx = min(max(rx0, rx1), max(rx0, x0 + dx))
                o = base + (yy * LM_SIZE + xx) * 3
                w = wx * wy
                out[0] += data[o] * w
                out[1] += data[o + 1] * w
                out[2] += data[o + 2] * w
        return out

    def closest_on_triangle(p, a, b, c):
        """Barycentrics (u, v, w) of the point of triangle abc nearest p
        (Ericson, Real-Time Collision Detection 5.1.5)."""
        ab = [b[i] - a[i] for i in range(3)]
        ac = [c[i] - a[i] for i in range(3)]
        ap = [p[i] - a[i] for i in range(3)]
        dot = lambda x, y: x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0 and d2 <= 0:
            return 1.0, 0.0, 0.0
        bp = [p[i] - b[i] for i in range(3)]
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0 and d4 <= d3:
            return 0.0, 1.0, 0.0
        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            v = d1 / (d1 - d3)
            return 1 - v, v, 0.0
        cp = [p[i] - c[i] for i in range(3)]
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0 and d5 <= d6:
            return 0.0, 0.0, 1.0
        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            w = d2 / (d2 - d6)
            return 1 - w, 0.0, w
        va = d3 * d6 - d5 * d4
        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return 0.0, 1 - w, w
        den = 1.0 / (va + vb + vc)
        v, w = vb * den, vc * den
        return 1 - v - w, v, w

    def ground_light(p, vn):
        """Lightmap colour of the lightmapped surface nearest p that faces
        the same way: the one directly under it (distance along the normal
        up to MAX_DISTANCE), or for a vertex just past a surface's edge the
        nearest point of it within EDGE_REACH."""
        key = (int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL)),
               int(math.floor(p[2] / CELL)))
        best = None
        for k in grid.get(key, ()):
            page, tp, st, n, rect = receivers[k]
            # BSP triangles wind clockwise seen from the front, so n as
            # computed above points out of the BACK; either way only the
            # plane matters here, so compare without the sign.
            if abs(n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2]) < MIN_FACING:
                continue
            d = sum((p[c] - tp[0][c]) * n[c] for c in range(3))
            if abs(d) > MAX_DISTANCE:
                continue
            u, v, w = closest_on_triangle(p, tp[0], tp[1], tp[2])
            q = [u * tp[0][c] + v * tp[1][c] + w * tp[2][c] for c in range(3)]
            dist = math.sqrt(sum((p[c] - q[c]) ** 2 for c in range(3)))
            if dist > max(MAX_DISTANCE, EDGE_REACH):
                continue
            if best is not None and dist >= best[0]:
                continue
            s = u * st[0][0] + v * st[1][0] + w * st[2][0]
            t = u * st[0][1] + v * st[1][1] + w * st[2][1]
            best = (dist, page, s, t, rect)
        if best is None:
            return None
        return sample(best[1], best[2], best[3], best[4])

    done = missed = 0
    for fv, nv in decal_surfs:
        for i in range(fv, fv + nv):
            c = ground_light(vert_pos(i), vert_normal(i))
            if c is None:
                missed += 1
                continue
            o = vo + i * VERT_SIZE + 64            # color[0] RGBA
            data[o] = min(255, int(round(c[0])))
            data[o + 1] = min(255, int(round(c[1])))
            data[o + 2] = min(255, int(round(c[2])))
            done += 1
    open(bsp_path, 'wb').write(bytes(data))
    print("bsp_decal_relight: {} decal surface(s), {} vertex colour(s) taken from the lightmap "
          "under them, {} with no lightmapped surface under them kept q3map2's colour."
          .format(len(decal_surfs), done, missed))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
