#!/usr/bin/env python3
"""
mdl_to_md3.py - Convert a Source studiomdl (.mdl + .vvd + .vtx) directly to
an idTech3 .md3, without Noesis.

Every piece of this has been individually verified against real, known-
correct Noesis output before being used here (see the geometry validation
work this script is built on): position, triangle connectivity, UV, and
normal decoding were each checked against real .md3 ground truth across
151 real models before this writer existed. Two real bugs were caught
that way before ever reaching this file - a normal-decode byte-order
swap, and a mesh vertexoffset that's invisible for a single-mesh model
but corrupts every mesh after the first in a multi-mesh one.

Handles LOD vertex fixup tables (vvd numFixups > 0) - about 13% of real
content in one tested batch had these. Confirmed against the Valve
Developer Community's documented vertexFileFixup_t layout and verified
against real ground truth across 20 real fixup-table models (all matching
the same quantization-only tolerance as non-fixup models).

Also bakes in the correct material name directly (via the same
skin-family-0 resolution already used by md3_shader_autofix.py), so
models converted through this script should not need a separate autofix
pass at all.

USAGE
-----
    python3 mdl_to_md3.py <model.mdl> <output.md3>

Expects <model>.vvd and <model>.dx90.vtx (falling back to .dx80.vtx or
.sw.vtx) next to the .mdl.
"""

import argparse
import json
import math
import os
import re
import struct
import sys


def cstr(data, off):
    return data[off:data.index(b'\x00', off)].decode('latin1', errors='replace')


# ------------------------------------------------------------------ read --

def join_material_path(cdmat, tex):
    """cdmaterials dir + texture name, as one forward-slash path. Either part
    can be empty or carry its own slashes: a model with cdmaterials "" and
    texture "models/props/de_dust/sign_street01" (de_mirage_cs2's signs and
    rugs) used to come out as "/models/...", which matched no .vmt and no
    shader."""
    parts = [p for p in re.split(r'[\\/]+', (cdmat or '') + '/' + (tex or '')) if p]
    return '/'.join(parts)


def _resolve_cdmat_for_texture(cdmaterials, texname, materials_dir):
    """Pick the cdmaterials directory that actually has a VMT for this
    texture, trying each one in the model's own listed order (matching
    how the engine itself, and vmf_asset_extractor.py, resolve these) -
    rather than assuming the first cdmaterials entry is always right.
    Confirmed against a real model: a multi-cdmaterials prop (a per-map
    override folder listed first, then shared fallback folders) had its
    ACTUAL vmt for one texture living under a LATER folder in the list,
    not the first - baking in cdmaterials[0] unconditionally embedded a
    shader name with no matching definition anywhere in the generated
    .shader file, while the correct name (matching where
    vmf_asset_extractor.py actually found and copied the file) was one
    directory further down the same list.

    Falls back to cdmaterials[0] (previous behavior) when materials_dir
    isn't given or none of the candidates can be found there, so a
    standalone conversion without an extracted materials/ folder handy
    still works exactly as before."""
    if materials_dir and os.path.isdir(materials_dir):
        for cdmat in cdmaterials:
            cdmat_norm = cdmat.rstrip('\\/').replace('\\', '/')
            candidate_dir = os.path.join(materials_dir, *cdmat_norm.split('/')) if cdmat_norm else materials_dir
            if os.path.isfile(os.path.join(materials_dir, *join_material_path(cdmat_norm, texname).split('/')) + '.vmt'):
                return cdmat_norm
            if os.path.isdir(candidate_dir):
                target = (texname + '.vmt').lower()
                try:
                    if any(fn.lower() == target for fn in os.listdir(candidate_dir)):
                        return cdmat_norm
                except OSError:
                    pass
    return cdmaterials[0].rstrip('\\/').replace('\\', '/') if cdmaterials else ''


def parse_mdl(path, materials_dir=None):
    d = open(path, 'rb').read()
    if d[0:4] != b'IDST':
        raise ValueError("not a Source studiomdl (bad magic)")
    checksum, = struct.unpack_from('<i', d, 8)
    numtextures, textureindex = struct.unpack_from('<ii', d, 204)
    numcdtextures, cdtextureindex = struct.unpack_from('<ii', d, 212)
    numskinref, numskinfamilies, skinindex = struct.unpack_from('<iii', d, 220)
    numbodyparts, bodypartindex = struct.unpack_from('<ii', d, 232)

    cdmaterials = [cstr(d, struct.unpack_from('<i', d, cdtextureindex+i*4)[0])
                   for i in range(numcdtextures)]
    texnames = [cstr(d, textureindex+i*64+struct.unpack_from('<i', d, textureindex+i*64)[0])
                for i in range(numtextures)]
    skin_family0 = [texnames[struct.unpack_from('<h', d, skinindex+i*2)[0]]
                    for i in range(numskinref)] if numskinref else texnames

    warnings = []
    bodyparts_out = []
    for bpi in range(numbodyparts):
        bp_off = bodypartindex + bpi * 16
        sznameindex, nummodels, base, modelindex = struct.unpack_from('<4i', d, bp_off)
        bp_name = cstr(d, bp_off + sznameindex)
        if nummodels > 1:
            warnings.append("bodypart '{}' has {} models - using model 0 only "
                             "(no bodygroup-selection mechanism for misc_model in SOF2, "
                             "so any alternates would never be selectable anyway)".format(
                                 bp_name, nummodels))
        model_off = bp_off + modelindex  # model 0 always first
        model_name = cstr(d, model_off)
        o = model_off + 64
        nummeshes, meshindex = struct.unpack_from('<ii', d, o+8)
        numvertices, vertexindex, tangentsindex = struct.unpack_from('<iii', d, o+16)
        # vertexindex is a BYTE offset (like most other mdl index fields),
        # not a vertex count - confirmed against real multi-bodypart data:
        # a second bodypart's raw vertexindex (1048560) divided by 48
        # (mstudiovertex_t's fixed size) gave exactly 21845, matching the
        # first bodypart's vertex count precisely - i.e. this bodypart's
        # vertices start immediately after the previous one's in the same
        # shared .vvd array. Every single-bodypart model tested earlier had
        # vertexindex=0, where this conversion is a no-op, which is exactly
        # why the need for it never surfaced until a real multi-bodypart
        # model was available to test against.
        if vertexindex % 48 != 0:
            raise NotImplementedError("model.vertexindex ({}) not a multiple of the 48-byte "
                                       "vertex stride (bodypart '{}')".format(vertexindex, bp_name))
        vertex_base = vertexindex // 48

        meshes = []
        for m in range(nummeshes):
            mesh_off = model_off + meshindex + m * 116
            material, _back = struct.unpack_from('<ii', d, mesh_off)
            mnumvertices, mvertexoffset = struct.unpack_from('<ii', d, mesh_off+8)
            mat_name = skin_family0[material] if material < len(skin_family0) else None
            meshes.append({'material_idx': material, 'numvertices': mnumvertices,
                            'vertexoffset': vertex_base + mvertexoffset, 'material_name': mat_name})
        bodyparts_out.append({'name': bp_name, 'model_name': model_name, 'meshes': meshes})

    for bp in bodyparts_out:
        for mesh in bp['meshes']:
            if mesh['material_name'] is not None:
                cdmat = _resolve_cdmat_for_texture(cdmaterials, mesh['material_name'], materials_dir)
                mesh['full_material_path'] = join_material_path(cdmat, mesh['material_name']).lower()
            else:
                mesh['full_material_path'] = None

    return {'checksum': checksum, 'bodyparts': bodyparts_out, 'warnings': warnings}


# ------------------------------------------------------------ bone pose --
#
# What Source draws is not the .vvd's vertex positions as stored. Every
# vertex is skinned: drawn = boneToModel(sequence 0, frame 0) x poseToBone x
# stored, per bone, by its weights. For a model compiled as a static prop
# that product is the identity, which is why ignoring it was invisible on
# most props. It is not the identity when the reference animation turns the
# root bone: de_overpass_csgo's overpass_swingset_seat.mdl stores its chains
# along +Y, its bone table says identity, and its "@ref" animation (one frame,
# RAWROT2) turns the root by quaternion (0.5, 0.5, 0.5, 0.5) - a 120 degree
# turn about (1,1,1) that maps Y to Z. Hammer, CS:GO and its .phy collision
# all show the chains hanging up to the hinge; our .md3 had them flat.
# So the pose is applied here, exactly as Source's bone setup does it.

STUDIO_ANIM_RAWPOS, STUDIO_ANIM_RAWROT, STUDIO_ANIM_ANIMPOS = 0x01, 0x02, 0x04
STUDIO_ANIM_ANIMROT, STUDIO_ANIM_DELTA, STUDIO_ANIM_RAWROT2 = 0x08, 0x10, 0x20


def _half(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


def _quat48(d, o):
    x, y, zw = struct.unpack_from('<HHH', d, o)
    qx = (x - 32768) * (1.0 / 32768.0)
    qy = (y - 32768) * (1.0 / 32768.0)
    qz = ((zw & 0x7FFF) - 16384) * (1.0 / 16384.0)
    qw = math.sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz))
    return (qx, qy, qz, -qw if zw & 0x8000 else qw)


def _quat64(d, o):
    v = struct.unpack_from('<Q', d, o)[0]
    qx = ((v & 0x1FFFFF) - 1048576) * (1.0 / 1048576.5)
    qy = (((v >> 21) & 0x1FFFFF) - 1048576) * (1.0 / 1048576.5)
    qz = (((v >> 42) & 0x1FFFFF) - 1048576) * (1.0 / 1048576.5)
    qw = math.sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz))
    return (qx, qy, qz, -qw if v >> 63 else qw)


def _euler_quat(rx, ry, rz):
    """Source AngleQuaternion(RadianEuler): x roll, y pitch, z yaw."""
    sr, cr = math.sin(rx * 0.5), math.cos(rx * 0.5)
    sp, cp = math.sin(ry * 0.5), math.cos(ry * 0.5)
    sy, cy = math.sin(rz * 0.5), math.cos(rz * 0.5)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat_mat(q, t):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), t[0]],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), t[1]],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), t[2]]]


def _mat_mul(a, b):
    """3x4 x 3x4 (affine)."""
    return [[a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + (a[i][3] if j == 3 else 0.0)
             for j in range(4)] for i in range(3)]


def _anim_value0(d, o_chan):
    """Frame 0 of one compressed channel (mstudioanimvalue_t run list)."""
    valid, total = d[o_chan], d[o_chan + 1]
    if total == 0:
        return 0.0
    return float(struct.unpack_from('<h', d, o_chan + 2)[0])


def mdl_bone_skin_matrices(d):
    """Per bone: boneToModel(sequence 0, frame 0) x poseToBone, as 3x4 rows;
    or None when every one is the identity (or the pose cannot be read - an
    external .ani block, which only animated characters use)."""
    try:
        numbones, boneindex = struct.unpack_from('<ii', d, 156)
        numanim, animindex = struct.unpack_from('<ii', d, 180)
        numseq, seqindex = struct.unpack_from('<ii', d, 188)
        if numbones < 1 or numbones > 256:
            return None
        bones = []
        for b in range(numbones):
            o = boneindex + b * 216
            parent = struct.unpack_from('<i', d, o + 4)[0]
            pos = struct.unpack_from('<3f', d, o + 32)
            quat = struct.unpack_from('<4f', d, o + 44)
            rot = struct.unpack_from('<3f', d, o + 60)
            posscale = struct.unpack_from('<3f', d, o + 72)
            rotscale = struct.unpack_from('<3f', d, o + 84)
            p2b = struct.unpack_from('<12f', d, o + 96)
            bones.append({'parent': parent, 'pos': list(pos), 'quat': quat, 'rot': rot,
                          'posscale': posscale, 'rotscale': rotscale,
                          'p2b': [list(p2b[0:4]), list(p2b[4:8]), list(p2b[8:12])]})
        pose = {b: (list(bones[b]['pos']), bones[b]['quat']) for b in range(numbones)}
        if numseq >= 1 and numanim >= 1:
            animindexindex = struct.unpack_from('<i', d, seqindex + 60)[0]
            ai = struct.unpack_from('<h', d, seqindex + animindexindex)[0]
            if 0 <= ai < numanim:
                ad = animindex + ai * 100
                animblock, aindex = struct.unpack_from('<ii', d, ad + 52)
                sectionindex, sectionframes = struct.unpack_from('<ii', d, ad + 80)
                if sectionframes:
                    animblock, aindex = struct.unpack_from('<ii', d, ad + sectionindex)
                if animblock != 0:
                    return None
                o = ad + aindex
                for _ in range(numbones):
                    if o + 4 > len(d):
                        break
                    bone, flags, nxt = struct.unpack_from('<BBh', d, o)
                    if bone >= numbones:
                        break
                    bb = bones[bone]
                    p, q = list(bb['pos']), bb['quat']
                    data = o + 4
                    if not flags & STUDIO_ANIM_DELTA:
                        if flags & STUDIO_ANIM_RAWROT2:
                            q = _quat64(d, data)
                            data += 8
                        elif flags & STUDIO_ANIM_RAWROT:
                            q = _quat48(d, data)
                            data += 6
                        elif flags & STUDIO_ANIM_ANIMROT:
                            offs = struct.unpack_from('<3h', d, data)
                            r = [bb['rot'][k] + (_anim_value0(d, data + offs[k]) * bb['rotscale'][k]
                                                 if offs[k] else 0.0) for k in range(3)]
                            q = _euler_quat(*r)
                            data += 6
                        if flags & STUDIO_ANIM_RAWPOS:
                            p = [_half(h) for h in struct.unpack_from('<3H', d, data)]
                        elif flags & STUDIO_ANIM_ANIMPOS:
                            offs = struct.unpack_from('<3h', d, data)
                            p = [bb['pos'][k] + (_anim_value0(d, data + offs[k]) * bb['posscale'][k]
                                                 if offs[k] else 0.0) for k in range(3)]
                        pose[bone] = (p, q)
                    if nxt == 0:
                        break
                    o += nxt
        b2m = []
        for b in range(numbones):
            p, q = pose[b]
            local = _quat_mat(q, p)
            par = bones[b]['parent']
            b2m.append(_mat_mul(b2m[par], local) if 0 <= par < b else local)
        mats = [_mat_mul(b2m[b], bones[b]['p2b']) for b in range(numbones)]
    except (struct.error, IndexError, ValueError):
        return None
    ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    if all(abs(m[i][j] - ident[i][j]) < 1e-4 for m in mats for i in range(3) for j in range(4)):
        return None
    return mats


def apply_bone_pose(verts, weights, mats):
    """Skin (pos, normal, uv) vertices by their bone weights."""
    out = []
    for (pos, nrm, uv), (w, bidx) in zip(verts, weights):
        P = [0.0, 0.0, 0.0]
        N = [0.0, 0.0, 0.0]
        tot = 0.0
        for wk, bk in zip(w, bidx):
            if wk <= 0 or bk >= len(mats):
                continue
            m = mats[bk]
            for i in range(3):
                P[i] += wk * (m[i][0] * pos[0] + m[i][1] * pos[1] + m[i][2] * pos[2] + m[i][3])
                N[i] += wk * (m[i][0] * nrm[0] + m[i][1] * nrm[1] + m[i][2] * nrm[2])
            tot += wk
        if tot <= 0:
            m = mats[0]
            P = [m[i][0] * pos[0] + m[i][1] * pos[1] + m[i][2] * pos[2] + m[i][3] for i in range(3)]
            N = [m[i][0] * nrm[0] + m[i][1] * nrm[1] + m[i][2] * nrm[2] for i in range(3)]
            tot = 1.0
        P = [x / tot for x in P]
        ln = math.sqrt(sum(x * x for x in N)) or 1.0
        out.append((tuple(P), tuple(x / ln for x in N), uv))
    return out


def parse_vvd(path):
    d = open(path, 'rb').read()
    if d[0:4] != b'IDSV':
        raise ValueError("not a valid .vvd (bad magic)")
    checksum, numLODs = struct.unpack_from('<ii', d, 8)
    numLODVertexes = struct.unpack_from('<8i', d, 16)
    numFixups, fixupTableStart, vertexDataStart, tangentDataStart = struct.unpack_from('<4i', d, 48)

    weights = []

    def read_vert(i):
        off = vertexDataStart + i*48
        pos = struct.unpack_from('<3f', d, off+16)
        normal = struct.unpack_from('<3f', d, off+28)
        uv = struct.unpack_from('<2f', d, off+40)
        # mstudioboneweight_t: float weight[3], char bone[3], byte numbones
        w = struct.unpack_from('<3f', d, off)
        bidx = struct.unpack_from('<3B', d, off + 12)
        nb = d[off + 15]
        weights.append((w[:nb], bidx[:nb]))
        return (pos, normal, uv)

    if numFixups == 0:
        verts = [read_vert(i) for i in range(numLODVertexes[0])]
    else:
        # vertexFileFixup_t = (lod, sourceVertexID, numVertexes) - confirmed
        # against the Valve Developer Community's documented struct layout,
        # and independently confirmed against real ground truth across 20
        # real fixup-table models (see project history: an earlier version
        # had sourceVertexID/numVertexes swapped, which is impossible to
        # detect from plausible-looking output alone - the vertex-count sum
        # simply can't reach numLODVertexes[0] under the wrong field order,
        # regardless of ordering/filtering assumptions). For LOD0 (all this
        # writer targets), concatenate every fixup whose lod >= 0, in table
        # order - which in practice is every fixup, since lod is never
        # negative, but the >= check is kept for fidelity to the spec.
        verts = []
        for i in range(numFixups):
            off = fixupTableStart + i*12
            lod, sourceVertexID, numVertexes = struct.unpack_from('<3i', d, off)
            if lod >= 0:
                verts.extend(read_vert(sourceVertexID + j) for j in range(numVertexes))

    return {'checksum': checksum, 'numFixups': numFixups, 'verts': verts, 'weights': weights}


def parse_vtx(path, lod=0):
    """Triangles of one LOD (0 = full detail). Every LOD's origMeshVertID
    indexes the same LOD0 vertex list of its mesh - Source renders lower LODs
    out of the root LOD's vertex buffer, it only switches index lists - so a
    lower LOD is simply fewer triangles over a subset of the same vertices."""
    d = open(path, 'rb').read()
    (version, vertCacheSize, maxBonesPerStrip, maxBonesPerTri, maxBonesPerVert,
     checkSum, numLODs, matReplOffset, numBodyParts, bodyPartOffset) = \
        struct.unpack_from('<iihhiiiiii', d, 0)

    bodyparts_out = []
    for bpi in range(numBodyParts):
        bp_off = bodyPartOffset + bpi * 8
        numModels, modelOffset = struct.unpack_from('<ii', d, bp_off)
        model_off = bp_off + modelOffset  # model 0 always first, matching mdl side
        numLODs_, lodOffset = struct.unpack_from('<ii', d, model_off)
        use_lod = max(0, min(lod, numLODs_ - 1))
        lod0_off = model_off + lodOffset + use_lod * 12   # sizeof(ModelLODHeader_t)
        numMeshes, meshOffset, switchPoint = struct.unpack_from('<iif', d, lod0_off)

        mesh_tris = []
        for m in range(numMeshes):
            mesh_off = lod0_off + meshOffset + m*9
            numStripGroups, sgOffset, mflags = struct.unpack_from('<iib', d, mesh_off)
            tris = []
            if numStripGroups > 0:
                sg_off = mesh_off + sgOffset
                numVerts, vertOffset, numIndices, indexOffset, numStrips, stripOffset, sgflags = \
                    struct.unpack_from('<iiiiiib', d, sg_off)
                vert_base = sg_off + vertOffset
                # origMeshVertID is MESH-LOCAL (0-based within this mesh's own
                # vertex range) - confirmed the hard way during validation, see
                # this file's module docstring. Used directly, unmodified, as
                # the local index MD3 itself wants for a per-surface vertex
                # array, which is convenient: no offset math needed here at all.
                orig_ids = [struct.unpack_from('<H', d, vert_base+i*9+4)[0] for i in range(numVerts)]
                idx_base = sg_off + indexOffset
                idxs = struct.unpack_from('<{}H'.format(numIndices), d, idx_base)
                tris = [(orig_ids[idxs[t]], orig_ids[idxs[t+1]], orig_ids[idxs[t+2]])
                        for t in range(0, numIndices, 3)]
            mesh_tris.append(tris)
        bodyparts_out.append(mesh_tris)

    return {'checksum': checkSum, 'bodyparts': bodyparts_out, 'num_lods': numLODs}


def find_vtx_path(mdl_path):
    base = mdl_path[:-4] if mdl_path.lower().endswith('.mdl') else mdl_path
    for ext in ('.dx90.vtx', '.dx80.vtx', '.sw.vtx'):
        if os.path.isfile(base + ext):
            return base + ext
    return None


# ----------------------------------------------------------------- write --

MAX_QPATH = 64
# largest coordinate an int16 in 1/64 units can hold (32767/64 = 511.98)
MD3_COORD_LIMIT = 32767 / 64.0


# Shader names longer than this get a deterministic short form instead of being
# cut at 63 (which also left no room for q3map2 -rename's _BSP). KEEP IDENTICAL
# to short_qpath in vmt2shader.py, which names the shader block the same way.
QPATH_NAME_MAX = 59


def short_qpath(name, limit=QPATH_NAME_MAX):
    if len(name) <= limit:
        return name
    import hashlib
    digest = hashlib.md5(name.lower().encode('utf-8')).hexdigest()[:8]
    return name[:limit - 9].rstrip('/_.') + '_' + digest


def encode_qpath(name):
    b = short_qpath(name or '').encode('ascii', errors='replace')[:MAX_QPATH - 1]
    return b + b'\x00' * (MAX_QPATH - len(b))


def encode_normal(nx, ny, nz):
    """Inverse of the validated decode formula (see mdl_to_md3 module
    docstring / project history: decode is lat=high_byte, lng=low_byte,
    x=cos(lat)sin(lng), y=sin(lat)sin(lng), z=cos(lng)). Round-trip tested
    against 2000 random unit vectors before use: worst-case error 0.017,
    consistent with the format's inherent 1-byte-per-axis quantization."""
    lng = math.acos(max(-1.0, min(1.0, nz)))
    lat = math.atan2(ny, nx)
    if lat < 0:
        lat += 2 * math.pi
    lat_b = round(lat * 255.0 / (2*math.pi)) & 0xff
    lng_b = round(lng * 255.0 / (2*math.pi)) & 0xff
    return lng_b, lat_b  # (low byte, high byte)


def split_oversized_md3_surface(verts, tris, max_verts=1000):
    """Split one surface's (verts, tris) into multiple (verts, tris) chunks,
    each with at most max_verts vertices - MD3's own numVerts field is a
    signed 16-bit count, so no single surface can exceed that regardless of
    engine limits, but the more immediate constraint here is SOF2's own
    renderer: SHADER_MAX_VERTEXES = 1000 (confirmed in the original Quake3
    engine source, qfiles.h - the same codebase era SOF2 is built on).
    q3map2 enforces that automatically for misc_model with forced-meta
    (spawnflags 4), splitting any oversized surface itself at compile time -
    but a model spawned at RUNTIME (e.g. SOF2's model_static, which bypasses
    q3map2's meta pipeline entirely) has no such safety net, so the MD3 file
    must already respect it. Confirmed directly: a real model_static test
    compiled fine except for surfaces over 1000 verts.

    Walks triangles in order, starting a new chunk whenever adding a
    triangle's own (up to 3 new) vertices would exceed the cap - never
    splits a triangle itself across chunks. Returns a list of (verts, tris)
    pairs; a single pair, unchanged, if already under the cap.
    """
    if len(verts) <= max_verts:
        return [(verts, tris)]
    chunks = []
    cur_verts, cur_tris, remap = [], [], {}
    for tri in tris:
        new_ids = [i for i in tri if i not in remap]
        if len(cur_verts) + len(new_ids) > max_verts and cur_tris:
            chunks.append((cur_verts, cur_tris))
            cur_verts, cur_tris, remap = [], [], {}
            new_ids = list(tri)
        new_tri = []
        for i in tri:
            if i not in remap:
                remap[i] = len(cur_verts)
                cur_verts.append(verts[i])
            new_tri.append(remap[i])
        cur_tris.append(tuple(new_tri))
    if cur_tris:
        chunks.append((cur_verts, cur_tris))
    return chunks


def build_surface(name, verts, tris, material_path, shift=(0.0, 0.0, 0.0)):
    """verts: list of (pos, normal, uv) already sliced to this mesh's own
    range. tris: list of 3-tuples, LOCAL indices into verts (0-based)."""
    numVerts = len(verts)
    numTriangles = len(tris)

    # Layout (relative to surface start), matching the validated read layout:
    # header(108) -> triangles -> shaders -> st -> xyznormal -> end
    ofs_triangles = 108
    tri_block = b''.join(struct.pack('<3i', *t) for t in tris)

    ofs_shaders = ofs_triangles + len(tri_block)
    shader_block = encode_qpath(material_path) + struct.pack('<i', 0)

    ofs_st = ofs_shaders + len(shader_block)
    st_block = b''.join(struct.pack('<2f', uv[0], uv[1]) for _pos, _n, uv in verts)

    ofs_xyznormal = ofs_st + len(st_block)
    xyz_block_parts = []
    for pos, normal, _uv in verts:
        x = max(-32768, min(32767, round((pos[0] - shift[0]) * 64.0)))
        y = max(-32768, min(32767, round((pos[1] - shift[1]) * 64.0)))
        z = max(-32768, min(32767, round((pos[2] - shift[2]) * 64.0)))
        low_b, high_b = encode_normal(*normal)
        xyz_block_parts.append(struct.pack('<3h2B', x, y, z, low_b, high_b))
    xyz_block = b''.join(xyz_block_parts)

    ofs_end = ofs_xyznormal + len(xyz_block)

    header = struct.pack('<4s64siiiiiiiiii',
                          b'IDP3',
                          name.encode('ascii', 'replace')[:63].ljust(64, b'\x00'),
                          0, 1, 1, numVerts, numTriangles,
                          ofs_triangles, ofs_shaders, ofs_st, ofs_xyznormal, ofs_end)

    return header + tri_block + shader_block + st_block + xyz_block


def convert(mdl_path, output_path, materials_dir=None, model_scale=1.0, lod=0):
    vtx_path = find_vtx_path(mdl_path)
    if not vtx_path:
        raise FileNotFoundError("no .dx90.vtx/.dx80.vtx/.sw.vtx found next to " + mdl_path)
    vvd_path = mdl_path[:-4] + '.vvd'
    if not os.path.isfile(vvd_path):
        raise FileNotFoundError("no .vvd found next to " + mdl_path)

    mdl = parse_mdl(mdl_path, materials_dir)
    vvd = parse_vvd(vvd_path)
    vtx = parse_vtx(vtx_path, lod)

    if mdl['checksum'] != vvd['checksum'] or mdl['checksum'] != vtx['checksum']:
        raise ValueError("checksum mismatch between .mdl/.vvd/.vtx - files don't match "
                          "(mdl={} vvd={} vtx={})".format(mdl['checksum'], vvd['checksum'], vtx['checksum']))
    if len(mdl['bodyparts']) != len(vtx['bodyparts']):
        raise ValueError("bodypart count mismatch between .mdl ({}) and .vtx ({})".format(
            len(mdl['bodyparts']), len(vtx['bodyparts'])))

    pose_mats = mdl_bone_skin_matrices(open(mdl_path, 'rb').read())
    flip_winding = False
    if pose_mats is not None:
        vvd['verts'] = apply_bone_pose(vvd['verts'], vvd['weights'], pose_mats)
        m0 = pose_mats[0]
        det = (m0[0][0] * (m0[1][1] * m0[2][2] - m0[1][2] * m0[2][1])
               - m0[0][1] * (m0[1][0] * m0[2][2] - m0[1][2] * m0[2][0])
               + m0[0][2] * (m0[1][0] * m0[2][1] - m0[1][1] * m0[2][0]))
        flip_winding = det < 0
        mdl['warnings'].append("drawn in its sequence-0 bone pose (Source skins every vertex "
                               "by it; this model's is not the identity)")

    if model_scale != 1.0:
        # For models going into model_static: unlike misc_model, it does
        # NOT honor the entity's own "modelscale" key at runtime (found the
        # hard way - a real in-game test showed model_static props at the
        # wrong, unscaled size). The only fix is to bake the scale directly
        # into the geometry here, before anything else runs - including the
        # MD3 range check right below, since scaling up can push a model
        # that was previously just inside MD3's +-512 limit over it.
        # Normals are unit vectors and stay untouched; only position scales.
        vvd['verts'] = [((p[0] * model_scale, p[1] * model_scale, p[2] * model_scale), n, uv)
                        for p, n, uv in vvd['verts']]

    all_min = [min(v[0][k] for v in vvd['verts']) for k in range(3)]
    all_max = [max(v[0][k] for v in vvd['verts']) for k in range(3)]

    # MD3 vertices are int16 in 1/64 units: nothing may be more than ~512
    # units from the model origin. Anything further used to be silently
    # clamped (e.g. palm_tree_trunk, 536 units tall: its bottom 24 units were
    # squashed flat at -512, so the tree floated). For each axis that
    # overflows, move the geometry so the axis is centred on the origin, and
    # return the shift: vmf2map.py --model-offsets moves the entity's origin
    # by exactly that much (rotated/scaled like the model), so it lands where
    # Source had it.
    warnings = list(mdl['warnings'])
    shift = [0.0, 0.0, 0.0]
    for k in range(3):
        if all_min[k] < -MD3_COORD_LIMIT or all_max[k] > MD3_COORD_LIMIT:
            shift[k] = float(round((all_min[k] + all_max[k]) / 2.0))
            if all_max[k] - all_min[k] > 2 * MD3_COORD_LIMIT:
                warnings.append("model is {:.0f} units along {} - larger than MD3's 1024-unit "
                                "range even when centred; the ends will still be clamped".format(
                                    all_max[k] - all_min[k], 'XYZ'[k]))
    all_min = [all_min[k] - shift[k] for k in range(3)]
    all_max = [all_max[k] - shift[k] for k in range(3)]
    radius = (max(math.sqrt(sum((v[0][k] - shift[k]) ** 2 for k in range(3))) for v in vvd['verts'])
              if vvd['verts'] else 0.0)

    surfaces = []
    materials = []
    surf_idx = 0
    for bpi, bp in enumerate(mdl['bodyparts']):
        vtx_meshes = vtx['bodyparts'][bpi]
        if len(bp['meshes']) != len(vtx_meshes):
            raise ValueError("mesh count mismatch in bodypart '{}': .mdl has {}, .vtx has {}".format(
                bp['name'], len(bp['meshes']), len(vtx_meshes)))
        for m, mesh in enumerate(bp['meshes']):
            lo, n = mesh['vertexoffset'], mesh['numvertices']
            mesh_verts = vvd['verts'][lo:lo+n]
            tris = vtx_meshes[m]
            if flip_winding:
                tris = [(t[0], t[2], t[1]) for t in tris]
            if lod:
                # keep only the vertices this LOD's triangles use
                used = sorted({i for t in tris for i in t})
                remap = {old: new for new, old in enumerate(used)}
                mesh_verts = [mesh_verts[i] for i in used]
                tris = [tuple(remap[i] for i in t) for t in tris]
                if not tris:
                    continue
            mat_path = mesh['full_material_path'] or 'models/unknown'
            for chunk_verts, chunk_tris in split_oversized_md3_surface(mesh_verts, tris,
                                                                       MD3_MAX_VERTS_PER_SURFACE):
                surf_name = "surf{}".format(surf_idx)
                surfaces.append(build_surface(surf_name, chunk_verts, chunk_tris, mat_path, shift))
                materials.append(mat_path)
                surf_idx += 1

    ofs_frames = 108
    frame = struct.pack('<3f3f3ff16s',
                         all_min[0], all_min[1], all_min[2],
                         all_max[0], all_max[1], all_max[2],
                         0.0, 0.0, 0.0,
                         radius,
                         b'\x00' * 16)
    ofs_tags = ofs_frames + len(frame)
    ofs_surfaces = ofs_tags  # numTags = 0, no tag data
    surfaces_blob = b''.join(surfaces)
    ofs_eof = ofs_surfaces + len(surfaces_blob)

    # No single model_name once there's more than one bodypart (each has its
    # own) - use the .mdl's own filename instead, which is unique and always
    # available regardless of bodypart count.
    header_name = os.path.splitext(os.path.basename(mdl_path))[0]
    header = struct.pack('<4si64s9i',
                          b'IDP3', 15,
                          header_name.encode('ascii', 'replace')[:63].ljust(64, b'\x00'),
                          0,               # flags
                          1,               # numFrames
                          0,               # numTags
                          len(surfaces),   # numSurfaces
                          0,               # numSkins
                          ofs_frames, ofs_tags, ofs_surfaces, ofs_eof)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(header + frame + surfaces_blob)

    # A model that's essentially flat in one axis (window/door inserts,
    # signage, decal-like props) is placed FLUSH against a wall brush in
    # Source, which has no problem with coplanar surfaces. idTech3 does:
    # the model and the wall fight for the depth test, so the prop either
    # z-fights or loses outright and disappears behind the wall. Reported
    # on several de_dust2 props (du_window_*_flat, door01a) sitting behind
    # or flickering against the same wall. Flagging them here lets
    # vmt2shader.py put 'polygonOffset' on their shaders - the same
    # mechanism decals already use to sit on a surface without fighting it.
    extents = sorted(all_max[k] - all_min[k] for k in range(3))
    planar = extents[0] <= PLANAR_MODEL_MAX_THICKNESS and extents[2] > extents[0] * 4
    # Largest dimension, not a diagonal/volume figure - a long thin prop
    # (a pipe, a plank) should count by its length, since that's what
    # determines whether it reads as "small clutter" vs a real structure.
    max_extent = extents[2]
    small = max_extent <= SMALL_MODEL_MAX_EXTENT

    return {'num_surfaces': len(surfaces), 'materials': materials, 'warnings': warnings,
            'origin_offset': tuple(shift), 'planar': planar, 'thickness': extents[0],
            'small': small, 'max_extent': max_extent, 'num_lods': vtx.get('num_lods', 1),
            'num_verts': sum(struct.unpack_from('<i', sb, 80)[0] for sb in surfaces)}


def find_vvd_vtx(mdl_path):
    """Return (vvd_path, vtx_path) if both exist next to this .mdl, else None."""
    base = mdl_path[:-4] if mdl_path.lower().endswith('.mdl') else mdl_path
    vvd_path = base + '.vvd'
    if not os.path.isfile(vvd_path):
        return None
    vtx_path = find_vtx_path(mdl_path)
    if not vtx_path:
        return None
    return vvd_path, vtx_path


def convert_batch(input_dir, output_dir, materials_dir=None, model_scale=None):
    """Walk input_dir for every .mdl with complete .vvd/.vtx siblings, convert
    each, and write the result to output_dir preserving the same relative
    path (just swapping the extension to .md3) - matching the same
    directory-mirroring convention used throughout this pipeline (e.g.
    md3_shader_autofix.py's --output).

    model_scale, if given, ALSO writes a second, "_static"-suffixed copy of
    every model with that scale baked directly into its geometry (e.g.
    "beer_bottle.md3" and "beer_bottle_static.md3") - for vmf2map.py to
    reference on any prop it routes to model_static (whether by the small-
    model heuristic or by skybox position), since that entity type does
    NOT honor the ordinary "modelscale" key at runtime the way misc_model
    does (confirmed via a real in-game test - see convert()'s own comment).
    The ordinary, unscaled file is unaffected either way, so misc_model
    props keep working exactly as before.
    """
    converted = []
    skipped_no_siblings = []
    skipped_unsupported = []
    failed = []

    for dirpath, _dirs, files in os.walk(input_dir):
        for fn in sorted(files):
            if not fn.lower().endswith('.mdl'):
                continue
            mdl_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(mdl_path, input_dir)

            siblings = find_vvd_vtx(mdl_path)
            if not siblings:
                skipped_no_siblings.append(rel)
                continue

            rel_out = os.path.splitext(rel)[0] + '.md3'
            out_path = os.path.join(output_dir, rel_out)
            try:
                result = convert(mdl_path, out_path, materials_dir)
                # Source's own lower LODs, as <name>_lod1.md3, _lod2 ... for
                # vmf2map.py's baked-geometry budget (see its
                # plan_prop_lods): a heavy prop that is placed many times can
                # be baked from one of these instead of LOD0.
                if result.get('num_lods', 1) > 1 and result.get('num_verts', 0) >= LOD_VARIANT_MIN_VERTS:
                    prev = result['num_verts']
                    for lod in range(1, result['num_lods']):
                        lod_out = os.path.splitext(out_path)[0] + '_lod{}.md3'.format(lod)
                        r = convert(mdl_path, lod_out, materials_dir, lod=lod)
                        if r['num_verts'] >= prev:      # no smaller than the one before
                            os.remove(lod_out)
                            break
                        prev = r['num_verts']
                if model_scale is not None:
                    static_out = os.path.splitext(out_path)[0] + '_static.md3'
                    convert(mdl_path, static_out, materials_dir, model_scale=model_scale)
                converted.append((rel, result))
            except NotImplementedError as e:
                # A real, known, documented limitation (currently: multiple
                # bodyparts or multiple models per bodypart) - not a bug,
                # but genuinely unsupported. Reported separately from real
                # failures so it's clear these specific models still need
                # Noesis, rather than looking like something broke.
                skipped_unsupported.append((rel, str(e)))
            except Exception as e:
                failed.append((rel, str(e)))

    return {
        'converted': converted,
        'skipped_no_siblings': skipped_no_siblings,
        'skipped_unsupported': skipped_unsupported,
        'failed': failed,
    }


OFFSETS_JSON_NAME = 'md3_origin_offsets.json'

# Models at least this heavy (LOD0 vertices) also get their lower Source LODs
# written as <name>_lodN.md3. Lighter ones cost too little to be worth it.
LOD_VARIANT_MIN_VERTS = 1000

# MD3's own numVerts field per surface is int16, but the constraint that
# actually bites first is SOF2's renderer (SHADER_MAX_VERTEXES=1000,
# confirmed in the original Quake3 engine source, qfiles.h - the same
# codebase era SOF2 is built on). q3map2 enforces that automatically for
# misc_model with forced-meta (spawnflags 4), splitting any oversized
# surface itself at compile time - but a model spawned at RUNTIME (SOF2's
# model_static, which bypasses q3map2's meta pipeline entirely) has no such
# safety net, so the MD3 file must already respect it. Confirmed directly:
# a real model_static test compiled fine except for surfaces over 1000
# verts. See split_oversized_md3_surface, used by convert() for every
# surface regardless of destination classname - harmless overhead for
# misc_model (which q3map2 would split anyway) and required for
# model_static (which it wouldn't).
#
# Set one under the true 1000, not at it: a real in-game test hit
# "RB_CheckOverflow: verts > MAX (1000 > 1000)" with a surface at exactly
# 1000 - the engine's own check is apparently >= rather than >, so 1000 is
# already one too many.
MD3_MAX_VERTS_PER_SURFACE = 999
PLANAR_MATERIALS_JSON_NAME = 'md3_planar_materials.json'
SMALL_MODELS_JSON_NAME = 'md3_small_models.json'

# A model thinner than this in its smallest axis counts as "flat" (see
# convert()). Source's own flat window/door inserts measure well under this;
# a genuinely 3D prop is far thicker, so this separates them cleanly without
# catching ordinary geometry.
PLANAR_MODEL_MAX_THICKNESS = 4.0

# Largest dimension (raw source-model units - mdl_to_md3.py bakes no scale
# factor of its own; that's applied later, at placement, via the prop's
# own modelscale key) under which a model counts as "small clutter"
# (bottles, cans, tires, debris) rather than a real structural prop. A
# heuristic, not a hard fact - tunable via --small-model-max-extent.
SMALL_MODEL_MAX_EXTENT = 64.0


def update_offsets_json(json_path, new_offsets):
    """Merge {model key: (x, y, z)} into json_path (created if missing). Only
    models whose geometry was actually moved are kept."""
    data = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    for key, off in new_offsets.items():
        if any(off):
            data[key] = [round(c, 4) for c in off]
        else:
            data.pop(key, None)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data


def model_key(path):
    """'.../models/props/x/y.md3' -> 'models/props/x/y.md3'; otherwise the
    path as given. vmf2map matches keys as a path suffix, so either works."""
    p = path.replace('\\', '/').lower()
    i = p.rfind('models/')
    return p[i:] if i >= 0 else p.lstrip('./')


def main():
    global SMALL_MODEL_MAX_EXTENT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mdl", help="a single .mdl file, OR a folder to batch-convert every .mdl "
                                 "found within it (recursively)")
    ap.add_argument("output", help="output .md3 path (single-file mode), OR output folder "
                                    "(batch mode) - folder structure is mirrored from the "
                                    "input folder, same convention as md3_shader_autofix.py")
    ap.add_argument("--materials-dir", help="the extracted materials/ folder (from "
                                             "vmf_asset_extractor.py). When a model lists more "
                                             "than one cdmaterials search folder, this lets the "
                                             "embedded shader name match whichever folder the real "
                                             ".vmt actually landed in, instead of always assuming "
                                             "the first one listed. Optional but recommended - "
                                             "without it, a model like this falls back to the "
                                             "first cdmaterials folder regardless of where its "
                                             ".vmt really is, which can embed a shader name with "
                                             "no matching definition.")
    ap.add_argument("--small-model-max-extent", type=float, default=SMALL_MODEL_MAX_EXTENT,
                    help="largest dimension (raw model units), at or under which a model counts "
                         "as small clutter (bottles, cans, tires) worth spawning as model_static "
                         "instead of misc_model. Default %(default)g. A heuristic on the model "
                         "file's own size, not its placed scale in the map.")
    ap.add_argument("--model-scale", type=float, help="batch mode only: vmf2map.py's own "
                    "--model-scale value. When given, ALSO writes a second, '<name>_static.md3' "
                    "copy of every model with this scale baked directly into the geometry, for "
                    "vmf2map.py to point any model_static prop at - model_static does NOT honor "
                    "the ordinary 'modelscale' entity key at runtime the way misc_model does "
                    "(confirmed via a real in-game test: props rendered at the wrong, unscaled "
                    "size). Omit this and model_static props will be the wrong size.")
    args = ap.parse_args()
    SMALL_MODEL_MAX_EXTENT = args.small_model_max_extent

    if os.path.isdir(args.mdl):
        summary = convert_batch(args.mdl, args.output, args.materials_dir, args.model_scale)
        print("Converted: {}".format(len(summary['converted'])), file=sys.stderr)
        for rel, result in summary['converted']:
            print("  {} -> {} surface(s)".format(rel, result['num_surfaces']), file=sys.stderr)

        if summary['skipped_unsupported']:
            print("\nSkipped - not yet supported ({}):".format(len(summary['skipped_unsupported'])),
                  file=sys.stderr)
            print("  (these still need Noesis for now)", file=sys.stderr)
            for rel, reason in summary['skipped_unsupported']:
                print("  {}: {}".format(rel, reason), file=sys.stderr)

        if summary['skipped_no_siblings']:
            print("\nSkipped - missing .vvd/.vtx sibling ({}):".format(
                len(summary['skipped_no_siblings'])), file=sys.stderr)
            for rel in summary['skipped_no_siblings']:
                print("  {}".format(rel), file=sys.stderr)

        offsets = {model_key(os.path.splitext(rel)[0] + '.md3'): r['origin_offset']
                   for rel, r in summary['converted']}
        moved = {k: v for k, v in offsets.items() if any(v)}
        update_offsets_json(os.path.join(args.output, OFFSETS_JSON_NAME), offsets)
        if moved:
            print("\nMoved back into MD3's 512-unit range ({}) - pass {} to vmf2map.py "
                  "--model-offsets:".format(len(moved), os.path.join(args.output, OFFSETS_JSON_NAME)),
                  file=sys.stderr)
            for k, v in sorted(moved.items()):
                print("  {} by ({:g} {:g} {:g})".format(k, *v), file=sys.stderr)

        planar_materials = sorted({m for _rel, r in summary['converted'] if r.get('planar')
                                    for m in r['materials']})
        if planar_materials:
            planar_path = os.path.join(args.output, PLANAR_MATERIALS_JSON_NAME)
            with open(planar_path, 'w', encoding='utf-8') as f:
                json.dump(planar_materials, f, indent=2)
            print("\nFlat (planar) model(s) found - their material(s) written to {}; pass it to "
                  "vmt2shader.py --planar-model-materials so they get polygonOffset (these props "
                  "sit flush against wall brushes and otherwise z-fight or hide behind them):"
                  .format(planar_path), file=sys.stderr)
            for rel, r in summary['converted']:
                if r.get('planar'):
                    print("  {} ({:.2g} units thick)".format(rel, r['thickness']), file=sys.stderr)

        small_models = sorted('models/' + (os.path.splitext(rel)[0] + '.md3').replace('\\', '/')
                              for rel, r in summary['converted'] if r.get('small'))
        if small_models:
            small_path = os.path.join(args.output, SMALL_MODELS_JSON_NAME)
            with open(small_path, 'w', encoding='utf-8') as f:
                json.dump(small_models, f, indent=2)
            print("\nSmall (<= {:g} unit) model(s) found - {} of them, written to {}; pass it to "
                  "vmf2map.py --model-static-list so they spawn as model_static instead of "
                  "misc_model by model identity rather than placement (except inside a 3D skybox "
                  "area, which always keeps misc_model - model_static is invisible to q3map2's "
                  "skybox marking and would never render as a backdrop). This is a heuristic on the "
                  "model FILE's own size (bottles, cans, tires, debris), not its placed scale - "
                  "spot-check a few if something looks off, and use "
                  "--small-model-max-extent to adjust.".format(
                      SMALL_MODEL_MAX_EXTENT, len(small_models), small_path), file=sys.stderr)

        if summary['failed']:
            print("\nFAILED ({}):".format(len(summary['failed'])), file=sys.stderr)
            for rel, reason in summary['failed']:
                print("  {}: {}".format(rel, reason), file=sys.stderr)
        return

    try:
        result = convert(args.mdl, args.output, args.materials_dir)
    except (NotImplementedError, ValueError, FileNotFoundError) as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        sys.exit(1)

    print("Wrote {} surface(s) to {}".format(result['num_surfaces'], args.output), file=sys.stderr)
    for mat in result['materials']:
        print("  material: {}".format(mat), file=sys.stderr)
    for w in result.get('warnings', []):
        print("  NOTE: {}".format(w), file=sys.stderr)
    json_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), OFFSETS_JSON_NAME)
    update_offsets_json(json_path, {model_key(args.output): result['origin_offset']})
    if any(result['origin_offset']):
        print("  NOTE: geometry moved by ({:g} {:g} {:g}) to fit MD3's 512-unit range - recorded in "
              "{}; pass it to vmf2map.py --model-offsets".format(*result['origin_offset'], json_path),
              file=sys.stderr)


if __name__ == "__main__":
    main()
