import struct, math, os, sys

def cstr(data, off):
    return data[off:data.index(b'\x00', off)].decode('latin1', errors='replace')

def decode_normal(low_b, high_b):
    lat = high_b * (2*math.pi) / 255.0
    lng = low_b * (2*math.pi) / 255.0
    return (math.cos(lat)*math.sin(lng), math.sin(lat)*math.sin(lng), math.cos(lng))

def dist(a, b):
    return sum((a[i]-b[i])**2 for i in range(len(a))) ** 0.5

def parse_mdl(path):
    d = open(path, 'rb').read()
    if d[0:4] != b'IDST':
        return None
    version, = struct.unpack_from('<i', d, 4)
    checksum, = struct.unpack_from('<i', d, 8)
    numtextures, textureindex = struct.unpack_from('<ii', d, 204)
    numcdtextures, cdtextureindex = struct.unpack_from('<ii', d, 212)
    numskinref, numskinfamilies, skinindex = struct.unpack_from('<iii', d, 220)
    numbodyparts, bodypartindex = struct.unpack_from('<ii', d, 232)

    texnames = [cstr(d, textureindex+i*64+struct.unpack_from('<i', d, textureindex+i*64)[0])
                for i in range(numtextures)]
    skin_family0 = [texnames[struct.unpack_from('<h', d, skinindex+i*2)[0]] for i in range(numskinref)] \
        if numskinref else []

    bodyparts = []
    for bp in range(numbodyparts):
        bp_off = bodypartindex + bp*16
        sznameindex, nummodels, base, modelindex = struct.unpack_from('<4i', d, bp_off)
        models = []
        for mi in range(nummodels):
            model_off = bp_off + modelindex + mi*148
            o = model_off + 64
            nummeshes, meshindex = struct.unpack_from('<ii', d, o+8)
            numvertices, vertexindex, tangentsindex = struct.unpack_from('<iii', d, o+16)
            meshes = []
            for m in range(nummeshes):
                mesh_off = model_off + meshindex + m * 116
                material, _back = struct.unpack_from('<ii', d, mesh_off)
                mnumvertices, mvertexoffset = struct.unpack_from('<ii', d, mesh_off+8)
                meshes.append({'material': material, 'numvertices': mnumvertices,
                                'vertexoffset': mvertexoffset})
            models.append({'meshes': meshes, 'numvertices': numvertices,
                            'vertexindex': vertexindex})
        bodyparts.append({'models': models})
    return {'version': version, 'checksum': checksum, 'bodyparts': bodyparts,
            'skin_family0': skin_family0}


def parse_vvd(path, target_lod=0):
    d = open(path, 'rb').read()
    if d[0:4] != b'IDSV':
        return None
    version, checksum, numLODs = struct.unpack_from('<iii', d, 4)
    numLODVertexes = struct.unpack_from('<8i', d, 16)
    numFixups, fixupTableStart, vertexDataStart, tangentDataStart = struct.unpack_from('<4i', d, 48)

    def read_vert(i):
        off = vertexDataStart + i*48
        pos = struct.unpack_from('<3f', d, off+16)
        normal = struct.unpack_from('<3f', d, off+28)
        uv = struct.unpack_from('<2f', d, off+40)
        return (pos, normal, uv)

    if numFixups == 0:
        verts = [read_vert(i) for i in range(numLODVertexes[0])]
    else:
        # vertexFileFixup_t is (lod, sourceVertexID, numVertexes) - confirmed
        # against the Valve Developer Community's documented struct layout
        # after an earlier version of this code had sourceVertexID and
        # numVertexes swapped, which silently produced a vertex-count sum
        # that could never match numLODVertexes[0] under any reordering.
        # Composition rule (also VDC-documented): concatenate, in table
        # order, every fixup whose lod >= the target LOD.
        verts = []
        for i in range(numFixups):
            off = fixupTableStart + i*12
            lod, sourceVertexID, numVertexes = struct.unpack_from('<3i', d, off)
            if lod >= target_lod:
                verts.extend(read_vert(sourceVertexID + j) for j in range(numVertexes))

    return {'checksum': checksum, 'numFixups': numFixups, 'verts': verts, 'total_n': len(verts)}


def parse_vtx(path):
    d = open(path, 'rb').read()
    (version, vertCacheSize, maxBonesPerStrip, maxBonesPerTri, maxBonesPerVert,
     checkSum, numLODs, matReplOffset, numBodyParts, bodyPartOffset) = \
        struct.unpack_from('<iihhiiiiii', d, 0)
    bodyparts = []
    for bp in range(numBodyParts):
        bp_off = bodyPartOffset + bp*8
        numModels, modelOffset = struct.unpack_from('<ii', d, bp_off)
        models = []
        for mi in range(numModels):
            model_off = bp_off + modelOffset + mi*8
            numLODs_, lodOffset = struct.unpack_from('<ii', d, model_off)
            lod0_off = model_off + lodOffset
            numMeshes, meshOffset, switchPoint = struct.unpack_from('<iif', d, lod0_off)
            meshes = []
            for m in range(numMeshes):
                mesh_off = lod0_off + meshOffset + m*9
                numStripGroups, sgOffset, mflags = struct.unpack_from('<iib', d, mesh_off)
                tris = []
                if numStripGroups > 0:
                    sg_off = mesh_off + sgOffset
                    numVerts, vertOffset, numIndices, indexOffset, numStrips, stripOffset, sgflags = \
                        struct.unpack_from('<iiiiiib', d, sg_off)
                    vert_base = sg_off + vertOffset
                    orig_ids = [struct.unpack_from('<H', d, vert_base+i*9+4)[0] for i in range(numVerts)]
                    idx_base = sg_off + indexOffset
                    idxs = struct.unpack_from('<{}H'.format(numIndices), d, idx_base)
                    tris = [(orig_ids[idxs[t]], orig_ids[idxs[t+1]], orig_ids[idxs[t+2]])
                            for t in range(0, numIndices, 3)]
                meshes.append(tris)
            models.append(meshes)
        bodyparts.append(models)
    return {'checksum': checkSum, 'bodyparts': bodyparts}


def parse_md3(path):
    d = open(path, 'rb').read()
    if d[0:4] != b'IDP3':
        return None
    num_surfaces, = struct.unpack_from('<i', d, 84)
    ofs_surfaces, = struct.unpack_from('<i', d, 100)
    surfaces = []
    off = ofs_surfaces
    for s in range(num_surfaces):
        shname = None
        numFrames, numShaders, numVerts, numTriangles = struct.unpack_from('<4i', d, off+72)
        ofs_triangles, ofs_shaders, ofs_st, ofs_xyznormal, ofs_end = struct.unpack_from('<5i', d, off+88)
        if numShaders:
            shname = cstr(d, off+ofs_shaders)
        verts = []
        for i in range(numVerts):
            x, y, z = struct.unpack_from('<3h', d, off+ofs_xyznormal+i*8)
            lb, hb = struct.unpack_from('<2B', d, off+ofs_xyznormal+i*8+6)
            s_, t_ = struct.unpack_from('<2f', d, off+ofs_st+i*8)
            verts.append(((x/64, y/64, z/64), decode_normal(lb, hb), (s_, t_)))
        tris_raw = struct.unpack_from('<{}i'.format(numTriangles*3), d, off+ofs_triangles)
        tris = [tuple(tris_raw[t+k] for k in range(3)) for t in range(0, len(tris_raw), 3)]
        surfaces.append({'shader': shname, 'verts': verts, 'tris': tris})
        off += ofs_end
    return surfaces


def cross_check_mesh(vvd_verts, vtx_tris, md3_surface):
    """Returns (pos_max, uv_max, norm_max, tri_count_ok) or None if structurally
    incompatible (different triangle count - a real mismatch worth flagging)."""
    md3_verts, md3_tris = md3_surface['verts'], md3_surface['tris']
    if len(vtx_tris) != len(md3_tris):
        return None
    if not vtx_tris:
        return (0.0, 0.0, 0.0, True)

    md3_centroids = [tuple(sum(md3_verts[i][0][k] for i in t)/3 for k in range(3)) for t in md3_tris]

    pos_errs, uv_errs, norm_errs = [], [], []
    for tri in vtx_tris:
        vpts = [vvd_verts[i] for i in tri]
        vc = tuple(sum(p[0][k] for p in vpts)/3 for k in range(3))
        best_i = min(range(len(md3_tris)), key=lambda i: dist(vc, md3_centroids[i]))
        md3_tri = md3_tris[best_i]
        for vpos, vnorm, vuv in vpts:
            mj = min(md3_tri, key=lambda j: dist(vpos, md3_verts[j][0]))
            mpos, mnorm, muv = md3_verts[mj]
            pos_errs.append(dist(vpos, mpos))
            norm_errs.append(dist(vnorm, mnorm))
            uv_errs.append(dist(vuv, muv))

    return (max(pos_errs), max(uv_errs), max(norm_errs), True)


def validate_model(base_path):
    mdl = parse_mdl(base_path + '.mdl')
    vvd = parse_vvd(base_path + '.vvd')
    vtx = parse_vtx(base_path + '.dx90.vtx')
    md3 = parse_md3(base_path + '.md3')
    if not all([mdl, vvd, vtx, md3]):
        return {'status': 'PARSE_FAIL', 'detail': 'one or more files failed to parse'}

    if mdl['checksum'] != vvd['checksum'] or mdl['checksum'] != vtx['checksum']:
        return {'status': 'CHECKSUM_MISMATCH'}

    if len(mdl['bodyparts']) != 1 or len(mdl['bodyparts'][0]['models']) != 1:
        return {'status': 'MULTI_BODYPART_OR_MODEL',
                'detail': '{} bodyparts, {} models'.format(
                    len(mdl['bodyparts']), len(mdl['bodyparts'][0]['models']) if mdl['bodyparts'] else 0)}

    meshes = mdl['bodyparts'][0]['models'][0]['meshes']
    vtx_meshes = vtx['bodyparts'][0][0]
    if len(meshes) != len(vtx_meshes):
        return {'status': 'MESH_COUNT_MISMATCH_MDL_VTX'}
    if len(meshes) != len(md3):
        return {'status': 'MESH_COUNT_MISMATCH_MDL_MD3',
                'detail': '{} mdl meshes vs {} md3 surfaces'.format(len(meshes), len(md3))}

    results = []
    for m, mesh in enumerate(meshes):
        lo = mesh['vertexoffset']
        # vtx's origMeshVertID is MESH-RELATIVE, not a global vvd index - it
        # must be offset by this mesh's own vertexoffset before it points at
        # the right vvd vertex. Confirmed the hard way: an earlier version of
        # this function omitted the offset, which is invisible for a mesh at
        # vertexoffset=0 (the offset is a no-op) but silently reads
        # completely wrong vertices for every subsequent mesh in a
        # multi-mesh model - exactly the kind of bug that would corrupt
        # every mesh after the first in a real write path.
        offset_tris = [tuple(lo + i for i in tri) for tri in vtx_meshes[m]]
        r = cross_check_mesh(vvd['verts'], offset_tris, md3[m])
        if r is None:
            return {'status': 'TRIANGLE_COUNT_MISMATCH', 'detail': 'mesh {}'.format(m)}
        results.append(r)

    pos_max = max(r[0] for r in results)
    uv_max = max(r[1] for r in results)
    norm_max = max(r[2] for r in results)
    return {'status': 'OK', 'pos_max': pos_max, 'uv_max': uv_max, 'norm_max': norm_max,
            'num_meshes': len(meshes)}


def main(root):
    bases = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith('.mdl'):
                base = os.path.join(dirpath, fn)[:-4]
                if all(os.path.isfile(base + ext) for ext in ('.vvd', '.dx90.vtx', '.md3')):
                    bases.append(base)

    print("Testing {} model(s) with complete file sets...\n".format(len(bases)))
    status_counts = {}
    worst_pos = worst_uv = worst_norm = 0.0
    failures = []
    ok_results = []
    for base in sorted(bases):
        name = os.path.relpath(base, root)
        try:
            r = validate_model(base)
        except Exception as e:
            r = {'status': 'EXCEPTION', 'detail': str(e)}
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 'OK':
            worst_pos = max(worst_pos, r['pos_max'])
            worst_uv = max(worst_uv, r['uv_max'])
            worst_norm = max(worst_norm, r['norm_max'])
            ok_results.append((name, r['pos_max'], r['uv_max'], r['norm_max']))
        else:
            failures.append((name, status, r.get('detail', '')))

    print("=== SUMMARY ===")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print("  {}: {}".format(status, count))
    print("\nAcross all OK models:")
    print("  worst position error: {:.4f}".format(worst_pos))
    print("  worst uv error:       {:.4f}".format(worst_uv))
    print("  worst normal error:   {:.4f}".format(worst_norm))

    print("\n=== TOP 10 WORST 'OK' MODELS BY POSITION ERROR ===")
    for name, p, u, n in sorted(ok_results, key=lambda r: -r[1])[:10]:
        print("  {}  pos={:.4f}  uv={:.4f}  norm={:.4f}".format(name, p, u, n))

    if failures:
        print("\n=== NON-OK MODELS ===")
        for name, status, detail in failures:
            print("  {}  [{}]  {}".format(name, status, detail))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
