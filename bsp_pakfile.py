#!/usr/bin/env python3
"""
bsp_pakfile.py - Read the embedded "pakfile" lump out of a Source .bsp -
the map-specific custom content (models/materials that only exist for this
one map, never shipped in the base game's VPKs) packed directly into the
compiled map.

Header layout and the PAKFILE lump index (40) are taken directly from the
Valve Developer Community's BSP page and confirmed independently against
the srctools library's own documentation, not from memory:

    struct dheader_t {
        int    ident;                 // "VBSP"
        int    version;
        lump_t lumps[64];             // 16 bytes each: fileofs, filelen,
                                       // version, fourCC[4]
        int    mapRevision;
    };

The pakfile lump (index 40) is a plain, NEVER-compressed, standard ZIP
archive (confirmed: srctools' own docs state only zipfile.ZIP_STORED is
allowed there) - so once located, this needs no custom parsing at all,
just handing the byte range straight to Python's zipfile module.

USAGE (as a library)
---------------------
    from bsp_pakfile import BSPPakfile

    pak = BSPPakfile("de_cache.bsp")
    data = pak.read("materials/de_cache/some_map_specific_texture.vmt")
    for path in pak.files:
        ...

USAGE (standalone)
-------------------
    python3 bsp_pakfile.py list <map.bsp>
    python3 bsp_pakfile.py extract <map.bsp> <internal/path> <output_file>
    python3 bsp_pakfile.py extract-all <map.bsp> <output_folder>
"""

import io
import os
import struct
import sys
import zipfile

LUMP_PAKFILE = 40
HEADER_LUMPS = 64


class BSPPakfile:
    def __init__(self, bsp_path):
        self.bsp_path = bsp_path
        with open(bsp_path, 'rb') as f:
            data = f.read()

        ident, version = struct.unpack_from('<4sI', data, 0)
        if ident != b'VBSP':
            raise ValueError("not a valid Source .bsp (bad ident, got {!r})".format(ident))
        self.version = version

        lump_off = 8 + LUMP_PAKFILE * 16
        fileofs, filelen, lump_version, fourCC = struct.unpack_from('<iiI4s', data, lump_off)
        self.fileofs = fileofs
        self.filelen = filelen

        if filelen == 0:
            self.files = {}
            self._zip = None
            return

        pak_bytes = data[fileofs:fileofs + filelen]
        self._zip = zipfile.ZipFile(io.BytesIO(pak_bytes))
        self.files = {}
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            key = info.filename.replace('\\', '/').lower().lstrip('/')
            self.files[key] = info.filename  # keep the real (original-case) name for reading

    def read(self, internal_path):
        """Return the full bytes of one file, or None if not present."""
        if self._zip is None:
            return None
        key = internal_path.replace('\\', '/').lower().lstrip('/')
        real_name = self.files.get(key)
        if real_name is None:
            return None
        with self._zip.open(real_name) as f:
            return f.read()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    pak = BSPPakfile(sys.argv[2])

    if cmd == 'list':
        print("BSP version: {}".format(pak.version))
        print("Pakfile lump: {} byte(s) at offset {}".format(pak.filelen, pak.fileofs))
        print("{} file(s):".format(len(pak.files)))
        for path in sorted(pak.files):
            print("  {}".format(path))
    elif cmd == 'extract':
        if len(sys.argv) < 5:
            print("usage: bsp_pakfile.py extract <map.bsp> <internal/path> <output_file>")
            sys.exit(1)
        data = pak.read(sys.argv[3])
        if data is None:
            print("ERROR: '{}' not found in pakfile".format(sys.argv[3]), file=sys.stderr)
            sys.exit(1)
        out_path = sys.argv[4]
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        print("Wrote {} byte(s) to {}".format(len(data), out_path))
    elif cmd == 'extract-all':
        if len(sys.argv) < 4:
            print("usage: bsp_pakfile.py extract-all <map.bsp> <output_folder>")
            sys.exit(1)
        out_root = sys.argv[3]
        count = 0
        for key in pak.files:
            data = pak.read(key)
            out_path = os.path.join(out_root, key.replace('/', os.sep))
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(data)
            count += 1
        print("Extracted {} file(s) to {}".format(count, out_root))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
