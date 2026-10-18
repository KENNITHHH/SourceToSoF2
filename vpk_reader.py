#!/usr/bin/env python3
"""
vpk_reader.py - Read files directly out of Source .vpk archives, without
needing them pre-extracted to loose files first.

Implements the format exactly as documented on the Valve Developer
Community wiki (VPK (file format)), not from memory - the tree structure
in particular has a couple of easy-to-miss details (an empty string means
"end of this list", a single space means "this level doesn't apply to
this entry", and preload bytes for small files live directly in the
_dir.vpk rather than a numbered archive) that are worth getting from the
spec rather than guessing.

USAGE (as a library)
---------------------
    from vpk_reader import VPKArchive

    vpk = VPKArchive("cstrike_pak_dir.vpk")
    data = vpk.read("materials/metal/metalwall014a.vmt")
    for path in vpk.files:
        ...

USAGE (standalone - list or extract one file, for testing)
-------------------------------------------------------------
    python3 vpk_reader.py list <name_dir.vpk>
    python3 vpk_reader.py extract <name_dir.vpk> <internal/path> <output_file>
"""

import os
import struct
import sys


class VPKArchive:
    """Indexes one VPK's directory tree (<name>_dir.vpk) and can extract any
    file's bytes on demand, pulling from the numbered archive chunks
    (<name>_000.vpk etc.), from preload data embedded in the directory
    file itself, or both concatenated together."""

    def __init__(self, dir_vpk_path):
        self.dir_path = dir_vpk_path
        # <name>_dir.vpk -> <name> (used to build sibling archive filenames)
        base = dir_vpk_path
        if base.lower().endswith('_dir.vpk'):
            base = base[:-len('_dir.vpk')]
        elif base.lower().endswith('.vpk'):
            base = base[:-len('.vpk')]
        self.base_name = base

        self.files = {}  # normalized path -> dict(crc, preload_bytes, archive_index,
                          #                          entry_offset, entry_length, preload_data)
        self._parse()

    def _archive_path(self, index):
        return "{}_{:03d}.vpk".format(self.base_name, index)

    def _parse(self):
        with open(self.dir_path, 'rb') as f:
            data = f.read()

        signature, version = struct.unpack_from('<II', data, 0)
        if signature != 0x55AA1234:
            raise ValueError("not a valid VPK directory file (bad signature)")

        if version == 1:
            tree_size, = struct.unpack_from('<I', data, 8)
            header_size = 12
        elif version == 2:
            (tree_size, file_data_section_size, archive_md5_section_size,
             other_md5_section_size, signature_section_size) = struct.unpack_from('<5I', data, 8)
            header_size = 28
        else:
            raise ValueError("unsupported VPK version {}".format(version))

        self.header_size = header_size
        self.tree_size = tree_size
        tree_end = header_size + tree_size  # where embedded file data (if any) begins

        pos = [header_size]

        def read_cstr():
            start = pos[0]
            end = data.index(b'\x00', start)
            pos[0] = end + 1
            return data[start:end].decode('utf-8', errors='replace')

        while True:
            ext = read_cstr()
            if ext == '':
                break
            while True:
                path = read_cstr()
                if path == '':
                    break
                while True:
                    filename = read_cstr()
                    if filename == '':
                        break

                    crc, preload_bytes, archive_index = struct.unpack_from('<IHH', data, pos[0])
                    pos[0] += 8
                    entry_offset, entry_length = struct.unpack_from('<II', data, pos[0])
                    pos[0] += 8
                    terminator, = struct.unpack_from('<H', data, pos[0])
                    pos[0] += 2
                    if terminator != 0xFFFF:
                        raise ValueError("VPK directory entry terminator mismatch at offset {} "
                                          "(got 0x{:04x}, expected 0xffff) - tree parsing has "
                                          "drifted out of sync".format(pos[0] - 2, terminator))

                    preload_data = None
                    if preload_bytes:
                        preload_data = data[pos[0]:pos[0] + preload_bytes]
                        pos[0] += preload_bytes

                    # Reassemble the real relative path. " " (a single space)
                    # at the path or extension level means "doesn't apply"
                    # (a file with no extension, or one directly at the
                    # archive root with no subdirectory) - per spec, this is
                    # deliberately distinct from "" (which means end-of-list).
                    dir_part = '' if path == ' ' else path
                    name_ext = filename if ext == ' ' else "{}.{}".format(filename, ext)
                    full_path = "{}/{}".format(dir_part, name_ext) if dir_part else name_ext
                    full_path = full_path.replace('\\', '/').lower()

                    self.files[full_path] = {
                        'crc': crc,
                        'preload_bytes': preload_bytes,
                        'archive_index': archive_index,
                        'entry_offset': entry_offset,
                        'entry_length': entry_length,
                        'preload_data': preload_data,
                    }

        self._tree_end = tree_end

    def read(self, internal_path):
        """Return the full bytes of one file, or None if not present."""
        key = internal_path.replace('\\', '/').lower().lstrip('/')
        entry = self.files.get(key)
        if entry is None:
            return None

        parts = []
        if entry['preload_data']:
            parts.append(entry['preload_data'])

        if entry['entry_length'] > 0:
            if entry['archive_index'] == 0x7FFF:
                # Data lives in the directory file itself, right after the tree.
                with open(self.dir_path, 'rb') as f:
                    f.seek(self._tree_end + entry['entry_offset'])
                    parts.append(f.read(entry['entry_length']))
            else:
                archive_path = self._archive_path(entry['archive_index'])
                if not os.path.isfile(archive_path):
                    raise FileNotFoundError(
                        "'{}' references archive chunk '{}', which wasn't found next to "
                        "the directory file".format(internal_path, archive_path))
                with open(archive_path, 'rb') as f:
                    f.seek(entry['entry_offset'])
                    parts.append(f.read(entry['entry_length']))

        return b''.join(parts)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    vpk = VPKArchive(sys.argv[2])

    if cmd == 'list':
        print("{} file(s) in {}:".format(len(vpk.files), sys.argv[2]))
        for path in sorted(vpk.files):
            print("  {}  ({} bytes)".format(path, vpk.files[path]['preload_bytes'] +
                                             vpk.files[path]['entry_length']))
    elif cmd == 'extract':
        if len(sys.argv) < 5:
            print("usage: vpk_reader.py extract <dir.vpk> <internal/path> <output_file>")
            sys.exit(1)
        data = vpk.read(sys.argv[3])
        if data is None:
            print("ERROR: '{}' not found in archive".format(sys.argv[3]), file=sys.stderr)
            sys.exit(1)
        out_path = sys.argv[4]
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        print("Wrote {} byte(s) to {}".format(len(data), out_path))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
