#!/usr/bin/env python3
"""
scan_multi_bodypart.py - Find which .mdl files in a folder have multiple
bodyparts or multiple models-per-bodypart - the one case mdl_to_md3.py
doesn't support yet (still needs Noesis + md3_shader_autofix.py for these
specific models).

Reads the same header fields already verified in mdl_to_md3.py's parser,
just reporting instead of refusing.

USAGE
-----
    python3 scan_multi_bodypart.py <folder> [--extract-to <folder>]

Walks the folder recursively for every .mdl file and reports which ones
have numbodyparts > 1, or any single bodypart with nummodels > 1.

--extract-to <folder>   Also copy every FLAGGED model's complete sibling
                        set (.mdl, .vvd, .dx90.vtx/.dx80.vtx/.sw.vtx, .phy,
                        .md3, .ani - whichever exist) into this folder,
                        preserving the same relative path as the source -
                        so you can zip the result and send it as a real
                        test case, the same way every other model example
                        in this project got shared. Only flagged models
                        are copied; everything else is left alone.
"""

import os
import shutil
import struct
import sys

SIBLING_EXTS = ('.mdl', '.vvd', '.dx90.vtx', '.dx80.vtx', '.sw.vtx', '.phy', '.md3', '.ani')


def scan_mdl(path):
    with open(path, 'rb') as f:
        d = f.read()
    if d[0:4] != b'IDST':
        return None
    numbodyparts, bodypartindex = struct.unpack_from('<ii', d, 232)
    bodyparts = []
    for bp in range(numbodyparts):
        bp_off = bodypartindex + bp * 16
        sznameindex, nummodels, base, modelindex = struct.unpack_from('<4i', d, bp_off)
        name = d[bp_off+sznameindex:].split(b'\x00')[0].decode('latin1', errors='replace')
        bodyparts.append({'name': name, 'nummodels': nummodels})
    return {'numbodyparts': numbodyparts, 'bodyparts': bodyparts}


def find_siblings(mdl_path):
    """Every file sharing this .mdl's base name across all known sibling
    extensions - only the ones that actually exist."""
    base = mdl_path[:-4] if mdl_path.lower().endswith('.mdl') else mdl_path
    found = []
    for ext in SIBLING_EXTS:
        p = base + ext
        if os.path.isfile(p):
            found.append(p)
    return found


def extract_flagged(flagged, root, extract_to):
    copied_models = 0
    copied_files = 0
    for rel, _info in flagged:
        mdl_path = os.path.join(root, rel)
        for src in find_siblings(mdl_path):
            src_rel = os.path.relpath(src, root)
            dst = os.path.join(extract_to, src_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied_files += 1
        copied_models += 1
    return copied_models, copied_files


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    extract_to = None
    if '--extract-to' in sys.argv:
        i = sys.argv.index('--extract-to')
        if i + 1 >= len(sys.argv):
            print("ERROR: --extract-to needs a folder argument", file=sys.stderr)
            sys.exit(1)
        extract_to = sys.argv[i + 1]

    total = 0
    simple = 0
    flagged = []
    failed = []

    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith('.mdl'):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            total += 1
            try:
                info = scan_mdl(path)
            except Exception as e:
                failed.append((rel, str(e)))
                continue
            if info is None:
                failed.append((rel, 'not a valid studiomdl (bad magic)'))
                continue

            multi_model_bodyparts = [bp for bp in info['bodyparts'] if bp['nummodels'] > 1]
            if info['numbodyparts'] > 1 or multi_model_bodyparts:
                flagged.append((rel, info))
            else:
                simple += 1

    print("Scanned {} .mdl file(s) under {}\n".format(total, root))
    print("Simple (1 bodypart, 1 model - supported by mdl_to_md3.py): {}".format(simple))
    print("Flagged (multi-bodypart or multi-model - needs Noesis for now): {}".format(len(flagged)))
    if failed:
        print("Failed to read: {}".format(len(failed)))

    if flagged:
        print("\n=== FLAGGED MODELS ===")
        for rel, info in flagged:
            print("\n{}".format(rel))
            print("  numbodyparts: {}".format(info['numbodyparts']))
            for bp in info['bodyparts']:
                flag = "  <-- multiple models" if bp['nummodels'] > 1 else ""
                print("    bodypart '{}': nummodels={}{}".format(bp['name'], bp['nummodels'], flag))

    if failed:
        print("\n=== FAILED TO READ ===")
        for rel, reason in failed:
            print("  {}: {}".format(rel, reason))

    if extract_to:
        if flagged:
            n_models, n_files = extract_flagged(flagged, root, extract_to)
            print("\nCopied {} flagged model(s), {} file(s) total, to: {}".format(
                n_models, n_files, extract_to))
        else:
            print("\nNothing flagged - nothing to extract.")


if __name__ == "__main__":
    main()
