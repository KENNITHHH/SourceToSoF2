#!/usr/bin/env python3
"""
shaderlist_for_compile.py - Keep q3map2 under its shader limit while one
converted map compiles.

WHY
---
q3map2 loads every shader file named in base/shaders/shaderlist.txt (or, with
no shaderlist.txt, every .shader it can find). Each converted map installs
its own .shader there, and a converted CS map defines 700-1500 shaders. A few
of them next to SOF2's own and the compile stops before it starts:

    MAX_SHADER_INFO exceeded. Remove some PK3 files or shader scripts from
    shaderlist.txt and try again.

(GtkRadiant 1.6.7's q3map2 2.5.17: 8192 shaders. NetRadiant's has a higher
limit, so it hits this later.) A converted map never needs ANOTHER converted
map's shader file: each one defines every shader it uses itself (shared CS
textures come out identical in all of them).

WHAT IT DOES
------------
    apply   <shaders dir> <map name> [shader file to keep] ...
        Saves shaderlist.txt as shaderlist.txt.s2s_backup and writes a
        shaderlist.txt without the other maps' shader files. A LOOSE
        .shader file is another map's when it holds one of the converters'
        fingerprints (so old conversions with no signature line, and CoD4
        conversions, are recognised too) or when base/maps holds a .map or
        .bsp of the same name. Anything inside a .pk3 is SOF2's own and is
        always kept, as is any name given after the map name. When there is
        no shaderlist.txt, one is written listing every shader file found
        loose and in base's .pk3s, minus those, and removed again by
        "restore".
    restore <shaders dir>
        Puts the original shaderlist.txt back.

The compile .bat runs "apply" before q3map2 and "restore" when it finishes or
fails. A backup left behind by an interrupted compile is restored first, so
the list never loses an entry.
"""
import os
import sys
import zipfile

BACKUP = 'shaderlist.txt.s2s_backup'
NONE_MARK = '// s2s: there was no shaderlist.txt\n'

# A shader file belongs to a converted map if any of these appears anywhere in
# it. The signature line is only in files written by a recent vmt2shader.py, so
# the rest are fingerprints of what the converters emit - older versions wrote
# no signature at all, and the CoD4 converter writes its own.
GENERATOR_MARKS = (
    'converted by vmt2shader',               # vmt2shader.py, current
    'cod4mat2shader',                        # the CoD4 (IW3) converter
    'SOF2 clamps vertex colour',             # the 2x vertex-colour decal stage
    'per-pixel lighting features',           # the $bumpmap/$envmap/$phong note
    'no classic idTech3 equivalent',
)
SCAN_BYTES = 4 * 1024 * 1024     # whole file in practice: the big ones are ~300 KB


def _map_names(shaders_dir):
    """Map names that have a source or compiled map in base/maps."""
    base = os.path.dirname(os.path.abspath(shaders_dir))
    names = set()
    try:
        for fn in os.listdir(os.path.join(base, 'maps')):
            low = fn.lower()
            for ext in ('.map', '.bsp'):
                if low.endswith(ext):
                    names.add(low[:-len(ext)])
    except OSError:
        pass
    return names


def _other_map_reason(shaders_dir, entry, map_name, map_names, keep):
    """Why this shaderlist entry is another map's, or None to keep it.

    Only loose .shader files are ever left out: an entry that resolves inside a
    .pk3 is SOF2's own content and stays.
    """
    low = entry.lower()
    if low == map_name.lower() or low in keep:
        return None
    path = os.path.join(shaders_dir, entry + '.shader')
    try:
        with open(path, 'rb') as f:
            data = f.read(SCAN_BYTES)
    except OSError:
        return None           # not loose: inside a .pk3
    text = data.decode('utf-8', 'ignore')
    for mark in GENERATOR_MARKS:
        if mark in text:
            return 'converted map'
    if low in map_names:
        return 'map in base/maps'
    return None


def _count_shaders(shaders_dir, entries):
    """Rough number of shader definitions the kept list loads, for the report."""
    total = 0
    pk3_texts = None
    for entry in entries:
        text = None
        try:
            with open(os.path.join(shaders_dir, entry + '.shader'), 'rb') as f:
                text = f.read(SCAN_BYTES).decode('utf-8', 'ignore')
        except OSError:
            if pk3_texts is None:
                pk3_texts = _pk3_shader_texts(shaders_dir)
            text = pk3_texts.get(entry.lower())
        if not text:
            continue
        depth = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('{'):
                depth += 1
                continue
            if line.startswith('}'):
                depth = max(0, depth - 1)
                continue
            if depth == 0:
                total += 1
    return total


def _pk3_shader_texts(shaders_dir):
    out = {}
    base = os.path.dirname(os.path.abspath(shaders_dir))
    try:
        pk3s = sorted(fn for fn in os.listdir(base) if fn.lower().endswith('.pk3'))
    except OSError:
        return out
    for fn in pk3s:
        try:
            with zipfile.ZipFile(os.path.join(base, fn)) as z:
                for n in z.namelist():
                    nl = n.lower()
                    if nl.startswith('shaders/') and nl.endswith('.shader') and '/' not in nl[8:]:
                        out.setdefault(nl[8:-7], z.read(n).decode('utf-8', 'ignore'))
        except (OSError, zipfile.BadZipFile, KeyError):
            continue
    return out


def _all_shader_files(shaders_dir):
    names = set()
    for fn in os.listdir(shaders_dir):
        if fn.lower().endswith('.shader'):
            names.add(fn[:-7])
    base = os.path.dirname(os.path.abspath(shaders_dir))
    for fn in os.listdir(base):
        if not fn.lower().endswith('.pk3'):
            continue
        try:
            with zipfile.ZipFile(os.path.join(base, fn)) as z:
                for n in z.namelist():
                    nl = n.lower()
                    if nl.startswith('shaders/') and nl.endswith('.shader') and '/' not in nl[8:]:
                        names.add(n[8:-7])
        except (OSError, zipfile.BadZipFile):
            continue
    return sorted(names, key=str.lower)


def restore(shaders_dir):
    lst = os.path.join(shaders_dir, 'shaderlist.txt')
    bak = os.path.join(shaders_dir, BACKUP)
    if not os.path.isfile(bak):
        return False
    with open(bak, 'rb') as f:
        data = f.read()
    if data == NONE_MARK.encode('ascii'):
        try:
            os.remove(lst)
        except OSError:
            pass
    else:
        with open(lst, 'wb') as f:      # byte for byte, line endings included
            f.write(data)
    os.remove(bak)
    return True


def apply(shaders_dir, map_name, keep_extra=()):
    keep = set(k.lower() for k in keep_extra)
    if restore(shaders_dir):
        print("shaderlist: put back the shaderlist.txt an interrupted compile left behind")
    lst = os.path.join(shaders_dir, 'shaderlist.txt')
    bak = os.path.join(shaders_dir, BACKUP)
    if os.path.isfile(lst):
        with open(lst, 'rb') as f:
            original = f.read()
        entries = [l.strip() for l in original.decode('utf-8', 'ignore').splitlines()]
        entries = [e for e in entries if e and not e.startswith('//')]
        with open(bak, 'wb') as f:
            f.write(original)
    else:
        entries = _all_shader_files(shaders_dir)
        with open(bak, 'wb') as f:
            f.write(NONE_MARK.encode('ascii'))
    if map_name.lower() not in (e.lower() for e in entries):
        entries.append(map_name)
    map_names = _map_names(shaders_dir)
    kept, dropped = [], []
    for entry in entries:
        why = _other_map_reason(shaders_dir, entry, map_name, map_names, keep)
        if why:
            dropped.append((entry, why))
        else:
            kept.append(entry)
    with open(lst, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('\n'.join(kept) + '\n')
    print("shaderlist: {} shader file(s) for this compile; left out {} other map(s) so "
          "q3map2 stays under its 8192-shader limit{}".format(
              len(kept), len(dropped),
              ": " + ", ".join("{} ({})".format(e, w) for e, w in dropped)
              if dropped else " (nothing to leave out)"))
    count = _count_shaders(shaders_dir, kept)
    if count:
        print("shaderlist: about {} shader(s) in those files (q3map2's limit is 8192)".format(count))
        if count > 7800:
            print("shaderlist: WARNING - that is close to the limit. If the compile still stops "
                  "with MAX_SHADER_INFO, move some unused loose .shader files out of "
                  "{} and run this again.".format(shaders_dir))
    return 0


def main(argv):
    if len(argv) >= 4 and argv[1] == 'apply':
        return apply(argv[2], argv[3], argv[4:])
    if len(argv) >= 3 and argv[1] == 'restore':
        restore(argv[2])
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
