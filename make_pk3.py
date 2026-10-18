"""
make_pk3.py - build the .pk3 archives install_to_sof2.py and the compile .bat
need. A .pk3 is an ordinary zip; SOF2 (and q3map2, and Radiant) mount every
.pk3 sitting in base/ and read assets straight out of it, so a converted map
does not need a single loose file.

VERIFIED against the real q3map2 (netradiant-custom 2.5.17n, -game sof2): with
a .pk3 in base/ holding only shaders/x.shader and textures/x/y.tga, q3map2
printed "entering shaders/x.shader" and compiled without complaint; with the
same .pk3 removed it printed "WARNING: Couldn't find image for shader
textures/x/y". So the compile reads shaders and images from inside the archive
exactly like the game does.

Two jobs:

    make_pk3.py pack <src_dir> <dest.pk3>
        Zip a staged tree (shaders/, textures/, models/, *.pcx) into dest.

    make_pk3.py release <sof2_base> <map_name> [--out <file.pk3>]
        Take base/<map_name>.pk3 (the assets, written at install time) and the
        compiled base/maps/<map_name>.bsp, and write the one archive you give
        to someone else: base/maps/<map_name>_release.pk3, assets + .bsp.

Why the compiled .bsp is NOT in the .pk3 that lives in base/: Quake3-family
filesystems search mounted .pk3s BEFORE loose files, so a .bsp inside base/
<map>.pk3 would quietly shadow the freshly compiled one in base/maps/ and you
would be testing the previous build. The install .pk3 therefore holds no
maps/ folder at all - the .bsp stays loose while you work on the map, and only
the release archive bundles the two together.
"""
import argparse
import os
import re
import sys
import zipfile
from collections import Counter

# Working files that live in the output folder but are not game assets.
SKIP_EXTS = {'.json', '.txt', '.mdl', '.vvd', '.vtx', '.phy', '.ani', '.bak',
             '.vmt', '.vtf', '.log', '.py', '.pyc', '.map', '.srf', '.prt',
             '.lin', '.bak1'}


# Source materials carry inputs for per-pixel lighting that idTech3 has no use
# for at all: normal maps, ssbumps, height, specular exponent and env masks,
# plus the odd extra mip level extracted as its own image. vmt2shader.py cannot
# reference them (there is no shader stage that could), so they are converted,
# installed and then sit in the release archive as dead weight - measured on
# de_overpass_csgo: 36 MiB of a 177 MiB archive. They are left out of the
# release, but ONLY after checking that nothing in the map refers to them.
DEAD_MAP_SUFFIXES = (
    '_normal', '_normals', '_nrm', '_n',
    '_ssbump', '-ssbump', '_height', '-height',
    '_exponent', '_envmask', '_env', '_ref', '_spec', '_specular',
    '_mip0', '_mip1', '_mip2',
)
TEXTURE_EXTS = ('.png', '.jpg', '.jpeg', '.tga')
# skyParms names the set; the engine appends the face, so sky_x_bk.tga is used
# by a shader that only ever says sky_x.
SKY_FACES = ('_bk', '_ft', '_lf', '_rt', '_up', '_dn')
_TEX_REF_RE = re.compile(rb'[A-Za-z0-9_\-/\.]*textures/[A-Za-z0-9_\-/\.]+', re.I)


def _texture_key(rel):
    low = rel.lower()
    for ext in TEXTURE_EXTS:
        if low.endswith(ext):
            return low[:-len(ext)]
    return None


def _looks_dead(rel):
    key = _texture_key(rel)
    return bool(key) and key.endswith(DEAD_MAP_SUFFIXES)


def _referenced_textures(read_bytes, names):
    """Every textures/... path named inside the shaders, models and .bsp."""
    out = set()
    for rel in names:
        if not rel.lower().endswith(('.shader', '.md3', '.lwo', '.bsp')):
            continue
        try:
            data = read_bytes(rel)
        except (OSError, KeyError):
            continue
        for m in _TEX_REF_RE.findall(data):
            s = m.decode('ascii', 'ignore').lower().lstrip('./')
            out.add(re.sub(r'\.(png|jpg|jpeg|tga)$', '', s))
    return out


def drop_dead_textures(names, read_bytes, bsp_bytes=None):
    """(kept, dropped) - textures with a Source-only map suffix that nothing
    in this map refers to. A file is dropped only when BOTH are true, so a
    shader that does name one keeps it."""
    candidates = [rel for rel in names if _looks_dead(rel)]
    if not candidates:
        return list(names), []
    refs = _referenced_textures(read_bytes, names)
    if bsp_bytes:
        for m in _TEX_REF_RE.findall(bsp_bytes):
            s = m.decode('ascii', 'ignore').lower().lstrip('./')
            refs.add(re.sub(r'\.(png|jpg|jpeg|tga)$', '', s))

    def used(rel):
        key = _texture_key(rel)
        if key in refs:
            return True
        for face in SKY_FACES:
            if key.endswith(face) and key[:-len(face)] in refs:
                return True
        return False

    dead = {rel for rel in candidates if not used(rel)}
    return [rel for rel in names if rel not in dead], sorted(dead)


def iter_files(src_dir):
    """Every file under src_dir as (abs_path, archive_path), sorted."""
    out = []
    for dirpath, _dirs, files in os.walk(src_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in SKIP_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, src_dir).replace(os.sep, '/')
            out.append((full, rel))
    out.sort(key=lambda p: p[1])
    return out


def write_dir_entries(z, arcnames):
    """Add a zero-byte entry for every folder the file paths imply.

    The zip format does not need these - a file at "textures/a/b.jpg" defines
    its own folders - and the engine never looks at them: Quake3's unzip walks
    the central directory and resolves assets by full path. Archivers write
    them anyway, and some file managers build their tree lazily from file paths
    instead, which showed up here as a blank row where "textures" should have
    been until the list was re-sorted. These cost nothing and make the archive
    look like any other pk3 the moment it is opened."""
    dirs = set()
    for rel in arcnames:
        parts = rel.split('/')[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add('/'.join(parts[:i]) + '/')
    for d in sorted(dirs):
        info = zipfile.ZipInfo(d)
        info.external_attr = (0o40755 << 16) | 0x10  # directory bit, both conventions
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, b'')
    return len(dirs)


def pack(src_dir, dest_pk3, extra=()):
    """Zip src_dir into dest_pk3. extra is (disk_path, archive_path) pairs
    added on top (used for the compiled .bsp)."""
    if not os.path.isdir(src_dir):
        print("ERROR: nothing to pack - {} does not exist".format(src_dir), file=sys.stderr)
        return None
    entries = iter_files(src_dir) + [(d, a.replace(os.sep, '/')) for d, a in extra]
    if not entries:
        print("ERROR: nothing to pack - {} is empty".format(src_dir), file=sys.stderr)
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest_pk3)) or '.', exist_ok=True)
    tmp = dest_pk3 + '.tmp'
    total = 0
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        write_dir_entries(z, [rel for _f, rel in entries])
        for full, rel in entries:
            z.write(full, rel)
            total += os.path.getsize(full)
    # Replace only once the new archive is complete, so an interrupted run
    # cannot leave a half-written .pk3 mounted in base/.
    if os.path.isfile(dest_pk3):
        os.remove(dest_pk3)
    os.replace(tmp, dest_pk3)
    print("  {} - {} file(s), {:.1f} MB raw -> {:.1f} MB".format(
        os.path.basename(dest_pk3), len(entries), total / 1048576.0,
        os.path.getsize(dest_pk3) / 1048576.0))
    # Read the finished archive back and say what is in it, so the log shows
    # the file's own table of contents rather than this function's intentions.
    verify(dest_pk3)
    return dest_pk3


def verify(pk3, quiet=False):
    """Re-read a finished .pk3 and report what is actually in it.

    Written because a run reported packing 690 textures while a zip viewer
    showed only two folders: the archive's own table of contents is the only
    thing that settles which is true. It also flags entries Windows cannot
    extract even though the archive is valid - a component ending in a space
    or a dot, a reserved device name, a character illegal in a Win32 path - and
    empty-named entries, which a viewer draws as a blank row. Those are real:
    the Source material names this pipeline converts have included trailing
    spaces before the extension before now."""
    try:
        with zipfile.ZipFile(pk3) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
    except (OSError, zipfile.BadZipFile):
        infos = []
    names = [i.filename for i in infos]
    empty = [i.filename for i in infos if i.file_size == 0]
    tops = Counter(n.split('/')[0] if '/' in n else '(root)' for n in names)
    bad = []
    for n in names:
        leaf = n.rsplit('/', 1)[-1]
        parts = n.split('/')
        if not leaf:
            bad.append((n, "empty name"))
        elif not leaf.split('.')[0]:
            bad.append((n, "no name before the extension - Explorer draws this as a blank row"))
        elif any(p != p.strip() or p.endswith('.') for p in parts):
            bad.append((n, "component ends in a space or dot - Windows cannot create it"))
        elif any(c in n for c in '<>:"|?*\\') or any(ord(c) < 32 for c in n):
            bad.append((n, "character illegal in a Windows path"))
        elif leaf.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL'} or \
                re.match(r'^(COM|LPT)[1-9]$', leaf.split('.')[0].upper()):
            bad.append((n, "reserved Windows device name"))
        elif len(n) > 200:
            bad.append((n, "very long path ({} chars)".format(len(n))))
    if not quiet:
        print("  contains {} file(s): {}".format(
            len(names), ", ".join("{} {}".format(c, t) for t, c in sorted(tops.items()))))
    if empty:
        print("  WARNING: {} packed file(s) are 0 bytes - whatever produced them wrote "
              "nothing: {}{}".format(len(empty), ", ".join(empty[:5]),
                                     ", ..." if len(empty) > 5 else ""), file=sys.stderr)
    if bad:
        print("  WARNING: {} entry name(s) Windows will struggle with - a zip viewer may show "
              "them blank or skip the folder they are in:".format(len(bad)), file=sys.stderr)
        for n, why in bad[:10]:
            print("    {!r}  ({})".format(n, why), file=sys.stderr)
        if len(bad) > 10:
            print("    ... and {} more".format(len(bad) - 10), file=sys.stderr)
    return names, bad


def archive_names(pk3):
    try:
        with zipfile.ZipFile(pk3) as z:
            return [n for n in z.namelist() if not n.endswith('/')]
    except (OSError, zipfile.BadZipFile):
        return []


def read_manifest(path):
    """The base-relative paths install_to_sof2.py copied for this map."""
    names = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                names.append(line)
    return names


def release(base, name, out=None, manifest=None):
    """This map's assets in base + base/maps/<name>.bsp -> one shareable archive.

    The assets come from <output>/<name>_files.txt, the manifest
    install_to_sof2.py writes when it copies them in. base holds loose files
    now and several converted maps share it, so that manifest is the only
    thing that says which files belong to this map - the directory tree does
    not. A base/<name>.pk3 is still used when one is there, so a base left
    over from the older archive-based install still releases correctly.
    """
    assets = os.path.join(base, name + '.pk3')
    bsp = os.path.join(base, 'maps', name + '.bsp')
    out = out or os.path.join(base, 'maps', name + '_release.pk3')
    # The compile .bat points this at the conversion's output folder. If that
    # folder has since been moved or deleted, fall back to sitting next to the
    # .bat rather than failing at the end of a finished compile.
    out_dir = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            out = os.path.join(base, 'maps', os.path.basename(out))
            print("  ({} is gone - writing the release archive to {} instead)".format(
                out_dir, os.path.dirname(out)))
    if not os.path.isfile(bsp):
        print("ERROR: {} not found - compile the map first.".format(bsp), file=sys.stderr)
        return 1
    # Prefer the manifest (loose install). Fall back to base/<name>.pk3 only
    # when one is actually present, from an install by the older step.
    from_pk3 = os.path.isfile(assets)
    names = []
    if not from_pk3:
        if not manifest:
            manifest = os.path.join(os.path.dirname(os.path.abspath(out)),
                                    name + '_files.txt')
        if not os.path.isfile(manifest):
            print("ERROR: no asset list for '{}'. Expected the manifest\n"
                  "         {}\n"
                  "       written by step 10 (install). Run step 10, then compile "
                  "again.".format(name, manifest), file=sys.stderr)
            return 1
        names = [rel for rel in read_manifest(manifest)
                 if not rel.lower().startswith('maps/')]
        missing = [rel for rel in names if not os.path.isfile(os.path.join(base, *rel.split('/')))]
        if missing:
            print("ERROR: {} file(s) the manifest lists are not in base, e.g. {} - "
                  "run step 10 (install) again.".format(len(missing), missing[0]),
                  file=sys.stderr)
            return 1
    with open(bsp, 'rb') as f:
        bsp_bytes = f.read()
    tmp = out + '.tmp'
    n = 0
    saved = 0
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        if from_pk3:
            with zipfile.ZipFile(assets) as src:
                keep = [i for i in src.infolist()
                        if not i.is_dir() and not i.filename.lower().startswith('maps/')]
                by_name = {i.filename: i for i in keep}
                kept_names, dropped = drop_dead_textures(
                    [i.filename for i in keep], src.read, bsp_bytes)
                saved = sum(by_name[r].file_size for r in dropped)
                write_dir_entries(z, kept_names + ['maps/x'])
                for rel in kept_names:
                    z.writestr(rel, src.read(rel))
                    n += 1
        else:
            def _read(rel):
                with open(os.path.join(base, *rel.split('/')), 'rb') as f:
                    return f.read()
            kept_names, dropped = drop_dead_textures(names, _read, bsp_bytes)
            saved = sum(os.path.getsize(os.path.join(base, *r.split('/'))) for r in dropped)
            write_dir_entries(z, kept_names + ['maps/x'])
            for rel in kept_names:
                z.write(os.path.join(base, *rel.split('/')), rel)
                n += 1
        z.write(bsp, 'maps/{}.bsp'.format(name))
        n += 1
    if dropped:
        print("  Left out {} texture(s) nothing in the map refers to - Source normal, "
              "ssbump, height, specular and env-mask images, which idTech3 has no stage "
              "for ({:.1f} MB smaller). They stay in base.".format(len(dropped),
                                                                   saved / 1048576.0))
    if os.path.isfile(out):
        os.remove(out)
    os.replace(tmp, out)
    print("Release archive: {}  ({} files, {:.1f} MB)".format(
        out, n, os.path.getsize(out) / 1048576.0))
    verify(out)
    print("That single .pk3 is the whole map - anyone who drops it in their own "
          "SOF2 base folder can load it.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack', help="zip a staged asset tree into a .pk3")
    p.add_argument('src_dir')
    p.add_argument('dest_pk3')
    v = sub.add_parser('verify', help="list what a .pk3 actually contains, and flag "
                                      "entry names Windows cannot extract")
    v.add_argument('pk3')
    r = sub.add_parser('release', help="this map's assets in base + compiled .bsp "
                                       "-> one shareable .pk3")
    r.add_argument('sof2_base')
    r.add_argument('map_name')
    r.add_argument('--out', help="where to write it (default: "
                                 "<base>/maps/<map>_release.pk3)")
    r.add_argument('--manifest', help="the <map>_files.txt install_to_sof2.py wrote, "
                                      "listing this map's files in base (default: next "
                                      "to --out). Ignored when a base/<map>.pk3 exists.")
    args = ap.parse_args()
    if args.cmd == 'pack':
        return 0 if pack(args.src_dir, args.dest_pk3) else 1
    if args.cmd == 'verify':
        print(os.path.basename(args.pk3) + ":")
        names, bad = verify(args.pk3)
        if not names:
            print("  nothing readable in it - not a zip, or truncated.", file=sys.stderr)
            return 1
        return 1 if bad else 0
    return release(args.sof2_base, args.map_name, args.out, args.manifest)


if __name__ == '__main__':
    sys.exit(main())
