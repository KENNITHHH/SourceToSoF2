#!/usr/bin/env python3
"""
VMF Asset Extractor
====================

Scans a .vmf file for every material, model, sound and skybox reference,
resolves the real files (including .vmt texture dependencies and .mdl
sibling files) across one or more "content root" folders, and copies
them into an output folder, preserving materials/models/sound structure.

USAGE
-----
    python vmf_asset_extractor.py <vmf_file> <sources_root_or_list> <output_folder>
                                  [--map-bsp <the map's own .bsp>]

<sources_root_or_list> can be any of:
  - Your Source game install: "...\\counter-strike source" or its game folder
    "...\\counter-strike source\\cstrike" - anything holding a gameinfo.txt,
    or one level above one. Content is then mounted exactly the way the game
    mounts it, read from gameinfo.txt's SearchPaths: for CS:S that is
    cstrike/custom, cstrike_pak.vpk, the hl2 VPKs (where props_junk,
    props_c17 and most shared props live), cstrike's loose files, hl2's,
    then cstrike/download - in that order, behind the map's own .bsp.
  - A folder that CONTAINS your extracted content trees as subfolders (e.g.
    "E:\\CSS\\Extracted" containing cstrike_pak_dir, hl2_misc_dir, ...). Every
    subfolder is auto-detected as a source root.
  - A ';'-separated explicit list of folders, .vpk files or .bsp files, e.g.
    "E:\\CSS\\Extracted\\cstrike_pak_dir;E:\\CSS\\Extracted\\hl2_misc_dir"

The map's own .bsp (found next to the .vmf, under the content roots, or given
with --map-bsp <file>) is always read first, as Source does; other maps' .bsp
files are ignored.

EXAMPLE
-------
    python vmf_asset_extractor.py "E:\\CSS\\Extracted\\de_vertigo.vmf" "E:\\CSS\\Extracted" "E:\\CSS\\VertigoAssets"

NOTES / LIMITATIONS
--------------------
- Sounds referenced through soundscript names (e.g. an entity's "noise" key
  pointing to a game_sounds.txt entry like "Weapon_AWP.Single") can't be
  resolved to a file without the soundscript manifest - only hardcoded
  raw .wav/.mp3 paths (e.g. ambient_generic's "message" key) are caught.
- info_particle_system "effect_name" values reference particle manifest
  entries, not files directly, so .pcf files are not pulled automatically.
- Model materials ARE now pulled from the .mdl's own texture table: each
  model's cdmaterials directories + texture names are read straight out
  of the compiled Source studiomdl (v44-49) binary header, and every
  candidate materials/<cdmaterials>/<texture>.vmt is resolved through the
  same .vmt dependency chain used for world/brush materials (so you get
  the .vmt AND its .vtf textures, not just the model files themselves).
  If a model uses an older/unsupported mdl layout (e.g. GoldSrc), or a
  texture genuinely isn't in any of your source folders, it's logged in
  _missing_assets.txt rather than silently skipped.
"""

import os
import re
import struct
import sys
import shutil
from collections import Counter

try:
    from vpk_reader import VPKArchive
except ImportError:
    print("ERROR: vpk_reader.py must be in the same folder as this script.", file=sys.stderr)
    sys.exit(1)

try:
    from bsp_pakfile import BSPPakfile
except ImportError:
    print("ERROR: bsp_pakfile.py must be in the same folder as this script.", file=sys.stderr)
    sys.exit(1)

try:
    import sof2_sounds
except ImportError:
    sof2_sounds = None          # soundscapes are then skipped, with a note

SOUND_EXTS = ('.wav', '.mp3')
MODEL_SINGLE_EXTS = ['.mdl', '.vvd', '.phy', '.ani']
MODEL_VTX_EXTS = ['.dx90.vtx', '.dx80.vtx', '.sw.vtx']


class SourceIndex(dict):
    """lowercased relative path -> source, plus where every entry came from.

    Still an ordinary dict to every caller (find_file() is index.get()), but
    it also remembers which source WON each path and which sources had the
    same path and LOST, so a run can report the files it actually used that
    existed in more than one place - and whether those copies differed."""
    def __init__(self):
        super().__init__()
        self.origin = {}     # key -> label of the source that supplied it
        self.shadowed = {}   # key -> [(label, source), ...] that lost to it
        self.skipped_bsps = []

    def add(self, rel, value, label):
        """Add one entry, plus a fallback entry under the same path with
        trailing whitespace stripped from the filename's stem.

        Valve content occasionally ships a file whose real name has a trailing
        space right before the extension - confirmed directly: a user's own
        CS:S install has "de_mirage_base_trim_ver1_diffuse .vmt" (space, then
        ".vmt"). Windows Explorer hides that space when you type or browse the
        name, so it looks ordinary, but os.walk() reports the real bytes, and
        every lookup built from a clean name (a .vmt's $basetexture, a brush
        face's material, cdmaterials + texture) never has that space - so an
        index keyed on the literal filename alone silently misses the file.
        Indexing both the literal and the space-stripped key fixes this
        without touching every caller, and never overwrites a real file that
        already has the clean name."""
        self._put(rel, value, label)
        base, ext = os.path.splitext(rel)
        stripped = base.rstrip() + ext
        if stripped != rel:
            self._put(stripped, value, label)

    def _put(self, key, value, label):
        if key in self:
            if self.origin.get(key) != label:
                self.shadowed.setdefault(key, []).append((label, value))
            return
        self[key] = value
        self.origin[key] = label


def _add_index_entry(index, rel, value, label='?'):
    """Kept for anything still calling the old helper."""
    if isinstance(index, SourceIndex):
        index.add(rel, value, label)
    elif rel not in index:
        index[rel] = value


def read_source(src):
    """The bytes behind an index entry, wherever it lives."""
    if isinstance(src, tuple) and src[0] in ('vpk', 'bsp'):
        return src[1].read(src[2])
    try:
        with open(src, 'rb') as f:
            return f.read()
    except OSError:
        return None


def is_source_bsp(path):
    """True for a Source engine .bsp (ident VBSP). A same-named idTech3 .bsp -
    the SOF2 map this pipeline compiles, ident RBSP - has no pakfile and must
    never be mistaken for the Source one."""
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'VBSP'
    except OSError:
        return False


def find_map_bsp(vmf_path, source_roots, map_bsp=None):
    """The Source .bsp that belongs to THIS map, or None.

    Order: an explicit --map-bsp; <map>.bsp next to the .vmf; then <map>.bsp
    anywhere under the content roots (a CS:S install keeps it in
    cstrike/maps/, and the GUI's content root is usually the whole game
    folder). The name is compared case-insensitively and the file must be a
    Source VBSP."""
    if map_bsp:
        return map_bsp if is_source_bsp(map_bsp) else None
    name = os.path.splitext(os.path.basename(vmf_path))[0].lower()
    sibling = os.path.join(os.path.dirname(os.path.abspath(vmf_path)), name + '.bsp')
    if is_source_bsp(sibling):
        return sibling
    for root in source_roots:
        if os.path.isfile(root):
            if os.path.basename(root).lower() == name + '.bsp' and is_source_bsp(root):
                return root
            continue
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower() == name + '.bsp':
                    full = os.path.join(dirpath, fn)
                    if is_source_bsp(full):
                        return full
    return None


_KV_TOKEN_RE = re.compile(r'"([^"]*)"|(\{|\})|([^\s{}"]+)')


def _kv_tokens(text):
    text = re.sub(r'//[^\n]*', '', text)
    for m in _KV_TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            yield m.group(1)
        elif m.group(2):
            yield m.group(2)
        else:
            yield m.group(3)


def gameinfo_searchpaths(gameinfo_path):
    """The (key, value) pairs of gameinfo.txt's SearchPaths block, in order."""
    try:
        with open(gameinfo_path, encoding='utf-8', errors='replace') as f:
            toks = list(_kv_tokens(f.read()))
    except OSError:
        return []
    for i, t in enumerate(toks):
        if t.lower() == 'searchpaths' and i + 1 < len(toks) and toks[i + 1] == '{':
            pairs, j = [], i + 2
            while j < len(toks) and toks[j] != '}':
                key = toks[j]
                val = toks[j + 1] if j + 1 < len(toks) else ''
                j += 2
                while j < len(toks) and toks[j].startswith('['):   # [$WIN32] conditions
                    j += 1
                if val not in ('{', '}'):
                    pairs.append((key, val))
            return pairs
    return []


def find_game_dir(root, vmf_path):
    """The Source game folder (the one holding gameinfo.txt) a content root
    points at, or None for anything else - a folder of extracted trees, say.

    Both natural answers to "where is the game" work: the game folder itself
    (counter-strike source/cstrike) or the install above it (counter-strike
    source), where the folder the .vmf sits in wins, then anything but the
    shared hl2/platform folders, which carry a gameinfo.txt of their own."""
    if os.path.isfile(os.path.join(root, 'gameinfo.txt')):
        return root
    try:
        subs = sorted(os.listdir(root))
    except OSError:
        return None
    cands = [os.path.join(root, d) for d in subs
             if os.path.isfile(os.path.join(root, d, 'gameinfo.txt'))]
    if not cands:
        return None
    vmf_abs = os.path.normcase(os.path.abspath(vmf_path))
    for c in cands:
        if vmf_abs.startswith(os.path.normcase(os.path.abspath(c)) + os.sep):
            return c
    others = [c for c in cands if os.path.basename(c).lower() not in ('hl2', 'platform', 'bin')]
    return (others or cands)[0]


def game_mounts(game_dir):
    """What Source itself mounts for this game, in its order, as a list of
    ('loose', folder) and ('vpk', *_dir.vpk) - read from gameinfo.txt.

    A Source game is not one folder. CS:S's gameinfo.txt mounts cstrike's own
    loose files and cstrike_pak.vpk, then the HL2 VPKs and loose files from
    the sibling hl2/ folder - which is where props_junk, props_c17,
    props_pipes and most other props CS:S maps use actually live - then
    cstrike/download. Pointing this script at cstrike/ alone used to find
    none of that: it treated each SUBFOLDER as a content tree, so the VPK
    sitting directly in cstrike/ was never opened and hl2/ was out of reach.
    Only keys naming the 'game' path ID are mounted (not gamebin, platform
    or the low-violence game_lv)."""
    game_dir = os.path.normpath(game_dir)
    game_root = os.path.dirname(game_dir)
    mounts, seen = [], set()

    def add(kind, path):
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            mounts.append((kind, path))

    def add_path(path):
        if path.endswith('*'):
            folder = path[:-1].rstrip('/\\') or '.'
            if os.path.isdir(folder):
                for n in sorted(os.listdir(folder)):
                    full = os.path.join(folder, n)
                    if os.path.isdir(full):
                        add('loose', full)
                    elif n.lower().endswith('_dir.vpk'):
                        add('vpk', full)
            return
        if path.lower().endswith('.vpk'):
            for cand in (path[:-4] + '_dir.vpk', path):
                if os.path.isfile(cand):
                    add('vpk', cand)
                    return
            return
        if os.path.isdir(path):
            add('loose', path)

    for key, value in gameinfo_searchpaths(os.path.join(game_dir, 'gameinfo.txt')):
        if 'game' not in [k.strip().lower() for k in key.split('+')]:
            continue
        v = value.replace('\\', '/')
        low = v.lower()
        if low.startswith('|gameinfo_path|'):
            base, rest = game_dir, v[len('|gameinfo_path|'):]
        elif low.startswith('|all_source_engine_paths|'):
            base, rest = game_root, v[len('|all_source_engine_paths|'):]
        else:
            base, rest = game_root, v
        rest = rest.strip('/')
        add_path(base if rest in ('', '.') else os.path.join(base, *rest.split('/')))

    if not mounts:   # no SearchPaths block: the game folder and hl2, as old Source did
        add('loose', game_dir)
        if os.path.isdir(os.path.join(game_root, 'hl2')):
            add('loose', os.path.join(game_root, 'hl2'))
    # Older gameinfo.txt files mount folders but not the VPKs inside them;
    # only then, mount any *_dir.vpk a loose folder holds right after that
    # folder. A gameinfo.txt that lists VPKs is taken at its word: anything
    # else lying in the folder (language packs, stray copies of the hl2 VPKs
    # in cstrike/) is not mounted by the game, so it is not mounted here.
    lists_vpks = any(kind == 'vpk' for kind, _ in mounts)
    out = []
    for kind, path in mounts:
        out.append((kind, path))
        if kind == 'loose' and not lists_vpks:
            for n in sorted(os.listdir(path)):
                full = os.path.join(path, n)
                if n.lower().endswith('_dir.vpk') and \
                        os.path.normcase(os.path.abspath(full)) not in seen:
                    seen.add(os.path.normcase(os.path.abspath(full)))
                    out.append(('vpk', full))
    return out


def build_index(source_roots, map_bsp=None):
    """Map lowercased relative path -> either a plain absolute file path, a
    ('vpk', VPKArchive, internal_path) tuple, or a ('bsp', BSPPakfile,
    internal_path) tuple - across all source roots.

    PRECEDENCE, matching what Source itself mounts while the map is played:
      1. the map's OWN .bsp (map_bsp). Source adds the map's pakfile to the
         HEAD of its search path when the map loads - that is how custom maps
         override stock materials - so anything it packs wins over the game's
         copy. Earlier versions of this script appended it LAST, so a CS:GO
         port that packs its own version of a material CS:S also ships got
         the CS:S one.
      2. the source roots in the order given, each walked top-down; within
         that, first found wins.
    Any OTHER .bsp met while walking a folder is skipped. Source never mounts
    another map's pakfile, so its content is not part of this map - and with
    the whole game folder as a content root, cstrike/maps/ holds every map
    installed, whose pakfiles were all being indexed and could supply a file
    to the wrong map (alphabetically first one wins). A .bsp passed directly
    as a source root is still honoured: that is a deliberate choice.

    A source root can be:
      - a folder of loose extracted files
      - a folder that CONTAINS one or more *_dir.vpk files anywhere within it
        (auto-detected and indexed directly, no pre-extraction needed)
      - a *_dir.vpk or *.bsp file given directly
    """
    index = SourceIndex()
    map_bsp_real = os.path.normcase(os.path.abspath(map_bsp)) if map_bsp else None

    def index_vpk(dir_vpk_path):
        try:
            vpk = VPKArchive(dir_vpk_path)
        except Exception as e:
            print(f"  ! failed to read VPK '{dir_vpk_path}': {e}")
            return
        label = os.path.basename(dir_vpk_path)
        for rel in vpk.files:
            index.add(rel, ('vpk', vpk, rel), label)

    def index_bsp(bsp_path, label):
        try:
            pak = BSPPakfile(bsp_path)
        except Exception as e:
            print(f"  ! failed to read BSP '{bsp_path}': {e}")
            return
        for rel in pak.files:
            index.add(rel, ('bsp', pak, rel), label)

    if map_bsp:
        index_bsp(map_bsp, os.path.basename(map_bsp) + " (the map's own)")

    for root in source_roots:
        if isinstance(root, tuple):
            kind, path = root
            if kind == 'vpk':
                index_vpk(path)
                continue
            # a mounted loose folder: its own files only - VPKs inside it are
            # mounted separately, at their own place in the order
            for dirpath, _, filenames in os.walk(path):
                for fn in filenames:
                    low = fn.lower()
                    full = os.path.join(dirpath, fn)
                    if low.endswith('.vpk'):
                        continue
                    if low.endswith('.bsp'):
                        if os.path.normcase(os.path.abspath(full)) != map_bsp_real:
                            index.skipped_bsps.append(fn)
                        continue
                    rel = os.path.relpath(full, path).replace('\\', '/').lower()
                    index.add(rel, full, path)
            continue
        if os.path.isfile(root) and root.lower().endswith('.vpk'):
            index_vpk(root)
            continue
        if os.path.isfile(root) and root.lower().endswith('.bsp'):
            if os.path.normcase(os.path.abspath(root)) != map_bsp_real:
                index_bsp(root, os.path.basename(root))
            continue
        if not os.path.isdir(root):
            print(f"  ! skipping missing source folder: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                low = fn.lower()
                if low.endswith('_dir.vpk'):
                    index_vpk(full)
                    continue
                if low.endswith('.vpk'):
                    continue  # numbered data archives (_000.vpk ...), read via their _dir
                if low.endswith('.bsp'):
                    if os.path.normcase(os.path.abspath(full)) != map_bsp_real:
                        index.skipped_bsps.append(fn)
                    continue
                rel = os.path.relpath(full, root).replace('\\', '/').lower()
                index.add(rel, full, root)
    return index


def find_file(rel_path, index):
    return index.get(rel_path.replace('\\', '/').lower())


# What this run did to the output folder - reset in main(), reported at the
# end, and the manifest is what lets the NEXT run remove what this one no
# longer needs.
STATS = {'new': 0, 'updated': 0, 'unchanged': 0}
MANIFEST = set()
MANIFEST_NAME = "_extracted_files.txt"
MISSING_REPORT = "_missing_assets.txt"


# Most of what a Source map "references" has no part in a SOF2 conversion, and
# listing it all turns the report into hundreds of lines that look like faults.
# .ani/.vtx/.phy are the companions of a .mdl (animations, LOD vertex data,
# collision - the converter reads the .mdl and the .vvd); the per-pixel map
# types have no idTech3 stage. Those are counted, not listed; everything else
# is listed in full, because that IS worth a look.
HARMLESS_MISSING = (
    ('.ani', "animation data (not used: SOF2 props are static)"),
    ('.vtx', "LOD vertex data (not used: the converter reads the .vvd)"),
    ('.phy', "collision hulls (not used: brush clips are built from the map)"),
    ('.dx80.vtx', "LOD vertex data (not used)"),
    ('.sw.vtx', "LOD vertex data (not used)"),
)
HARMLESS_MISSING_PATTERNS = (
    ('_normal', 'normal map'), ('_normals', 'normal map'), ('-ssbump', 'ssbump'),
    ('_height', 'height map'), ('_exponent', 'specular exponent'),
    ('_ref', 'reflection mask'), ('_envmask', 'env mask'), ('_nomral', 'normal map (sic)'),
)


def _missing_bucket(rel):
    low = rel.lower()
    base = low.rsplit('.', 1)[0]
    for ext, why in HARMLESS_MISSING:
        if low.endswith(ext):
            return why
    for pat, why in HARMLESS_MISSING_PATTERNS:
        if base.endswith(pat):
            return why + " (no idTech3 equivalent)"
    return None


def format_missing_report(missing):
    """The report, with the kinds that never matter collapsed to a count."""
    buckets, real = {}, []
    for rel in sorted(missing):
        why = _missing_bucket(rel)
        if why:
            buckets.setdefault(why, []).append(rel)
        else:
            real.append(rel)
    out = ["Files referenced by the map that were not found in any source folder.",
           ""]
    if real:
        out.append("NOT FOUND - worth a look ({}):".format(len(real)))
        out.append("  These are files the conversion would have used. Usually they belong")
        out.append("  to a game you do not have installed (a CS:GO-only sound in a map")
        out.append("  converted with CS:S content, say), and the map works without them.")
        out.append("")
        out += ["  " + r for r in real]
        out.append("")
    else:
        out += ["Nothing missing that the conversion would have used.", ""]
    if buckets:
        total = sum(len(v) for v in buckets.values())
        out.append("Not needed by a SOF2 conversion ({} file(s), listed at the end):".format(total))
        for why in sorted(buckets):
            out.append("  {:>5}  {}".format(len(buckets[why]), why))
        out.append("")
        out.append("-" * 70)
        out.append("The full list of those, for reference:")
        for why in sorted(buckets):
            out.append("")
            out.append("# " + why)
            out += ["  " + r for r in buckets[why]]
    return "\n".join(out) + "\n"


def _same_bytes(path, data):
    try:
        if os.path.getsize(path) != len(data):
            return False
        with open(path, 'rb') as f:
            return f.read() == data
    except OSError:
        return False


def copy_asset(rel_path, index, output_root, done, missing, src_override=None):
    """Extracts the file to output_root (from a loose source file, a VPK
    archive, or a BSP's pakfile lump, whichever it came from) and returns
    the DESTINATION path - not the source - since every caller immediately
    opens and reads whatever this returns, and the destination is always a
    real, local, readable file regardless of which kind of source it came
    from, while an archive source reference on its own is not directly
    open()-able.

    An existing destination is only kept when it still matches its source.
    This used to skip any file that already existed, so re-running against
    updated content - or after fixing which source wins - silently kept the
    old copy, and every later step converted the stale file."""
    key = rel_path.replace('\\', '/').lower()
    dst = os.path.join(output_root, rel_path.replace('/', os.sep))
    if key in done:
        return dst if os.path.isfile(dst) else None
    done.add(key)
    src = src_override or find_file(rel_path, index)
    if not src:
        missing.add(rel_path.replace('\\', '/'))
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    existed = os.path.isfile(dst)
    if isinstance(src, tuple) and src[0] in ('vpk', 'bsp'):
        data = read_source(src)
        if data is None:
            missing.add(rel_path.replace('\\', '/'))
            return None
        if existed and _same_bytes(dst, data):
            STATS['unchanged'] += 1
        else:
            with open(dst, 'wb') as f:
                f.write(data)
            STATS['updated' if existed else 'new'] += 1
    else:
        same = False
        if existed:
            try:
                a, b = os.stat(src), os.stat(dst)
                # copy2 preserves the modification time, so an untouched
                # source has the same size AND the same mtime as its copy.
                same = a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)
            except OSError:
                same = False
        if same:
            STATS['unchanged'] += 1
        else:
            shutil.copy2(src, dst)
            STATS['updated' if existed else 'new'] += 1
    MANIFEST.add(os.path.relpath(dst, output_root).replace(os.sep, '/'))
    return dst


def report_sources(index, done, limit=12):
    """Where the files this map uses came from, and which of them existed in
    more than one source with DIFFERENT content. Identical duplicates are
    everywhere (hl2 and cstrike VPKs overlap) and are not worth a line; a
    differing one means the precedence rule decided what the map looks like."""
    used = Counter(index.origin.get(k, '?') for k in done if k in index)
    if used:
        print("Files used, by source:")
        for label, n in used.most_common():
            print(f"  {n:5d}  {label}")
    differing = []
    for k in sorted(done):
        losers = index.shadowed.get(k)
        if not losers or k not in index:
            continue
        won = read_source(index[k])
        if won is None:
            continue
        for label, src in losers:
            other = read_source(src)
            if other is not None and other != won:
                differing.append((k, index.origin.get(k, '?'), label))
    if differing:
        print(f"\n{len(differing)} file(s) this map uses exist in more than one source with "
              f"DIFFERENT content. The first source listed was used:")
        for k, won, lost in differing[:limit]:
            print(f"  {k}\n      used {won}, ignored {lost}")
        if len(differing) > limit:
            print(f"  ... and {len(differing) - limit} more")
    if index.skipped_bsps:
        names = sorted(set(index.skipped_bsps))
        print(f"\nSkipped {len(names)} other map(s)' .bsp file(s) under the content roots - "
              f"Source only mounts the pakfile of the map being played: "
              f"{', '.join(names[:6])}{', ...' if len(names) > 6 else ''}")


def remove_obsolete(output_root):
    """Delete what the PREVIOUS run extracted that this run did not.

    Only files named in the previous run's own manifest are candidates, so
    nothing the user or another tool put in the folder is ever touched. Then
    write this run's manifest for the next one."""
    old_path = os.path.join(output_root, MANIFEST_NAME)
    removed = 0
    if os.path.isfile(old_path):
        with open(old_path, encoding='utf-8') as f:
            old = set(l.strip() for l in f if l.strip())
        for rel in sorted(old - MANIFEST):
            full = os.path.join(output_root, rel.replace('/', os.sep))
            if os.path.isfile(full):
                try:
                    os.remove(full)
                    removed += 1
                except OSError:
                    pass
        for top in ('materials', 'models', 'sound'):
            base = os.path.join(output_root, top)
            for dirpath, _d, _f in sorted(os.walk(base), key=lambda t: -len(t[0])):
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
    with open(old_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(MANIFEST)) + '\n')
    return removed


def required_missing(materials, models, missing):
    """(missing, total) for the models the .vmf places and the materials it
    names directly - the files a map cannot do without. Optional companions
    (.sw.vtx, .dx80.vtx, .ani, .phy) are absent for many perfectly good
    models and say nothing about the setup, so they are not counted."""
    miss = set(m.replace('\\', '/').lower() for m in missing)

    def mdl(v):
        v = v.strip().replace('\\', '/')
        if not v.lower().startswith('models/'):
            v = 'models/' + v
        return re.sub(r'\.mdl$', '', v, flags=re.IGNORECASE).lower() + '.mdl'
    m_paths = set(mdl(v) for v in models)
    t_paths = set(norm_material_rel(v).lower() for v in materials)
    return (len(m_paths & miss), len(m_paths)), (len(t_paths & miss), len(t_paths))


# More than this share of the map's own models or materials missing is a
# content-root problem, not a conversion result: de_overpass_csgo pointed at
# cstrike/ alone (before the gameinfo.txt mounting) lost 96 of its 315 models
# and every converted step after it ran on regardless. A healthy run of a
# ported map is a few percent at most.
REQUIRED_MISSING_LIMIT = 0.20


def parse_vmt_dependencies(vmt_path):
    """Return list of ('material'|'texture', value) referenced by a .vmt file."""
    deps = []
    try:
        with open(vmt_path, 'r', errors='ignore') as f:
            text = f.read()
    except OSError:
        return deps

    for m in re.finditer(r'\binclude\b\s*"([^"]+)"', text, re.IGNORECASE):
        deps.append(('material', m.group(1)))

    # Key and value as Source's own KeyValues reader accepts them, which is
    # looser than it looks. Both of these are in de_mirage_cs2's real .vmts
    # and Source loads them fine:
    #   "$basetexture""models\props\de_mirage\couch_a\couch_a_diffuse"
    #       - no space between the two quoted strings (couch_a_diffuse.vmt,
    #         and $basetexture2 in de_mirage_ground_tileh_blend_diffuse .vmt)
    #   $basetexture concrete/blacktop_ext_01c
    #       - value not quoted at all
    # The old pattern required whitespace AND a quoted value, so these
    # textures were never extracted: the couch, and the second blend layer
    # of the mirage ground (the black/white terrain in game).
    for m in re.finditer(r'"?(\$[a-zA-Z0-9_]+)"?[ \t]*(?:"([^"\n]*)"|([^\s"{}]+))', text):
        _, qval, bval = m.groups()
        val = (qval if qval is not None else bval).strip().replace('\\', '/')
        if '/' in val and not val.startswith('['):
            val = re.sub(r'\.vtf$', '', val, flags=re.IGNORECASE)
            deps.append(('texture', val))
    return deps


def join_material_path(cdmat, tex):
    """cdmaterials dir + texture name, as one forward-slash path. Either part
    can be empty or carry its own slashes: a model with cdmaterials "" and
    texture "models/props/de_dust/sign_street01" (de_mirage_cs2's signs and
    rugs) used to come out as "/models/...", which matched no .vmt and no
    shader."""
    parts = [p for p in re.split(r'[\\/]+', (cdmat or '') + '/' + (tex or '')) if p]
    return '/'.join(parts)


def norm_material_rel(mat_path):
    v = mat_path.strip().replace('\\', '/')
    if v.lower().startswith('materials/'):
        v = v[len('materials/'):]
    if v.lower().endswith('.vmt'):
        v = v[:-4]
    return f"materials/{v}.vmt"


def norm_texture_rel(tex_path):
    v = tex_path.strip().replace('\\', '/')
    if v.lower().startswith('materials/'):
        v = v[len('materials/'):]
    if v.lower().endswith('.vtf'):
        v = v[:-4]
    return f"materials/{v}.vtf"


def cstr(data, off):
    end = data.index(b'\x00', off)
    return data[off:end].decode('latin1')


def parse_mdl_materials(mdl_path):
    """Read a Source studiomdl (v44-49) binary header and return
    (cdmaterials_dirs, texture_names). Returns ([], []) if the file
    doesn't look like a supported studiomdl (e.g. GoldSrc, or truncated)."""
    try:
        with open(mdl_path, 'rb') as f:
            data = f.read()
    except OSError:
        return [], []

    if len(data) < 240 or data[0:4] != b'IDST':
        return [], []

    try:
        numtextures, textureindex = struct.unpack_from('<ii', data, 204)
        numcdtextures, cdtextureindex = struct.unpack_from('<ii', data, 212)

        cdmaterials = []
        for i in range(numcdtextures):
            off = struct.unpack_from('<i', data, cdtextureindex + i * 4)[0]
            cdmaterials.append(cstr(data, off).rstrip('\\/').replace('\\', '/'))

        texnames = []
        for i in range(numtextures):
            tex_struct_off = textureindex + i * 64
            sznameindex = struct.unpack_from('<i', data, tex_struct_off)[0]
            texnames.append(cstr(data, tex_struct_off + sznameindex))

        return cdmaterials, texnames
    except (struct.error, ValueError, IndexError):
        return [], []


def process_model_materials(mdl_src_path, index, output_root, done, missing, queue):
    """Resolve every texture a model references (via its own cdmaterials +
    texture table) to a materials/*.vmt path, trying each cdmaterials
    directory in order (matching how the engine itself searches them),
    and queue whichever ones actually exist for full .vmt dependency
    resolution alongside world/brush materials."""
    cdmaterials, texnames = parse_mdl_materials(mdl_src_path)
    for tex in texnames:
        found = False
        for cdmat in (cdmaterials or ['']):
            rel = join_material_path(cdmat, tex)
            if find_file("materials/" + rel + ".vmt", index):
                queue.append(('material', rel))
                found = True
                break
        if not found:
            # Record what we tried so it shows up in _missing_assets.txt
            for cdmat in (cdmaterials or ['']):
                missing.add("materials/" + join_material_path(cdmat, tex) + ".vmt")


def process_material(mat_path, index, output_root, done, missing, queue):
    rel = norm_material_rel(mat_path)
    src = copy_asset(rel, index, output_root, done, missing)
    if src:
        for kind, val in parse_vmt_dependencies(src):
            queue.append((kind, val))


def process_texture(tex_path, index, output_root, done, missing):
    copy_asset(norm_texture_rel(tex_path), index, output_root, done, missing)


# Where each compiled-model file keeps the checksum studiomdl stamps into every
# file of one build (verified on real CS:S and CS:GO files: .mdl and .vvd at
# byte 8, every .vtx variant at byte 16). .phy is left out: its checksum
# legitimately differs even inside one consistent set.
CHECKSUM_OFFSET = {'.mdl': 8, '.vvd': 8, '.dx90.vtx': 16, '.dx80.vtx': 16, '.sw.vtx': 16,
                   '.vtx': 16}


def _checksum(data, ext):
    off = CHECKSUM_OFFSET.get(ext)
    if off is None or data is None or len(data) < off + 4:
        return None
    return struct.unpack_from('<i', data, off)[0]


def pick_companion(base, ext, index, want, mdl_label):
    """The source for <base><ext> that belongs to the SAME build as the .mdl.

    A model is a set - .mdl, .vvd, .vtx - and the three must come from one
    compile. With the map's own .bsp read first, a CS:GO port can supply the
    .mdl and .vvd while the .dx90.vtx comes from the game's VPK: CS:GO packs
    its strip data as a plain .vtx, which no CS:S lookup asks for. On
    de_overpass_csgo that broke props_junk/plasticcrate01a ("checksum
    mismatch between .mdl/.vvd/.vtx") and the prop vanished from the map.

    Every source holding the path is a candidate - the one that won the
    index and any it shadowed - and for .dx90.vtx the plain .vtx too. The
    first whose checksum matches the .mdl's is used; for files without a
    checksum (.phy, .ani), the .mdl's own source is preferred. None means:
    let the ordinary lookup decide (and report a mismatch downstream)."""
    rels = [base + ext]
    if ext == '.dx90.vtx':
        rels.append(base + '.vtx')
    cands = []
    for r in rels:
        k = r.replace('\\', '/').lower()
        if k in index:
            cands.append((index.origin.get(k), index[k]))
            cands.extend(index.shadowed.get(k, []) if isinstance(index, SourceIndex) else [])
    if not cands:
        return None
    if want is not None and ext in CHECKSUM_OFFSET:
        for label, src in cands:
            if _checksum(read_source(src), ext) == want:
                return src
        return None
    for label, src in cands:
        if label == mdl_label:
            return src
    return None


def process_model(model_path, index, output_root, done, missing, queue):
    v = model_path.strip().replace('\\', '/')
    if not v.lower().startswith('models/'):
        v = 'models/' + v
    base = re.sub(r'\.mdl$', '', v, flags=re.IGNORECASE)

    mdl_src = copy_asset(base + '.mdl', index, output_root, done, missing)
    want, mdl_label = None, None
    if mdl_src:
        try:
            with open(mdl_src, 'rb') as f:
                want = _checksum(f.read(12), '.mdl')
        except OSError:
            pass
        mdl_label = getattr(index, 'origin', {}).get((base + '.mdl').replace('\\', '/').lower())
    for ext in [e for e in MODEL_SINGLE_EXTS if e != '.mdl'] + MODEL_VTX_EXTS:
        src = pick_companion(base, ext, index, want, mdl_label) if mdl_src else None
        copy_asset(base + ext, index, output_root, done, missing, src_override=src)

    if mdl_src:
        process_model_materials(mdl_src, index, output_root, done, missing, queue)


def process_sound(snd_path, index, output_root, done, missing):
    v = snd_path.strip().replace('\\', '/').lstrip('/')
    if not v.lower().startswith('sound/'):
        v = 'sound/' + v
    copy_asset(v, index, output_root, done, missing)


def process_raw(raw_path, index, output_root, done, missing):
    """Copy something whose relative path (relative to a content root) is already
    fully known, e.g. sprites/xyz.spr referenced directly by a 'model' key."""
    copy_asset(raw_path.strip().replace('\\', '/'), index, output_root, done, missing)


def parse_vmf(vmf_path):
    with open(vmf_path, 'r', errors='ignore') as f:
        text = f.read()

    pairs = re.findall(r'"([^"]+)"\s+"([^"]*)"', text)

    materials, models, sounds, skynames, raw_files = set(), set(), set(), set(), set()

    for key, val in pairs:
        k = key.lower()
        v = val.strip()
        if not v:
            continue

        # A bare number is never a real material path (every real Source
        # material reference is a folder/name path) - confirmed against
        # real content: func_breakable's own "material" key is a NUMERIC
        # enum (0=Glass, 1=Wood, 2=Metal, etc, Hammer's own FGD-defined
        # Material Type), unrelated to any actual .vmt file, and the naive
        # "any key literally named material" check was catching it as a
        # false "missing asset".
        if k == 'material' and not v.isdigit():
            materials.add(v)
        elif k == 'detailmaterial' and not v.isdigit():
            materials.add(v)
        elif k == 'texture' and not v.isdigit() and '/' in v:
            # Source's 'infodecal' entity names its decal in a 'texture' key,
            # NOT 'material' - so scanning only for 'material' silently misses
            # every one of them. Measured on de_dust2: 135 infodecals using 21
            # distinct textures, including 33 'decals/win_square' and 16
            # 'decals/win_rectang', none of which were being extracted - which
            # is why those decals had no .vmt/.vtf on the SOF2 side at all and
            # could never show up, independently of whether the .map placed
            # them. Requiring a '/' keeps this from catching unrelated keys
            # that happen to be called 'texture' with a non-path value.
            materials.add(v)
        elif k == 'skyname':
            skynames.add(v)
        elif k == 'model':
            lv = v.lower()
            if lv.endswith('.vmt'):
                materials.add(v)
            elif lv.endswith('.mdl') or '.' not in os.path.basename(lv):
                models.add(v)
            else:
                raw_files.add(v)  # e.g. .spr sprites, or already-full relative paths

        if v.lower().endswith(SOUND_EXTS):
            sounds.add(v)

    return materials, models, sounds, skynames, raw_files


def main():
    argv = list(sys.argv[1:])
    allow_missing = '--allow-missing' in argv
    argv = [a for a in argv if a != '--allow-missing']
    map_bsp_arg = None
    if '--map-bsp' in argv:
        i = argv.index('--map-bsp')
        map_bsp_arg = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if len(argv) < 3:
        print(__doc__)
        sys.exit(1)

    vmf_path, sources_arg, output_root = argv[0], argv[1], argv[2]
    STATS.update(new=0, updated=0, unchanged=0)
    MANIFEST.clear()
    # A report left by an earlier run must not outlive the problem it
    # described: delete it now, and only write a new one if this run has
    # something missing.
    stale_report = os.path.join(output_root, MISSING_REPORT)
    if os.path.isfile(stale_report):
        os.remove(stale_report)

    game_dir = find_game_dir(sources_arg, vmf_path) if os.path.isdir(sources_arg) else None
    if game_dir:
        # A real Source install: mount exactly what the game mounts.
        source_roots = game_mounts(game_dir)
    elif os.path.isdir(sources_arg):
        # A folder of extracted content trees (e.g. E:\\CSS\\Extracted holding
        # cstrike_pak_dir, hl2_misc_dir, ...): each subfolder is one tree.
        subdirs = [os.path.join(sources_arg, d) for d in sorted(os.listdir(sources_arg))
                   if os.path.isdir(os.path.join(sources_arg, d))]
        source_roots = subdirs if subdirs else [sources_arg]
    else:
        source_roots = [s for s in sources_arg.split(';') if s.strip()]

    # The map's own .bsp carries whatever custom content the map packs into
    # its pakfile lump - for a stock map that is little more than cubemaps,
    # for a ported or community map it is most of its unique materials and
    # models. Found next to the .vmf or anywhere under the content roots, and
    # indexed FIRST (see build_index for why).
    map_bsp = find_map_bsp(vmf_path, [r[1] if isinstance(r, tuple) else r for r in source_roots],
                           map_bsp_arg)
    if map_bsp:
        print(f"Map's own .bsp: {map_bsp}\n  (its packed content takes priority over the game's, "
              f"as it does in Source)")
    elif map_bsp_arg:
        print(f"WARNING: --map-bsp {map_bsp_arg} is missing or not a Source .bsp - "
              f"continuing without it.")
    else:
        print("No .bsp for this map found next to the .vmf or under the content roots. Fine "
              "for a stock map; a ported or community map keeps its custom content inside "
              "its .bsp, and that content will be reported missing below.")

    if game_dir:
        print(f"Source game folder: {game_dir}\n  mounted the way its gameinfo.txt does, in "
              f"this order (after the map's own .bsp):")
    else:
        print("Source folders:")
    for s in source_roots:
        print(f"  - {s[1]}  ({'VPK' if s[0] == 'vpk' else 'loose files'})"
              if isinstance(s, tuple) else f"  - {s}")

    print("\nIndexing source content (this can take a bit on large trees)...")
    index = build_index(source_roots, map_bsp)
    print(f"Indexed {len(index)} files.\n")

    print(f"Parsing VMF: {vmf_path}")
    materials, models, sounds, skynames, raw_files = parse_vmf(vmf_path)
    materials_referenced = set(materials)

    for sky in skynames:
        for suffix in ('bk', 'dn', 'ft', 'lf', 'rt', 'up'):
            materials.add(f"skybox/{sky}{suffix}")

    print(f"  materials referenced: {len(materials)}")
    print(f"  models referenced:    {len(models)}")
    print(f"  sounds referenced:    {len(sounds)}")
    print(f"  skynames referenced:  {len(skynames)}")
    print(f"  raw/other refs:       {len(raw_files)}\n")

    done = set()
    missing = set()
    # Materials and models share one queue now, since resolving a model's
    # own textures (via its .mdl texture table) discovers more materials
    # mid-run, same as a .vmt's $basetexture discovering more textures.
    queue = [('material', m) for m in materials] + [('model', m) for m in models]

    while queue:
        kind, val = queue.pop()
        if kind == 'material':
            process_material(val, index, output_root, done, missing, queue)
        elif kind == 'texture':
            process_texture(val, index, output_root, done, missing)
        elif kind == 'model':
            process_model(val, index, output_root, done, missing, queue)

    # Soundscapes: the env_soundscape entities name soundscapes defined in
    # scripts/soundscapes*.txt (the map's .bsp usually packs its own). Every
    # such script is extracted - vmf2map.py reads them to place the sounds -
    # and every wave the map's soundscapes play is pulled like any sound.
    soundscape_names = set(re.findall(r'"soundscape"\s+"([^"]+)"',
                                      open(vmf_path, 'r', errors='ignore').read(), flags=re.I))
    if soundscape_names:
        scripts = sorted(k for k in index if k.startswith('scripts/soundscapes') and k.endswith('.txt'))
        script_files = [p for p in (copy_asset(k, index, output_root, done, missing) for k in scripts) if p]
        if sof2_sounds is None:
            print("NOTE: sof2_sounds.py not found next to this script - soundscape sounds skipped.")
        else:
            defs = sof2_sounds.load_soundscapes(script_files)
            waves = sof2_sounds.soundscape_waves(soundscape_names, defs)
            unknown = sorted(n for n in soundscape_names if n.lower() not in defs)
            print(f"  soundscapes: {len(soundscape_names)} used, from {len(script_files)} script(s); "
                  f"{len(waves)} sound file(s)" +
                  (f"; not defined in any script: {', '.join(unknown)}" if unknown else ""))
            sounds |= {w[len('sound/'):] if w.lower().startswith('sound/') else w for w in waves}

    for snd in sounds:
        process_sound(snd, index, output_root, done, missing)

    for raw in raw_files:
        process_raw(raw, index, output_root, done, missing)

    os.makedirs(output_root, exist_ok=True)
    if missing:
        with open(os.path.join(output_root, MISSING_REPORT), "w") as f:
            f.write(format_missing_report(missing))

    print()
    report_sources(index, done - missing)
    removed = remove_obsolete(output_root)

    found = len(done) - len(missing)
    print(f"\nDone. {found} files in {output_root}: {STATS['new']} new, "
          f"{STATS['updated']} updated because their source changed, "
          f"{STATS['unchanged']} unchanged.")
    if removed:
        print(f"Removed {removed} file(s) an earlier run extracted that this map no longer uses.")
    if missing:
        real = [r for r in missing if _missing_bucket(r) is None]
        if real:
            print(f"{len(real)} of {len(missing)} referenced files were NOT found and the "
                  f"conversion would have used them; the other {len(missing) - len(real)} "
                  f"are Source files a SOF2 conversion has no use for.")
        else:
            print(f"{len(missing)} referenced files were not found, none of which a SOF2 "
                  f"conversion needs (model animation/LOD/collision companions and "
                  f"per-pixel map types).")
        print(f"See: {os.path.join(output_root, MISSING_REPORT)}")
    (mm, mt), (tm, tt) = required_missing(materials_referenced, models, missing)
    print(f"Of what the .vmf itself uses: {mm} of {mt} model(s) and {tm} of {tt} material(s) "
          f"not found.")
    if not allow_missing and ((mt and mm > mt * REQUIRED_MISSING_LIMIT) or
                              (tt and tm > tt * REQUIRED_MISSING_LIMIT)):
        print(f"\nERROR: more than {REQUIRED_MISSING_LIMIT:.0%} of the map's own models or materials "
              f"are missing. That is almost always the content root, not the map: point it at "
              f"your game install (e.g. ...\\counter-strike source, or its cstrike folder) so the "
              f"game's VPKs are read. Every later step would convert an incomplete map, so the "
              f"run stops here. (Run this script by hand with --allow-missing to carry on "
              f"anyway.)")
        sys.exit(1)


if __name__ == '__main__':
    main()
