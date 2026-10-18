"""
install_to_sof2.py - Put a converted map into SOF2's base folder so it is ready
to compile and test. What lands there, as LOOSE FILES:

    <base>/maps/<map>.map
    <base>/shaders/<map>.shader
    <base>/textures/...      every converted texture
    <base>/models/...        every .md3, the welded terrain .lwo(s)
    <base>/*.pcx             any terrain alphamap

The shader name is also appended to shaders/shaderlist.txt, because that file
is q3map2's own list of which shaders to load (the engine loads every
shaders/*.shader it can find, archives included, and ignores the list).

Loose rather than one base/<map>.pk3, which an earlier version of this step
produced. The archive was tidier but wrong to regenerate on every Convert:

  - A Quake3-family filesystem mounts an archive whole, so re-converting
    replaces the single file holding ALL of this map's assets in one go.
  - The paths inside a converted CS map are not private to it. CS:S content is
    shared, so props_urban and cs_office textures sit at identical paths in
    several maps, and this map's archive shadows the copies another converted
    map is reading.

Loose files avoid both: a re-convert overwrites exactly the files it
regenerates and leaves every other map's assets alone, and q3map2's VFS plus
the engine both read real directories BEFORE mounted .pk3s, so what is on disk
is what compiles and what runs.

One shareable archive is still produced, just not in base: the compile .bat
packs the finished .bsp together with the assets into
<output>/<map>_release.pk3, which is the single file to hand to someone else.
The .bsp belongs in an archive only there - inside base/<map>.pk3 it would
shadow the one you just recompiled into base/maps/.

If a base/<map>.pk3 from the older scheme is still present, it is MOVED into
<output>/_moved_out_of_base/ rather than deleted: loose files already win over
it for this map, but its paths can still feed stale assets to another map, and
it is not this script's to destroy.

Works on the pipeline output folder either before or after clean_output.py
has tidied it (files at the top level, or in maps/, shaders/, models/).

USAGE
-----
    python install_to_sof2.py <output_dir> <map_name> <sof2_base_dir>

e.g.
    python install_to_sof2.py "E:\\CSS\\SOF2\\de_dust2" de_dust2 ^
        "E:\\Soldier of Fortune II - Double Helix\\Soldier of Fortune II - MODDING\\base"
"""
import argparse
import glob
import importlib.util
import os
import re
import shutil
import struct
import sys


def helper_dirs():
    """Where the helper scripts this one hands to the compile .bat live, in
    order: next to this file (which is also where the GUI's .exe unpacks them,
    and where someone can drop a newer copy), then PyInstaller's bundle dir,
    then next to the .exe itself."""
    dirs = [os.path.dirname(os.path.abspath(__file__))]
    for extra in (getattr(sys, '_MEIPASS', None),
                  os.path.dirname(os.path.abspath(sys.executable))
                  if getattr(sys, 'frozen', False) else None):
        if extra and extra not in dirs:
            dirs.append(extra)
    return dirs


def helper_path(filename):
    for d in helper_dirs():
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


def _load_helper(module_name):
    """A helper script (make_pk3.py, audit_release.py), however this script is
    being run. A plain import works on the command line and inside a
    PyInstaller bundle, but the GUI loads its steps straight from a file path,
    so fall back to doing the same here."""
    try:
        return __import__(module_name)
    except ImportError:
        pass
    path = helper_path(module_name + '.py')
    if path is None:
        raise FileNotFoundError(
            "{}.py not found in: {} - step 10 needs it next to the other scripts.".format(
                module_name, "; ".join(helper_dirs())))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_make_pk3():
    return _load_helper('make_pk3')


make_pk3 = None  # set in main(), so importing this module never fails

# Must match vmf2map.py's --terrain-lwo-model-path default, which is what the
# .map's misc_model "model" key points at.
TERRAIN_MODEL_DIR = os.path.join('models', 'mapobjects', 'terrain')

# Model-side files that are pipeline working data, not game assets.
SKIP_MODEL_EXTS = {'.json', '.txt', '.mdl', '.vvd', '.vtx', '.phy', '.ani', '.bak'}


def first_existing(*paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def copy_file(src, dest_dir, base):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    existed = os.path.isfile(dest)
    shutil.copy2(src, dest)
    print("  {} {}".format("replaced" if existed else "added   ", os.path.relpath(dest, base)))
    return dest


def ensure_in_shaderlist(shaders_dir, shader_name):
    """q3map2 loads only the .shader files named in shaders/shaderlist.txt.

    When that file does NOT exist, this deliberately does nothing. q3map2 then
    falls back to loading every shader it can find ("No shaderlist.txt found:
    loading all shaders"), which is exactly right - whereas creating one
    holding just this map would be far worse than its absence: the map's own
    materials would become the ONLY shaders loaded, with no common.shader and
    no tools.shader, so no caulk, nodraw, clip or hint, and the compile is
    ruined. The game never reads this file; it loads every shaders/*.shader
    it can find, inside .pk3s included."""
    path = os.path.join(shaders_dir, 'shaderlist.txt')
    if not os.path.isfile(path):
        print("  (no shaders/shaderlist.txt in base - leaving it that way, so q3map2 "
              "loads every shader it finds, this map's included)")
        return
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    if any(l.strip().lower() == shader_name.lower() for l in text.splitlines()):
        print("  shaderlist.txt already lists '{}'".format(shader_name))
        return
    with open(path, 'a', encoding='utf-8') as f:
        if text and not text.endswith(('\n', '\r')):
            f.write('\n')
        f.write(shader_name + '\n')
    print("  added '{}' to shaders/shaderlist.txt".format(shader_name))


def _light_flags_from_shader(shader_file):
    """The '-gamma G -gridscale S -bounce 1' vmt2shader.py solved for this map,
    read back from the comment it writes in the sky shader."""
    import re
    try:
        text = open(shader_file, encoding='utf-8', errors='ignore').read()
    except (OSError, TypeError):
        return '-gamma 2.2 -bounce 1'
    m = re.search(r'COMPILE:.*?-light \.\.\.\s*([^\n]*)', text)
    return m.group(1).strip() if m else '-gamma 2.2 -bounce 1'


def q3map2_strings(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except OSError:
        return None


def q3map2_supports_gridscale(path):
    """True when the q3map2 build knows -gridscale (netradiant / netradiant-
    custom do, GtkRadiant's older one does not): the option's name is in the
    executable's own strings."""
    data = q3map2_strings(path)
    return None if data is None else b'-gridscale' in data


def q3map2_bitness(path):
    """32 or 64 from the executable's PE header, or None.

    It matters: -light asks for ONE block for the trace nodes (measured on
    de_overpass_csgo: 224,919,552 bytes = 214.5 MiB), and a 32-bit q3map2
    often cannot get a block that size once the .bsp is loaded - it stops with
    "safe_malloc failed on allocation of N bytes" in SetupTraceNodes. The
    machine has nothing to do with it; the address space does.
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(0x400)
    except OSError:
        return None
    if len(head) < 0x40 or head[:2] != b'MZ':
        return None
    off = struct.unpack_from('<I', head, 0x3C)[0]
    if off + 6 > len(head) or head[off:off + 4] != b'PE\0\0':
        return None
    machine = struct.unpack_from('<H', head, off + 4)[0]
    return {0x014c: 32, 0x8664: 64, 0xaa64: 64, 0x0200: 64}.get(machine)


def q3map2_build(path):
    """(name, growable) for a q3map2 executable.

    growable True  - the NetRadiant fork: the .bsp draw-vertex and draw-index
                     lumps grow on demand, so a big converted map compiles.
             False - stock ydnar 2.5.17 lumps (GtkRadiant 1.6.7 ships one):
                     MAX_MAP_DRAW_VERTS / MAX_MAP_DRAW_INDEXES are fixed at
                     0x80000, and a big map stops with "MAX_MAP_DRAW_INDEXES"
                     in FilterDrawsurfsIntoTree (and earlier, at 8192 shaders,
                     with MAX_SHADER_INFO).
             None  - could not tell; say nothing.
    -gridscale and the "NetRadiant" banner both came from that fork, so either
    one identifies it.
    """
    data = q3map2_strings(path)
    if data is None:
        return None, None
    net = b'NetRadiant' in data or b'netradiant' in data or b'-gridscale' in data
    gtk = b'GtkRadiant' in data
    if net:
        return 'NetRadiant', True
    if gtk:
        return 'GtkRadiant', False
    return None, None


FIXED_LUMP_NOTE = (
    "that build keeps the .bsp draw-vertex and draw-index lumps in fixed arrays "
    "(0x80000 = 524,288 each), so a large converted map stops with "
    "\"MAX_MAP_DRAW_INDEXES\" part way through -bsp. NetRadiant's q3map2 grows them "
    "as needed and compiles the same map unchanged. Use it for the compile; "
    "GtkRadiant is fine for editing."
)
BIT32_NOTE = (
    "it is a 32-bit program. -light asks for one block of a couple of hundred MB for "
    "the trace nodes on a map this size (measured: 214.5 MiB), which a 32-bit process "
    "often cannot give it - the compile then stops with \"safe_malloc failed on "
    "allocation\" in SetupTraceNodes. A 64-bit q3map2 has no such trouble."
)


def write_compile_bat(base, name, shader_file, out_dir=None, q3map2_path=None, runner=None,
                      tool_version=None):
    """maps/compile_<name>.bat: the full compile, with bsp_fix_leafbounds.py
    between -bsp and -vis. q3map2 writes leaf boxes as truncated integers and
    -light then drops walls lying on fractional leaf boundaries out of the leaf
    in front of them (black bands along wall bottoms) - the script widens the
    boxes by the one unit the truncation can lose. -vis keeps them."""
    maps_dir = os.path.join(base, 'maps')
    for helper in ('bsp_fix_leafbounds.py', 'bsp_decal_relight.py', 'bsp_shader_count.py',
                   'make_pk3.py', 'shaderlist_for_compile.py'):
        src = helper_path(helper)
        if src:
            copy_file(src, maps_dir, base)
        else:
            print("  WARNING: {} not found in {} - the compile .bat calls it".format(
                helper, "; ".join(helper_dirs())), file=sys.stderr)
    game_root = os.path.dirname(os.path.normpath(base))
    q3map2 = q3map2_path if q3map2_path and os.path.isfile(q3map2_path) else None
    if q3map2_path and not q3map2:
        print("  WARNING: q3map2 not found at {} - looking in the usual places".format(
            q3map2_path), file=sys.stderr)
    for cand in (() if q3map2 else (
            os.path.join(os.path.dirname(game_root), 'Netradiant', 'q3map2.exe'),
            os.path.join(game_root, 'Netradiant', 'q3map2.exe'),
            os.path.join(os.path.dirname(game_root), 'netradiant-custom', 'q3map2.exe'))):
        if os.path.isfile(cand):
            q3map2 = cand
            break
    light = _light_flags_from_shader(shader_file)
    if q3map2 and q3map2_supports_gridscale(q3map2) is False:
        light = re.sub(r'\s*-gridscale\s+\S+', '', light)
        print("  NOTE: {} has no -gridscale, so it is left off the -light line - players and "
              "networked props will read a bit darker than the brushwork.".format(q3map2),
              file=sys.stderr)
    build_name, growable = q3map2_build(q3map2) if q3map2 else (None, None)
    fixed_lumps = growable is False
    bits = q3map2_bitness(q3map2) if q3map2 else None
    if fixed_lumps:
        print("  WARNING: {} looks like {}'s q3map2 - {}".format(
            q3map2, build_name or 'GtkRadiant', FIXED_LUMP_NOTE), file=sys.stderr)
    if bits == 32:
        print("  WARNING: {} - {}".format(q3map2, BIT32_NOTE), file=sys.stderr)
    # The four helper scripts run through SourceToSoF2.exe itself when it wrote
    # this .bat (--run): the .exe carries its own Python, so whoever compiles
    # needs nothing installed. A hand run of this script keeps calling python.
    if runner:
        def run(script):
            return '"%S2S%" --run {}'.format(script)
    else:
        def run(script):
            return 'python "%~dp0{}"'.format(script)
    head = [
        '@echo off',
        'rem Written by {}. Edit Q3MAP2 if it is not found.'.format(
            tool_version or 'install_to_sof2.py'),
        'set Q3MAP2={}'.format(q3map2 or 'q3map2.exe'),
    ] + (['set S2S={}'.format(runner),
          'if not exist "%S2S%" goto nos2s'] if runner else []) + [
        'set GAMEROOT={}'.format(game_root),
        'set BASEDIR={}'.format(os.path.normpath(base)),
        'set OUTDIR={}'.format(os.path.normpath(out_dir or os.path.join(base, 'maps'))),
        'set MAP=%~dp0{}'.format(name),
        'rem The leaf-box fix after -bsp is not optional: without it -light drops walls that',
        'rem lie on fractional leaf boundaries out of the leaf in front of them, leaving black',
        'rem bands along them.',
    ] + ([] if runner else [
        'python -c "pass" >nul 2>nul',
        'if errorlevel 1 goto nopython',
    ]) + [
        'rem -gridscale on the -light line is left off automatically for a q3map2 that does',
        'rem not have it; players and networked props then read a bit darker.',
    ] + ([
        'echo.',
        'echo NOTE: Q3MAP2 above is {}\'s build. It keeps the .bsp draw-vertex and'.format(
            build_name or 'GtkRadiant'),
        'echo draw-index lumps in fixed arrays (524,288 each), so a large converted map',
        'echo stops part way through -bsp with "MAX_MAP_DRAW_INDEXES". NetRadiant\'s q3map2',
        'echo grows them as needed and compiles the same map unchanged - set the q3map2.exe',
        'echo field to it and run step 10 again. GtkRadiant stays fine for editing.',
        'echo.',
    ] if fixed_lumps else []) + ([
        'echo.',
        'echo NOTE: Q3MAP2 above is a 32-bit program. -light asks for ONE block of a few',
        'echo hundred MB for the trace nodes on a map this size, which a 32-bit process',
        'echo often cannot give it - the compile then stops with "safe_malloc failed on',
        'echo allocation of N bytes" in SetupTraceNodes. Use a 64-bit q3map2',
        'echo (NetRadiant ships one) and run step 10 again.',
        'echo.',
    ] if bits == 32 else []) + [
        'rem -nohint: Source hint brushes are tuned for VBSP/VVIS. In q3map2 they multiply',
        'rem vis clusters, and the vis data is clusters x clusters / 8 bytes inside the .bsp',
        'rem (SOF2 loads the whole .bsp as one block). de_overpass_csgo: 7608 clusters with',
        'rem them, 4955 without - 7.2 MB of vis data down to 3.1 MB, a third fewer portals.',
        'rem de_mirage_cs2 overflowed MAX_MAP_VISIBILITY with them. Visuals are unaffected;',
        'rem only how finely the engine culls what is out of sight.',
        'rem q3map2 loads every shader file in shaders/shaderlist.txt, and every converted map',
        'rem adds one of 700-1500 shaders: a few of them and q3map2 stops with "MAX_SHADER_INFO',
        'rem exceeded". A map never needs another converted map\'s shaders, so for this compile',
        'rem the list leaves them out; the original list is put back at the end.',
        run('shaderlist_for_compile.py') + ' apply "%BASEDIR%\\shaders" {} || goto fail'.format(name),
        '"%Q3MAP2%" -game sof2 -fs_basepath "%GAMEROOT%" -meta -rename -nohint -v "%MAP%.map" || goto fail',
        run('bsp_fix_leafbounds.py') + ' "%MAP%.bsp" || goto fail',
    ]
    relight = [
        'rem Decals are vertex lit, and q3map2 lights each decal vertex with one unfiltered',
        'rem sample, so a decal could come out darker or lighter than the ground it is on,',
        'rem and a shadow edge crossing it turned into a row of triangles. This copies the',
        'rem lightmap under every decal vertex into its colour, so decal and ground match.',
        run('bsp_decal_relight.py') + ' "%MAP%.bsp" "%BASEDIR%\\shaders\\{}.shader" || goto fail'
        .format(name),
    ]
    restore = [run('shaderlist_for_compile.py') + ' restore "%BASEDIR%\\shaders"']
    tail = restore + relight + [
        "rem How close the finished map is to SOF2's 2048-shader table. Over it, surfaces",
        "rem draw with the default shader (GeneratePermanentShader - MAX_SHADERS hit). The fix",
        'rem it prints is a key in this .map - worldspawn\'s "_lightmapsamplesize" - so it can',
        'rem be raised in Radiant and recompiled without converting the map again.',
        run('bsp_shader_count.py') + ' "%MAP%.bsp"',
        'rem One .pk3 holding the assets AND this .bsp: the single file to give to someone',
        'rem else. It goes to the conversion output folder, not base - base is where the',
        'rem game mounts archives, and a .bsp inside a mounted one would shadow the loose',
        'rem .bsp above, so you would keep testing the previous build.',
        'rem --manifest names this map"s own files in base. base is shared loose storage',
        'rem now, with several converted maps side by side in it, so the file list is what',
        'rem keeps another map"s textures out of this archive.',
        (run('make_pk3.py') + ' release "%BASEDIR%" {} --out "%OUTDIR%\\{}_release.pk3" '
         '--manifest "%OUTDIR%\\{}_files.txt"').format(name, name, name),
        'echo.',
        'echo To play: set com_hunkMegs to 256 (in your config, or +set com_hunkMegs 256 on',
        'echo the SoF2MP.exe shortcut) and restart the game. SOF2 loads the whole .bsp as one',
        'echo block of memory; a very large hunk such as 1024 reserves half of the 2 GB the',
        'echo game can use, and the map then fails with "Z_Malloc(): Failed to malloc".',
        'echo Done.',
        'pause',
        'exit /b 0',
        ':fail',
    ] + restore + [
        'echo Compile FAILED.',
        'pause',
        'exit /b 1',
        ':nos2s',
        'echo %S2S% was not found. This .bat was written by that program, which runs',
        'echo the compile helpers; it has been moved or renamed since. Run step 10 in it',
        'echo again to write a new .bat, or edit the S2S line at the top of this file.',
        'pause',
        'exit /b 1',
        ':nopython',
        'echo Python was not found on PATH, and this compile needs it.',
        'echo Install Python from python.org, ticking Add python.exe to PATH in the',
        'echo installer, then run this file again.',
        'pause',
        'exit /b 1',
    ]
    rem_ss = ['rem -lightmapsearchblocksize 2: the engine makes one shader per material PER',
              'rem lightmap page, and by default q3map2 drops each lightmap into the FIRST page',
              'rem with room, scattering every material across dozens of pages. Searching only',
              'rem the newest pages keeps a material\'s lightmaps together: measured world',
              'rem shaders 574 -> 438 on de_dust2, 483 -> 404 on de_cache, 1031 -> 847 on',
              'rem de_overpass_csgo, for a few more pages (SOF2 holds 256; each is 48 KB).',
              'rem No -samplesize here: -light takes it from the .srf -bsp wrote, and the .map\'s',
              'rem own _lightmapsamplesize keys (this map\'s Source lightmapscale x map scale)',
              'rem are what put it there. Raise that key in Radiant to trade lighting detail for',
              'rem lightmap pages, and so for shaders, if the count below comes out over budget.']
    lines = head + [
        '"%Q3MAP2%" -game sof2 -fs_basepath "%GAMEROOT%" -vis -saveprt -v "%MAP%.bsp" || goto fail',
    ] + rem_ss + [
        '"%Q3MAP2%" -game sof2 -fs_basepath "%GAMEROOT%" -light {} -lightmapsearchblocksize 2 -filter -patchshadows -shade '
        '-samples 2 -threads %NUMBER_OF_PROCESSORS% -v "%MAP%.bsp" || goto fail'.format(light),
    ] + tail
    # Same build for testing: -vis -fast (no full portal flow - minutes instead of
    # hours, everything still drawn, just less culled) and no -bounce.
    quick_light = re.sub(r'\s*-bounce\s+\d+', '', light)
    quick = head + [
        '"%Q3MAP2%" -game sof2 -fs_basepath "%GAMEROOT%" -vis -fast -saveprt -v "%MAP%.bsp" || goto fail',
    ] + rem_ss + [
        '"%Q3MAP2%" -game sof2 -fs_basepath "%GAMEROOT%" -light {} -lightmapsearchblocksize 2 -filter -patchshadows -shade '
        '-samples 2 -threads %NUMBER_OF_PROCESSORS% -v "%MAP%.bsp" || goto fail'.format(quick_light),
    ] + tail
    quick_path = os.path.join(maps_dir, 'compile_{}_quick.bat'.format(name))
    with open(quick_path, 'w', newline='\r\n') as f:
        f.write('\n'.join(quick) + '\n')
    print("  wrote {} (test build: -vis -fast, no -bounce)".format(os.path.relpath(quick_path, base)))
    path = os.path.join(maps_dir, 'compile_{}.bat'.format(name))
    with open(path, 'w', newline='\r\n') as f:
        f.write('\n'.join(lines) + '\n')
    print("  wrote {}{}".format(os.path.relpath(path, base),
                                "" if q3map2 else "  (set Q3MAP2 in it - q3map2.exe not found)"))
    return path


def _stage(src, stage_dir, arc_dir, base_for_print=None):
    """Copy one file into the staging tree at arc_dir/<name>."""
    dest_dir = os.path.join(stage_dir, arc_dir) if arc_dir else stage_dir
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    shutil.copy2(src, dest)
    return dest


def _stage_tree(src_root, stage_dir, arc_dir, skip_exts=()):
    n = 0
    for dirpath, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        for fn in files:
            if os.path.splitext(fn)[1].lower() in skip_exts:
                continue
            dest_dir = os.path.normpath(os.path.join(stage_dir, arc_dir, rel))
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(dest_dir, fn))
            n += 1
    return n


def copy_tree_into(src_root, dest_root):
    """Copy the staged tree into base as loose files, preserving relative paths.

    Only files the conversion produced are touched; anything already in base
    under a path this map does not use is left exactly as it was. Replacements
    are counted rather than printed one per line - a converted CS map stages
    hundreds of textures, and the per-file list buried the real messages.

    Returns the list of archive-style relative paths copied, which main()
    writes out as this map's manifest.
    """
    added = replaced = 0
    copied = []
    for dirpath, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        dest_dir = os.path.normpath(os.path.join(dest_root, rel)) if rel != '.' else dest_root
        os.makedirs(dest_dir, exist_ok=True)
        for fn in files:
            dest = os.path.join(dest_dir, fn)
            if os.path.isfile(dest):
                replaced += 1
            else:
                added += 1
            shutil.copy2(os.path.join(dirpath, fn), dest)
            arc = fn if rel == '.' else (rel.replace(os.sep, '/') + '/' + fn)
            copied.append(arc)
    print("  {} added, {} replaced".format(added, replaced))
    return copied


def write_manifest(path, rel_paths):
    """Record which files in base belong to this map.

    Needed because base is now shared loose storage: several converted maps
    have their textures and models side by side there, so nothing in the
    directory tree says which ones are this map's. make_pk3.py release reads
    this to build <map>_release.pk3 - the single shareable file - out of base
    without dragging in another map's assets. It replaced reading
    base/<map>.pk3, which no longer exists.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# files this map installed into SOF2's base folder, one per\n"
                "# line, relative to base. Written by install_to_sof2.py and\n"
                "# read by make_pk3.py release. Safe to delete; re-run step 10.\n")
        for rel in sorted(rel_paths):
            f.write(rel + "\n")
    return path


def tidy_loose(base, pk3_path, quarantine_dir):
    """UNUSED since base went back to loose files - kept for reference only.

    Nothing calls this: there is no base/<map>.pk3 to supersede the loose files
    any more, so moving them aside would remove the installed map. Left in
    place because the reasoning below is the record of why a map's archive must
    never be treated as owning its paths.

    MOVE loose files in base/ that the .pk3 now supplies into quarantine_dir.

    Why move and not delete. The paths in a map's .pk3 are not that map's
    private property: CS:S content is shared, so textures/models/props/
    cs_office/..., building_template and props_urban sit at identical paths in
    several maps. Deleting every loose file that appears in THIS map's archive
    therefore pulled assets out from under another converted map that was still
    reading them loose - which is exactly what happened the first time this ran
    against a base that already held de_mirage_cs2's loose install.

    Why do anything at all. q3map2's VFS searches real directories BEFORE the
    .pk3s it mounted (vfsLoadFile walks g_strDirs first), so a loose file from
    an older conversion silently wins over the corrected one in the archive at
    COMPILE time - the opposite of the engine, which prefers the archive. A
    stale loose texture is a wrong image size, a missing alpha channel, or an
    old shader, and nothing in the log would say so.

    Moving satisfies both: the compile sees only the .pk3, and every file is
    still there to put back. Nothing is destroyed, and the map each file
    belonged to gets its own .pk3 the next time step 10 runs for it."""
    moved, failed = 0, 0
    for rel in make_pk3.archive_names(pk3_path):
        loose = os.path.join(base, *rel.split('/'))
        if not os.path.isfile(loose):
            continue
        dest = os.path.join(quarantine_dir, *rel.split('/'))
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(loose, dest)
            moved += 1
        except OSError as exc:
            failed += 1
            print("  could not move {}: {}".format(rel, exc), file=sys.stderr)
    if moved:
        print("  moved {} loose file(s) out of base and into\n    {}\n"
              "  They are the same assets this .pk3 now carries, and q3map2 reads loose\n"
              "  files BEFORE .pk3s, so leaving them would compile this map against the\n"
              "  older copies. Nothing is deleted: if another converted map was still\n"
              "  using them loose, either run step 10 for that map too (it gets its own\n"
              "  .pk3) or copy this folder back over base.".format(moved, quarantine_dir))
    elif failed:
        print("  {} loose file(s) could not be moved - see above.".format(failed))
    # Tidy up the empty folders that leaves behind, deepest first.
    for root in ('textures', 'models'):
        top = os.path.join(base, root)
        if not os.path.isdir(top):
            continue
        for dirpath, _d, _f in sorted(os.walk(top), key=lambda t: -len(t[0])):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="the pipeline's output folder")
    ap.add_argument("map_name", help="map name without extension, e.g. de_dust2")
    ap.add_argument("sof2_base", help="SOF2's base folder (the one containing maps/, shaders/, ...)")
    # Both accepted and ignored, so an older GUI, .bat or shell history keeps
    # working. --clean-loose moved loose files aside because the archive was
    # meant to supersede them; the loose files ARE the install now, so there is
    # nothing to move and doing it would delete the map.
    ap.add_argument("--clean-loose", "--tidy-loose", dest="tidy_loose", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--with-assets", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--q3map2", help="q3map2.exe to write into the compile .bat (default: "
                                      "looked for next to the SOF2 install)")
    ap.add_argument("--runner", help="SourceToSoF2.exe: the .bat runs the helper scripts "
                                      "through it (--run) instead of through python")
    ap.add_argument("--tool-version", help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true",
                    help="pack even when the release audit finds errors (names over the "
                         "engine's 63-character limit). Missing assets never stop packing; "
                         "they are reported.")
    args = ap.parse_args()

    # No make_pk3 import: this script no longer packs anything. The compile
    # .bat still runs make_pk3.py from disk to build <map>_release.pk3, and
    # helper_dirs() still copies that script into place for it.

    out, name, base = args.output_dir, args.map_name, args.sof2_base
    if not os.path.isdir(out):
        print("ERROR: output folder not found: {}".format(out), file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(base):
        print("ERROR: SOF2 base folder not found: {}".format(base), file=sys.stderr)
        sys.exit(1)
    looks_like_base = (any(os.path.isdir(os.path.join(base, d))
                           for d in ('maps', 'shaders', 'textures', 'models'))
                       or glob.glob(os.path.join(base, '*.pk3')))
    if not looks_like_base:
        print("WARNING: {} has no maps/ or shaders/ folder and no .pk3 files - double-check "
              "this really is SOF2's base folder.".format(base), file=sys.stderr)

    print("Installing '{}' into {}".format(name, base))
    missing = []

    # The .map is the one thing that stays loose: q3map2 compiles a file on
    # disk, and writes the .bsp/.srf/.prt next to it.
    map_file = first_existing(os.path.join(out, 'maps', name + '.map'), os.path.join(out, name + '.map'))
    if map_file:
        copy_file(map_file, os.path.join(base, 'maps'), base)
    else:
        missing.append(name + '.map')

    # Everything else goes into one base/<map>.pk3, staged here first.
    stage = os.path.join(out, '_pk3_staging')
    if os.path.isdir(stage):
        shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)

    shader_file = first_existing(os.path.join(out, 'shaders', name + '.shader'),
                                 os.path.join(out, name + '.shader'))
    if shader_file:
        _stage(shader_file, stage, 'shaders')
        # q3map2 - not the game - loads only the .shader files named in
        # shaderlist.txt, so the name still has to go in there. The engine
        # itself reads every shaders/*.shader it can find, inside .pk3s
        # included, which is why the file itself does not need to be loose.
        ensure_in_shaderlist(os.path.join(base, 'shaders'), name)
    else:
        missing.append(name + '.shader')

    # Terrain alphamap .pcx files (vmf2map.py --terrain-mode patch/lwo's
    # skybox-region fallback): q3map2's LoadPCX resolves the bare filename
    # in a func_group's "alphamap" key via vfsLoadFile - the SAME lookup it
    # uses for every other VFS-relative path, which means the VFS ROOT, not
    # maps/ (confirmed directly in q3map2's own source: LoadPCX calls
    # vfsLoadFile with no path prefix added). In a .pk3 that is the archive
    # root. clean_output.py leaves these at the output folder's own root.
    pcx_files = glob.glob(os.path.join(out, '*.pcx'))
    for pcx_file in pcx_files:
        _stage(pcx_file, stage, '')

    # The welded terrain models, wherever the run left them.
    for lwo_name, needed_for in (
            (name + '_terrain.lwo', "terrain mode 'lwo'"),
            (name + '_terrain_sky.lwo', "the 3D skybox's own terrain"),
            (name + '_terrain_overlays.lwo', "info_overlays sitting on terrain"),
            (name + '_terrain_overlays_sky.lwo', "overlays in the 3D skybox")):
        found = first_existing(os.path.join(out, TERRAIN_MODEL_DIR, lwo_name),
                               os.path.join(out, lwo_name))
        if found:
            _stage(found, stage, TERRAIN_MODEL_DIR)
        elif lwo_name == name + '_terrain.lwo':
            # Older runs named it this before the map name was used.
            legacy = first_existing(os.path.join(out, TERRAIN_MODEL_DIR, 'welded_terrain.lwo'),
                                    os.path.join(out, 'welded_terrain.lwo'))
            if legacy:
                _stage(legacy, stage, TERRAIN_MODEL_DIR)
            else:
                print("  (no {} - fine unless you used {})".format(lwo_name, needed_for))

    # vmf2map.py's extra models cut around water volumes (find_fog_volume_boxes):
    # <map>_terrain_partN.lwo / <map>_terrain_overlays_partN.lwo. The .map
    # references each one, so a missing part is a missing piece of ground.
    part_re = re.compile(r'^' + re.escape(name) + r'_terrain(_overlays)?_part\d+\.lwo$', re.I)
    seen_parts = set()
    for folder in (os.path.join(out, TERRAIN_MODEL_DIR), out):
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if part_re.match(fn) and fn.lower() not in seen_parts:
                seen_parts.add(fn.lower())
                _stage(os.path.join(folder, fn), stage, TERRAIN_MODEL_DIR)

    tex = os.path.join(out, 'textures')
    n_tex = _stage_tree(tex, stage, 'textures') if os.path.isdir(tex) else 0

    # vmf2map.py's converted sounds (sound_sof2/sound/...), for its
    # target_speakers - installed as sound/... The extracted Source originals
    # in out/sound are NOT installed: SOF2 may not play their formats.
    snd = os.path.join(out, 'sound_sof2')
    n_snd = _stage_tree(snd, stage, '') if os.path.isdir(snd) else 0
    if n_snd:
        print("  {} converted sound file(s) from sound_sof2/".format(n_snd))

    # The finished .md3s are in models_fixed/ until clean_output.py renames it
    # to models/. Both are staged - models/ first, then models_fixed/ over it,
    # so for any path in both the newer conversion wins and nothing is lost.
    # This used to take models_fixed/ ALONE whenever it existed: re-run step 3
    # after step 9 had already cleaned up and models_fixed/ comes back almost
    # empty (step 9 deleted the .mdl files it converts from), so the .pk3
    # quietly shipped without its props. Raw Source files in a pre-cleanup
    # models/ (.mdl/.vvd/.vtx/.phy) are filtered out by SKIP_MODEL_EXTS.
    for sub in ('models', 'models_fixed'):
        src_models = os.path.join(out, sub)
        if os.path.isdir(src_models):
            _stage_tree(src_models, stage, 'models', SKIP_MODEL_EXTS)
    n_mdl = sum(len(f) for _d, _s, f in os.walk(os.path.join(stage, 'models')))

    if missing:
        print("ERROR: not found in {}: {} - run the earlier steps first.".format(out, ", ".join(missing)),
              file=sys.stderr)
        sys.exit(1)

    # Release audit: follow every texture, shader and model the finished map
    # names to the file the engine (or q3map2, for baked props) will look for,
    # BEFORE anything is packed. Errors - names the engine cannot hold - stop
    # here; missing assets are reported and the .pk3 is still built, because a
    # map with one untextured decal is still worth testing.
    audit = _load_helper('audit_release')
    print("  auditing what the map references...")
    findings = audit.run(map_file, stage,
                         shader_file=os.path.join(stage, 'shaders', name + '.shader'),
                         base=base, map_name=name,
                         report_path=os.path.join(out, name + '_audit.txt'))
    if findings.count('ERROR') and not args.force:
        print("ERROR: the audit found {} name(s) the engine cannot load (listed above) - "
              "nothing was packed. Fix them, or run step 10 by hand with --force to pack "
              "anyway.".format(findings.count('ERROR')), file=sys.stderr)
        shutil.rmtree(stage, ignore_errors=True)
        sys.exit(1)

    print("  copying {} texture file(s), {} model file(s), the .shader{} into base".format(
        n_tex, n_mdl, " and {} .pcx alphamap(s)".format(len(pcx_files)) if pcx_files else ""))
    # Loose files, not a base/<map>.pk3. A single archive in base is the tidier
    # end state but the wrong thing to produce on every Convert. A Quake3-family
    # filesystem mounts an archive whole, so re-converting replaces the one file
    # carrying ALL of this map's assets at once - and because the paths inside a
    # converted CS map are NOT unique to that map (CS:S content is shared), that
    # archive also shadows the copies another converted map is using. Keeping
    # base consistent then meant moving files in and out by hand.
    #
    # Loose files have neither problem: a re-convert overwrites exactly the
    # files it regenerates and leaves every other map's assets alone, and loose
    # files are read BEFORE mounted .pk3s, so what is on disk is what compiles.
    # The single shareable archive is still produced - the compile .bat packs
    # the finished .bsp and the assets into <output>/<map>_release.pk3 - it just
    # isn't what base runs on.
    installed = copy_tree_into(stage, base)
    n_copied = len(installed)
    shutil.rmtree(stage, ignore_errors=True)
    manifest = write_manifest(os.path.join(out, name + '_files.txt'), installed)

    # A base/<map>.pk3 from the version of this step that packed an archive is
    # now dead weight. Loose files win over it for THIS map, but its paths can
    # still feed stale assets to any other map that shares them. Moved aside
    # rather than deleted, same reasoning as tidy_loose: it is not ours to
    # delete, and the user may want it back.
    stale = os.path.join(base, name + '.pk3')
    if os.path.isfile(stale):
        quarantine = os.path.join(out, '_moved_out_of_base')
        os.makedirs(quarantine, exist_ok=True)
        shutil.move(stale, os.path.join(quarantine, name + '.pk3'))
        print("  moved base/{}.pk3 (left by an earlier version of this step) into\n"
              "  {}\n  - it is no longer used, and its paths can feed stale assets to other "
              "maps.".format(name, quarantine))

    bat = write_compile_bat(base, name, shader_file, out, q3map2_path=args.q3map2,
                            runner=args.runner, tool_version=args.tool_version)
    print("\nDone. {} file(s) in base, including maps/{}.map. Compile with maps/{} - it runs "
          "-bsp, the leaf-box fix, -vis and -light, prints how close the map is to SOF2's "
          "2048-shader table, and packs the finished .bsp together with the assets into "
          "{}_release.pk3 in your output folder, the single file you give to anyone else. "
          "(Or compile in Radiant/Q3map2GUI, with bsp_fix_leafbounds.py run on the .bsp "
          "between the BSP and LIGHT stages.)".format(
              n_copied + 1, name, os.path.basename(bat), name))


if __name__ == "__main__":
    main()
