#!/usr/bin/env python3
"""
audit_release.py - before a converted map is packed, check that everything the
finished map names will actually resolve in game.

Every texture, shader and model the .map refers to is followed to the file the
engine (or q3map2, for what is baked at compile time) will look for, and that
file is looked up in three places, in this order:

  this map's .pk3        what step 10 is about to pack
  the game               every .pk3 in base/ that is NOT another converted map,
                         plus loose shaders/*.shader (stock shaders a modding
                         install keeps extracted for Radiant)
  local only             loose files in base/ and other converted maps' .pk3s
                         and .shader files. They make the map look right on
                         THIS machine, and are not there for anyone you give
                         the .pk3 to - so they are reported separately, not
                         counted as found. Old loose installs share paths with
                         this map (CS:S props live at the same path in every
                         map), which is exactly how a gap in the .pk3 hides.

The lookups mirror the tools, not a guess at them:

  brush faces and patches  shader "textures/<name>"; with no shader of that
                           name, the engine loads the image of the same name.
  baked models             misc_model surfaces as netradiant-custom's q3map2
                           (the compiler compile_<map>.bat runs) sees them:
                           the md3 shader / lwo surface name, extension
                           stripped, given the model's own folder when it has
                           no '/' ("terrain0" -> "models/mapobjects/terrain/
                           terrain0", model.cpp), then -rename appends _BSP
                           (_RMG_BSP with spawnflag 1). There is no fallback:
                           a missing _BSP shader becomes a default shader whose
                           image does not exist (shaders.cpp,
                           ShaderInfoForShader).
  runtime models           model_static and any other entity the ENGINE loads:
                           md3 shader names used as they are, extension
                           stripped, no _BSP.
  shader stages            map / clampMap / animMap frames / skyParms boxes
                           (_bk _dn _ft _lf _rt _up), for this map's own
                           shaders that something actually uses. An image
                           matches as .tga, .jpg, .jpeg or .png.

Severity:
  ERROR    cannot work in any install: a name over MAX_QPATH (63 characters)
           where the engine or the .bsp stores it in a 64-byte field.
           install_to_sof2.py does not pack while there are errors (--force
           packs anyway).
  MISSING  renders as the default texture, or a model is absent.
  LOCAL    resolves only through files on this machine (see above).
  NOTE     known approximations of the conversion, e.g. props that use a
           Source skin other than 0 (every model is converted with skin
           family 0).

USAGE
-----
    python audit_release.py <map file> <assets: folder or .pk3> [--base <SOF2 base>]
                            [--report <file>]

    python audit_release.py base/maps/de_overpass_csgo.map ^
        base/de_overpass_csgo.pk3 --base base
"""
import argparse
import os
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict

MAX_QPATH = 64              # engine path buffers, including the terminating NUL
IMAGE_EXTS = ('.tga', '.jpg', '.jpeg', '.png')
# How a .shader written by this pipeline can be recognised: the signature line
# vmt2shader.py writes now, plus two things every version of it has emitted
# for months (its own helper shaders, and the -light flags comment).
CONVERTED_MARKERS = ('// converted by vmt2shader.py', 'textures/vmf2map/', 'COMPILE: -bsp')


# --- paths ---------------------------------------------------------------------

def norm(p):
    return p.replace('\\', '/').strip().lstrip('/').lower()


def extless(p):
    """q3map2's PathExtensionless: drop the extension of the last component."""
    head, _, tail = p.rpartition('/')
    if '.' in tail:
        tail = tail[:tail.rindex('.')]
    return head + '/' + tail if head else tail


def cap(s):
    """A fixed 64-byte buffer (q3map2's String64 / StringStream<64>)."""
    return s[:MAX_QPATH - 1]


# --- reading an asset tree: a staging folder or a .pk3 ---------------------------

class Tree:
    """A set of game paths and a way to read them, from a folder or a .pk3."""
    def __init__(self, path):
        self.path = path
        self.files = {}          # normalized path -> real name / archive name
        if os.path.isdir(path):
            for dirpath, _d, names in os.walk(path):
                for n in names:
                    full = os.path.join(dirpath, n)
                    rel = os.path.relpath(full, path).replace(os.sep, '/')
                    self.files[norm(rel)] = full
            self._zip = None
        else:
            self._zip = zipfile.ZipFile(path)
            for n in self._zip.namelist():
                if not n.endswith('/'):
                    self.files[norm(n)] = n

    def read(self, key):
        real = self.files.get(key)
        if real is None:
            return None
        if self._zip is not None:
            return self._zip.read(real)
        with open(real, 'rb') as f:
            return f.read()


# --- shader files ----------------------------------------------------------------

_COMMENT_RE = re.compile(r'/\*.*?\*/|//[^\n]*', re.S)
_TOKEN_RE = re.compile(r'\s*([^\s{}]+|\{|\})')
_STAGE_RE = re.compile(r'^[ \t]*(map|clampmap|animmap|animclampmap|implicitmap|implicitmask|'
                       r'implicitblend|skyparms)\b[ \t]*([^\n]*)', re.I | re.M)


def parse_shaders(text):
    """{normalized shader name: body text} for one .shader file."""
    text = _COMMENT_RE.sub('', text)
    out, i, n, pending = {}, 0, len(text), None
    while True:
        m = _TOKEN_RE.match(text, i)
        if not m:
            break
        tok, i = m.group(1), m.end()
        if tok == '{':
            depth, j = 1, i
            while j < n and depth:
                c = text[j]
                depth += (c == '{') - (c == '}')
                j += 1
            if pending:
                out.setdefault(norm(extless(pending)), text[i:j - 1])
            pending, i = None, j
        elif tok != '}':
            pending = tok
    return out


def shader_images(name, body):
    """Every image a shader's stages load."""
    imgs = []
    for kw, rest in _STAGE_RE.findall(body):
        kw, args = kw.lower(), rest.split()
        if kw == 'skyparms':
            if args and args[0] != '-':
                imgs += [args[0] + '_' + s for s in ('bk', 'dn', 'ft', 'lf', 'rt', 'up')]
            continue
        if kw in ('map', 'clampmap'):
            cands = args[:1]
        elif kw in ('animmap', 'animclampmap'):
            cands = args[1:]
        else:  # implicit*: '-' or nothing means the shader's own name
            cands = [name if not args or args[0] == '-' else args[0]]
        imgs += [c for c in cands if not c.startswith(('$', '*'))]
    return imgs


def is_converted_shader(text):
    return any(m in text for m in CONVERTED_MARKERS)


# --- model files -------------------------------------------------------------------

def md3_shader_names(data):
    names = []
    if not data or len(data) < 108 or data[:4] != b'IDP3':
        return None
    nsurf = struct.unpack_from('<i', data, 84)[0]
    off = struct.unpack_from('<i', data, 100)[0]
    for _ in range(nsurf):
        if off + 108 > len(data) or data[off:off + 4] != b'IDP3':
            break
        nshader = struct.unpack_from('<i', data, off + 76)[0]
        sh_off = struct.unpack_from('<i', data, off + 92)[0]   # ofsShaders (@96 is ofsSt)
        end = struct.unpack_from('<i', data, off + 104)[0]
        for s in range(nshader):
            p = off + sh_off + s * 68
            if p + 64 <= len(data):
                names.append(data[p:p + 64].split(b'\0')[0].decode('latin-1'))
        if end <= 0:
            break
        off += end
    return names


def lwo_surface_names(data):
    """SURF chunk names of an LWO2 - the material names the model loader sees."""
    if not data or data[:4] != b'FORM' or data[8:12] != b'LWO2':
        return None
    names, i = [], 12
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        size = struct.unpack('>I', data[i + 4:i + 8])[0]
        if cid == b'SURF':
            names.append(data[i + 8:i + 8 + size].split(b'\0', 1)[0].decode('latin-1'))
        i += 8 + size + (size & 1)
    return names


def model_names(path, data):
    low = path.lower()
    if low.endswith('.md3'):
        return md3_shader_names(data)
    if low.endswith(('.lwo', '.lwo2')):
        return lwo_surface_names(data)
    return []      # other formats: existence checked, surfaces not followed


def baked_shader_name(raw, model_path, spawnflags):
    """What q3map2 -rename looks up for one misc_model surface (see module doc)."""
    s = cap(norm(extless(raw)))
    if '/' not in s:
        folder = model_path.rsplit('/', 1)[0] + '/' if '/' in model_path else ''
        s = cap(folder + s)
    return cap(extless(s) + ('_rmg_bsp' if spawnflags & 1 else '_bsp'))


# --- the .map ----------------------------------------------------------------------

_FACE_RE = re.compile(r'^\(\s*\S+\s+\S+\s+\S+\s*\)\s*\(\s*\S+\s+\S+\s+\S+\s*\)\s*'
                      r'\(\s*\S+\s+\S+\s+\S+\s*\)\s*(\S+)', re.M)
_PATCH_RE = re.compile(r'patchDef[23]?\s*\{\s*(\S+)')
_ENTITY_RE = re.compile(r'^\{\s*\n((?:"[^"\n]*"\s+"[^"\n]*"\s*\n)+)', re.M)
_KV_RE = re.compile(r'"([^"\n]*)"\s+"([^"\n]*)"')


def read_map(text):
    faces = Counter()
    for m in _FACE_RE.finditer(text):
        faces['textures/' + norm(m.group(1))] += 1
    for m in _PATCH_RE.finditer(text):
        faces['textures/' + norm(m.group(1))] += 1
    entities = []
    for m in _ENTITY_RE.finditer(text):
        kv = {}
        for k, v in _KV_RE.findall(m.group(1)):
            kv.setdefault(k.lower(), v)
        if 'classname' in kv:
            entities.append(kv)
    return faces, entities


# --- where things can come from ------------------------------------------------------

class Sources:
    """The three places an asset can resolve, in the order they are tried."""
    def __init__(self, tree, own_shader_text, base, map_name):
        self.tree = tree
        self.own_shaders = parse_shaders(own_shader_text)
        self.game_files, self.local_files = set(), set()
        self.game_shaders, self.local_shaders = {}, {}
        self.notes = []
        if base and os.path.isdir(base):
            self._index_base(base, map_name.lower())
        self._img = {}
        for label, files in (('pk3', set(tree.files)), ('game', self.game_files),
                             ('local', self.local_files)):
            self._img[label] = set(extless(f) for f in files if f.endswith(IMAGE_EXTS))

    def _add_shader_file(self, text, game):
        target = self.game_shaders if game and not is_converted_shader(text) else self.local_shaders
        for k, v in parse_shaders(text).items():
            target.setdefault(k, v)

    def _index_base(self, base, map_name):
        n_game = n_conv = 0
        for fn in sorted(os.listdir(base)):
            full = os.path.join(base, fn)
            if not fn.lower().endswith('.pk3') or not os.path.isfile(full):
                continue
            stem = fn[:-4].lower()
            if stem == map_name:
                continue    # the previous build of this very map
            try:
                z = zipfile.ZipFile(full)
            except (OSError, zipfile.BadZipFile):
                continue
            names = [n for n in z.namelist() if not n.endswith('/')]
            lowered = set(norm(n) for n in names)
            converted = ('shaders/%s.shader' % stem) in lowered
            (self.local_files if converted else self.game_files).update(lowered)
            n_conv += converted
            n_game += not converted
            for n in names:
                if norm(n).startswith('shaders/') and n.lower().endswith('.shader'):
                    try:
                        text = z.read(n).decode('latin-1')
                    except Exception:
                        continue
                    self._add_shader_file(text, game=not converted)
        for dirpath, dirs, names in os.walk(base):
            rel_dir = os.path.relpath(dirpath, base).replace(os.sep, '/')
            if rel_dir == '.':
                dirs[:] = [d for d in dirs if d.lower() != 'maps']
            for n in names:
                rel = norm(os.path.join(rel_dir, n) if rel_dir != '.' else n)
                if rel.endswith('.pk3'):
                    continue
                if rel.startswith('shaders/') and rel.endswith('.shader'):
                    if rel == 'shaders/%s.shader' % map_name:
                        continue    # a stale loose copy of this map's shader
                    try:
                        with open(os.path.join(dirpath, n), encoding='latin-1') as f:
                            self._add_shader_file(f.read(), game=True)
                    except OSError:
                        pass
                    continue
                self.local_files.add(rel)
        self.notes.append("base: {} game .pk3(s), {} other converted map(s) treated as "
                          "local only".format(n_game, n_conv))

    # Each lookup answers 'pk3', 'game', 'local' or None.
    def shader(self, name):
        if name in self.own_shaders:
            return 'pk3'
        if name in self.game_shaders:
            return 'game'
        if name in self.local_shaders:
            return 'local'
        return None

    def image(self, name):
        key = extless(norm(name))
        for label in ('pk3', 'game', 'local'):
            if key in self._img[label]:
                return label
        return None

    def file(self, path):
        key = norm(path)
        if key in self.tree.files:
            return 'pk3'
        if key in self.game_files:
            return 'game'
        if key in self.local_files:
            return 'local'
        return None


# --- the audit -----------------------------------------------------------------------

class Findings:
    def __init__(self):
        self.items = defaultdict(lambda: defaultdict(Counter))  # sev -> what -> {context: n}

    def add(self, sev, what, context, n=1):
        self.items[sev][what][context] += n

    def count(self, sev):
        return len(self.items.get(sev, {}))


def audit(map_text, tree, own_shader_text, base=None, map_name=''):
    src = Sources(tree, own_shader_text, base, map_name)
    fx = Findings()
    faces, entities = read_map(map_text)
    used_own = {}        # own shader name -> where it is used (for image checks)

    def resolve_material(name, context, n=1):
        """A shader name the engine will look up: a shader, else an image."""
        if len(name) >= MAX_QPATH:
            fx.add('ERROR', name, "{} - {} characters, the limit is {}".format(
                context, len(name), MAX_QPATH - 1), n)
            return
        where = src.shader(name)
        if where == 'pk3':
            used_own.setdefault(name, context)
            return
        if where == 'game':
            return
        img = src.image(name)
        if img in ('pk3', 'game'):
            return
        if where == 'local' or img == 'local':
            fx.add('LOCAL', name, context, n)
        else:
            fx.add('MISSING', name, context + " - no shader and no image of that name", n)

    for name, n in faces.items():
        resolve_material(name, "brush faces/patches", n)

    model_cache = {}
    skins = Counter()
    for ent in entities:
        cls = ent.get('classname', '').lower()
        model = ent.get('model', '')
        if model.startswith('*'):
            model = ''
        if ent.get('alphamap'):
            if not src.file(ent['alphamap']):
                fx.add('MISSING', norm(ent['alphamap']), "terrain alphamap (" + cls + ")")
        if not model:
            continue
        mpath = norm(model)
        baked = cls == 'misc_model'
        if not baked and len(mpath) >= MAX_QPATH:
            fx.add('ERROR', mpath, "{} model path - {} characters, the limit is {}".format(
                cls, len(mpath), MAX_QPATH - 1))
            continue
        where = src.file(mpath)
        if where is None:
            fx.add('MISSING', mpath, "{} model{}".format(
                cls, " (q3map2 drops it from the compile)" if baked else ""))
            continue
        if where == 'local':
            fx.add('LOCAL', mpath, cls + " model")
        try:
            skin = int(ent.get('skin', '0') or 0)
        except ValueError:
            skin = 0
        if skin > 0 and mpath.endswith('.md3'):
            skins[(mpath, skin)] += 1
        if mpath not in model_cache:
            data = tree.read(mpath) if where == 'pk3' else None
            model_cache[mpath] = model_names(mpath, data) if data is not None else []
        try:
            flags = int(ent.get('spawnflags', '0') or 0)
        except ValueError:
            flags = 0
        for raw in model_cache[mpath] or []:
            if baked:
                resolve_material(baked_shader_name(raw, mpath, flags),
                                 "baked surface of " + mpath)
            else:
                resolve_material(cap(norm(extless(raw))), "surface of " + mpath)

    for name, context in used_own.items():
        for img in shader_images(name, src.own_shaders[name]):
            ip = norm(img)
            if len(extless(ip)) >= MAX_QPATH - 4:     # room for the extension
                fx.add('ERROR', ip, "image of shader {} - {} characters".format(name, len(ip)))
                continue
            where = src.image(ip)
            if where is None:
                fx.add('MISSING', ip, "image of shader " + name)
            elif where == 'local':
                fx.add('LOCAL', ip, "image of shader " + name)

    for (mpath, skin), n in sorted(skins.items()):
        fx.add('NOTE', mpath, "Source skin {} on {} prop(s); converted with skin 0".format(
            skin, n))
    return fx, src, faces, entities


SEVERITY_TEXT = {
    'ERROR':   "cannot work in any install",
    'MISSING': "will render as the default texture, or the model is absent",
    'LOCAL':   "found only in files on THIS machine - missing for anyone you give the .pk3 to",
    'NOTE':    "known approximations",
}


def format_report(fx, limit=None):
    lines = []
    for sev in ('ERROR', 'MISSING', 'LOCAL', 'NOTE'):
        group = fx.items.get(sev)
        if not group:
            continue
        lines.append("{} {} - {}:".format(len(group), sev, SEVERITY_TEXT[sev]))
        ordered = sorted(group.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        for what, ctx in ordered[:limit] if limit else ordered:
            uses = sum(ctx.values())
            first = ctx.most_common(1)[0][0]
            more = len(ctx) - 1
            lines.append("  {}\n      {}{}{}".format(
                what, first, " (x{})".format(uses) if uses > 1 else "",
                "  +{} other use(s)".format(more) if more > 0 else ""))
        if limit and len(ordered) > limit:
            lines.append("  ... and {} more".format(len(ordered) - limit))
    return "\n".join(lines)


def run(map_file, assets, shader_file=None, base=None, map_name=None, report_path=None,
        limit=15):
    """Audit and print. Returns the Findings."""
    map_name = map_name or os.path.splitext(os.path.basename(map_file))[0]
    with open(map_file, encoding='utf-8', errors='replace') as f:
        map_text = f.read()
    tree = Tree(assets)
    if shader_file:
        with open(shader_file, encoding='latin-1') as f:
            shader_text = f.read()
    else:
        key = 'shaders/%s.shader' % map_name.lower()
        data = tree.read(key)
        shader_text = data.decode('latin-1') if data else ''
    fx, src, faces, entities = audit(map_text, tree, shader_text, base, map_name)
    n_models = sum(1 for e in entities if e.get('model', '') and not e['model'].startswith('*'))
    print("  audit: {} material(s) on brushes/patches, {} model placement(s), {} own "
          "shader(s); {}".format(len(faces), n_models, len(src.own_shaders),
                                 "; ".join(src.notes) or "no SOF2 base given"))
    if not any(fx.items.values()):
        print("  audit: every reference resolves inside this .pk3 or the game.")
    else:
        print(format_report(fx, limit))
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Release audit for {}\n\n".format(map_name))
            f.write(format_report(fx) or "Every reference resolves inside this .pk3 or the game.")
            f.write("\n")
        if any(fx.items.values()):
            print("  full list: {}".format(report_path))
    return fx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_file')
    ap.add_argument('assets', help="the staged asset folder, or a built .pk3")
    ap.add_argument('--shader', help="the map's .shader (default: shaders/<map>.shader "
                                     "inside the assets)")
    ap.add_argument('--base', help="SOF2 base folder, to resolve the game's own assets")
    ap.add_argument('--report', help="write the full list here")
    args = ap.parse_args()
    fx = run(args.map_file, args.assets, args.shader, args.base, report_path=args.report,
             limit=None if args.report is None else 15)
    return 1 if fx.count('ERROR') else 0


if __name__ == '__main__':
    sys.exit(main())
