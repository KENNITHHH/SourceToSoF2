#!/usr/bin/env python3
"""
vmt2shader.py - Convert Source engine .vmt material scripts into an idTech3
.shader script compatible with SOF2 (and other Q3-derived engines).

------------------------------------------------------------------------------
WHAT THIS DOES
------------------------------------------------------------------------------
- Parses each .vmt (a simple KeyValues file: one root shader class, e.g.
  LightmappedGeneric/VertexLitGeneric/WorldVertexTransition/Water, plus
  $-prefixed parameters).
- Emits one idTech3 shader block per .vmt, named "textures/<relative path>"
  (matching the .vmt's path under your materials root, extension stripped) -
  this MUST match exactly what your converted .map file's brush texture
  field, and your converted texture image's path, both use.
- Points the diffuse stage at "textures/<relative path>.<ext>" using the
  image extension you specify (default: png, per your export workflow).
- Derives blend/alpha/cull state from common VMT parameters:
    $translucent          -> blend stage (GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA),
                              surfaceparm trans, sort priority
    $alphatest / $alphatestreference
                           -> alphaFunc GE128, surfaceparm alphashadow
    $additive              -> blend stage (GL_ONE GL_ONE)
    $nocull / $selfillum   -> cull none / extra rgbGen stage
- Derives a surfaceparm from $surfaceprop using a best-effort table (see
  SURFACEPROP_MAP below) for footstep sounds / material type.
- "nodraw"/"tools" style VMTs (shader "Tools", or a name matching common
  Source tool material patterns) are skipped, since your .map conversion
  already points those brush faces straight at SOF2's textures/tools/_*
  shaders rather than needing a generated shader script.

------------------------------------------------------------------------------
WHAT THIS CANNOT DO
------------------------------------------------------------------------------
- It has no idea what SOF2's actual shader stage syntax quirks or engine
  extensions are beyond the common idTech3 baseline - treat this as a
  strong starting point, not a drop-in replacement for hand-tuned shaders,
  especially for anything with $bumpmap/$envmap/$phong (Source per-pixel
  lighting has no direct classic-idTech3 equivalent; those parameters are
  read but only used to leave a comment, not translated into real stages).
- $surfaceprop -> surfaceparm mapping is a best-effort guess for the most
  common Source surface names; verify against SOF2's actual supported
  surfaceparm keywords and adjust via --surfaceprop-map if needed.
- Doesn't touch or convert the actual texture image files - it assumes
  you've already exported/converted them (e.g. VTF -> PNG) to the same
  relative path the .vmt lives at.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 vmt2shader.py materials_dir output.shader
    python3 vmt2shader.py materials_dir output.shader --ext png
    python3 vmt2shader.py materials_dir output.shader --surfaceprop-map spmap.json

"materials_dir" should be the root folder containing your .vmt files
(searched recursively). The shader name for each is derived from its path
relative to this root, e.g. materials_dir/brick/brickwall051d.vmt becomes
shader "textures/brick/brickwall051d" - which must match the path you used
(or will use) for the matching converted texture image, and the material
name your .map file already references.

A summary of skipped materials and unmapped $surfaceprop values is printed
to stderr at the end of the run.
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from collections import OrderedDict, Counter

# reuse the same tokenizer/parser approach as vmf2map.py, since VMT is the
# same KeyValues format VMF uses, just with a single root key.
# The comment alternative MUST come before the \S+ fallback: without it, a
# leading "// comment" line gets tokenized as a real token, so the FIRST
# token (which parse_vmt takes as the shader class name) becomes '//' and
# the whole file mis-parses into nothing. Confirmed against real content -
# this silently dropped every .vmt that happened to start with a comment.
# Quoted strings are matched first, so a '//' inside a quoted value (e.g. a
# URL or an odd path) is still preserved correctly.
_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(//[^\n]*)|([{}])|(\S+)')


# ==============================================================================
# idTech3 name length (MAX_QPATH)
# ==============================================================================
# Every shader name and image path in idTech3 lives in a 64-byte field
# (MAX_QPATH: 63 characters + NUL): the MD3 surface shader field, the BSP shader
# lump, q3map2's own shaderInfo (String64 - longer names are silently cut), and
# the engine's image loader. Two things get appended to names after we write
# them:
#   - q3map2 -rename appends "_BSP" to every misc_model surface (model.cpp:
#     String64(name, "_BSP")), so a 60-63 character name loses its suffix and
#     the baked prop gets no shader;
#   - the engine appends ".jpg"/".tga" to an extensionless image path.
# Both take four characters, so 59 is the longest name that survives both.
# CS:GO/CS2 ports (de_mirage_cs2's models/props/de_dust/hr_dust/...) exceed it;
# the CS:S content in de_dust2 never did.
#
# Over-long names are shortened DETERMINISTICALLY - the same input always gives
# the same output - so every script applies it on its own and they agree
# without passing a table around: md3_shader_autofix / mdl_to_md3 (MD3 surface
# names), vmf2map (.map faces, patches, .lwo tags) and this script (shader block
# names, image paths - it also copies the image to the short file name). The
# front of the path is kept so the name still reads and still starts with
# models/... or textures/...; the hash is of the whole lower-cased name, so two
# long names sharing a prefix never collide.
#
# KEEP IDENTICAL to the copies in vmf2map.py and mdl_to_md3.py.
QPATH_NAME_MAX = 59


def short_qpath(name, limit=QPATH_NAME_MAX):
    if len(name) <= limit:
        return name
    import hashlib
    digest = hashlib.md5(name.lower().encode('utf-8')).hexdigest()[:8]
    return name[:limit - 9].rstrip('/_.') + '_' + digest


def short_shader_name(name):
    """short_qpath for a shader name that may carry -rename's _bsp suffix: q3map2
    appends that to the (already shortened) MD3/LWO name, so it is shortened as
    base + suffix, not as one string."""
    if name.lower().endswith('_bsp'):
        return short_qpath(name[:-4]) + name[-4:]
    return short_qpath(name)


def tokenize(text):
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            tokens.append(m.group(1).replace('\\"', '"'))
        elif m.group(2) is not None:
            continue  # line comment - skip entirely
        elif m.group(3) is not None:
            tokens.append(m.group(3))
        else:
            tokens.append(m.group(4))
    return tokens


def parse_vmt(tokens):
    """A VMT has exactly one root key (the shader class name) whose value is
    a block of $key/value pairs (plus possibly nested blocks like
    "Proxies" or "> or <dx9 fallback" blocks, which we mostly ignore)."""
    pos = [0]

    def parse_block():
        node = OrderedDict()
        while pos[0] < len(tokens):
            tok = tokens[pos[0]]
            if tok == '}':
                pos[0] += 1
                return node
            key = tok
            pos[0] += 1
            if pos[0] >= len(tokens):
                break
            nxt = tokens[pos[0]]
            if nxt == '{':
                pos[0] += 1
                value = parse_block()
            else:
                value = nxt
                pos[0] += 1
            node.setdefault(key, []).append(value)
        return node

    root = parse_block()
    if not root:
        return None, {}
    shader_class = list(root.keys())[0]
    body = root[shader_class][0]
    if not isinstance(body, dict):
        return shader_class, {}
    return shader_class, body


def get_param(body, *names, default=None):
    """VMT params are case-insensitive and always $-prefixed; look up by
    any of the given names (without the $)."""
    lower_map = {k.lstrip('$').lower(): v for k, v in body.items()}
    for name in names:
        key = name.lstrip('$').lower()
        if key in lower_map:
            v = lower_map[key][0]
            if isinstance(v, dict):
                continue
            return v
    return default


def is_truthy(value):
    if value is None:
        return False
    v = str(value).strip()
    return v not in ('0', '', 'false', 'no')


_TEXTURES_PREFIX_RE = re.compile(r'^(textures/)+', re.IGNORECASE)

# Top-level roots that must NOT get a "textures/" prefix on the SHADER NAME
# (and qer_editorimage): these are model-owned materials, and the shader
# name has to exactly match what fix_md3_shaders.py/md3_shader_autofix.py
# already burns into the .md3 surface shader field - the bare
# "<cdmaterials>/<texture>" string read straight out of the source .mdl's
# texture table. Everything else (world/brush materials) keeps the
# "textures/" prefix as before.
#
# 'models/' is included as a safe default since it covers the overwhelming
# majority of Source content, but it's still a GUESS about naming
# convention, not a fact - some content (e.g. CS:GO's FMPONE-authored
# de_cache assets) has .mdls whose own cdmaterials use a completely
# different top-level folder ("fmpone/tires/", "fmpone/roof/", etc.), which
# no fixed guess can anticipate. Pass --mdl-dir on the command line to
# scan your actual .mdl files and add whatever top-level folders they
# really use - always prefer that over relying on this default alone.
NO_TEXTURES_PREFIX_ROOTS = ['models/']


def collect_model_cdmaterials_roots(mdl_root):
    """Scan every .mdl under mdl_root and return the real set of top-level
    folders (first path segment) their own cdmaterials directories use -
    the authoritative alternative to guessing folder names like 'models/'."""
    roots = set()
    mdl_count = 0
    for dirpath, _dirs, filenames in os.walk(mdl_root):
        for fn in filenames:
            if not fn.lower().endswith('.mdl'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                if len(data) < 240 or data[0:4] != b'IDST':
                    continue
                mdl_count += 1
                numcdtextures, cdtextureindex = struct.unpack_from('<ii', data, 212)
                for i in range(numcdtextures):
                    off = struct.unpack_from('<i', data, cdtextureindex + i * 4)[0]
                    end = data.index(b'\x00', off)
                    cdmat = data[off:end].decode('latin1').strip().replace('\\', '/').lower()
                    top = cdmat.split('/', 1)[0] if cdmat else ''
                    if top:
                        roots.add(top + '/')
            except (struct.error, ValueError, IndexError, OSError):
                continue
    return roots, mdl_count


def with_textures_prefix(path_value):
    """Normalize to the SHADER IDENTIFIER idTech3/the .md3 expects: bare (no
    prefix) for model materials under a known model-owned root, a single
    leading 'textures/' for everything else, regardless of how many
    'textures/' prefixes (if any) the input already had."""
    v = path_value.strip().replace('\\', '/').lower()
    v = _TEXTURES_PREFIX_RE.sub('', v)
    if v.startswith(tuple(NO_TEXTURES_PREFIX_ROOTS)):
        return v
    return "textures/" + v


def is_model_owned_root(path_value):
    """True if this path falls under a folder known to be a model's own
    cdmaterials root (i.e. with_textures_prefix gives it a bare, no-prefix
    shader name) - the case where the SAME material might ALSO be used on
    a world brush face, which the engine always resolves through an
    implicit 'textures/'-prefixed name regardless of what's literally
    written in the .map file. Used to decide whether a shader needs to be
    emitted under BOTH names."""
    v = path_value.strip().replace('\\', '/').lower()
    v = _TEXTURES_PREFIX_RE.sub('', v)
    return v.startswith(tuple(NO_TEXTURES_PREFIX_ROOTS))


# --- Source "Water" materials -----------------------------------------------
#
# A Source Water .vmt has NO $basetexture: the surface is drawn entirely by
# Source's water shader from a reflection/refraction render target, a normal
# map ($normalmap) and fog parameters. convert_vmt_to_shader therefore bailed
# out on it ("has no $basetexture, skipped"), leaving the brush with no shader
# at all - which is why converted maps showed the missing-texture grid on
# every water surface.
#
# idTech3 has no render-to-texture water, so the honest conversion is SOF2's
# own water: a scrolling/turbulent texture plus fogparms for the underwater
# tint. These three stock SOF2 textures are used as the base, VERIFIED present
# in a real SOF2 install's base/shaders/common.shader rather than guessed:
#
#     textures/common/water01        clear-ish blue water   ("water02", "water_murky_dm")
#     textures/common/water_murky    dark green/brown        ("water_murky", "water_murky_new")
#     textures/common/water_pool     bright pool blue        ("water_pool")
#
# The choice between them is made from the map's OWN $fogcolor, so a green
# canal and a blue pool do not come out the same. $fogcolor is also what
# drives fogparms, so the underwater tint matches what Source rendered.
WATER_BASE_CLEAR = 'textures/common/water01'
WATER_BASE_MURKY = 'textures/common/water_murky'
WATER_BASE_POOL = 'textures/common/water_pool'

# The fixed Source -> SOF2 world scale this pipeline converts at (vmf2map.py's
# --scale, pinned at 1.25 since the GUI stopped exposing it). Used here only to
# turn Source's $fogend, which is in Source units, into map units.
SOURCE_TO_SOF2_SCALE = 1.25


def _parse_vec3(value, default=None):
    """'[.15 .1 0]' or '{38 25 0}' or '.15 .1 0' -> (r, g, b) as 0..1."""
    if not value:
        return default
    txt = value.strip().strip('[]{}').replace(',', ' ')
    try:
        parts = [float(x) for x in txt.split()]
    except ValueError:
        return default
    if len(parts) < 3:
        return default
    r, g, b = parts[0], parts[1], parts[2]
    # Source writes {} vectors in 0..255 and [] vectors in 0..1.
    if '{' in value or max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))


def water_shader(rel_path, body, warnings):
    """A SOF2 water shader for a Source 'Water' material.

    Source gives us $fogcolor (the underwater tint), $fogstart/$fogend (how
    far you can see through it) and $abovewater. All of those map onto
    idTech3's 'fogparms', which is the one part of Source's water idTech3 can
    reproduce exactly rather than approximate.
    """
    fog = _parse_vec3(get_param(body, 'fogcolor', '$fogcolor'), None)
    if fog is None:
        fog = (0.12, 0.14, 0.12)
        warnings.append("'{}' is a Water material with no $fogcolor - using a neutral "
                        "green-grey tint for its fog.".format(rel_path))
    # Pick the base texture from the water's own colour: strongly blue water
    # gets the pool/clear texture, anything green or brown gets the murky one.
    r, g, b = fog
    if b > r and b > g * 1.1:
        base = WATER_BASE_POOL if (b > 0.35) else WATER_BASE_CLEAR
    elif max(r, g, b) < 0.08:
        base = WATER_BASE_MURKY
    elif g >= r:
        base = WATER_BASE_MURKY
    else:
        base = WATER_BASE_CLEAR

    # $fogend is how far light carries through the water, in SOURCE units, so
    # it scales with the map exactly like every other distance this pipeline
    # converts. Source's default when unset is a few hundred units.
    def _f(name, dflt):
        v = get_param(body, name, '$' + name)
        try:
            return float(v)
        except (TypeError, ValueError):
            return dflt
    fog_end = _f('fogend', 300.0)
    if fog_end <= 0:
        fog_end = 300.0
    fog_dist = fog_end * SOURCE_TO_SOF2_SCALE

    name = with_textures_prefix(rel_path)
    lines = [name, '{']
    lines.append('\tqer_editorimage {}'.format(base))
    lines.append('\t// Source "Water" material: no $basetexture exists in the original -')
    lines.append('\t// Source draws water from a render target, which idTech3 has no')
    lines.append('\t// equivalent for. Base texture is SOF2 stock; the tint and view')
    lines.append('\t// distance below are this map\'s own $fogcolor/$fogend.')
    lines.append('\tsurfaceparm water')
    lines.append('\tsurfaceparm trans')
    lines.append('\tsurfaceparm nonsolid')
    lines.append('\tsurfaceparm nolightmap')
    lines.append('\tq3map_material Water')
    lines.append('\tqer_trans 0.5')
    # fogparms is what makes it read as water from under the surface, and is a
    # direct translation of Source's own numbers.
    lines.append('\tfogparms ( {:.4g} {:.4g} {:.4g} ) {:.0f}'.format(r, g, b, fog_dist))
    lines.append('\tcull none')
    lines.append('\t{')
    lines.append('\t\tmap {}'.format(base))
    lines.append('\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA')
    lines.append('\t\trgbGen identity')
    lines.append('\t\ttcMod turb 0 0.03 0 0.08')
    lines.append('\t\ttcMod scroll 0.012 0.008')
    lines.append('\t}')
    lines.append('}')
    return '\n'.join(lines)


def emit_dual_shader_blocks(rel_path, lines):
    """Given a built shader block ('lines', with lines[0] the primary shader
    name already computed via with_textures_prefix), also emit a second,
    identical block under the 'textures/'-prefixed name whenever the
    material falls under a model-owned cdmaterials root.

    This is the fix for a real, confirmed engine bug: a material that is
    BOTH a model's own skin (looked up bare, verbatim, from the .md3) AND
    an ordinary world-brush texture (always looked up by the engine with
    an implicit 'textures/' prefix, regardless of what's literally written
    in the .map file) needs a shader definition under EACH name - one
    shader block can only ever satisfy one of the two lookups. Confirmed
    against a real GtkRadiant load log: model-only materials in the same
    folder loaded fine, but ones also used on world brushes failed with
    'Activate shader failed for textures/<path>' even though the
    (bare-named) shader was right there in the .shader file the whole
    time. Emitting the harmless second block for every model-owned-root
    material costs nothing when it isn't needed (an unused shader
    definition is never an error) and fixes every case where it is."""
    if not is_model_owned_root(rel_path):
        return "\n".join(lines)
    alt_lines = list(lines)
    alt_lines[0] = "textures/" + lines[0]
    return "\n".join(lines) + "\n\n" + "\n".join(alt_lines)


def physical_texture_path(path_value):
    """Normalize to the ACTUAL ON-DISK image path for a 'map' stage. Unlike
    the shader name above, every converted texture - including the ones
    backing a model's own shader - physically lives under idTech3's
    textures/ root, so this ALWAYS gets exactly one leading 'textures/'.
    (The shader name vs. real file path split is exactly the Source
    materials/ vs. idTech3 textures/ root mismatch - the shader name has to
    match the .md3, the file path has to match where the image really is.)"""
    v = path_value.strip().replace('\\', '/').lower()
    v = _TEXTURES_PREFIX_RE.sub('', v)
    # Some .vmt files spell the extension out - confirmed on CS:S's own
    # models/props_urban/curbs001.vmt: $baseTexture "models/props_urban/
    # curbs001.vtf". Source strips it; idTech3 does not. q3map2 drops any
    # extension and tries .tga/.jpg itself, so the compile found the image
    # and stayed quiet - but SOF2's R_FindImageFile takes an explicit
    # extension literally, looked for curbs001.vtf, and drew the surface
    # with no image. The converted file is always .jpg/.png with the same
    # stem, so drop the extension and let the engine pick it.
    v = _IMAGE_EXT_RE.sub('', v)
    return "textures/" + v


_IMAGE_EXT_RE = re.compile(r'\.(vtf|tga|jpg|jpeg|png|dds)$', re.IGNORECASE)


# ==============================================================================
# "patch" material resolution
# ==============================================================================
# Source auto-generates per-instance "patch" VMTs (most commonly for baked
# per-brush $envmap/cubemap overrides near a reflective surface). These have
# no $basetexture of their own - they "include" a real base material and
# "replace" a handful of params on top of it. We resolve the include chain
# so the generated shader still gets the correct base texture.

def build_vmt_index(materials_dir):
    """Map lowercased relative .vmt path (with and without extension) -> the
    real path on disk, for case-insensitive include resolution."""
    index = {}
    for root, _dirs, files in os.walk(materials_dir):
        for fname in files:
            if not fname.lower().endswith('.vmt'):
                continue
            full_path = os.path.join(root, fname)
            rel = os.path.relpath(full_path, materials_dir).replace('\\', '/').lower()
            index[rel] = full_path
            index[os.path.splitext(rel)[0]] = full_path
    return index


def resolve_include_path(include_value, vmt_index):
    # include values look like "materials/Foo/Bar.vmt" - strip a leading
    # "materials/" since our index is already rooted there, then normalize.
    v = include_value.strip().replace('\\', '/').lower()
    if v.startswith('materials/'):
        v = v[len('materials/'):]
    return vmt_index.get(v) or vmt_index.get(os.path.splitext(v)[0])


def resolve_vmt(full_path, vmt_index, depth=0, seen=None):
    """Parse a .vmt file, following 'patch' include/replace chains until a
    real (non-patch) shader class + body is reached. Returns (shader_class,
    merged_body) or (None, {}) if unresolvable."""
    if seen is None:
        seen = set()
    if depth > 8 or full_path in seen:
        return None, {}
    seen.add(full_path)

    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    tokens = tokenize(text)
    shader_class, body = parse_vmt(tokens)

    if shader_class and shader_class.strip().lower() == 'patch':
        include_value = get_param(body, 'include')
        if not include_value:
            return shader_class, body
        included_path = resolve_include_path(include_value, vmt_index)
        if not included_path:
            return shader_class, body
        base_class, base_body = resolve_vmt(included_path, vmt_index, depth + 1, seen)
        merged = OrderedDict(base_body)
        replace_values = body.get('replace') or body.get('Replace')
        if replace_values and isinstance(replace_values[0], dict):
            for k, v in replace_values[0].items():
                merged[k] = v
        return base_class, merged

    return shader_class, body


# ==============================================================================
# $surfaceprop -> idTech3 surfaceparm mapping
# ==============================================================================
# Best-effort guess based on common idTech3 footstep/material surfaceparms.
# NOT confirmed against SOF2 specifically - verify and override with
# --surfaceprop-map if SOF2 uses different keywords for some of these.
SURFACEPROP_MAP = {
    'metal': 'metalsteps',
    'metal_box': 'metalsteps',
    'metalvent': 'metalsteps',
    'metalgrate': 'metalsteps',
    'metalpanel': 'metalsteps',
    'panel': 'metalsteps',
    'chainlink': 'metalsteps',
    'iron': 'metalsteps',
    'grate': 'metalsteps',
    'flesh': 'flesh',
    'blood': 'flesh',
    'antlion': 'flesh',
    'zombieflesh': 'flesh',
    'glass': 'nomarks',
    'glassbottle': 'nomarks',
    'ladder': 'ladder',
    # NOTE: these used to map to 'nosteps', which is WRONG - in standard
    # Quake3 surfaceparm semantics, nosteps means COMPLETELY SILENT (no
    # footstep sound at all - used for things like the underside of water),
    # not "hard/concrete-like footsteps" as the name misleadingly suggests.
    # That was an early, unverified guess; confirmed wrong by real testing
    # (concrete floors were silent in-game). Removed entirely rather than
    # given a replacement value: omitting a surfaceparm here means the
    # engine's own DEFAULT footstep sound plays, which is the correct
    # behavior for ordinary walkable materials like these. q3map_material
    # (below, in Q3MAP_MATERIAL_MAP) is the CONFIRMED, verified mechanism
    # SOF2 actually uses for material-specific footstep/impact behavior
    # (checked directly against its own tools.shader/mp_office_tools.shader),
    # so it does the real work here, not this table.
    # dirt/gravel/sand/grass/mud/snow used to map to 'dust' here - also
    # WRONG, but in a different way than the nosteps case above: 'dust' IS
    # a real Quake3 surfaceparm (falling dust when walked on), but SOF2's
    # own q3map2 build doesn't recognize it - confirmed by a real compile
    # log flagging it as unknown on every single occurrence. Removed for
    # the same reason as nosteps: omitting it means the engine's own
    # default footstep sound plays, which is correct for ordinary
    # ground materials like these, rather than a surfaceparm SOF2 just
    # silently ignores anyway.
    'water': 'water',
    'slime': 'slime',
    'wade': 'water',
    'solidmetal': 'metalsteps',
    'default': None,  # Source's generic fallback surfaceprop - no useful signal
}


def convert_surfaceparm(surfaceprop, surfaceprop_map, unmapped_counter):
    if not surfaceprop:
        return None
    key = surfaceprop.strip().lower()
    if key in surfaceprop_map:
        return surfaceprop_map[key]
    if key in SURFACEPROP_MAP:
        return SURFACEPROP_MAP[key]
    unmapped_counter[surfaceprop] += 1
    return None


# ==============================================================================
# $surfaceprop -> SOF2 q3map_material mapping
# ==============================================================================
# q3map_material is SEPARATE from surfaceparm above - it's SOF2's own
# surface-type keyword (drives footstep sounds, bullet impact effects/decals,
# not idTech3's generic surfaceparm system). Valid values per SOF2's own
# material list: None, Solidwood, Hollowwood, Solidmetal, Hollowmetal,
# Shortgrass, Longgrass, Dirt, Sand, Gravel, Glass, Concrete, Marble, Water,
# Snow, Ice, Flesh, Mud, BPGlass, DryLeaves, GreenLeaves, Fabric, Canvas,
# Rock, Rubber, Plastic, Tiles, Carpet, Plaster, ShatterGlass, Armor, Computer.
# Best-effort guess from common Source $surfaceprop values - NOT confirmed
# against SOF2 specifically, verify and override with --material-map.
Q3MAP_MATERIAL_MAP = {
    'metal': 'Solidmetal',
    'solidmetal': 'Solidmetal',
    'iron': 'Solidmetal',
    'metal_box': 'Hollowmetal',
    'metalpanel': 'Hollowmetal',
    'panel': 'Solidmetal',  # confirmed: used by metal/prodventa in real content
    'metalvent': 'Hollowmetal',
    'metalgrate': 'Hollowmetal',
    'chainlink': 'Hollowmetal',
    'grate': 'Hollowmetal',
    'combine_metal': 'Solidmetal',
    'flesh': 'Flesh',
    'blood': 'Flesh',
    'antlion': 'Flesh',
    'zombieflesh': 'Flesh',
    'alienflesh': 'Flesh',
    'glass': 'Glass',
    'glassbottle': 'Glass',
    'combine_glass': 'BPGlass',
    'wood': 'Solidwood',
    'wood_plank': 'Solidwood',
    'wood_panel': 'Solidwood',
    'wood_box': 'Hollowwood',
    'wood_crate': 'Hollowwood',
    'cardboard': 'Hollowwood',  # no true cardboard/paper option in SOF2's list - closest available
    'plaster': 'Plaster',
    'drywall': 'Plaster',
    'concrete': 'Concrete',
    'concrete_block': 'Concrete',
    'brick': 'Concrete',
    'stone': 'Rock',
    'rock': 'Rock',
    'boulder': 'Rock',
    'marble': 'Marble',
    'tile': 'Tiles',
    'dirt': 'Dirt',
    'gravel': 'Gravel',
    'sand': 'Sand',
    'grass': 'Shortgrass',
    'foliage': 'GreenLeaves',
    'mud': 'Mud',
    'snow': 'Snow',
    'ice': 'Ice',
    'water': 'Water',
    'slime': 'Water',
    'wade': 'Water',
    'plastic': 'Plastic',
    'plastic_barrel': 'Plastic',
    'rubber': 'Rubber',
    'rubbertire': 'Rubber',
    'upholstery': 'Fabric',
    'fabric': 'Fabric',
    'carpet': 'Carpet',
    'canvas': 'Canvas',
    'computer': 'Computer',
    'armor': 'Armor',
    'combine_armor': 'Armor',
    'default': None,  # Source's generic fallback surfaceprop - no useful signal
}


# Smoothing angle written as 'q3map_shadeAngle' on the terrain (and
# terrain-following overlay) shaders. q3map2 smooths the normals of adjacent
# meta triangles whose normals differ by LESS than this angle; with no shader
# in the map setting it at all, q3map2 skips smoothing entirely and reports
# "SmoothMetaTriangles: No smoothing angles specified, aborting" - which
# leaves converted displacement terrain lit facet-by-facet, each triangle
# visibly flat.
#
# Only terrain shaders get it, deliberately. Source computes smooth normals
# across a displacement (and across neighbouring displacements), so smoothing
# here restores what the original map had. Putting it on ordinary wall
# materials would be wrong: at a high angle it would smooth straight across a
# 90-degree brush corner and make hard edges look rounded.
TERRAIN_SHADE_ANGLE = [120.0]


# Lightmap luxel size, in world units, for the terrain and overlay-mesh
# shaders only. -light's cost scales with the luxel COUNT, i.e. with 1/N^2, and
# terrain is where that bites hardest: on de_dust2 the terrain model alone is
# 1.83e7 square map units, which at the global default of 16 is 71k luxels and
# at a -samplesize of 4 is 1.14 MILLION. It is also the surface that needs fine
# lightmaps least - sand lit by a sun is low-frequency, with no small-scale
# occluders to resolve - so giving it a coarser size of its own buys back a
# large multiple of compile time at almost no visual cost, while leaving the
# brushwork (where shadow detail actually shows) at the global size.
#
# 0 = leave it to the global -samplesize. Deliberately NOT derived from the
# VMF's own per-face lightmapscale values: measured on de_dust2 those are
# irreducibly per-face (919 of 941 non-default faces are one material, and even
# that material mixes 64 with 16), so nothing per-shader or per-entity can
# reproduce them. This value is an authoring choice instead.
TERRAIN_LIGHTMAP_SAMPLESIZE = [0.0]


# Light the terrain properly (real lightmaps and real shadows) instead of the
# vertex-lit, forced-full-sun form SOF2's own RMG terrain uses.
#
# The default form - "surfaceparm nolightmap" + "q3map_novertexshadows" +
# "q3map_forcesunlight" + "rgbGen vertex" - was copied from a confirmed-working
# SOF2 map, and it is right FOR THAT MAP: SOF2's RMG terrain is generated at
# runtime, so it cannot carry a baked lightmap and is deliberately painted at
# full sun. A converted Source displacement is the opposite: a static mesh that
# holds a lightmap perfectly well. Inheriting the RMG convention makes the
# ground ignore the sun entirely and render fullbright and shadowless, while
# every brush around it is lit for real - which reads in game as "the buildings
# are too dark" no matter how far the sun's intensity is raised, because the
# ground was never responding to the sun in the first place.
#
# The lightmapped form keeps vertex ALPHA for the material blend (that is what
# Source's WorldVertexTransition encodes, and it has nothing to do with
# lighting) but moves colour off the vertices: rgbGen identity, with a final
# $lightmap stage modulating the result. Requires the terrain misc_model to be
# flagged lightmapped, which vmf2map.py already does via spawnflags 4.
TERRAIN_LIGHTMAPPED = [False]


def terrain_rgbgen():
    """rgbGen for a terrain texture stage.

    Vertex colour carries the LIGHTING in the vertex-lit form, so it has to stop
    doing so once a lightmap supplies it - otherwise the two multiply together
    and the ground goes dark."""
    return '\t\trgbGen identity' if TERRAIN_LIGHTMAPPED[0] else '\t\trgbGen vertex'


def terrain_unlit_header_lines():
    """The three keywords that opt terrain out of lighting - or none of them."""
    if TERRAIN_LIGHTMAPPED[0]:
        return []
    return ['\tsurfaceparm nolightmap', '\tq3map_novertexshadows', '\tq3map_forcesunlight']


def terrain_lightmap_stage():
    """Final modulate-by-lightmap stage, for the lightmapped form only."""
    if not TERRAIN_LIGHTMAPPED[0]:
        return []
    return ['\t{', '\t\tmap $lightmap', '\t\trgbGen identity',
            '\t\tblendFunc GL_DST_COLOR GL_ZERO', '\t}']


# Decals, overlays and other translucent surfaces get no lightmap stage - they
# are blended over geometry that already has one, so a second $lightmap pass
# would multiply the wall behind them by its own lightmap twice. That leaves
# their lighting to rgbGen, and the default for a stage carrying a blendFunc is
# identityLighting: FULL BRIGHTNESS, ignoring every light in the map.
#
# Confirmed in game: with the sun at its derived 117 the whole map went dark
# while the road-line overlays stayed pure white, drawing a glowing grid across
# a black scene. It also explains the overlays appearing in front of a muzzle
# flash - they are fullbright AND "sort banner".
#
# rgbGen vertex fixes it: q3map2 bakes per-vertex lighting for every surface
# (the -light log's "IlluminateVertexes" stage, 160,669 vertexes on de_dust2),
# so the decal is modulated by real light. It is coarser than a lightmap -
# lighting is interpolated from the polygon's corners - but a decal is small
# and flat, so corner interpolation is close to right, and "slightly coarse" is
# an enormous improvement on "ignores the sun completely".
DECAL_LIGHTING = ['vertex']


def decal_rgbgen_line():
    """rgbGen for a blended/alpha-tested stage, or None to leave it fullbright."""
    mode = DECAL_LIGHTING[0]
    if not mode or mode == 'none':
        return None
    return '\t\trgbGen {}'.format(mode)


def lightmap_samplesize_line():
    """The q3map_lightmapSampleSize line for terrain shaders, or None."""
    size = TERRAIN_LIGHTMAP_SAMPLESIZE[0]
    if not size or size <= 0:
        return None
    # Absolute, in world units, unlike the _lightmapscale entity key which is a
    # multiplier - so this does not track -samplesize and is not affected by it.
    return '\tq3map_lightmapSampleSize {:g}'.format(size)


def shade_angle_line():
    """The q3map_shadeAngle line for terrain shaders, or None when disabled."""
    angle = TERRAIN_SHADE_ANGLE[0]
    if not angle or angle <= 0:
        return None
    return '\tq3map_shadeAngle {:.6g}'.format(angle)


# SOF2's whole material list, in its own order (q3map2 game_sof2: S_MAT_*,
# "q3map_material <name>" sets the low 5 surface-flag bits).
SOF2_MATERIALS = ('None', 'Solidwood', 'Hollowwood', 'Solidmetal', 'Hollowmetal', 'Shortgrass',
                  'Longgrass', 'Dirt', 'Sand', 'Gravel', 'Glass', 'Concrete', 'Marble', 'Water',
                  'Snow', 'Ice', 'Flesh', 'Mud', 'BPGlass', 'DryLeaves', 'GreenLeaves', 'Fabric',
                  'Canvas', 'Rock', 'Rubber', 'Plastic', 'Tiles', 'Carpet', 'Plaster',
                  'ShatterGlass', 'Armor', 'Computer')

# For a $surfaceprop the table above doesn't name: Source names its surface
# properties by what they are made of, so the material word in the name is
# the answer (metal_barrel, metalvehicle, wood_furniture, plastic_box,
# rubbertire, porcelain...). Checked in order; the first word found wins.
_SURFACEPROP_WORDS = (
    ('glassbottle', 'Glass'), ('glass', 'Glass'),
    ('computer', 'Computer'), ('electronic', 'Computer'),
    ('metal_box', 'Hollowmetal'), ('metal_barrel', 'Hollowmetal'), ('metalvent', 'Hollowmetal'),
    ('metalpanel', 'Hollowmetal'), ('metal_bouncy', 'Hollowmetal'), ('popcan', 'Hollowmetal'),
    ('canvas', 'Canvas'), ('can', 'Hollowmetal'), ('vehicle', 'Solidmetal'), ('metal', 'Solidmetal'), ('iron', 'Solidmetal'),
    ('steel', 'Solidmetal'), ('chain', 'Solidmetal'), ('grate', 'Hollowmetal'),
    ('wood_box', 'Hollowwood'), ('wood_crate', 'Hollowwood'), ('wood_furniture', 'Hollowwood'),
    ('wood', 'Solidwood'), ('cardboard', 'Hollowwood'), ('paper', 'Hollowwood'),
    ('plastic', 'Plastic'), ('rubber', 'Rubber'), ('tire', 'Rubber'),
    ('porcelain', 'Tiles'), ('ceramic', 'Tiles'), ('pottery', 'Tiles'), ('tile', 'Tiles'),
    ('concrete', 'Concrete'), ('brick', 'Concrete'), ('cinder', 'Concrete'), ('asphalt', 'Concrete'),
    ('plaster', 'Plaster'), ('drywall', 'Plaster'),
    ('marble', 'Marble'), ('stone', 'Rock'), ('rock', 'Rock'), ('boulder', 'Rock'),
    ('gravel', 'Gravel'), ('sand', 'Sand'), ('mud', 'Mud'), ('dirt', 'Dirt'),
    ('grass', 'Shortgrass'), ('foliage', 'GreenLeaves'), ('leaves', 'GreenLeaves'),
    ('carpet', 'Carpet'), ('rug', 'Carpet'), ('canvas', 'Canvas'), ('upholstery', 'Fabric'),
    ('fabric', 'Fabric'), ('cloth', 'Fabric'), ('cushion', 'Fabric'), ('mattress', 'Fabric'),
    ('flesh', 'Flesh'), ('water', 'Water'), ('slime', 'Water'), ('snow', 'Snow'),
)


def q3map_material_for(surfaceprop, material_map=None):
    """SOF2 material name for a Source $surfaceprop, or None: --material-map
    first, then Q3MAP_MATERIAL_MAP, then the material word in the name."""
    if not surfaceprop:
        return None
    key = surfaceprop.strip().lower()
    if material_map and key in material_map:
        return material_map[key]
    if key in Q3MAP_MATERIAL_MAP:
        return Q3MAP_MATERIAL_MAP[key]
    for word, mat in _SURFACEPROP_WORDS:
        if word in key:
            return mat
    if key == 'ice':            # not a substring test: office, device, slice...
        return 'Ice'
    return None


def convert_q3map_material(surfaceprop, material_map, unmapped_counter):
    if not surfaceprop:
        return None
    mat = q3map_material_for(surfaceprop, material_map)
    if mat is None:
        unmapped_counter[surfaceprop] += 1
    return mat


SHOOT_CLIP_PREFIX = 'vmf2map/shoot_'


def shoot_clip_name(material):
    """Map-texture name (no textures/ prefix) of the collision shader for a
    SOF2 material - see generate_shoot_clip_shaders."""
    return SHOOT_CLIP_PREFIX + material.lower()


def generate_shoot_clip_shaders():
    """One invisible collision shader per SOF2 material, for phy2clip.py's
    model collision. Built to the same flags as SOF2's own tools/_shoot (read
    out of a compiled .bsp: surface 0x200000 = nodraw, contents 0x800000B0 =
    translucent | shotclip | monsterclip | playerclip) plus that material's
    q3map_material, which is all that SOF2's own _shoot_metal/_wood/_stone add.
    SOF2 ships only a few of those; this gives every material one, so a prop's
    collision rings and marks like the prop."""
    blocks = []
    for mat in SOF2_MATERIALS:
        lines = [
            'textures/' + shoot_clip_name(mat),
            '{',
            '\tqer_editorimage textures/tools/_shoot',
            '\tqer_trans 0.4',
            '\tsurfaceparm nodraw',
            '\tsurfaceparm trans',
            '\tsurfaceparm playerclip',
            '\tsurfaceparm monsterclip',
            '\tsurfaceparm shotclip',
        ]
        if mat != 'None':
            lines.append('\tq3map_material {}'.format(mat))
        lines.append('}')
        blocks.append('\n'.join(lines))
    return blocks


# ==============================================================================
# Tool-material detection (these are skipped - already handled directly by
# vmf2map.py's DEFAULT_TOOL_TEXTURES table, no shader script needed)
# ==============================================================================
TOOL_SHADER_CLASSES = {'tools'}
TOOL_NAME_PATTERNS = re.compile(
    r'^(tools[/\\])|(nodraw|toolsnodraw|toolsclip|toolstrigger|toolshint|'
    r'toolsskip|toolsskybox|toolsareaportal|toolsinvisible|toolsorigin)$',
    re.IGNORECASE)


def is_tool_material(shader_class, rel_path):
    if shader_class and shader_class.strip().lower() in TOOL_SHADER_CLASSES:
        return True
    name = os.path.basename(rel_path)
    if TOOL_NAME_PATTERNS.match(rel_path) or TOOL_NAME_PATTERNS.match(name):
        return True
    return False


# ==============================================================================
# Sky material handling
# ==============================================================================
# Source's "Sky" shader class VMTs are per-face (bk/dn/ft/lf/rt/up) cubemap
# textures for the background skybox. idTech3 does NOT want one shader per
# face - it wants a single shader on the sky brush using a "skyParms
# <basename> <cloudheight> -" directive, which auto-loads
# <basename>_bk.*, <basename>_dn.*, <basename>_ft.*, <basename>_lf.*,
# <basename>_rt.*, <basename>_up.* (note the underscore before the suffix -
# Source's own basetexture names for these usually do NOT have that
# underscore, e.g. "skybox/vertigobk", so the actual image files will need
# renaming to "vertigo_bk.png" etc. for skyParms to find them).
SKY_SUFFIX_RE = re.compile(r'^(?P<base>.+?)(?P<suffix>bk|dn|ft|lf|rt|up)$', re.IGNORECASE)


def is_sky_material(shader_class, rel_path):
    """Confirmed against real content: Source skybox VMTs use an ordinary
    shader class (unlitGeneric + $nofog 1 + $ignorez 1), never a distinct
    'Sky' class - checking shader_class alone missed every real skybox VMT
    and let all 6 faces fall through as plain wall-texture shaders instead
    of being grouped into one skyParms shader. The reliable signal is the
    material's own path: Source's skybox/ folder convention is what the
    engine itself uses to find these textures, so it's a better test than
    any per-VMT content heuristic (which could false-positive on unrelated
    materials that also happen to use $nofog/$ignorez)."""
    if shader_class and shader_class.strip().lower() == 'sky':
        return True
    normalized = rel_path.strip().replace('\\', '/').lower()
    return normalized.startswith('skybox/')


def sky_group_key(basetexture_value):
    v = basetexture_value.strip().replace('\\', '/').lower()
    m = SKY_SUFFIX_RE.match(v)
    if m:
        return m.group('base').rstrip('_')
    return v


def parse_kv_blocks(tokens):
    """Generic KeyValues parse for a file with MULTIPLE top-level blocks
    (a VMF's 'world'/'entity' blocks), unlike parse_vmt's single-root-key
    assumption. Reuses the identical block-parsing logic."""
    pos = [0]

    def parse_block():
        node = OrderedDict()
        while pos[0] < len(tokens):
            tok = tokens[pos[0]]
            if tok == '}':
                pos[0] += 1
                return node
            key = tok
            pos[0] += 1
            if pos[0] >= len(tokens):
                break
            nxt = tokens[pos[0]]
            if nxt == '{':
                pos[0] += 1
                value = parse_block()
            else:
                value = nxt
                pos[0] += 1
            node.setdefault(key, []).append(value)
        return node

    return parse_block()


def get1(node, key, default=None):
    v = node.get(key)
    return v[0] if v else default


# --- EXPOSURE: Source's lighting pipeline, reproduced in q3map2 --------------
#
# NOTE: vmf2map.py carries an identical copy of this model (it writes the
# worldspawn fill and the point lights; this file writes the sun and sky). The
# two MUST agree - change both or neither.
#
# Every number below is taken from the code of the tools involved, not tuned.
#
# SOURCE (source-sdk-2013, src/utils/vrad + src/mathlib):
#   * lightmap.cpp LightForString:  linear = pow(c/255, 2.2) * brightness
#     "// convert to linear". VRAD lights in LINEAR space.
#   * color_conversion.cpp BuildGammaTable: the engine shows a lightmap as
#     pow(linear, 1/gamma), gamma = 2.2 by default. Linear light, gamma display.
#   * GatherSampleAmbientSkySSE: _ambient is a SKY DOME - hemisphere samples,
#     each traced for sky visibility and cosine-weighted, normalised so a fully
#     open floor receives exactly _ambient. An open wall gets ~half (its lower
#     hemisphere faces the ground), a covered ceiling gets none.
#   * ParseLightGeneric: a point light's stated brightness is its value at 100
#     units ("// scale intensity for unit 100 distance").
#
# Q3MAP2 (both GtkRadiant 1.6's and netradiant-custom's, checked in source):
#   * ColorToBytes: byte = 255 * pow(value/255, 1/lightmapGamma). With SOF2's
#     game default gamma 1.0 (game_sof2.h) q3map2 writes LINEAR light straight
#     to the lightmap and the engine displays it as-is - so compiling the same
#     physical light gives far darker shadows than Source shows. -gamma 2.2
#     reproduces Source's pow(linear, 1/2.2) exactly; both compilers implement
#     it the same way and both apply it to the light grid too.
#   * q3map_skylight <value> <iterations>: a dome of occluded directional suns
#     - the same estimator as Source's sky ambient. Its response on an open
#     floor is computed below from q3map2's own dome construction.
#   * point lights: light * 7500 / d^2 (pointScale).
#
# So the conversion is: take Source's light in linear units, scale every light
# by ONE exposure factor k, and let -gamma 2.2 do what Source's display does.
# k is the only choice, fixed by TARGET_SUNLIT_LEVEL - what a sun-facing floor
# in the open should read on screen. Everything else keeps the ratios Source
# authored.
#
# !! The map MUST be lit with -gamma 2.2. Without it every value here is too
# !! dark in the shadows, by design.
SOURCE_GAMMA = 2.2   # Source's display gamma (BuildGammaTable); the COMPILE gamma is derived below

# On screen, 0-255. 235 leaves headroom under the clamp. SOF2 has no HDR and
# (as far as can be measured) no overbright, so Source's sunlit floor - which
# its LDR overbright lets go past 1.0 - has to land here instead.
TARGET_SUNLIT_LEVEL = 235.0

# q3map_skylight iterations. Source's estimator gives an open wall exactly half
# an open floor; q3map2's dome gives 0.479 at 3, 0.450 at 4, 0.407 at 6 (its
# rings are spaced evenly in elevation, not solid angle). 4 is within 10% of
# Source at 37 suns; 3 is closer on paper but its 17 suns show as visible
# stepped shadow copies.
SKYLIGHT_ITERATIONS = 4

# Source's radiosity (VRAD bounces until converged) is what lights a room the
# sky cannot see; q3map2 can't reproduce that without -bounce, so a flat
# worldspawn ambient stands in for it. MEASURED, not estimated: in CS:S's own
# compiled de_dust2.bsp, downward-facing surfaces - which can only be reached by
# bounce - have a median on-screen level of 0.293, i.e. linear 0.0672, against
# an _ambient luminance of 1.539 linear. Bounce = 0.0437 x _ambient.
# (The first version of this estimated 0.175 - four times too bright - which is
# why test D's ceilings sat at 0.31 of sunlit where CS:S has 0.17.)
# Measured again with -bounce 1 (same CS:S reference, position-matched luxels
# between a bounced and an unbounced bake): with one real bounce doing part of
# the work, the fill that best matches CS:S's walls and ceilings is 0.0130 x
# _ambient - walls then land at 0.20/0.52/0.87/1.12/1.25 (p10..p90) against
# CS:S's 0.21/0.49/0.81/1.05/1.20. Ceilings stay a little dark (median 0.20 vs
# 0.29): one bounce barely reaches them, VRAD bounces until converged.
COMPILE_BOUNCES = 1
BOUNCE_FILL = 0.0130 if COMPILE_BOUNCES >= 1 else 0.0437

# How SOF2 displays a lightmap byte - MEASURED, from the game's own cvars:
# r_mapOverBrightBits 1, r_overBrightBits 0. SOF2's renderer is Quake 3's here:
# R_ColorShiftLightingBytes shifts every lightmap, vertex-colour and light-grid
# byte left by (mapOverBrightBits - overBrightBits) = 1 AT LOAD TIME and clamps
# to 255 with colour normalisation. So on screen:
#     multiplier = min(1.0, 2 * byte/255)
# Bytes above 128 are all full brightness. Test E stored its open ground at
# 0.6-0.9 and every one of those came out clamped at 1.0 - sunlit and shaded
# sand identical, which is why the courtyard read flat.
# (A player running fullscreen with r_overBrightBits 1 gets shift 0 plus a x2
# hardware gamma instead - the same factor of 2, so the same result as long as
# nothing is stored above 128, which this model guarantees.)
ENGINE_LIGHTMAP_OVERBRIGHT = 2

# What the SHADERS add on top. Clamping at load caps the lightmap at 1.0x the
# texture - but CS:S's dust2 needs 1.75x, and unlike SOF2, Source clamps the
# FINAL PIXEL, not the lightmap. Measured from its .bsp (texture reflectivity x
# lightmap): only 0.3% of CS:S's luxels clip to white, so its sunlit sand really
# is ~0.97 on screen, where a 1.0x-capped lightmap can never take that texture
# past 0.58. No compile setting can close that gap; the shader can. The lightmap
# stage blends with GL_DST_COLOR GL_SRC_COLOR - dst*src + src*dst = 2 * texture
# * lightmap, clamped on the final pixel exactly where Source clamps (the
# standard Quake 3 "2x modulate", as id's own detail-texture stages use).
# Vertex-lit surfaces get the same 2x by repeating their stage additively.
# Lightmap range on screen becomes 0-2.0x; dust2's 1.75 fits, so the
# conversion is exact 1:1 at Source's own gamma.
SHADER_LIGHTMAP_SCALE = 2

# q3map2 pointScale, and Source's 100-unit reference distance.
Q3_POINT_SCALE = 7500.0
SOURCE_POINT_REFERENCE = 100.0


def _source_linear(rgb, brightness):
    """VRAD's LightForString: per-channel linear light."""
    return [((max(c, 0.0) / 255.0) ** 2.2) * brightness for c in rgb]


def _luminance(v):
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def skylight_floor_response(iterations=SKYLIGHT_ITERATIONS):
    """What an open, up-facing surface receives from q3map_skylight <value>,
    per unit of value - built the way CreateSkyLights builds its dome (same in
    GtkRadiant and netradiant-custom for the default 0-90 horizon)."""
    step = 90.0 / iterations
    angle_steps = (iterations - 1) * 4
    total = 0.0
    count = 0
    for i in range(iterations - 1):
        e = math.radians(step * (0.5 + i))
        total += angle_steps * math.sin(e)
        count += angle_steps
    total += 1.0      # the zenith sun
    count += 1
    return total / count


# --- Light headroom ------------------------------------------------------------
#
# q3map2 gives every entity and surface light its "Fast" flag (light.cpp:
# LightFlags::DefaultQ3A, whatever the command line says) and then drops, luxel
# by luxel, any contribution of <= 1.0 (on its 0-255 linear scale) before
# -gamma. At -gamma 2.2 a dropped 1.0 would have shown as 20/255, so every
# light ends in a hard cut instead of VRAD's soft tail. Measured with this
# q3map2: a light whose floor should read 15.9 reads 0; at 21 and up it is
# exact. On de_dust2 the tunnel lamps (brightness ~50) are cut off about 150
# Source units out - their pools never reach the floor, which is why the
# tunnels read 0.57-0.74 of CS:S while sunlit sand is at 0.94.
#
# The cure is exact, not a boost: every light (k) is made LIGHT_HEADROOM times
# stronger, and the compile undoes it - -brightness X^(-1/G) after gamma for
# lightmaps (gamma is a power, so (X*L)^(1/G) * X^(-1/G) = L^(1/G)),
# -vertexscale 1/X and -gridscale / X before it for vertex light and the light
# grid. Per-shader q3map_lightmapBrightness / q3map_vertexScale multiply those
# defaults, so no shared shader changes. What stays cut is <= 1/X: at 16 that
# is 6.5/255 after gamma, against 20 now (16 recovers 0.96 of the lost light
# on de_dust2, 64 would give 1.00 for twice the light reach and compile time).
LIGHT_HEADROOM = 16.0

# The worldspawn "ambient" stood in for bounce light that q3map2 lost: until
# LIGHT_HEADROOM, its cutoff dropped most of the bounce pass's dim diffuse
# lights, so k * BOUNCE_FILL was added as a flat floor. With the headroom the
# real bounce arrives and the floor counted twice: measured against Valve's
# lightmaps (luxel by luxel, the fill taken back out), dim interiors on
# de_dust2 went 1.46 -> 1.11 and dark corners 4.91 -> 2.11 with no fill at all,
# sunlit and lamp-lit unchanged; de_cbble 1.33 -> 1.24 and 2.00 -> 1.55.
# BOUNCE_FILL stays as the solve's anchor for Source's bounce level.
AMBIENT_FILL = 0.0


def solve_exposure(sun_rgb, sun_brightness, amb_rgb, amb_brightness, elevation):
    """Return the q3map2 values - and the -gamma to compile with - that put this
    light_environment on SOF2's screen as close to CS:S as SOF2 can show.

    Source shows a lightmap as M = pow(linear, 1/2.2), and on de_dust2 M reaches
    1.67 on sunlit sand. SOF2 shows min(1.0, 2*byte/255): nothing above 1.0. So
    the range has to be compressed, and the curve is fixed by two measured
    anchors instead of by eye:

      1. the sunlit open floor lands on TARGET_SUNLIT_LEVEL - the top of what
         SOF2 can show, with a little headroom under the clamp;
      2. a room lit only by bounce light stays EXACTLY as bright as in CS:S
         (0.293 on screen, measured from its compiled .bsp).

    One power curve through both, screen = c * M^p, spreads everything else.
    A power curve on screen is simply a different compile gamma: with
    -gamma G, q3map2 stores 255 * pow(k*L/255, 1/G), so G = 2.2/p. If Source's
    own sunlit level already fits, p = 1, G = 2.2 and the match is exact.
    """
    S = _source_linear(sun_rgb, sun_brightness)
    A = _source_linear(amb_rgb, amb_brightness)
    incidence = max(math.sin(math.radians(max(0.0, min(90.0, elevation)))), 0.30)
    floor_src = (_luminance(S) * incidence + _luminance(A) * (1.0 + BOUNCE_FILL)) / 255.0
    bounce_src = max(_luminance(A) * BOUNCE_FILL / 255.0, 1e-6)
    m_sun = floor_src ** (1.0 / SOURCE_GAMMA)
    m_bounce = bounce_src ** (1.0 / SOURCE_GAMMA)
    # the lightmap is clamped at 1.0 after the engine's doubling, then scaled by
    # the shader: SHADER_LIGHTMAP_SCALE is the most a surface can show
    max_m = float(SHADER_LIGHTMAP_SCALE)
    target = TARGET_SUNLIT_LEVEL / 255.0 * max_m
    if m_sun <= target or m_sun <= m_bounce * 1.01 or target <= m_bounce:
        p = 1.0
        c = 1.0 if m_sun <= target else target / m_sun
    else:
        p = math.log(target / m_bounce) / math.log(m_sun / m_bounce)
        p = min(1.0, max(0.35, p))
        c = target / (m_sun ** p)
    gamma = SOURCE_GAMMA / p
    O = float(ENGINE_LIGHTMAP_OVERBRIGHT * SHADER_LIGHTMAP_SCALE)
    # q3map2 value = k * Source linear (255 units), with the light headroom in it
    k = (c / O) ** gamma * LIGHT_HEADROOM
    s_peak = max(S + [1e-9])
    a_peak = max(A + [1e-9])
    screen = lambda lin255: min(max_m, c * (max(lin255, 0.0) / 255.0) ** (p / SOURCE_GAMMA))
    return {
        'k': k,
        'compile_gamma': round(gamma, 2),
        # the light grid lights players and model_static props with SOF2's normal
        # 1x shaders, so it must not carry the shaders' 2x: scaling it up by
        # SHADER^gamma before gamma (q3map2 -gridscale) puts it back on 1x.
        'grid_scale': round(float(SHADER_LIGHTMAP_SCALE) ** gamma / LIGHT_HEADROOM, 4),
        # undo LIGHT_HEADROOM: after gamma for lightmaps, before it for vertices
        'brightness': round(LIGHT_HEADROOM ** (-1.0 / gamma), 5),
        'vertex_scale': round(1.0 / LIGHT_HEADROOM, 5),
        'curve_p': p,
        'exact': p >= 0.999 and c >= 0.999,
        'sun_intensity': k * s_peak,
        'sun_color': [x / s_peak for x in S],
        'skylight_value': k * a_peak / skylight_floor_response(),
        'sky_color': [x / a_peak for x in A],
        'bounce_fill': k * a_peak * BOUNCE_FILL * AMBIENT_FILL,
        # on-screen multipliers of the texture - same scale as the CS:S measurement
        'predicted': {
            'sunlit floor': screen(_luminance(S) * incidence + _luminance(A) * (1 + BOUNCE_FILL)),
            'open floor in shadow': screen(_luminance(A) * (1 + BOUNCE_FILL)),
            'open wall in shadow': screen(_luminance(A) * (0.5 + BOUNCE_FILL)),
            'covered room': screen(_luminance(A) * BOUNCE_FILL),
        },
        'source': {
            'sunlit floor': m_sun,
            'open floor in shadow': (_luminance(A) * (1 + BOUNCE_FILL) / 255.0) ** (1 / SOURCE_GAMMA),
            'open wall in shadow': (_luminance(A) * (0.5 + BOUNCE_FILL) / 255.0) ** (1 / SOURCE_GAMMA),
            'covered room': m_bounce,
        },
    }


def parse_light_environment(vmf_path):
    """Pull the map's light_environment entity (Source's sun/skylight
    definition) and return the Q3-side sun parameters derived from it, or
    None if the VMF has no such entity.

    Conversion notes - confidence varies per value:
    - elevation = -pitch: SOLID. Source's own documented convention for
      light_environment pitch is -90 = straight down (noon), 0 = horizontal
      (sunrise/sunset); Q3's altitude is 0 = horizon, 90 = noon - an exact
      sign-flip of the same physical quantity, not a guess.
    - degrees = yaw + 180: BEST-EFFORT, NOT independently verified. Source's
      "angles" yaw describes the direction light TRAVELS; Q3's "degrees"
      describes where the sun SITS (roughly opposite, hence +180). This is
      the commonly-used community conversion, but unlike elevation there is
      no directly-documented Source constant to confirm the rotational
      handedness against - check the sun's visual direction in-game (does
      it match shadow directions you'd expect from the original map?) and
      adjust if it looks mirrored.
    - keyword: q3map_sun. SOLID, and worth stating because getting it wrong
      is silent: a bare "sun" is not a q3map2 shader keyword, and q3map2
      ignores unrecognised keywords without a warning, so the shader reads as
      though the map is sunlit while compiling with no sun whatsoever.
    - color: q3map2's own doc says "Color is normalized to a 0.0 to 1.0 range,
      so it doesn't matter what range you use", so the 0-255 triple needs no
      conversion. It is written normalized anyway, since a 0-255 triple in a
      field that looks like a 0..1 one invites exactly the misreading it
      deserves; the raw values go in a comment above it.
    - intensity and the worldspawn ambient that goes with it: solved for a
      target lightmap level rather than scaled from Source's brightness,
      because the two engines' brightness units have no conversion between
      them. See derive_exposure.
    """
    try:
        with open(vmf_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except OSError:
        return None
    root = parse_kv_blocks(tokenize(text))
    for ent in root.get('entity', []):
        if not isinstance(ent, dict):
            continue
        if (get1(ent, 'classname') or '').lower() != 'light_environment':
            continue
        pitch = get1(ent, 'pitch')
        angles = get1(ent, 'angles')
        light = get1(ent, '_light')
        ambient = get1(ent, '_ambient')
        try:
            pitch_v = float(pitch) if pitch is not None else 0.0
            yaw_v = float(angles.split()[1]) if angles else 0.0
            lparts = [float(x) for x in light.split()] if light else [255, 255, 255, 200]
            aparts = [float(x) for x in ambient.split()] if ambient else None
        except (ValueError, IndexError):
            return None
        elevation = -pitch_v
        degrees = (yaw_v + 180.0) % 360.0
        r, g, b = lparts[0:3]
        brightness = lparts[3] if len(lparts) > 3 else 200.0
        if not (aparts and len(aparts) > 3):
            # VRAD's own fallback when _ambient is missing: half the sun.
            aparts = [r, g, b, brightness * 0.5]
        raw_ambient = aparts[3]
        sol = solve_exposure([r, g, b], brightness, aparts[0:3], aparts[3], elevation)
        return {
            'r': r, 'g': g, 'b': b,
            'sun_color': sol['sun_color'],
            'sky_color': sol['sky_color'],
            'intensity': float('%.3g' % sol['sun_intensity']),
            'skylight_value': float('%.3g' % sol['skylight_value']),
            'degrees': round(degrees, 1),
            'elevation': round(elevation, 1),
            # Reported so it can be compared with what vmf2map.py writes.
            'bounce_fill': sol['bounce_fill'],
            'exposure_k': sol['k'],
            'compile_gamma': sol['compile_gamma'],
            'grid_scale': sol['grid_scale'],
            'brightness': sol['brightness'],
            'vertex_scale': sol['vertex_scale'],
            'source_levels': sol['source'],
            'predicted': sol['predicted'],
            # Unrounded, so --light-scale rounds ONCE at the end.
            'intensity_raw': sol['sun_intensity'],
            'raw_brightness': brightness,
            'raw_ambient': raw_ambient,
            'ambient_rgb': (aparts[0:3] if (aparts and len(aparts) >= 3) else None),
        }
    return None


def generate_sky_shaders(sky_materials, sun_data=None):
    """sky_materials: list of (rel_path, basetexture) for every 'Sky'-class
    VMT found. Groups them by basename and emits one skyParms shader per
    group, plus a list of (shader_name, basename, face_count) for reporting."""
    groups = OrderedDict()
    for rel_path, basetexture in sky_materials:
        key = sky_group_key(basetexture)
        groups.setdefault(key, []).append(rel_path)

    blocks = []
    report = []
    for base_key, rel_paths in groups.items():
        shader_name = with_textures_prefix(base_key)
        sky_basename = shader_name
        lines = [
            shader_name,
            "{",
        ]
        if sun_data:
            # Sky fill as Source computes it: _ambient is a dome whose light
            # reaches a surface only along directions that actually see sky
            # (VRAD GatherSampleAmbientSkySSE). q3map_skylight is the same
            # estimator in q3map2 - a dome of occluded directional suns - so
            # open ground gets the full fill, walls about half, and roofed
            # spaces none, exactly as in the original.
            #
            # This is the second attempt at skylight in this converter. The
            # first made things worse, and that was not skylight's fault: at
            # the time the brush planes sat off q3map2's 1/8 grid, which filed
            # every downward-facing sky face inside its own brush, so the
            # dome's suns could not see the sky ceiling any more than the real
            # sun could (see vmf2map.py snap_brush_planes_to_grid).
            #
            # q3map_lightRGB carries _ambient's own colour (linearised, as VRAD
            # does); without it q3map2 averages the sky IMAGE for the dome's
            # colour instead.
            #
            # !! Compile -light with the -gamma written below; it is derived. !!
            if sun_data.get('surfacelight_override'):
                lines.append("\tq3map_surfacelight\t{:g}".format(
                    sun_data['surfacelight_override']))
                lines.append("\tq3map_lightsubdivide\t512")
            p = sun_data['predicted']
            sl = sun_data['source_levels']
            lines.append("\t// Source light_environment, converted in linear space (VRAD's own")
            lines.append("\t// pow 2.2), k = {:.5f}.  COMPILE: -bsp ... -rename   -light ... -gamma {:g}"
                         " -gridscale {:g} -brightness {:g} -vertexscale {:g} -bounce 1".format(
                             sun_data['exposure_k'], sun_data['compile_gamma'], sun_data['grid_scale'],
                             sun_data['brightness'], sun_data['vertex_scale']))
            lines.append("\t// (-brightness / -vertexscale undo the x{:g} light headroom that keeps q3map2 from"
                         " cutting dim light off - see LIGHT_HEADROOM)".format(LIGHT_HEADROOM))
            lines.append("\t// (-gridscale is netradiant-only; GtkRadiant 1.6's q3map2 lacks it and players"
                         " will read darker)")
            lines.append("\t// On screen (x texture), SOF2 vs CS:S: sunlit {:.2f}/{:.2f}  shade "
                         "{:.2f}/{:.2f}  wall {:.2f}/{:.2f}  roofed {:.2f}/{:.2f}".format(
                             p['sunlit floor'], sl['sunlit floor'],
                             p['open floor in shadow'], sl['open floor in shadow'],
                             p['open wall in shadow'], sl['open wall in shadow'],
                             p['covered room'], sl['covered room']))
            lines.append("\t// sky fill from _ambient {:g}: occluded dome, like VRAD's".format(
                sun_data['raw_ambient']))
            lines.append("\tq3map_lightRGB {:.4g} {:.4g} {:.4g}".format(*sun_data['sky_color']))
            # "0 90 0" = horizon_min, horizon_max, sample_color. netradiant's
            # q3map2 defaults sample_color to TRUE (q3map2.h skylight_t), which
            # colours every dome sun from the six sky-box images and ignores
            # q3map_lightRGB - measured on de_dust2 G: open shade came out
            # 1 : 1.05 : 1.02 (blue sky) where _ambient asks for 1 : 0.885 : 0.593,
            # which is most of why the map read grey next to CS:S's gold.
            # GtkRadiant's q3map2 reads only the first two numbers and moves on.
            lines.append("\tq3map_skyLight {:g} {} 0 90 0".format(sun_data['skylight_value'],
                                                            SKYLIGHT_ITERATIONS))
            lines.append("\t// sun from _light {:g} {:g} {:g} brightness {:g}".format(
                sun_data['r'], sun_data['g'], sun_data['b'], sun_data['raw_brightness']))
            # q3map_sun rather than the bare "sun" Raven's own shaders use - both
            # are accepted in the same shaders.c branch. q3map_sunExt adds
            # deviance/samples for soft shadow edges; Source has no key to
            # convert that from (its softness is runtime), so it stays opt-in.
            sc = sun_data['sun_color']
            sun_args = "{:.4g} {:.4g} {:.4g} {:g} {:g} {:g}".format(
                sc[0], sc[1], sc[2],
                sun_data['intensity'], sun_data['degrees'], sun_data['elevation'])
            deviance = sun_data.get('deviance') or 0.0
            samples = sun_data.get('sun_samples') or 16
            if deviance > 0:
                lines.append("\tq3map_sunExt {} {:g} {}".format(sun_args, deviance, samples))
                lines.append("\t// last two values are deviance (degrees of angular spread, so")
                lines.append("\t// softer shadow edges) and samples. For hard edges and a much")
                lines.append("\t// faster compile, drop back to: q3map_sun {}".format(sun_args))
            else:
                lines.append("\tq3map_sun {}".format(sun_args))
                lines.append("\t// for soft shadow edges, replace the line above with")
                lines.append("\t// q3map_sunExt {} 2 16".format(sun_args))
                lines.append("\t// (deviance 2 degrees, 16 samples - costs compile time)")
        lines += [
            "\tsurfaceparm noimpact",
            "\tsurfaceparm nolightmap",
            "\tsurfaceparm sky",
            "\tqer_editorimage textures/tools/editor_images/qer_sky",
            "\tnopicmip",
            "\tnomipmaps",
            "\t// the game loads {}_rt/_bk/_lf/_ft/_up/_dn.tga - exactly".format(sky_basename),
            "\t// .tga, and draws no sky at all if _rt fails. vtf_to_png_jpg.py",
            "\t// writes the faces that way (HDR decoded, half-height sides",
            "\t// unfolded per the face VMT's $basetexturetransform).",
            "\tskyParms {} 512 -".format(sky_basename),
            "}",
        ]
        blocks.append(emit_dual_shader_blocks(base_key, lines))
        # ParseSkyParms formats "%s_%s.tga" into a MAX_QPATH (64) buffer in
        # id's code (retail SOF2 builds the same "%s_%s.tga" name):
        # a longer name is cut short and no face is ever found.
        if len(sky_basename) + len('_rt.tga') >= 64:
            print("WARNING: sky name '{}' is too long for the game's 64-character "
                  "image path; its faces will not load. Shorten the skybox name."
                  .format(sky_basename))
        report.append((shader_name, sky_basename, len(rel_paths)))
    return blocks, report



DECAL_MODULATE_SEEN = []


def convert_vmt_to_shader(rel_path, shader_class, body, surfaceprop_map,
                           unmapped_surfaceprops, material_map, unmapped_materials, warnings):
    basetexture = get_param(body, 'basetexture')
    if not basetexture:
        # Report WHY, not just that it failed - the cause differs per file and
        # determines the fix. Common ones: an unresolved 'patch' whose include
        # target is missing from this materials tree (so the base $basetexture
        # was never inherited), or a shader type that legitimately has no
        # $basetexture at all (e.g. a pure $bumpmap/water/refract material).
        keys_present = ', '.join(sorted(k for k in body.keys() if k.startswith('$'))) or '(no $ params at all)'
        cls = shader_class or '(unknown)'
        extra = ''
        if cls.strip().lower() == 'patch':
            include_value = get_param(body, 'include')
            extra = (" - this is an UNRESOLVED 'patch' material; its include target "
                     "{} wasn't found in this materials tree, so the base material's "
                     "$basetexture couldn't be inherited (extract the included .vmt "
                     "into --materials-dir to fix)".format(
                         repr(include_value) if include_value else "(no include key)"))
        warnings.append("'{}' has no $basetexture, skipped. shader class={}, params present: {}{}".format(
            rel_path, cls, keys_present, extra))
        return None

    # A .vmt's OWN file location and its $basetexture value aren't always
    # the same path - Source content often has a per-map "override" .vmt
    # that reuses a shared texture living somewhere else entirely (e.g. a
    # map-specific materials/props/de_train/truck001c_01.vmt whose
    # $basetexture points at the shared models/props_vehicles/truck001c_01
    # texture). The shader IDENTITY still has to be the .vmt's own file
    # location, though, not $basetexture: Source resolves a model's
    # cdmaterials + texture name to the .vmt's OWN path first (that's the
    # file it actually loads), and only follows $basetexture as a second,
    # internal step to find the real pixel data - so that's exactly what
    # md3_shader_autofix.py independently derives from the .mdl and burns
    # into the .md3's shader field, and this has to match it exactly, or
    # the shader can never activate no matter how correct its contents
    # are. (An earlier version of this function switched to $basetexture
    # for model materials, which was wrong for exactly this override case
    # - confirmed by the .md3 side needing the .vmt's own path all along.)
    # $basetexture is still exactly right for the ACTUAL image location
    # (map_path, below), since that's genuinely wherever the pixels are.
    shader_name = with_textures_prefix(rel_path)

    # idTech3 shader "map"/"qer_editorimage" references never include a file
    # extension - the engine tries known image extensions automatically at
    # load time. Confirmed against SOF2's own shipped shader scripts (zero
    # exceptions across ~80k lines of reference shaders).
    #
    # map_path (always textures/-prefixed) is where the image actually
    # lives on disk, and is used for BOTH the map stage and
    # qer_editorimage - a bare "models/..." path (no textures/ prefix)
    # doesn't resolve to any real image file, so it can't drive the editor
    # preview even though it's the correct value for the shader name itself.
    map_path = physical_texture_path(basetexture)

    # Source's DecalModulate does not draw its texture - it MULTIPLIES what is
    # already on screen by 2 x texture, so mid-grey (128) leaves the wall
    # untouched, darker texels darken it and lighter ones brighten it. It is how
    # dust2's stains, dirt and window insets are made. Drawn as an ordinary
    # alpha-blended decal, its mostly-light-grey texture shows up as a pale
    # patch, and against a correctly 2x-lit sunlit wall it clips to white.
    # The idTech3 equivalent is the same 2x modulate the lightmap stage uses:
    # GL_DST_COLOR GL_SRC_COLOR = dst*src + src*dst. It needs no lighting of its
    # own - it modulates the already-lit surface underneath, exactly like Source.
    if (shader_class or '').strip().lower() == 'decalmodulate':
        DECAL_MODULATE_SEEN.append(rel_path)
        lines = [shader_name, "{",
                 '\tqer_editorimage {}'.format(map_path),
                 '\t// Source DecalModulate: multiplies the surface under it by 2 x texture',
                 '\tpolygonOffset',
                 '\tsurfaceparm trans',
                 '\tsurfaceparm nonsolid',
                 '\tsurfaceparm nomarks',
                 '\tsurfaceparm nolightmap',
                 '\tq3map_nolightmap',
                 '\tsort banner',
                 '\t{',
                 '\t\tmap {}'.format(map_path),
                 '\t\tblendFunc GL_DST_COLOR GL_SRC_COLOR',
                 '\t\trgbGen identity',
                 '\t}',
                 "}"]
        return emit_dual_shader_blocks(rel_path, lines)

    translucent = is_truthy(get_param(body, 'translucent'))
    alphatest = is_truthy(get_param(body, 'alphatest'))
    additive = is_truthy(get_param(body, 'additive'))
    nocull = is_truthy(get_param(body, 'nocull'))
    selfillum = is_truthy(get_param(body, 'selfillum'))
    surfaceprop = get_param(body, 'surfaceprop')

    lines = []
    lines.append(shader_name)
    lines.append("{")
    lines.append('\tqer_editorimage {}'.format(map_path))

    surfaceparm = convert_surfaceparm(surfaceprop, surfaceprop_map, unmapped_surfaceprops)
    if surfaceparm:
        lines.append('\tsurfaceparm {}'.format(surfaceparm))

    q3map_material = convert_q3map_material(surfaceprop, material_map, unmapped_materials)
    if q3map_material:
        lines.append('\tq3map_material {}'.format(q3map_material))

    if translucent:
        # trans only - NOT nonsolid. Translucency says nothing about collision
        # in Source, and q3map2 gives a whole brush the contents of its first
        # non-solid side (map.cpp SetBrushContents: contentShader), so one
        # translucent face made the entire brush walk-through. On de_dust2
        # that was 8 stair brushes carrying a dev_measurestairs01a face
        # ("stairs have no collision for a part"). Decals/overlays still get
        # nonsolid, from add_polygon_offset.
        lines.append('\tsurfaceparm trans')
        lines.append('\tsort banner')
    if nocull:
        lines.append('\tcull none')

    lines.append('\t{')
    lines.append('\t\tmap {}'.format(map_path))
    if alphatest:
        lines.append('\t\talphaFunc GE128')
        lines.append('\t\tdepthWrite')
        _dg = decal_rgbgen_line()
        if _dg:
            lines.append(_dg)
    elif translucent or additive:
        blend = 'GL_ONE GL_ONE' if additive else 'GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA'
        lines.append('\t\tblendFunc {}'.format(blend))
        _dg = decal_rgbgen_line()
        if _dg:
            lines.append(_dg)
    else:
        lines.append('\t\trgbGen identity')
    lines.append('\t}')

    if not translucent and not alphatest:
        lines.append('\t{')
        lines.append('\t\tmap $lightmap')
        lines.append('\t\trgbGen identity')
        lines.append('\t\tblendFunc GL_DST_COLOR GL_ZERO')
        lines.append('\t}')

    if selfillum:
        # Source $selfillum: the texture glows where its alpha (or a separate
        # $selfillummask) is white, independent of lighting. Added on top of the
        # lit result with GL_SRC_ALPHA GL_ONE - the dust2 tunnel's star lamps,
        # which rendered black without it.
        mask = get_param(body, 'selfillummask')
        lines.append('\t{')
        lines.append('\t\tmap {}'.format(map_path))
        lines.append('\t\tblendFunc GL_SRC_ALPHA GL_ONE')
        lines.append('\t\trgbGen identity')
        lines.append('\t\t// Source $selfillum glow')
        if mask:
            lines.append('\t\t// Source took the glow mask from $selfillummask {} - copy it'.format(mask))
            lines.append('\t\t// into this texture\'s alpha channel for the same result')
        lines.append('\t}')

    if get_param(body, 'bumpmap') or get_param(body, 'envmap') or get_param(body, 'phong'):
        lines.append('\t// NOTE: source material used per-pixel lighting features')
        lines.append('\t// ($bumpmap/$envmap/$phong) with no classic idTech3 equivalent')

    lines.append("}")
    return emit_dual_shader_blocks(rel_path, lines)


def convert_worldvertextransition_to_terrain_shaders(rel_path, body, surfaceprop_map,
                                                       unmapped_surfaceprops, material_map,
                                                       unmapped_materials, warnings,
                                                       texture_sizes=None, texel_scale=0.25,
                                                       base_shader_name=None,
                                                       u_scale_override=None, v_scale_override=None,
                                                       worldalign_units=None):
    """A Source WorldVertexTransition material (displacement blend between
    $basetexture and $basetexture2) needs a real SOF2 terrain shader
    TRIPLET, not a single flat shader - matching the exact structure of a
    confirmed-working SOF2 map's own terrain shaders (q3map_material,
    surfaceparm nolightmap, q3map_novertexshadows, q3map_forcesunlight,
    rgbGen vertex for the single-layer shaders, plus alphaGen vertex and
    blendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA on the blended pair, and
    a tcmod scale line on every texture stage - confirmed present on every
    stage of that reference map's own terrain shaders).
    vmf2map.py's build_displacement_terrain()/build_displacement_terrain_patch()
    emit surfaces referencing exactly this shader base name + '/terrain_0'
    / '_1' / '_0to1', and generate the paired .pcx alphamap whose raw pixel
    values are the layer indices these shader names encode (0 =
    $basetexture, 1 = $basetexture2) - the three pieces only work together.

    tcmod scale: texel_scale (default 0.25, matching Source's own common
    uaxis/vaxis scale default) is applied directly as the tcmod scale
    value on every stage. A theory that this double-scales an
    already-complete baked UV (since vmf2map.py's build_patch_uv also
    divides by this same axis scale) was tried and DISPROVEN by direct
    in-game testing: removing tcmod made the texture tile smaller and
    more repetitive, the opposite of what real-world comparison called
    for, not the "already correct" result that theory predicted. So
    tcmod scale is kept as an active, non-1.0 multiplier by default; the
    actual relationship between the baked UV and this value is evidently
    more involved than straightforward double-application, and is
    unresolved - don't remove or default this to a no-op again without
    first confirming against real in-game comparison, the same way this
    reversal was confirmed.

    worldalign_units: OPT-IN, off by default. When given a number of world
    units, emits 'q3map_tcGen ivector (N 0 0) (0 N 0)' at the SHADER level
    (not per-stage) instead of the per-stage tcmod scale line, and skips
    tcmod entirely for that shader. This is a real, documented Q3Map2
    directive for alphamap terrain shaders (confirmed via Q3Map2's own
    shader manual) that generates texture coordinates from world X/Y
    position AT COMPILE TIME, independent of whatever per-vertex UV a
    patch stores - which would guarantee texture-pattern continuity across
    any patch boundary regardless of how each patch's own UV was computed,
    unlike the current approach where continuity depends on every
    adjacent patch's stored UV agreeing (which they mathematically do, but
    a visible "doesn't blend across patches" symptom persisted anyway).
    UNVERIFIED whether SOF2's own compiler (a customized, older idTech3
    derivative) honors this the same way modern q3map2 does - the
    confirmed-working reference terrain shader this whole shader structure
    is modeled on does NOT use q3map_terrain, q3map_indexed, or any tcGen
    at all, so this is a genuine experiment, not a known-good fix. Test by
    itself (one displacement's material) before applying broadly; if it
    breaks the alphamap blending, that's real evidence SOF2 does NOT use
    this same terrain scheme and this parameter should stay off.

    texture_sizes: currently unused (kept as a parameter for a possible
    future per-material override).

    Returns a list of shader blocks (4 entries) or None if $basetexture
    or $basetexture2 is missing (falls back to a normal flat shader in
    that case, handled by the caller)."""
    basetexture = get_param(body, 'basetexture')
    basetexture2 = get_param(body, 'basetexture2')
    if not basetexture or not basetexture2:
        return None

    if base_shader_name is None:
        base_shader_name = with_textures_prefix(rel_path) + '/terrain'
    else:
        base_shader_name = with_textures_prefix(base_shader_name)
    map0 = physical_texture_path(basetexture)
    map1 = physical_texture_path(basetexture2)
    surfaceprop = get_param(body, 'surfaceprop')
    q3map_material = convert_q3map_material(surfaceprop, material_map, unmapped_materials)

    # tcmod scale is applied directly here (see this function's own
    # docstring above) - do not change this to a no-op default without
    # first confirming against real in-game comparison. u_scale_override/
    # v_scale_override, when supplied (from vmf2map.py's own
    # --terrain-scales JSON), take priority over texel_scale - this is
    # this specific displacement's own real uaxis/vaxis scale rather than
    # one flat guess shared by the whole material. Skipped entirely when
    # worldalign_units is set - see this function's own docstring above.
    tcmod_u = u_scale_override if u_scale_override is not None else texel_scale
    tcmod_v = v_scale_override if v_scale_override is not None else texel_scale

    def tcmod_line(source_material_name):
        if worldalign_units:
            return None
        return '\t\ttcmod scale {:.6g} {:.6g}'.format(tcmod_u, tcmod_v)

    tcmod0 = tcmod_line(basetexture)
    tcmod1 = tcmod_line(basetexture2)

    def common_header():
        lines = []
        if q3map_material:
            lines.append('\tq3map_material {}'.format(q3map_material))
        lines.extend(terrain_unlit_header_lines())
        _sa = shade_angle_line()
        if _sa:
            lines.append(_sa)
        _ls = lightmap_samplesize_line()
        if _ls:
            lines.append(_ls)
        if worldalign_units:
            lines.append('\tq3map_tcGen ivector ( {:.6g} 0 0 ) ( 0 {:.6g} 0 )'.format(
                worldalign_units, worldalign_units))
        return lines


    blocks = []

    terrain_0 = [base_shader_name + '_0', '{']
    terrain_0.append('\tqer_editorimage {}'.format(map0))
    terrain_0.extend(common_header())
    terrain_0 += ['\t{', '\t\tmap {}'.format(map0), terrain_rgbgen()]
    if tcmod0:
        terrain_0.append(tcmod0)
    terrain_0 += ['\t}'] + terrain_lightmap_stage() + ['}']
    blocks.append('\n'.join(terrain_0))

    terrain_1 = [base_shader_name + '_1', '{']
    terrain_1.append('\tqer_editorimage {}'.format(map1))
    terrain_1.extend(common_header())
    terrain_1 += ['\t{', '\t\tmap {}'.format(map1), terrain_rgbgen()]
    if tcmod1:
        terrain_1.append(tcmod1)
    terrain_1 += ['\t}'] + terrain_lightmap_stage() + ['}']
    blocks.append('\n'.join(terrain_1))

    terrain_blend = [base_shader_name + '_0to1', '{']
    terrain_blend.append('\tqer_editorimage {}'.format(map1))
    terrain_blend.extend(common_header())
    terrain_blend += ['\t{', '\t\tmap {}'.format(map0), terrain_rgbgen(), '\t\talphaGen vertex']
    if tcmod0:
        terrain_blend.append(tcmod0)
    terrain_blend += ['\t}', '\t{', '\t\tmap {}'.format(map1), terrain_rgbgen(),
                       '\t\talphaGen vertex']
    if tcmod1:
        terrain_blend.append(tcmod1)
    terrain_blend += ['\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA', '\t}']
    terrain_blend += terrain_lightmap_stage() + ['}']
    blocks.append('\n'.join(terrain_blend))

    # The worldspawn "vertexremapshaderN" keys (built in vmf2map.py) all
    # point back at this one shader as their common reference target -
    # confirmed required by matching a real, working SOF2 map's own
    # worldspawn keys exactly. Identical structure to terrain_0 in the
    # reference map.
    terrain_vertex = [base_shader_name + '.vertex', '{']
    terrain_vertex.append('\tqer_editorimage {}'.format(map0))
    terrain_vertex.extend(common_header())
    terrain_vertex += ['\t{', '\t\tmap {}'.format(map0), terrain_rgbgen()]
    if tcmod0:
        terrain_vertex.append(tcmod0)
    terrain_vertex += ['\t}'] + terrain_lightmap_stage() + ['}']
    blocks.append('\n'.join(terrain_vertex))

    return blocks


def convert_worldvertextransition_to_lwo_shader(rel_path, body, material_map,
                                                  unmapped_materials, shader_name,
                                                  u_scale, v_scale, worldalign_units=None):
    """A single plain 2-layer alphaGen-vertex terrain shader for
    --terrain-mode lwo's welded misc_model path.

    Unlike convert_worldvertextransition_to_terrain_shaders above (which
    emits a vertexremapshader-based _0/_1/_0to1/.vertex quadruple for
    func_group patch/brush terrain - required there because q3map2's
    terrain-baking pass substitutes real shaders for a placeholder name),
    a LWO misc_model's own SURF/TAGS name is looked up directly - no remap
    indirection needed. So this emits just ONE shader block, named EXACTLY
    shader_name, which vmf2map.py's LWOTerrainBuilder already generates as
    a short, flat name (e.g. "terrain0").

    NO "textures/" prefix on the shader's own name (unlike every other
    shader this script generates): CONFIRMED by a real in-game failure -
    "Activate shader failed for terrain0" / "Failed to load shader terrain0
    for models/mapobjects/terrain/welded_terrain.lwo", with the bare,
    unprefixed tag name in the actual failure message, followed by SOF2's
    own fallback attempt to load "models/mapobjects/terrain/skin.tga" (a
    classic idTech3 "give up and guess a nearby image" behavior) - i.e. a
    model's own baked-in surface tag is looked up as an exact shader name,
    with no implicit "textures/" added the way ordinary brush-face
    materials get. An EARLIER test that seemed to show nested shader names
    failing for models was actually just this same kind of name mismatch
    (the .lwo's own tag was "alphatest", the shader tried was
    "textures/sof2test/alphatest" - unrelated to nesting depth at all).
    Do not reintroduce a prefix here without re-testing against this
    specific evidence. The internal `map` lines below are genuine texture
    file paths (not shader-name lookups) and correctly keep their
    textures/ prefix via physical_texture_path.

    Structurally modeled on convert_worldvertextransition_to_terrain_shaders's
    own terrain_0to1 block (both stages carry rgbGen vertex + alphaGen
    vertex, blendFunc only on the second) - that convention comes from a
    confirmed-working real SOF2 map's own terrain shaders, so this matches
    it rather than inventing a different style.

    Returns a shader block string, or None if $basetexture/$basetexture2
    is missing (the caller should skip this entry in that case - the
    displacement will show as an untextured/fallback model on that surface,
    same failure mode as the other terrain modes when a VMT can't be read).
    """
    basetexture = get_param(body, 'basetexture')
    basetexture2 = get_param(body, 'basetexture2')
    if not basetexture or not basetexture2:
        return None

    full_shader_name = shader_name
    map0 = physical_texture_path(basetexture)
    map1 = physical_texture_path(basetexture2)
    surfaceprop = get_param(body, 'surfaceprop')
    q3map_material = convert_q3map_material(surfaceprop, material_map, unmapped_materials)

    # NO tcmod scale here, unlike convert_worldvertextransition_to_terrain_shaders's
    # patch/brush shaders (which need it - see that function's own comment).
    # CONFIRMED by a real in-game report of textures looking too big after
    # the shader-name fix: build_patch_uv (vmf2map.py) already computes
    # dot(point,u_dir)/u_scale + u_shift, THEN divides by the texture's
    # real pixel width/height - that's the complete, normalized UV, using
    # the EXACT same formula Source itself uses. It needs no further
    # scaling. The patch/brush path's own need for tcmod most plausibly
    # comes from q3map2's patchDef2 Bezier tessellation specifically - a
    # Bezier curve's parameter doesn't correspond linearly to arc length,
    # so tessellated points between control points can end up with UV
    # interpolated along the curve's parameter space rather than the
    # surface's real distance, a distortion tcmod was empirically tuned to
    # compensate for. A LWO's triangles are flat and interpolate UV
    # linearly with no such distortion, so they shouldn't inherit a
    # correction factor that exists to fix something triangles don't do.
    # If textures still look off after this fix, the next thing to check
    # is whether --texture-sizes has this material's REAL pixel dimensions
    # (build_patch_uv falls back to a guessed default otherwise, which
    # would proportionally mis-scale the baked UV independent of this).

    lines = [full_shader_name, '{']
    lines.append('\tqer_editorimage {}'.format(map1))
    if q3map_material:
        lines.append('\tq3map_material {}'.format(q3map_material))
    lines.extend(terrain_unlit_header_lines())
    # When vertex-lit (nolightmap + rgbGen vertex), smoothed
    # normals are exactly what its shading reads - without this the welded
    # displacement mesh is lit one flat triangle at a time.
    _sa = shade_angle_line()
    if _sa:
        lines.append(_sa)
    lines.extend(TERRAIN_MESH_LINES)
    _ls = lightmap_samplesize_line()
    if _ls:
        lines.append(_ls)
    if worldalign_units:
        lines.append('\tq3map_tcGen ivector ( {:.6g} 0 0 ) ( 0 {:.6g} 0 )'.format(
            worldalign_units, worldalign_units))
    lines += ['\t{', '\t\tmap {}'.format(map0), terrain_rgbgen(), '\t\talphaGen vertex']
    lines += ['\t}', '\t{', '\t\tmap {}'.format(map1), terrain_rgbgen(), '\t\talphaGen vertex']
    lines += ['\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA', '\t}']
    lines += terrain_lightmap_stage() + ['}']
    return '\n'.join(lines)



    """Tool materials (clip, trigger, hint, nodraw, etc.) were previously
    skipped from shader generation entirely - but with no shader defined at
    all, the engine has nothing telling it the surface should be invisible,
    so it falls back to loading whatever raw image happens to exist at that
    path (confirmed in-game: tools/toolsblockbullets showed up fully
    visible). Every one of these needs AT LEAST surfaceparm nodraw; the
    specific ones below get their well-established standard idTech3
    surfaceparm too. Anything not specifically matched still gets nodraw as
    a safe baseline (invisible, solid by default) even if its exact
    specialized compiler behavior isn't replicated 1:1."""
    name = os.path.basename(rel_path).lower()
    shader_name = with_textures_prefix(rel_path)
    lines = [shader_name, "{", "\tqer_editorimage {}".format(physical_texture_path(rel_path)),
             "\tsurfaceparm nodraw"]

    if 'trigger' in name:
        lines.append("\tsurfaceparm nonsolid")
    elif 'hint' in name:
        lines.append("\tsurfaceparm hint")
    elif 'skip' in name:
        lines.append("\tsurfaceparm skip")
    elif 'areaportal' in name:
        lines.append("\tsurfaceparm areaportal")
        lines.append("\tsurfaceparm nonsolid")
    # clip/invisible/origin/blocklight/blockbullets: no confirmed idTech3
    # surfaceparm distinguishes these from each other or from plain solid
    # collision, so they fall through to the nodraw-only (invisible, solid
    # by default) baseline above - correct for clip/invisible, an
    # approximation for blockbullets (blocks everything, not just bullets)
    # and blocklight/origin (no lighting-block or pivot-point equivalent).

    lines.append("}")
    return emit_dual_shader_blocks(rel_path, lines)


def convert_directory(materials_dir, surfaceprop_map, material_map, sun_data=None,
                       texture_sizes=None, terrain_texel_scale=0.25, terrain_scales=None,
                       terrain_worldalign_units=None, lwo_shaders=None):
    warnings = []
    unmapped_surfaceprops = Counter()
    unmapped_materials = Counter()
    skipped = []
    shader_blocks = []
    sky_materials = []

    # Reverse index: material name (lowercase) -> [(base_shader_name,
    # u_scale, v_scale), ...] - from vmf2map.py's --terrain-scales JSON,
    # one entry per actual displacement using that material (a material
    # can have several, each with its own base_shader_name already made
    # unique per-displacement there).
    terrain_scales_by_material = {}
    if terrain_scales:
        for shader_base, info in terrain_scales.items():
            mat = info.get('material', '').strip().lower()
            terrain_scales_by_material.setdefault(mat, []).append(
                (shader_base, info.get('u_scale'), info.get('v_scale')))

    # Reverse index for --terrain-mode lwo's shaders JSON (vmf2map.py's
    # "<mapname>_terrain_lwo_shaders.json"): material name (lowercase) ->
    # [(flat_shader_name, u_scale, v_scale), ...]. Same shape as
    # terrain_scales_by_material above, different naming convention on the
    # shader side - see convert_worldvertextransition_to_lwo_shader's own
    # comment for why these must stay flat, unlike the nested names above.
    lwo_shaders_by_material = {}
    if lwo_shaders:
        for shader_name, info in lwo_shaders.items():
            if info.get('plain'):
                continue     # ordinary textures/ shader; only needs a _bsp copy (see main)
            mat = info.get('material', '').strip().lower()
            lwo_shaders_by_material.setdefault(mat, []).append(
                (shader_name, info.get('u_scale'), info.get('v_scale')))

    vmt_index = build_vmt_index(materials_dir)

    for root, _dirs, files in os.walk(materials_dir):
        for fname in files:
            if not fname.lower().endswith('.vmt'):
                continue
            full_path = os.path.join(root, fname)
            # Normalize to forward slashes immediately - os.path.relpath
            # returns backslashes on Windows, but every other path-like
            # value in this pipeline (a VMF's own "material" key, this
            # script's own generated shader names) always uses forward
            # slashes. Left un-normalized here, the terrain-scales lookup
            # below compares "de_dust\groundsand_blend" against the JSON's
            # "de_dust/groundsand_blend" and silently never matches on
            # Windows - confirmed by a real run where terrain shaders kept
            # falling back to the old, non-per-displacement naming despite
            # a correctly-loaded, correctly-formatted --terrain-scales file.
            rel_path = os.path.relpath(full_path, materials_dir).replace('\\', '/')
            rel_path_noext = os.path.splitext(rel_path)[0]

            shader_class, body = resolve_vmt(full_path, vmt_index)

            if is_tool_material(shader_class, rel_path_noext):
                # Deliberately NOT generating a shader for these. SOF2 ships
                # its own tools.shader with proper definitions (_caulk,
                # _clip, _hint, _shoot, _ladder, _sky, _areaportal, ...)
                # carrying full surfaceparm sets, qer_editorimage, qer_trans
                # and q3map_material - far better than anything derivable
                # from a Source .vmt. vmf2map.py remaps every tool face to
                # those native names (see DEFAULT_TOOL_TEXTURES there), so a
                # shader under the SOURCE name would never be referenced by
                # anything anyway. An earlier version generated minimal
                # nodraw shaders here to fix tool textures showing up
                # visible in-game; that was fixing it at the wrong layer -
                # the real cause was an incomplete remap table in
                # vmf2map.py, which is where it belongs.
                skipped.append(rel_path_noext)
                continue

            if is_sky_material(shader_class, rel_path_noext):
                basetexture = get_param(body, 'basetexture')
                if basetexture:
                    sky_materials.append((rel_path_noext, basetexture))
                else:
                    warnings.append("'{}' is a Sky material with no $basetexture, skipped".format(rel_path_noext))
                continue

            if (shader_class or '').strip().lower() == 'water':
                block = water_shader(rel_path_noext, body, warnings)
                if block:
                    shader_blocks.append(block)
                continue

            if shader_class and shader_class.strip().lower() == 'worldvertextransition':
                per_displacement = terrain_scales_by_material.get(rel_path_noext.strip().lower())
                if per_displacement:
                    # Real displacement(s) using this material exist in the
                    # map - generate each one's own shader set with its own
                    # actual uaxis/vaxis scale, rather than one guess shared
                    # by every displacement of this material. REVERTED here
                    # after real in-game testing: a prior version of this
                    # code removed u_scale_override/v_scale_override on the
                    # theory that vmf2map.py's own build_patch_uv already
                    # divides this same scale into the baked UV, making
                    # tcmod a double-application - direct testing disproved
                    # that (removing it made the texture tile smaller AND
                    # more repetitive, the wrong direction, not the
                    # predicted "already correct" result). Restored to the
                    # previously-working behavior; the actual interaction
                    # between the baked UV and tcmod scale here is evidently
                    # more involved than that theory accounted for, and is
                    # unresolved - don't re-remove this without first
                    # confirming a fix against real in-game comparison.
                    for shader_base, u_scale, v_scale in per_displacement:
                        terrain_blocks = convert_worldvertextransition_to_terrain_shaders(
                            rel_path_noext, body, surfaceprop_map, unmapped_surfaceprops,
                            material_map, unmapped_materials, warnings, texture_sizes,
                            terrain_texel_scale, base_shader_name=shader_base,
                            u_scale_override=u_scale, v_scale_override=v_scale,
                            worldalign_units=terrain_worldalign_units)
                        if terrain_blocks:
                            shader_blocks.extend(terrain_blocks)
                else:
                    # No per-displacement data (e.g. --terrain-scales wasn't
                    # given, or this material is never actually used on a
                    # displacement) - fall back to one shader set for the
                    # whole material, using the flat --terrain-texel-scale.
                    terrain_blocks = convert_worldvertextransition_to_terrain_shaders(
                        rel_path_noext, body, surfaceprop_map, unmapped_surfaceprops,
                        material_map, unmapped_materials, warnings, texture_sizes,
                        terrain_texel_scale, worldalign_units=terrain_worldalign_units)
                    if terrain_blocks:
                        shader_blocks.extend(terrain_blocks)
                # Deliberately NOT skipping the normal shader below even when
                # terrain_blocks succeeded: this same material name can also
                # be used directly on ordinary, flat (non-displacement) brush
                # faces in the same map - confirmed happening in a real map
                # (de_dust2's own rockwall_blend/groundsand_blend materials
                # are used on 60+ regular world-brush faces, not just
                # displacements). Those faces reference the material's bare
                # name, not the '/terrain_N' suffixed ones, so without this
                # normal shader too they'd have no shader at all ("Activate
                # shader failed" in the editor). Harmless duplication when
                # the material is ONLY ever used on displacements.

                lwo_entries = lwo_shaders_by_material.get(rel_path_noext.strip().lower())
                if lwo_entries:
                    for shader_name, u_scale, v_scale in lwo_entries:
                        lwo_block = convert_worldvertextransition_to_lwo_shader(
                            rel_path_noext, body, material_map, unmapped_materials,
                            shader_name, u_scale if u_scale is not None else terrain_texel_scale,
                            v_scale if v_scale is not None else terrain_texel_scale,
                            worldalign_units=terrain_worldalign_units)
                        if lwo_block:
                            shader_blocks.append(lwo_block)
                        else:
                            warnings.append(
                                "'{}' (lwo terrain shader '{}') has no $basetexture2 - skipped, "
                                "the welded misc_model will show that surface untextured/"
                                "fallback.".format(rel_path_noext, shader_name))

            block = convert_vmt_to_shader(
                rel_path_noext, shader_class, body,
                surfaceprop_map, unmapped_surfaceprops,
                material_map, unmapped_materials, warnings)
            if block:
                shader_blocks.append(block)

    sky_report = []
    if sky_materials:
        sky_blocks, sky_report = generate_sky_shaders(sky_materials, sun_data)
        shader_blocks.extend(sky_blocks)

    # Always emitted: phy2clip.py's collision brushes use these.
    shader_blocks.extend(generate_shoot_clip_shaders())

    # Always emitted: vmf2map.py retextures nodraw slabs that displacement
    # terrain is built on to this (see its scan_terrain_base_slabs). nodraw +
    # nonsolid but NOT trans: q3map2 seals with any opaque structural brush,
    # so these still seal the map, they just no longer collide.
    shader_blocks.append("\n".join([
        "textures/vmf2map/terrain_base",
        "{",
        "\tqer_editorimage textures/tools/_caulk",
        "\tqer_trans 0.35",
        "\tsurfaceparm nodraw",
        "\tsurfaceparm nonsolid",
        "\tsurfaceparm nolightmap",
        "\tsurfaceparm nomarks",
        "}",
    ]))
    # Always emitted: vmf2map.py maps Source's toolsskip here. SOF2's own
    # tools/_skip lacks "surfaceparm skip", so q3map2 used the skip sides of
    # every hint brush as BSP splitters; this one is ignored like in VBSP.
    shader_blocks.append("\n".join([
        "textures/vmf2map/skip",
        "{",
        "\tqer_editorimage textures/tools/_skip",
        "\tqer_nocarve",
        "\tqer_trans 0.5",
        "\tsurfaceparm nodraw",
        "\tsurfaceparm nonsolid",
        "\tsurfaceparm trans",
        "\tsurfaceparm noimpact",
        "\tsurfaceparm skip",
        "}",
    ]))
    # Always emitted: the non-solid stand-ins vmf2map.py uses for the tool
    # faces of a never-solid brush entity (func_brush Solidity 1): SOF2's
    # tools/_caulk and tools/_black are both solid, and contents in idTech3
    # come from the shader alone (q3map2 ORs every side's contents into the
    # brush, so one solid face makes the whole brush solid).
    shader_blocks.append("\n".join([
        "textures/vmf2map/nodraw_nonsolid",
        "{",
        "\tqer_editorimage textures/tools/_caulk",
        "\tqer_trans 0.35",
        "\tsurfaceparm nodraw",
        "\tsurfaceparm nonsolid",
        "\tsurfaceparm trans",
        "\tsurfaceparm nolightmap",
        "\tsurfaceparm nomarks",
        "}",
    ]))
    shader_blocks.append("\n".join([
        "textures/vmf2map/black_nonsolid",
        "{",
        "\t// SOF2's own tools/_black, plus nonsolid",
        "\tqer_editorimage textures/tools/editor_images/qer_black",
        "\tsurfaceparm nonsolid",
        "\tsurfaceparm noimpact",
        "\tsurfaceparm nomarks",
        "\tq3map_nolightmap",
        "\t{",
        "\t\tmap textures/tools/_black",
        "\t\tblendFunc GL_ONE GL_ZERO",
        "\t\trgbGen identity",
        "\t}",
        "}",
    ]))
    shader_text = "\n\n".join(shader_blocks) + "\n"
    return shader_text, warnings, unmapped_surfaceprops, unmapped_materials, skipped, sky_report


# ==============================================================================
# CLI
# ==============================================================================

def load_json_map(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {k.strip().lower(): v for k, v in data.items()}


DECAL_SHADER_SUFFIX = '_decal'      # vmf2map.py decal_texture_name


def decal_shader_copies(shader_text, shader_names):
    """<name>_decal copies of the named shaders, for decal/overlay patches:
    the plain shader is left exactly as it is, so it comes out the same on
    every map whether or not that map also uses the material as a decal
    (shader names are global across converted maps, first loaded wins).
    The copy gets add_polygon_offset's lines. A name with no shader here (no
    .vmt was found) gets a vertex-lit, alpha-blended form drawing the image
    of the same name. Returns (blocks, how many were of that implicit kind)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    blocks, implicit = [], 0
    for name in sorted(shader_names):
        target = name + DECAL_SHADER_SUFFIX
        if target in have:
            continue
        src = have.get(name)
        if src is not None:
            text = _join_shader(target, src[2])
        else:
            implicit += 1
            text = "\n".join([
                target, '{', '\tqer_editorimage {}'.format(name), '\tsurfaceparm trans',
                '\tq3map_nolightmap', '\tq3map_onlyvertexlighting',
                '\t{', '\t\tmap {}'.format(name), '\t\tblendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA',
                '\t\trgbGen vertex', '\t}', '}'])
        text, _n = add_polygon_offset(text, {target})
        blocks.append(text)
    return blocks, implicit


def add_polygon_offset(shader_text, shader_names, add_cull=True):
    """Insert 'polygonOffset' and 'cull none' as the first lines of each
    named shader block (names lowercase, with textures/ prefix). Blocks
    that already have either are left alone for that one. Returns (new
    text, number of blocks changed).

    cull none: an overlay's front-facing direction (Source's uv0..uv3
    corner order, preserved as-is - see convert_overlay) was never
    independently verified against patchDef2's own winding convention, and
    a previous "correction" attempt caused real, reported flipped/skewed
    overlays. Double-siding removes the need to get that convention right
    at all, at the cost of the surface being visible from both sides -
    harmless for a thin decal nobody was meant to see the back of anyway.
    """
    out, changed, i = [], 0, 0
    lines = shader_text.split('\n')
    while i < len(lines):
        out.append(lines[i])
        name = lines[i].strip().lower()
        if (name in shader_names and i + 1 < len(lines) and lines[i + 1].strip() == '{'):
            depth, k, has_offset, has_cull, has_nonsolid = 0, i + 1, False, False, False
            while k < len(lines):
                s = lines[k].strip()
                depth += s.count('{') - s.count('}')
                if s.lower() == 'polygonoffset':
                    has_offset = True
                if s.lower().startswith('cull '):
                    has_cull = True
                if ' '.join(s.lower().split()) == 'surfaceparm nonsolid':
                    has_nonsolid = True
                if depth == 0:
                    break
                k += 1
            out.append(lines[i + 1])
            if not has_offset:
                out.append('\tpolygonOffset')
                changed += 1
            if add_cull and not has_cull:
                out.append('\tcull none')
            if not has_nonsolid:
                # decal patches sit half a unit off their wall; solid, they
                # would be patch collision players and bullets snag on
                out.append('\tsurfaceparm nonsolid')
            i += 2
            continue
        i += 1
    return '\n'.join(out), changed


def generate_overlay_mesh_shaders(mesh_materials, materials_dir, vmt_index, surfaceprop_map,
                                   unmapped_surfaceprops, material_map, unmapped_materials, warnings):
    """One shader per entry in vmf2map.py's --terrain-mode lwo overlay-mesh
    JSON (flat shader name -> material), for info_overlay entities that sit
    on terrain and got built as a dense conforming mesh instead of a patch
    (see vmf2map.py's build_overlay_conforming_mesh). Needs a FLAT name
    (like the terrain shaders - see convert_worldvertextransition_to_lwo_shader's
    own comment: a model's own surface name is looked up as an exact shader
    name, confirmed by a real in-game failure on a nested one), unlike the
    ordinary "textures/<material>" shader the same material already gets
    for any flat (non-terrain) instances of the same overlay.

    Reuses convert_vmt_to_shader for the actual body (texture, translucency,
    surfaceprop, etc. - identical to the ordinary shader), just swaps its
    header line for the flat name and adds polygonOffset/cull none directly,
    since add_polygon_offset's own name-matching (on the "textures/..."
    header) wouldn't find these by their flat name anyway.

    Returns list of shader block strings.
    """
    blocks = []
    for flat_name, material in mesh_materials.items():
        key = material.strip().lower().replace('\\', '/')
        full_path = vmt_index.get(key) or vmt_index.get(os.path.splitext(key)[0])
        if not full_path:
            warnings.append("overlay mesh shader '{}': material '{}' not found in "
                            "--materials-dir, skipped".format(flat_name, material))
            continue
        shader_class, body = resolve_vmt(full_path, vmt_index)
        block = convert_vmt_to_shader(material, shader_class, body, surfaceprop_map,
                                      unmapped_surfaceprops, material_map, unmapped_materials,
                                      warnings)
        if not block:
            continue
        lines = block.split('\n')
        lines[0] = flat_name
        # insert AFTER the opening brace, not before it - lines[0] is the
        # name, lines[1] should be "{" (from convert_vmt_to_shader's own
        # name/{/body/} structure), so new directives go at index 2. Found
        # this the hard way: an earlier version used insert(1, ...) here,
        # which put both lines BEFORE the brace, outside the shader block
        # entirely - confirmed directly from a real generated .shader file
        # ("cull none"/"polygonOffset" sitting above "{"), and matches
        # exactly the reported "Activate shader failed" on every one of
        # these shaders in-game, since Radiant's parser expects "{"
        # immediately after the name and desyncs on anything else there.
        brace_idx = next((i for i, l in enumerate(lines) if l.strip() == '{'), 1)
        insert_at = brace_idx + 1
        # No 'cull none' here (unlike the patch-based overlays): the mesh's
        # triangle winding is now matched to the terrain mesh's own,
        # measured from a known-good in-game build, so the correct face is
        # the one that draws. Double-siding it would just bring back the
        # reported "texture is double sided instead of the correct face".
        if not any(l.strip().lower() == 'polygonoffset' for l in lines[:brace_idx + 2]):
            lines.insert(insert_at, '\tpolygonOffset')
        # These sit ON the terrain and now share its exact triangulation, so
        # they must be smoothed the same way or a road would be lit flat while
        # the ground under it is smooth.
        _sa = shade_angle_line()
        if _sa and not any('shadeangle' in l.strip().lower() for l in lines):
            lines.insert(insert_at, _sa)
        _ls = lightmap_samplesize_line()
        if _ls and not any('lightmapsamplesize' in l.strip().lower() for l in lines):
            lines.insert(insert_at, _ls)
        blocks.append('\n'.join(lines))
    return blocks


LIGHTMAP_BRIGHTNESS_SLOTS = 4000
LIGHTMAP_BRIGHTNESS_STEP = 1.2e-6     # x 4000 slots = at most +0.48%


def lightmap_brightness_for(name):
    """q3map_lightmapBrightness for a shader, from its name alone (see
    split_lightmap_groups): the same on every map that defines it."""
    import zlib
    slot = zlib.crc32(name.strip().lower().encode('utf-8')) % LIGHTMAP_BRIGHTNESS_SLOTS
    return 1.0 + (slot + 1) * LIGHTMAP_BRIGHTNESS_STEP


def split_lightmap_groups(shader_text):
    """Give every lightmapped shader its own lightmap pages.

    The engine makes one shader per (material, lightmap page) pair, and
    q3map2 decides those pages WITHOUT looking at the material: surfaces
    share a lightmap when their plane, axis, sample size and a handful of
    per-shader lighting fields match (AddSurfaceToRawLightmap in
    lightmaps_ydnar.cpp). Every wall built from several textures on one plane
    therefore puts all of them on the same page, and a texture used all over
    a map collects a page from each wall it appears on. Measured on
    de_overpass_csgo: 175 ordinary brush materials cost 1128 shaders, 6.4
    pages each, while props and decals - which are vertex lit and never share
    - cost 1.1 each.

    q3map_lightmapBrightness is one of the fields that comparison tests, and
    it is a plain multiplier on the finished lightmap colour, so a difference
    of a few thousandths splits the grouping while being invisible. Each
    lightmapped shader gets its own value a hair above 1.0; surfaces of one
    material still share lightmaps with each other, so the pages stay nearly
    as full.

    MEASURED on de_overpass_csgo, same .map, same -light flags: 1383 world
    shaders -> 519, with lightmap pages 62 -> 67 and the lightmap data's mean
    brightness 22.3 -> 22.7 (i.e. the lighting itself is unchanged). The
    whole map went from 1509 shaders to 645 against the engine's 2048.

    The total spread is kept under half a percent, and the steps stay far
    enough apart for a 32-bit float to tell them apart (its resolution near
    1.0 is about 1.2e-7).

    The value comes from the shader's NAME (a CRC32 slot), not from where the
    shader sits in this file. Shader names are global: every converted map
    ships its own .shader, SOF2 loads them all at startup and the first
    definition of a name wins - and several people convert maps on their own.
    A CS texture used by two maps must come out byte-identical in both files,
    or which map loads first decides how the other one looks (seen with
    de_cbble_go + de_overpass_csgo, 2026-10-17: 25 shared shaders differed in
    this value alone). Two names landing in the same slot only lose the
    split between those two."""
    blocks = list(re.finditer(r'(?m)^([^\s{}/][^\s{}]*)[ \t]*\r?\n\s*\{', shader_text))
    targets = []
    for m in blocks:
        start = m.end() - 1
        depth, i = 0, start
        while i < len(shader_text):
            if shader_text[i] == '{':
                depth += 1
            elif shader_text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = shader_text[start:i]
        if re.search(r'(?mi)^\s*map\s+\$lightmap\b', body):
            targets.append((m.group(1), start + 1))
    if not targets:
        return shader_text, 0
    out, prev = [], 0
    for name, pos in targets:
        out.append(shader_text[prev:pos])
        out.append('\n\tq3map_lightmapBrightness {:.7f}'.format(lightmap_brightness_for(name)))
        prev = pos
    out.append(shader_text[prev:])
    return ''.join(out), len(targets)


def add_model_shader_aliases(shader_text, model_dir):
    """Duplicate every bare-named shader under "<model_dir>/<name>".

    The LWO terrain and overlay-mesh shaders are deliberately BARE - "terrain0",
    "overlay0", with no "textures/" prefix - because a misc_model's SURF/TAGS
    name is looked up verbatim, and prefixing it produced a real in-game failure
    ("Activate shader failed for terrain0").

    netradiant's q3map2 does not agree. It rewrites a misc_model's shader names
    by prefixing the model's own directory, and says so in its log:

        substituting: terrain0 -> models/mapobjects/terrain/terrain0
        WARNING: Couldn't find image for shader models/mapobjects/terrain/terrain0

    The BSP then references a shader the .shader file never defines, the engine
    falls back, and the result in game is black terrain with fullbright white
    overlay lines - reproduced side by side against a GtkRadiant 2.5.17 build of
    the same .map, which emits no substitution lines and renders correctly.

    Emitting BOTH names costs about ten extra shader blocks and makes the output
    compile correctly under either tool, with no setting for anyone to get
    wrong. A shader that already carries a path (everything under "textures/")
    is left alone - only the bare model-shader names are aliased."""
    if not model_dir:
        return shader_text
    prefix = model_dir.strip().strip('/').replace('\\', '/')
    if not prefix:
        return shader_text

    aliases = []
    name = None
    depth = 0
    body = []
    for line in shader_text.splitlines():
        stripped = line.strip()
        if stripped == '{':
            depth += 1
            body.append(line)
            continue
        if stripped == '}':
            depth -= 1
            body.append(line)
            if depth == 0 and name is not None:
                if '/' not in name:
                    aliases.append('{}/{}\n{}'.format(prefix, name, '\n'.join(body)))
                name = None
                body = []
            continue
        if depth == 0:
            if stripped and not stripped.startswith('//'):
                name = stripped
                body = []
        else:
            body.append(line)

    if not aliases:
        return shader_text
    return (shader_text.rstrip('\n') + '\n\n' +
            '// --- aliases of the bare model shaders above, prefixed with the terrain model\'s\n'
            '// own directory. netradiant\'s q3map2 rewrites a misc_model\'s shader names that\n'
            '// way ("substituting: terrain0 -> models/mapobjects/terrain/terrain0") while\n'
            '// GtkRadiant 2.5.17 leaves them bare; defining both means the map renders under\n'
            '// either compiler instead of falling back to a missing shader.\n' +
            '\n\n'.join(aliases) + '\n')



# ==============================================================================
# SOF2 lighting conventions - applied to the finished shader text
# ==============================================================================
#
# Three facts, each checked against SOF2's own content or engine behaviour:
#
# 1. Lightmaps are doubled and CLAMPED at load (r_mapOverBrightBits 1 minus
#    r_overBrightBits 0), so a lightmap can never take a texture past 1.0x.
#    CS:S shows its sunlit sand at 1.75x. The lightmap stage therefore blends
#    GL_DST_COLOR GL_SRC_COLOR (2 x texture x lightmap), which moves the clamp to
#    the final pixel - where Source has it - and gives the 0-2x range the
#    exposure model in this file assumes (SHADER_LIGHTMAP_SCALE).
#
# 2. A $lightmap stage on a surface that has no lightmap binds a white image.
#    Baked misc_model props are vertex-lit (q3map2 only lightmaps triangle
#    models forced to meta) and model_static props are lit by the light grid -
#    so every prop shader written in the lightmapped form rendered with no
#    lighting at all. Measured on your E and F screenshots: between the two
#    bakes brushwork and terrain changed by 0.55-0.78x, the crates and car by
#    1.01-1.08x - i.e. not at all.
#
#    Raven's answer (threewave.shader, "Misc Model Shaders"): baked props use a
#    separate shader named <name>_bsp - q3map_nolightmap, q3map_onlyvertexlighting,
#    rgbGen vertex - and q3map2 switches misc_models onto those names with
#    -rename ("Appending _bsp suffix to misc_model shaders (SOF2)"). The plain
#    <name> is then free for runtime entities, which need rgbGen lightingDiffuse.
#    COMPILE -bsp WITH -rename.
#
# 3. Vertex-lit alpha-blended surfaces (decals, overlays, translucent props)
#    cannot use the 2x blend - but repeating the stage with GL_SRC_ALPHA GL_ONE
#    adds the same colour again under the same alpha: a*2*tex*vc + (1-a)*dst.
#    Without it, road overlays on sunlit sand would show as dark stripes.

_STAGE_OPEN = re.compile(r'^\s*\{\s*$')
_STAGE_CLOSE = re.compile(r'^\s*\}\s*$')


def _split_shader_text(text):
    """-> list of ('raw', [lines]) and ('shader', name, [elements]) where each
    element is ('line', str) or ('stage', [str])."""
    lines = text.split('\n')
    out = []
    raw = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < n else ''
        if (stripped and not stripped.startswith('//') and not stripped.startswith('{')
                and not stripped.startswith('}') and nxt == '{'):
            if raw:
                out.append(('raw', raw))
                raw = []
            name = stripped
            i += 2
            elems = []
            while i < n and not _STAGE_CLOSE.match(lines[i]):
                if _STAGE_OPEN.match(lines[i]):
                    stage = []
                    i += 1
                    while i < n and not _STAGE_CLOSE.match(lines[i]):
                        stage.append(lines[i])
                        i += 1
                    elems.append(('stage', stage))
                    i += 1
                else:
                    elems.append(('line', lines[i]))
                    i += 1
            i += 1  # closing brace of the shader
            out.append(('shader', name, elems))
            continue
        raw.append(line)
        i += 1
    if raw:
        out.append(('raw', raw))
    return out


def _join_shader(name, elems):
    body = [name, '{']
    for kind, val in elems:
        if kind == 'line':
            body.append(val)
        else:
            body.append('\t{')
            body.extend(val)
            body.append('\t}')
    body.append('}')
    return '\n'.join(body)


def _stage_has(stage, pattern):
    return any(re.search(pattern, l, re.I) for l in stage)


def _stage_set(stage, key, value):
    """Replace (or add) a 'key value' line in a stage."""
    out = [l for l in stage if not re.match(r'^\s*' + key + r'\b', l.strip(), re.I)]
    out.append('\t\t{} {}'.format(key, value))
    return out


def _is_lightmap_stage(stage):
    return _stage_has(stage, r'^\s*map\s+\$lightmap\b')


def _is_vertex_alpha_stage(stage):
    return (_stage_has(stage, r'^\s*rgbGen\s+vertex\b')
            and _stage_has(stage, r'blendFunc\s+GL_SRC_ALPHA\s+GL_ONE_MINUS_SRC_ALPHA'))


_TWIN_MARKER = 'this is the 2x'


def _double_world_stages(elems):
    """World form: 2x lightmap blend, additive twin for vertex-lit alpha stages."""
    already = any(k == 'stage' and any(_TWIN_MARKER in l for l in v) for k, v in elems)
    out = []
    for kind, val in elems:
        if kind == 'stage' and _is_lightmap_stage(val):
            val = [re.sub(r'blendFunc\s+GL_DST_COLOR\s+GL_ZERO', 'blendFunc GL_DST_COLOR GL_SRC_COLOR',
                          l, flags=re.I) for l in val]
            out.append(('stage', val))
        elif kind == 'stage' and _is_vertex_alpha_stage(val) and not already:
            out.append(('stage', val))
            twin = [re.sub(r'blendFunc\s+GL_SRC_ALPHA\s+GL_ONE_MINUS_SRC_ALPHA',
                           'blendFunc GL_SRC_ALPHA GL_ONE', l, flags=re.I) for l in val]
            twin.append('\t\t// same stage again, added: SOF2 clamps vertex colour at 1.0x, this is the 2x')
            out.append(('stage', twin))
        else:
            out.append((kind, val))
    return out


def _needs_vertex_compile(elems):
    """A shader that takes its light from rgbGen vertex and has no $lightmap
    stage of its own."""
    stages = [v for k, v in elems if k == 'stage']
    if any(_is_lightmap_stage(v) for v in stages):
        return False
    return any(_stage_has(v, r'^\s*rgbGen\s+(exact)?vertex\b') for v in stages)


# NOT q3map_vertexScale. Tried and measured, 2026-10-21: baked props were
# thought to be too dark because their vertex colours (11-32/255) were
# compared with a LIGHTMAPPED surface's vertex colours - which are never drawn.
# Compared properly - each prop vertex against the lightmap of the wall right
# beside it - they were already at parity: 0.95 on de_dust2, 1.07 on de_cbble.
# q3map_vertexScale acts before -gamma, so 4.59 became 4.59^(1/2.2) = 2.0x on
# screen: window frames and dust2's wall-top teeth glowed white. Baked props and
# lightmaps are lit by the same q3map2 pass and need no correction between them.
VERTEX_LIGHT_SCALE = [None]


def set_vertex_light_scale(sun_data):
    """Kept as a no-op so callers stay valid; see the note above."""
    VERTEX_LIGHT_SCALE[0] = None


def _vertex_scale_lines():
    return []


def _ensure_vertex_compile(elems):
    """SOF2 only has real vertex colours on surfaces q3map2 compiled as
    vertex-lit. A surface that also got a lightmap draws rgbGen vertex as
    white - the "fullbright" decals and road overlays - so a vertex-lit
    shader has to say q3map_nolightmap + q3map_onlyvertexlighting, exactly
    as the prop _bsp form does. Raven's own shaders never break this: all 274
    of their textures/ shaders that use rgbGen vertex without a $lightmap
    stage carry q3map_nolightmap (259 with q3map_onlyvertexlighting too), and
    KENNITH's hand test on textures/decals/dict1 - the same two lines - is
    what stopped it glowing. Any q3map_lightmapSampleSize goes, since there is
    no lightmap left for it to size."""
    if not _needs_vertex_compile(elems):
        return elems
    header = [(k, v) for k, v in elems if k == 'line' and not re.match(
        r'^\s*(q3map_nolightmap|q3map_onlyvertexlighting|q3map_lightmapsamplesize)\b', v.strip(), re.I)]
    rest = [(k, v) for k, v in elems if k != 'line']
    return header + [('line', '\tq3map_nolightmap'),
                     ('line', '\tq3map_onlyvertexlighting')] + _vertex_scale_lines() + rest


def _prop_bsp_form(elems):
    """Raven's baked-prop form (<name>_bsp): vertex lit, no lightmap, plus the 2x."""
    header = [(k, v) for k, v in elems if k == 'line' and v.strip()
              and not re.match(r'^\s*q3map_(nolightmap|onlyvertexlighting|bounce\w*)\b', v.strip(), re.I)]
    stages = [v for k, v in elems if k == 'stage' and not _is_lightmap_stage(v)]
    # q3map_bounceScale 0: a baked prop receives bounced light but does not
    # emit it. That is VRAD's own rule (static props are not radiosity
    # patches), and it is what keeps -bounce affordable: on de_mirage_cs2 the
    # props' 435,891 triangles each became a bounce light (505,104 in all)
    # and the bounce pass ran for hours.
    out = header + [('line', '\tq3map_nolightmap'), ('line', '\tq3map_onlyvertexlighting'),
                    ('line', '\tq3map_bounceScale 0')] + _vertex_scale_lines()
    if not stages:
        return out
    first = _stage_set(stages[0], 'rgbGen', 'vertex')
    if _is_vertex_alpha_stage(first):
        return _double_world_stages(out + [('stage', first)] + [('stage', s) for s in stages[1:]])
    out.append(('stage', first))
    twin = [l for l in first if not re.match(r'^\s*(blendFunc|alphaFunc|depthWrite)\b', l.strip(), re.I)]
    twin.append('\t\tblendFunc GL_ONE GL_ONE')
    if _stage_has(first, r'^\s*alphaFunc\b'):
        twin.append('\t\tdepthFunc equal')
    twin.append('\t\t// same stage again, added: SOF2 clamps vertex colour at 1.0x, this is the 2x')
    out.append(('stage', twin))
    out.extend(('stage', s) for s in stages[1:])
    return out


def _prop_entity_form(elems):
    """Runtime-entity form (model_static): lit by the light grid, like players."""
    header = [(k, v) for k, v in elems if k == 'line']
    stages = [v for k, v in elems if k == 'stage' and not _is_lightmap_stage(v)]
    out = list(header)
    if stages:
        out.append(('stage', _stage_set(stages[0], 'rgbGen', 'lightingDiffuse')))
        out.extend(('stage', s) for s in stages[1:])
    return out


def _shader_class(name):
    n = name.lower()
    if n.endswith('_bsp') or n.startswith('textures/'):
        return 'world'
    if '/' not in n or n.startswith('models/mapobjects/terrain/'):
        return 'terrain'          # baked, lightmapped misc_models (terrain, overlays)
    if n.startswith('models/'):
        return 'prop'
    return 'world'


def collect_md3_shader_names(root_dir):
    """Every shader name any .md3 under root_dir references (lower-case, no
    extension). Read straight from the MD3 surface headers."""
    names = set()
    for dirpath, _dirs, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith('.md3'):
                continue
            try:
                with open(os.path.join(dirpath, fn), 'rb') as f:
                    data = f.read()
            except OSError:
                continue
            if data[:4] != b'IDP3' or len(data) < 108:
                continue
            num_surf = struct.unpack_from('<i', data, 84)[0]
            ofs = struct.unpack_from('<i', data, 100)[0]
            for _ in range(max(0, min(num_surf, 256))):
                if ofs <= 0 or ofs + 108 > len(data) or data[ofs:ofs + 4] != b'IDP3':
                    break
                n_sh = struct.unpack_from('<i', data, ofs + 76)[0]
                ofs_sh = struct.unpack_from('<i', data, ofs + 92)[0]
                for k in range(max(0, min(n_sh, 64))):
                    o = ofs + ofs_sh + k * 68
                    if o + 64 > len(data):
                        break
                    nm = data[o:o + 64].split(b'\0')[0].decode('latin-1').strip()
                    if nm:
                        names.add(re.sub(r'\.(tga|jpg|jpeg|png|dds)$', '', nm.replace('\\', '/'),
                                         flags=re.I).lower())
                ofs += struct.unpack_from('<i', data, ofs + 104)[0]
    return names


# q3map_nonplanar on every shader of the welded terrain model. q3map2's meta
# pass only merges a triangle into a surface on the SAME plane unless the
# shader is nonplanar (surface_meta.cpp AddMetaTriangleToSurface). Hilly
# terrain mostly escapes that, but near-flat ground - de_overpass_csgo's
# sandpit, 828 surfaces averaging 3.2 vertices - came out one surface per
# triangle: each its own lightmap island, filtered on its own, so the ground
# showed every triangle edge. With it: 31 surfaces averaging 29 vertices on
# the same compile. (q3map_terrain, the old Team Arena keyword, sets the same
# flag for the same reason.)
TERRAIN_MESH_LINES = ['\tq3map_nonplanar']


def lightmapped_bsp_copies(shader_text, names):
    """Exact copies of the world shader of each plain (non-blended) terrain
    surface in the .lwo, for surfaces that are part of a lightmapped
    misc_model and so get -rename's _BSP suffix, but must keep being
    lightmapped exactly like the brush version.

    names: {lwo tag: Source material}. The tag is usually textures/<material>
    itself and only needs the _bsp copy, but a material with more than 1800
    triangles is split across extra tags (textures/<material>_1, _2, ...),
    which need the shader under their own name as well - before this they had
    none and drew as missing textures (de_mirage_cs2: 10 such tags)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    out = []
    for tag, material in names.items():
        tag = tag.strip()
        src = have.get(tag.lower())
        if src is None and material:
            src = have.get(with_textures_prefix(material).lower())
        if src is None:
            continue
        _, _src_name, elems = src
        # These names are only ever drawn on the terrain model, never on a
        # brush face, so they get what the blended terrain shaders have:
        # smoothed normals (without them every triangle is lit flat), the
        # terrain lightmap size, and nonplanar (TERRAIN_MESH_LINES).
        extra = [l for l in [shade_angle_line(), lightmap_samplesize_line()] if l]
        extra += TERRAIN_MESH_LINES
        drop = re.compile(r'^\s*(q3map_shadeangle|q3map_lightmapsamplesize|q3map_nonplanar)\b', re.I)
        kept = [(k, v) for k, v in elems if not (k == 'line' and drop.match(v))]
        n_head = 0
        while n_head < len(kept) and kept[n_head][0] == 'line':
            n_head += 1
        terrain_elems = kept[:n_head] + [('line', l) for l in extra] + kept[n_head:]
        if tag.lower() not in have:
            out.append(_join_shader(tag, terrain_elems))
        if tag.lower() + '_bsp' not in have:
            out.append(_join_shader(tag + '_bsp', terrain_elems))
    return out


NONSOLID_SUFFIX = '_nonsolid'


def nonsolid_copies(shader_text, names):
    """<name>_nonsolid copies of world shaders, with "surfaceparm nonsolid".

    vmf2map.py retextures the faces of brush entities Source never collides
    with (func_brush "Solidity" "1", func_illusionary) to these, because
    idTech3 has no per-entity solidity: contents come from the shader only.
    names: shader names WITHOUT the suffix (textures/<material>)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    out, missing = [], []
    for base in sorted(names):
        base = base.strip()
        if (base + NONSOLID_SUFFIX).lower() in have:
            continue
        src = have.get(base.lower())
        if src is None:
            missing.append(base)
            continue
        elems = [e for e in src[2]
                 if not (e[0] == 'line' and re.match(r'^\s*surfaceparm\s+nonsolid\b', e[1], re.I))]
        out.append(_join_shader(base + NONSOLID_SUFFIX,
                                [('line', '\tsurfaceparm nonsolid')] + elems))
    return out, missing


def _is_glowing_part(elems):
    """Which surfaces of a lit prop glow: its glass. Source's surfaceprop glass
    is what q3map_material Glass is made from; a lantern's panes and a lamp's
    shade carry it, the iron frame around them doesn't."""
    return any(k == 'line' and re.match(r'^\s*q3map_material\s+glass\b', v, re.I)
               for k, v in elems)


def _glow_stage(elems):
    stages = [v for k, v in elems if k == 'stage']
    img = None
    for st in stages:
        for l in st:
            m = re.match(r'^\s*(?:map|clampmap)\s+(\S+)', l, re.I)
            if m and not m.group(1).startswith('$'):
                img = m.group(1)
                break
        if img:
            break
    if not img:
        return None
    return ('stage', ['\t\tmap {}'.format(img), '\t\tblendFunc GL_ONE GL_ONE',
                      '\t\trgbGen identity',
                      '\t\t// lit: a light sits inside this prop (vmf2map unshadow_light_fixtures)'])


def lit_copies(shader_text, targets):
    """<target> and <target>_bsp copies for the shaders of props that have a
    light inside them (vmf2map.py remaps those placements' shaders to
    <name>_lit, see unshadow_light_fixtures). The glass ones get one extra
    stage that adds the texture again, unlit - the same thing SOF2's own lit
    models do with a _glow stage (models/objects/hongkong/paper_lanterns). The
    rest are plain copies, so a lantern's frame stays iron.
    Returns (blocks, n_glowing, names with nothing to copy)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    out, missing, glowing = [], [], 0
    for name, target in sorted(targets.items()):
        base = re.sub(r'\.(tga|jpg|jpeg|png)$', '', name.strip(), flags=re.I)
        made = False
        for suffix in ('', '_bsp'):
            if (target + suffix).lower() in have:
                made = True
                continue
            src = have.get((base + suffix).lower())
            if src is None:
                continue
            elems = list(src[2])
            if _is_glowing_part(elems):
                g = _glow_stage(elems)
                if g:
                    elems.append(g)
                    if suffix == '_bsp':
                        glowing += 1
            out.append(_join_shader(target + suffix, elems))
            made = True
        if not made:
            missing.append(base)
    return out, glowing, missing


_TEXLIGHT_DROP = re.compile(
    r'^\s*(q3map_surfacelight|q3map_lightrgb|q3map_backsplash|q3map_lightimage|'
    r'surfaceparm\s+nolightmap)\b', re.I)


def texlight_copies(shader_text, specs):
    """The emitting copies of Source texture lights (vmf2map.py
    emit_texture_lights, <map>_texlights.json: {copy: {base, surfacelight,
    rgb}}). Source makes such a face an area light AND draws it fullbright with
    no lightmap (SURF_LIGHT | SURF_NOLIGHT), so the copy is the base shader's
    header plus q3map_surfacelight / q3map_lightRGB (q3map2 would otherwise
    take the colour from the image), q3map_backsplash 0 0 (VRAD has no
    backsplash, and the value was calibrated without it), surfaceparm
    nolightmap, and one stage drawing the texture as it is.
    Returns (blocks, names with nothing to copy)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    out, missing = [], []
    for name, spec in sorted(specs.items()):
        if name.lower() in have:
            continue
        src = have.get(str(spec.get('base', '')).lower())
        img = None
        if src is not None:
            g = _glow_stage(src[2])
            img = g[1][0].split(None, 1)[1] if g else None
        if src is None or not img:
            missing.append(spec.get('base', name))
            continue
        head = [e for e in src[2] if e[0] == 'line' and not _TEXLIGHT_DROP.match(e[1])]
        rgb = spec.get('rgb') or [1.0, 1.0, 1.0]
        head += [('line', '\tq3map_surfacelight {:g}'.format(float(spec['surfacelight']))),
                 ('line', '\tq3map_lightRGB {:g} {:g} {:g}'.format(*rgb)),
                 ('line', '\tq3map_backsplash 0 0'),
                 ('line', '\tsurfaceparm nolightmap'),
                 ('line', '\t// texture light from the Source .bsp (vmf2map emit_texture_lights)')]
        out.append(_join_shader(name, head + [('stage', ['\t\tmap {}'.format(img),
                                                         '\t\trgbGen identity'])]))
        have[name.lower()] = ('shader', name, head)
    return out, missing


def nofog_copies(shader_text, targets):
    """vmf2map.py remaps the shaders of a baked prop that stands mostly out of
    a water volume to <target> (misc_model "_remap"); q3map2 -rename then looks
    up <target>_bsp. Each is a copy of the model shader's own <name>_bsp form
    plus q3map_noFog, so q3map2 never puts that prop in the water's fog (see
    vmf2map.py keep_baked_props_out_of_fog). targets: {md3 shader name: target}.
    Returns (blocks, names with no _bsp form to copy)."""
    items = _split_shader_text(shader_text)
    have = {it[1].lower(): it for it in items if it[0] == 'shader'}
    out, missing = [], []
    for name, target in sorted(targets.items()):
        base = re.sub(r'\.(tga|jpg|jpeg|png)$', '', name.strip(), flags=re.I)
        if (target + '_bsp').lower() in have:
            continue
        src = have.get((base + '_bsp').lower())
        if src is None:
            missing.append(base)
            continue
        elems = [e for e in src[2]
                 if not (e[0] == 'line' and re.match(r'^\s*q3map_nofog\b', e[1], re.I))]
        out.append(_join_shader(target + '_bsp', [('line', '\tq3map_noFog')] + elems))
        have[(target + '_bsp').lower()] = ('shader', target + '_bsp', elems)
    return out, missing


MISSING_TWIN_SKIPPED = []


def missing_bsp_twins(shader_text, md3_names, textures_root=None):
    """-rename makes q3map2 look up <name>_bsp for every surface of every baked
    misc_model. A model can name a shader this file never defines - an MD3 whose
    surface points at a stock SOF2 texture, for instance (G's log: "Couldn't find
    image for shader textures/nature/rockwall009b_BSP"). Such a surface would
    render with a missing texture, so give it Raven's vertex-lit _bsp form,
    drawing the same image, with the 2x the rest of the map uses."""
    defined = {n.lower() for n in re.findall(r'(?m)^([^\s{}/][^\s{}]*)\s*\n\s*\{', shader_text)}
    out = []
    for name in sorted(md3_names):
        if (name + '_bsp') in defined or name.startswith('models/mapobjects/terrain/'):
            continue
        img = physical_texture_path(name)
        # A CS model's material (models/...) with no .vmt and no converted
        # image: the guessed form would draw a texture that does not exist,
        # and since shader names are global across converted maps it would
        # also override another map's correct definition when this map's
        # file loads first - de_cbble_go's foliage did that to
        # de_overpass_csgo's (2026-10-17). Leave such a name undefined.
        if textures_root and not name.startswith('textures/') and not any(
                os.path.isfile(os.path.join(textures_root, *img.split('/')) + ext)
                for ext in _IMAGE_EXTS):
            MISSING_TWIN_SKIPPED.append(name)
            continue
        out.append("\n".join([
            name + '_bsp', '{',
            '\t// generated for -rename: an .md3 names this shader but no _bsp form existed',
            '\tqer_editorimage {}'.format(img),
            '\tq3map_nolightmap', '\tq3map_onlyvertexlighting', '\tq3map_bounceScale 0',
        ] + [v for _k, v in _vertex_scale_lines()] + [
            '\t{', '\t\tmap {}'.format(img), '\t\trgbGen vertex', '\t}',
            '\t{', '\t\tmap {}'.format(img), '\t\trgbGen vertex', '\t\tblendFunc GL_ONE GL_ONE',
            '\t\t// same stage again, added: SOF2 clamps vertex colour at 1.0x, this is the 2x',
            '\t}', '}']))
    return out


_IMAGE_KEYWORDS = ('map', 'clampmap', 'qer_editorimage', 'q3map_lightimage', 'animmap')
_IMAGE_EXTS = ('.jpg', '.png', '.tga', '.jpeg', '.dds')


def shorten_long_names(shader_text, textures_dir=None):
    """Make every shader name and image path in the finished file fit
    MAX_QPATH (see short_qpath). Shader names are renamed in place. An image
    path that is too long is pointed at a short copy of the same file, made
    here: image paths are 'textures/...' relative to the folder that holds
    textures_dir. Returns (text, renamed_shaders, copied_images, missing_images)."""
    renamed = []
    image_map = {}
    out = []
    for line in shader_text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            out.append(line)
            continue
        toks = stripped.split()
        if len(toks) == 1:
            # a shader name line (keywords that stand alone are all short, so
            # short_shader_name leaves them as they are)
            new = short_shader_name(stripped)
            if new != stripped:
                renamed.append((stripped, new))
                line = line.replace(stripped, new, 1)
            out.append(line)
            continue
        if toks[0].lower() in _IMAGE_KEYWORDS:
            paths = toks[2:] if toks[0].lower() == 'animmap' else toks[1:2]
            for p in paths:
                if p.startswith('$') or p.startswith('*'):
                    continue
                base, ext = os.path.splitext(p)
                if ext.lower() not in _IMAGE_EXTS:
                    base, ext = p, ''
                new = short_qpath(base)
                if new != base:
                    image_map[base] = new
                    line = re.sub(r'(?<!\S)' + re.escape(p) + r'(?!\S)',
                                  lambda _m: new + ext, line, count=1)
        out.append(line)

    copied, missing = [], []
    if image_map and textures_dir:
        out_root = os.path.dirname(os.path.abspath(textures_dir))
        import shutil
        for long_p, short_p in sorted(image_map.items()):
            src = src_ext = None
            for ext in ('.jpg', '.png', '.tga', '.jpeg'):
                cand = os.path.join(out_root, *(long_p + ext).split('/'))
                if os.path.isfile(cand):
                    src, src_ext = cand, ext
                    break
            if not src:
                missing.append(long_p)
                continue
            dst = os.path.join(out_root, *(short_p + src_ext).split('/'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append((long_p, short_p))
    return '\n'.join(out), renamed, copied, missing


def apply_sof2_lighting_conventions(shader_text):
    """See the block comment above. Idempotent: already-converted names are
    recognised and skipped."""
    items = _split_shader_text(shader_text)
    names = {it[1].lower() for it in items if it[0] == 'shader'}
    out = []
    for it in items:
        if it[0] == 'raw':
            out.append('\n'.join(it[1]))
            continue
        _, name, elems = it
        cls = _shader_class(name)
        if cls == 'world':
            out.append(_join_shader(name, _double_world_stages(_ensure_vertex_compile(elems))))
        elif cls == 'terrain':
            conv = _double_world_stages(_ensure_vertex_compile(elems))
            out.append(_join_shader(name, conv))
            if name.lower() + '_bsp' not in names:
                out.append('')
                out.append(_join_shader(name + '_bsp', conv))
        else:  # prop
            out.append(_join_shader(name, _prop_entity_form(elems)))
            if name.lower() + '_bsp' not in names:
                out.append('')
                out.append(_join_shader(name + '_bsp', _prop_bsp_form(elems)))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Convert a folder of Source .vmt materials into an "
                     "idTech3 .shader script compatible with SOF2.")
    ap.add_argument("materials_dir", help="root folder containing .vmt files (searched recursively)")
    ap.add_argument("output", help="output .shader file")
    ap.add_argument("--light-scale", type=float, default=1.0, metavar="N",
                    help="Multiply the calculated q3map_sun intensity by N. Default 1.0, and you "
                         "should not normally need it: the intensity is solved from the map's own "
                         "light_environment so that a sun-facing surface lands at lightmap level "
                         "TARGET_SUNLIT_LEVEL (235 of 255) at this map's sun elevation. Raising "
                         "this alone brightens sunlight WITHOUT touching shadow, which is what "
                         "produces the blown-highlights-and-black-shadows look; to change overall "
                         "exposure, edit TARGET_SUNLIT_LEVEL at the top of this file AND of "
                         "vmf2map.py - the two carry the same exposure model and must agree.")
    ap.add_argument("--decal-lighting", choices=["vertex", "none"], default="vertex",
                    help="How decals, overlays and other translucent surfaces are lit. 'vertex' "
                         "(default) modulates them by q3map2's baked per-vertex lighting; 'none' "
                         "leaves them fullbright, which is what they were before this option "
                         "existed and is almost certainly not what you want. A translucent surface "
                         "gets no $lightmap stage - it blends over geometry that already has one - "
                         "so its lighting falls to rgbGen, and the default for a stage with a "
                         "blendFunc is identityLighting: full brightness, ignoring every light in "
                         "the map. Confirmed in game on de_dust2: with the sun at 117 the whole "
                         "map went dark while the road-line overlays stayed pure white, drawing a "
                         "glowing grid across a black scene. Vertex lighting is coarser than a "
                         "lightmap, being interpolated from each polygon's corners, but decals are "
                         "small and flat so that is close to right.")
    ap.add_argument("--terrain-lightmapped", action="store_true",
                    help="Light the terrain for real - baked lightmaps and cast shadows - instead "
                         "of the vertex-lit, forced-full-sun form SOF2's own RMG terrain uses. "
                         "Without this the terrain shaders carry 'surfaceparm nolightmap', "
                         "'q3map_novertexshadows' and 'q3map_forcesunlight', copied from a real "
                         "SOF2 map where they are correct: RMG terrain is generated at runtime and "
                         "cannot hold a baked lightmap. A converted Source displacement is a static "
                         "mesh that can, so inheriting that convention leaves the ground fullbright "
                         "and shadowless while every brush around it is lit properly - which reads "
                         "as 'the buildings are too dark', and cannot be fixed by raising the sun, "
                         "because the ground is not responding to the sun at all. With this on, "
                         "vertex ALPHA still carries the material blend (Source's "
                         "WorldVertexTransition data, unrelated to lighting) while colour comes "
                         "from a $lightmap stage. Pair it with --terrain-lightmap-samplesize 32, "
                         "since the terrain now needs luxels of its own.")
    ap.add_argument("--terrain-lightmap-samplesize", type=float, default=0.0, metavar="N",
                    help="Put q3map_lightmapSampleSize N on the terrain and overlay-mesh shaders, "
                         "giving them a coarser lightmap than the rest of the map. 0 (default) = "
                         "leave them on the global -samplesize. This is the cheapest large saving "
                         "in -light: cost scales with luxel COUNT, so with 1/N^2, and terrain is "
                         "both the biggest surface in the map and the one needing fine lightmaps "
                         "least (sand under a sun is low-frequency, with no small occluders to "
                         "resolve). Measured on de_dust2: the terrain model is 1.83e7 square map "
                         "units, which is 1.14 MILLION luxels at -samplesize 4, 71k at 16 and 18k "
                         "at 32. Try 32. The value is absolute world units, not a multiplier, so "
                         "it does not track -samplesize.")
    ap.add_argument("--sun-deviance", type=float, default=0.0, metavar="DEG",
                    help="Emit q3map_sunExt instead of q3map_sun, spreading the sun over DEG "
                         "degrees so shadow edges get a penumbra that widens with distance from "
                         "the occluder. 0 (default) = a plain hard-edged sun. 2 is a good starting "
                         "point for an outdoor map. This has no Source equivalent to convert from "
                         "- Source softens shadows at runtime via shadow_control, not in the "
                         "light_environment entity - so it is an authoring choice, not a "
                         "conversion. Costs compile time in proportion to --sun-samples.")
    ap.add_argument("--sun-samples", type=int, default=16, metavar="N",
                    help="Sample count for --sun-deviance (default 16). Ignored unless "
                         "--sun-deviance is above 0. Too few gives banded, stippled shadow edges "
                         "rather than smooth ones; each sample is another per-luxel sun trace.")
    ap.add_argument("--light-offset", type=float, default=0.0, metavar="N",
                    help="Add N to the derived sun intensity and sky fill AFTER --light-scale. "
                         "Default 0. Together with --light-scale this expresses the "
                         "'brightness * 2.5 + 150' remap that circulates for Source-to-idTech3 "
                         "conversion: the values here are derived as brightness/6, so "
                         "--light-scale 15 --light-offset 150 IS that formula (15/6 = 2.5). On "
                         "de_dust2 it gives sun 1900 and sky 1400, against Source's authored 700 "
                         "and 500. Unlike --point-light-offset, an offset is harmless here: 150 is "
                         "about a tenth of numbers in the several-hundreds, so the sun-to-fill "
                         "ratio survives nearly intact (1.36 against Source's 1.4), where the same "
                         "offset applied to lamps authored at 10 and 150 would squash a 15x range "
                         "into 3x. The formula has no documented derivation - it is an empirical "
                         "remap - so treat these as a starting point and judge in game.")
    ap.add_argument("--sun-intensity", type=float, default=None, metavar="N",
                    help="Absolute override for the sky shader's q3map_sun intensity, ignoring "
                         "the solved value. Raising this ALONE brightens sunlight without "
                         "touching shadow. Shadow fill is worldspawn ambient, written by "
                         "vmf2map.py (--ambient there).")
    ap.add_argument("--sky-light", type=float, default=None, metavar="N",
                    help="Put q3map_surfacelight back on the sky shader at this value. OFF by "
                         "default, and deliberately: on de_dust2 it created 41,924 area lights, "
                         "36,611 of which q3map2 culled at SetupEnvelopes before tracing a single "
                         "luxel, because a converted Source map's sky shell sits 1,100+ units "
                         "above the architecture rather than just over the rooftops the way "
                         "Raven's own maps do. It was the largest single cost in -light and left "
                         "ceilings at 1.1 of 255. Worth trying only on a map whose sky hugs the "
                         "geometry; Raven's own skies use 6-1000 here, typically 30-100.")
    ap.add_argument("--vmf", help="path to the map's .vmf file. When given, its light_environment "
                                   "entity (Source's sun/skylight definition) is converted into a "
                                   "q3map_sun directive on the generated sky shader, instead of a "
                                   "plain skyParms block with no lighting. Pass the same .vmf to "
                                   "vmf2map.py so the matching worldspawn ambient is written too - "
                                   "the sun alone gives you black shadows.")
    ap.add_argument("--surfaceprop-map", help="JSON file mapping $surfaceprop value -> surfaceparm keyword")
    ap.add_argument("--terrain-shade-angle", type=float, default=120.0, metavar="DEGREES",
                    help="Write 'q3map_shadeAngle DEGREES' on the terrain and terrain-following "
                         "overlay shaders (default %(default)g; 0 disables). q3map2 smooths the "
                         "normals of adjacent triangles whose normals differ by less than this "
                         "angle - and if NO shader in the map sets it, q3map2 skips smoothing "
                         "altogether, reporting 'SmoothMetaTriangles: No smoothing angles "
                         "specified, aborting'. That leaves converted displacement terrain lit "
                         "one flat triangle at a time, which is visible as faceting across what "
                         "should be a smooth surface. Source computes smooth normals across a "
                         "displacement and across its neighbours, so this restores the original "
                         "map's shading. Only terrain shaders get it: on ordinary wall materials a "
                         "high angle would smooth straight across 90-degree brush corners and "
                         "round off hard edges. Lower it (60-90) if a cliff edge that should stay "
                         "sharp looks softened.")
    ap.add_argument("--material-map", help="JSON file mapping $surfaceprop value -> SOF2 q3map_material keyword "
                                            "(the surface-type value used for footstep sounds/impact effects/"
                                            "decals - separate from the surfaceparm above). Valid SOF2 values: "
                                            "None, Solidwood, Hollowwood, Solidmetal, Hollowmetal, Shortgrass, "
                                            "Longgrass, Dirt, Sand, Gravel, Glass, Concrete, Marble, Water, Snow, "
                                            "Ice, Flesh, Mud, BPGlass, DryLeaves, GreenLeaves, Fabric, Canvas, "
                                            "Rock, Rubber, Plastic, Tiles, Carpet, Plaster, ShatterGlass, Armor, "
                                            "Computer.")
    ap.add_argument("--mdl-dir", help="folder of your .mdl files (searched recursively) - scanned for their "
                                       "REAL cdmaterials top-level folders, so model materials that don't live "
                                       "under 'models/' (e.g. some CS:GO-authored content uses author-specific "
                                       "folders like 'fmpone/...') still get the correct bare shader name "
                                       "instead of a wrongly-added 'textures/' prefix. Strongly recommended.")
    ap.add_argument("--texture-sizes", help="the same texture-pixel-size JSON file passed to vmf2map.py's own "
                                             "--texture-sizes. Currently unused by terrain shader generation "
                                             "(kept as a parameter for a possible future per-material override).")
    ap.add_argument("--terrain-texel-scale", type=float, default=0.25,
                     help="applied DIRECTLY as the 'tcmod scale' value on every WorldVertexTransition "
                          "terrain shader's texture stage (e.g. 0.25 produces 'tcmod scale 0.25 0.25'). "
                          "Default 0.25 matches Source's own common uaxis/vaxis scale default, confirmed "
                          "by real in-game testing to look close to correct on one real map - this most "
                          "plausibly should match whatever uaxis/vaxis scale value Source actually used "
                          "on your map's own terrain displacements, which can vary by map/mapper. If your "
                          "textures look too big, try a SMALLER value; if too small/repetitive, try LARGER. "
                          "No need to regenerate the .map for this, just re-run this script and recompile. "
                          "Ignored for any displacement covered by --terrain-scales below, which takes "
                          "priority.")
    ap.add_argument("--terrain-scales", help="the '<mapname>_terrain_scales.json' file vmf2map.py writes "
                                              "alongside its output whenever any WorldVertexTransition "
                                              "displacements exist. Maps each displacement's own unique "
                                              "shader name to its ACTUAL uaxis/vaxis scale from the .vmf, "
                                              "so each displacement's terrain shader gets a tcmod scale "
                                              "matching what Source itself used for it - correctly handling "
                                              "a map where different displacements of the same material "
                                              "were painted with different texture scales in Hammer, "
                                              "instead of forcing all of them to share one flat, guessed "
                                              "--terrain-texel-scale value. Strongly recommended whenever "
                                              "it exists; falls back to --terrain-texel-scale for any "
                                              "material with no matching entry.")
    ap.add_argument("--planar-model-materials", help="the 'md3_planar_materials.json' "
                     "mdl_to_md3.py writes listing materials belonging to FLAT models (window/door "
                     "inserts, signage). Those props sit flush against wall brushes, which Source "
                     "handles fine but idTech3 does not - they z-fight with the wall or vanish "
                     "behind it. This adds 'polygonOffset' to their shaders (the same mechanism "
                     "decals use), without touching culling since they are real geometry.")
    ap.add_argument("--nofog-shaders", help="the '<mapname>_nofog_shaders.json' vmf2map.py "
                    "writes: baked props mostly out of a water volume, whose shaders get a "
                    "q3map_noFog copy so the water's fog stays off them.")
    ap.add_argument("--nonsolid-materials", help="the '<mapname>_nonsolid_materials.json' "
                     "vmf2map.py writes when the map has brush entities Source never collides "
                     "with (func_brush Solidity 1, func_illusionary). Lists the shaders whose "
                     "<name>_nonsolid copy (+ surfaceparm nonsolid) their faces now use.")
    ap.add_argument("--overlay-mesh-materials", help="the '<mapname>_overlay_mesh_materials.json' "
                     "vmf2map.py writes when any info_overlay sits on terrain and got built as a "
                     "dense conforming mesh (--terrain-mode lwo). Maps each flat shader name baked "
                     "into that mesh's SURF/TAGS (overlayN) to its material; generates one flat-"
                     "named shader per entry, reusing --materials-dir.")
    ap.add_argument("--overlay-materials", help="the '<mapname>_overlay_materials.json' vmf2map.py "
                                                 "writes when it converts info_overlay entities to "
                                                 "q3map2 decals. Adds 'polygonOffset' to those "
                                                 "materials' shaders: a decal lies exactly on the "
                                                 "surface it's projected onto, so without it the two "
                                                 "z-fight.")
    ap.add_argument("--model-shader-prefix", default="models/mapobjects/terrain",
                    help="Directory the LWO terrain/overlay models live in, used to emit an ALIAS "
                         "of every bare model shader under '<prefix>/<name>'. Must match "
                         "vmf2map.py's --terrain-lwo-model-path (same default). netradiant's "
                         "q3map2 rewrites a misc_model's shader names by prefixing the model's own "
                         "directory and logs 'substituting: terrain0 -> "
                         "models/mapobjects/terrain/terrain0', so the BSP ends up naming a shader "
                         "the script never defined - black terrain and fullbright overlays in "
                         "game. GtkRadiant 2.5.17 leaves the names bare. Defining both makes the "
                         "output correct under either. Empty string disables the aliases.")
    ap.add_argument("--textures-dir", default=None,
                    help="the converted textures/ folder (step 2's output). Images whose path is "
                         "too long for idTech3 get a short-named copy made in it. Default: "
                         "'textures' next to the output .shader, which is where the GUI puts it.")
    ap.add_argument("--texlights", help="the '<mapname>_texlights.json' vmf2map.py writes: "
                    "emitting copies of the map's Source texture lights (q3map_surfacelight, "
                    "fullbright), made from the shaders they copy.")
    ap.add_argument("--lit-shaders", help="the '<mapname>_lit_shaders.json' vmf2map.py writes: "
                    "shaders of props that have a light inside them, remapped to <name>_lit. "
                    "Copies are made, with a glow stage on the glass ones.")
    ap.add_argument("--terrain-lwo-shaders", help="the '<mapname>_terrain_lwo_shaders.json' file "
                                                   "vmf2map.py writes alongside its output when "
                                                   "--terrain-mode lwo is used. Maps each welded "
                                                   "terrain shader's short flat name (e.g. 'terrain0' "
                                                   "- see vmf2map.py's LWOTerrainBuilder for why these "
                                                   "are flat rather than nested) to its material and "
                                                   "actual uaxis/vaxis scale, generating ONE plain "
                                                   "2-layer alphaGen-vertex shader per entry (not the "
                                                   "vertexremapshader-based quadruple --terrain-scales "
                                                   "above drives - a LWO misc_model's SURF name is "
                                                   "looked up directly, no remap indirection needed).")
    ap.add_argument("--terrain-worldalign-units", type=float, default=None,
                     help="EXPERIMENTAL, opt-in, off by default. Emits 'q3map_tcGen ivector (N 0 0) "
                          "(0 N 0)' at the shader level for every WorldVertexTransition terrain "
                          "shader (N being this value - try 128 first, matching Source's common "
                          "0.25 uaxis/vaxis scale on a 512px texture), and skips tcmod scale entirely "
                          "for those shaders. This is a real, documented Q3Map2 directive for "
                          "alphamap terrain shaders that generates texture coordinates from world "
                          "X/Y position at compile time, independent of each patch's own stored UV - "
                          "which would guarantee texture-pattern continuity across any patch "
                          "boundary. UNVERIFIED against SOF2's own compiler specifically - the "
                          "confirmed-working reference terrain shader this whole structure is based "
                          "on does not use this or any tcGen. Test on one material first; if "
                          "alphamap blending breaks, that's real evidence SOF2 doesn't use this "
                          "scheme and this should stay unset.")
    args = ap.parse_args()
    # Must be set BEFORE convert_directory runs - the shaders are generated in
    # there, so assigning it afterwards silently had no effect.
    TERRAIN_SHADE_ANGLE[0] = args.terrain_shade_angle
    TERRAIN_LIGHTMAP_SAMPLESIZE[0] = args.terrain_lightmap_samplesize
    TERRAIN_LIGHTMAPPED[0] = args.terrain_lightmapped
    DECAL_LIGHTING[0] = args.decal_lighting

    if not os.path.isdir(args.materials_dir):
        print("ERROR: '{}' is not a directory".format(args.materials_dir), file=sys.stderr)
        sys.exit(1)

    texture_sizes = {}
    if args.texture_sizes:
        with open(args.texture_sizes) as f:
            raw_sizes = json.load(f)
        texture_sizes = {k.lower(): tuple(v) for k, v in raw_sizes.items()}

    terrain_scales = None
    if args.terrain_scales:
        with open(args.terrain_scales) as f:
            terrain_scales = json.load(f)
        # Easy mistake: vmf2map.py writes two differently-shaped JSON
        # files that sound similar - "<mapname>_sizes.json" (texture
        # pixel sizes, for --texture-sizes) and
        # "<mapname>_terrain_scales.json" (per-displacement uaxis/vaxis
        # scale, for --terrain-scales here). Catch the mix-up explicitly
        # rather than crash deep inside convert_directory with an
        # unhelpful AttributeError.
        if not all(isinstance(v, dict) and 'material' in v for v in terrain_scales.values()):
            print("ERROR: '{}' doesn't look like a terrain-scales file (expected each entry "
                  "to be an object with a 'material' key). Did you accidentally pass the "
                  "texture-SIZES json (the one for --texture-sizes) instead? The file you want "
                  "here is '<mapname>_terrain_scales.json', written by vmf2map.py alongside its "
                  "output whenever WorldVertexTransition displacements exist.".format(
                      args.terrain_scales), file=sys.stderr)
            sys.exit(1)

    lwo_shaders = None
    if args.terrain_lwo_shaders:
        with open(args.terrain_lwo_shaders) as f:
            lwo_shaders = json.load(f)
        if not all(isinstance(v, dict) and 'material' in v for v in lwo_shaders.values()):
            print("ERROR: '{}' doesn't look like a terrain-lwo-shaders file (expected each "
                  "entry to be an object with a 'material' key). The file you want here is "
                  "'<mapname>_terrain_lwo_shaders.json', written by vmf2map.py alongside its "
                  "output when --terrain-mode lwo is used.".format(
                      args.terrain_lwo_shaders), file=sys.stderr)
            sys.exit(1)

    if args.mdl_dir:
        if not os.path.isdir(args.mdl_dir):
            print("ERROR: '{}' is not a directory".format(args.mdl_dir), file=sys.stderr)
            sys.exit(1)
        discovered_roots, mdl_count = collect_model_cdmaterials_roots(args.mdl_dir)
        new_roots = discovered_roots - set(NO_TEXTURES_PREFIX_ROOTS)
        NO_TEXTURES_PREFIX_ROOTS.extend(sorted(new_roots))
        print("Scanned {} .mdl file(s) under {}: found cdmaterials root(s) {}".format(
            mdl_count, args.mdl_dir, sorted(discovered_roots)), file=sys.stderr)
        if new_roots:
            print("  added to the no-textures-prefix list: {}".format(sorted(new_roots)), file=sys.stderr)
    else:
        print("NOTE: no --mdl-dir given, falling back to the 'models/' guess only. "
              "If any of your models use a different cdmaterials root (this happens with some "
              "CS:GO-authored content), pass --mdl-dir to detect it automatically instead.",
              file=sys.stderr)

    surfaceprop_map = load_json_map(args.surfaceprop_map)
    material_map = load_json_map(args.material_map)

    sun_data = None
    if args.vmf:
        sun_data = parse_light_environment(args.vmf)
        if sun_data:
            if args.light_scale != 1.0 or args.light_offset != 0.0:
                base_i = sun_data.get('intensity_raw', sun_data['intensity'])
                sun_data['intensity'] = float('%.3g' % max(0.01,
                    base_i * args.light_scale + args.light_offset))
            if args.sun_intensity is not None:
                sun_data['intensity'] = float('%.3g' % max(0.01, args.sun_intensity))
            if args.sky_light is not None:
                sun_data['surfacelight_override'] = max(0.0, args.sky_light)
            sun_data['deviance'] = max(0.0, args.sun_deviance)
            sun_data['sun_samples'] = max(1, args.sun_samples)
            # baked props are vertex lit, so they never see -gridscale
            set_vertex_light_scale(sun_data)
            print("Found light_environment in {}: sun _light {:g} {:g} {:g} brightness {:g}, "
                  "_ambient brightness {}".format(
                      args.vmf, sun_data['r'], sun_data['g'], sun_data['b'],
                      sun_data['raw_brightness'],
                      "{:g}".format(sun_data['raw_ambient'])
                      if sun_data.get('raw_ambient') is not None else "(none)"),
                  file=sys.stderr)
            pr = sun_data['predicted']
            print("  -> exposure k {:.4f}: q3map_sun {:g} at {:g} deg / elevation {:g}, "
                  "q3map_skyLight {:g} {}".format(sun_data['exposure_k'], sun_data['intensity'],
                                              sun_data['degrees'], sun_data['elevation'],
                                              sun_data['skylight_value'], SKYLIGHT_ITERATIONS),
                  file=sys.stderr)
            print("     predicted on screen: sunlit floor {:.2f}, open shade {:.2f}, shaded wall "
                  "{:.2f}, roofed room {:.2f} (x texture)".format(pr['sunlit floor'], pr['open floor in shadow'],
                                                     pr['open wall in shadow'], pr['covered room']),
                  file=sys.stderr)
            print("     !! compile: -bsp ... -rename, then bsp_fix_leafbounds.py on the .bsp, then "
                  "-light ... -gamma {:g} -gridscale {:g} -brightness {:g} -vertexscale {:g} -bounce 1 "
                  "(derived for this map; needs netradiant's q3map2) !!".format(
                      sun_data['compile_gamma'], sun_data['grid_scale'], sun_data['brightness'],
                      sun_data['vertex_scale']), file=sys.stderr)
            print("NOTE: elevation is a direct, confirmed conversion of Source's pitch "
                  "convention. degrees (the compass direction) is best-effort - check the "
                  "sun's direction in-game against the original map's shadows and flip it "
                  "by hand if it reads mirrored.", file=sys.stderr)
        else:
            print("--vmf given but no light_environment entity found in it - generating "
                  "sky shader(s) without sun data.", file=sys.stderr)

    shader_text, warnings, unmapped_surfaceprops, unmapped_materials, skipped, sky_report = convert_directory(
        args.materials_dir, surfaceprop_map, material_map, sun_data, texture_sizes,
        args.terrain_texel_scale, terrain_scales, args.terrain_worldalign_units, lwo_shaders)
    if args.overlay_materials:
        with open(args.overlay_materials, encoding='utf-8') as f:
            overlay_names = {'textures/' + m.strip().lower().replace('\\', '/') for m in json.load(f)}
        decal_blocks, n_implicit = decal_shader_copies(shader_text, overlay_names)
        shader_text += '\n\n' + '\n\n'.join(decal_blocks) + '\n'
        print("Added {} decal shader(s) (<material>_decal, with polygonOffset; {} of them for a "
              "material with no .vmt, drawing its image directly)".format(
                  len(decal_blocks), n_implicit), file=sys.stderr)

    if args.planar_model_materials:
        with open(args.planar_model_materials, encoding='utf-8') as f:
            planar_names = {'textures/' + m.strip().lower().replace('\\', '/') for m in json.load(f)}
        shader_text, n_planar = add_polygon_offset(shader_text, planar_names, add_cull=False)
        print("Added polygonOffset to {} flat-model shader(s) (of {} listed)".format(
            n_planar, len(planar_names)), file=sys.stderr)

    if args.overlay_mesh_materials:
        with open(args.overlay_mesh_materials, encoding='utf-8') as f:
            mesh_materials = json.load(f)
        vmt_index = build_vmt_index(args.materials_dir)
        mesh_blocks = generate_overlay_mesh_shaders(
            mesh_materials, args.materials_dir, vmt_index, surfaceprop_map, unmapped_surfaceprops,
            material_map, unmapped_materials, warnings)
        if mesh_blocks:
            shader_text += '\n\n' + '\n\n'.join(mesh_blocks) + '\n'
        print("Generated {} overlay-mesh shader(s) (of {} listed)".format(
            len(mesh_blocks), len(mesh_materials)), file=sys.stderr)

    with open(args.output, 'w', encoding='utf-8') as f:
        final_text = apply_sof2_lighting_conventions(
            add_model_shader_aliases(shader_text, args.model_shader_prefix))
        if lwo_shaders:
            plain = {n: info.get('material', '') for n, info in lwo_shaders.items()
                     if info.get('plain')}
            extra = lightmapped_bsp_copies(final_text, plain)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
                print("Added {} shader(s) for non-blended terrain surfaces in the .lwo (their _bsp "
                      "forms, since -rename renames every misc_model surface, and the _1, _2 ... "
                      "tags a material is split into past 1800 triangles): {}".format(
                          len(extra), ", ".join(sorted(plain))), file=sys.stderr)
        if args.nonsolid_materials and os.path.isfile(args.nonsolid_materials):
            with open(args.nonsolid_materials, encoding='utf-8') as nsf:
                ns_names = json.load(nsf)
            extra, ns_missing = nonsolid_copies(final_text, ns_names)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
            print("Added {} non-solid shader copies (of {} listed) for brush entities Source never "
                  "collides with (func_brush Solidity 1, func_illusionary){}".format(
                      len(extra), len(ns_names),
                      "; no source shader for: " + ", ".join(ns_missing) if ns_missing else ""),
                  file=sys.stderr)
        if args.mdl_dir and os.path.isdir(args.mdl_dir):
            # An .md3 already carries the SHORT form of an over-long name (see
            # short_qpath); look it up under the full name this file uses.
            defined = re.findall(r'(?m)^([^\s{}/][^\s{}]*)\s*\n\s*\{', final_text)
            long_by_short = {short_qpath(n.lower()): n.lower() for n in defined}
            md3_names = {long_by_short.get(n, n) for n in collect_md3_shader_names(args.mdl_dir)}
            _tex_dir = args.textures_dir or os.path.join(
                os.path.dirname(os.path.abspath(args.output)), 'textures')
            extra = missing_bsp_twins(final_text, md3_names,
                                      os.path.dirname(os.path.abspath(_tex_dir))
                                      if os.path.isdir(_tex_dir) else None)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
                print("Added {} _bsp shader(s) for .md3 surfaces whose shader this file does not "
                      "otherwise define (needed by -rename).".format(len(extra)), file=sys.stderr)
            if MISSING_TWIN_SKIPPED:
                print("NOT defined, {} .md3 shader name(s) with no .vmt and no converted image "
                      "(these surfaces will show a missing texture; a guessed definition would "
                      "also override another converted map's correct one): {}".format(
                          len(MISSING_TWIN_SKIPPED), ", ".join(MISSING_TWIN_SKIPPED[:20]) +
                          (" ..." if len(MISSING_TWIN_SKIPPED) > 20 else "")), file=sys.stderr)
        if args.lit_shaders and os.path.isfile(args.lit_shaders):
            with open(args.lit_shaders, encoding='utf-8') as lf:
                lit_targets = json.load(lf)
            extra, n_glow, lit_missing = lit_copies(final_text, lit_targets)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
            if lit_targets:
                print("Added {} lit shader copies for props with a light inside them (lanterns, "
                      "lamps); {} of them glass, given a glow stage{}".format(
                          len(extra), n_glow,
                          "; nothing to copy for: " + ", ".join(lit_missing) if lit_missing else ""),
                      file=sys.stderr)
        if args.texlights and os.path.isfile(args.texlights):
            with open(args.texlights, encoding='utf-8') as tf:
                tl_specs = json.load(tf)
            extra, tl_missing = texlight_copies(final_text, tl_specs)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
            if tl_specs:
                print("Added {} texture-light shader(s) (Source lights.rad faces: "
                      "q3map_surfacelight, drawn fullbright){}".format(
                          len(extra), "; nothing to copy for: " + ", ".join(tl_missing)
                          if tl_missing else ""), file=sys.stderr)
        if args.nofog_shaders and os.path.isfile(args.nofog_shaders):
            with open(args.nofog_shaders, encoding='utf-8') as nff:
                nf_targets = json.load(nff)
            extra, nf_missing = nofog_copies(final_text, nf_targets)
            if extra:
                final_text += '\n\n' + '\n\n'.join(extra) + '\n'
            if nf_targets:
                print("Added {} no-fog shader copies for baked props standing at water (of {} "
                      "listed){}".format(len(extra), len(nf_targets),
                                         "; no _bsp form to copy for: " + ", ".join(nf_missing)
                                         if nf_missing else ""), file=sys.stderr)
        textures_dir = args.textures_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.output)), 'textures')
        final_text, n_split = split_lightmap_groups(final_text)
        if n_split:
            print("Gave {} lightmapped shader(s) their own lightmap pages "
                  "(q3map_lightmapBrightness, see split_lightmap_groups): the engine makes one "
                  "shader per material PER LIGHTMAP PAGE, and without this every texture on a "
                  "multi-texture wall inherits that wall's pages. Measured on de_overpass_csgo: "
                  "1383 world shaders -> 519, lighting unchanged.".format(n_split),
                  file=sys.stderr)
        final_text, renamed, copied, missing_imgs = shorten_long_names(final_text, textures_dir)
        if renamed or copied or missing_imgs:
            print("\nShortened {} shader name(s) and {} image path(s) longer than {} characters "
                  "(idTech3 names are 64 bytes; -rename's _BSP and the engine's .jpg need the "
                  "last 4). The .md3 files and the .map use the same short names.".format(
                      len(renamed), len(copied) + len(missing_imgs), QPATH_NAME_MAX), file=sys.stderr)
            for old, new in renamed[:40]:
                print("  {} -> {}".format(old, new), file=sys.stderr)
            if len(renamed) > 40:
                print("  ... and {} more".format(len(renamed) - 40), file=sys.stderr)
            for old, new in copied:
                print("  image {} -> copied to {}".format(old, new), file=sys.stderr)
            for old in missing_imgs:
                print("  ! image {} not found under {} - its short copy could not be made".format(
                    old, os.path.dirname(os.path.abspath(textures_dir))), file=sys.stderr)
        # Signature line: audit_release.py uses it to tell a converted map's
        # shaders from the game's own, so assets that only resolve through
        # ANOTHER converted map (an old loose install sharing CS:S paths) are
        # not mistaken for game content. The parser skips // comments.
        f.write("// converted by vmt2shader.py - Source materials for SOF2\n\n")
        f.write(final_text)
    if DECAL_MODULATE_SEEN:
        print("Converted {} DecalModulate material(s) as 2x modulate decals (they multiply the "
              "lit surface instead of drawing): {}".format(
                  len(DECAL_MODULATE_SEEN), ", ".join(sorted(DECAL_MODULATE_SEEN))), file=sys.stderr)
    print("NOTE: prop shaders now come in Raven's two forms - <name> for runtime "
          "entities and <name>_bsp for baked misc_models. COMPILE -bsp WITH -rename, "
          "or baked props will use the entity form and light wrongly.", file=sys.stderr)

    print("Wrote {}".format(args.output), file=sys.stderr)

    if sky_report:
        print("\n--- Sky materials: {} shader(s) generated from {} face VMTs ---"
              .format(len(sky_report), sum(c for _, _, c in sky_report)), file=sys.stderr)
        for shader_name, basename, face_count in sky_report:
            print("  - {} ({} faces) -> rename your exported images to".format(shader_name, face_count), file=sys.stderr)
            print("    {}_bk / _dn / _ft / _lf / _rt / _up".format(basename), file=sys.stderr)
        print("  Also assign this shader to the sky brush faces in your .map,", file=sys.stderr)
        print("  and adjust the cloud height (default 512) as needed.", file=sys.stderr)

    if skipped:
        print("\n--- Skipped {} tool materials (SOF2's own tools.shader defines these) ---"
              .format(len(skipped)), file=sys.stderr)
        print("vmf2map.py remaps these faces to SOF2's native tool shaders", file=sys.stderr)
        print("(tools/_caulk, _clip, _hint, _shoot, _sky, ...), so no shader is", file=sys.stderr)
        print("generated here. Check vmf2map.py's warnings for any tool material", file=sys.stderr)
        print("that has no remap entry - that's the case that needs attention.", file=sys.stderr)
        for s in skipped[:20]:
            print("  - {}".format(s), file=sys.stderr)
        if len(skipped) > 20:
            print("  ... and {} more".format(len(skipped) - 20), file=sys.stderr)

    if warnings:
        print("\n--- Warnings ({}) ---".format(len(warnings)), file=sys.stderr)
        for w in warnings:
            print("  - {}".format(w), file=sys.stderr)

    if unmapped_surfaceprops:
        print("\n--- $surfaceprop values with no surfaceparm mapping ({} unique) ---"
              .format(len(unmapped_surfaceprops)), file=sys.stderr)
        print("These materials got no surfaceparm (footstep sound/material type).", file=sys.stderr)
        print("Supply --surfaceprop-map to fill these in.", file=sys.stderr)
        for prop, count in unmapped_surfaceprops.most_common():
            print("  - {}  ({} material(s))".format(prop, count), file=sys.stderr)

    if unmapped_materials:
        print("\n--- $surfaceprop values with no q3map_material mapping ({} unique) ---"
              .format(len(unmapped_materials)), file=sys.stderr)
        print("These materials got no q3map_material (footstep sound/impact effect/decal type).", file=sys.stderr)
        print("Supply --material-map to fill these in.", file=sys.stderr)
        for prop, count in unmapped_materials.most_common():
            print("  - {}  ({} material(s))".format(prop, count), file=sys.stderr)


if __name__ == "__main__":
    main()
