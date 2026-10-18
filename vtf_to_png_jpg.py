#!/usr/bin/env python3
"""
VTF -> PNG/JPG Batch Converter (all-in-one, no external VTF tools)
====================================================================

Pure-Python VTF decoder (handles VTF versions 7.0 through 7.5, including
the 7.3+ resource-chunk header layout) covering the pixel formats you'll
actually hit in Source engine map content: DXT1 / DXT1_ONEBITALPHA / DXT3
/ DXT5, RGBA8888 family, RGB888 / BGR888, RGB565 / BGR565, BGRA4444,
BGRA5551 / BGRX5551, I8 / IA88 / A8, UV88.

Then branches each decoded texture to:
  - PNG : if it has a meaningfully-used alpha channel, OR the filename
          looks like a normal map, OR it's an unusual/paletted format.
  - JPG : everything else (opaque diffuse/spec/etc.), to save space.
  - TGA : the six faces of a sky box (skybox/<name><bk|dn|ft|lf|rt|up>),
          written as <name>_<suffix>.tga because that exact file name is what
          the game asks for when it loads a sky; see write_sky_groups.

Folder structure is preserved relative to the input root.

REQUIREMENTS
------------
    pip install numpy pillow

USAGE
-----
    python vtf_to_png_jpg.py <input_folder> <output_folder>

Example:
    python vtf_to_png_jpg.py "E:\\CSS\\Extracted\\de_vertigo_files\\materials" "E:\\CSS\\Extracted\\de_vertigo_files\\textures"

NOTES
-----
- Not implemented: P8 (paletted), UVWQ8888, UVLX8888, RGBA16161616(F)
  (HDR). These are rare in map/prop diffuse-type content; files using
  them are logged and skipped rather than guessed at.
- Only the first frame / first face / first depth-slice is extracted
  (fine for ordinary static textures; animated/cubemap textures will
  only give you frame 0 / face 0).
- TEST ON A SMALL SUBFOLDER FIRST. This decoder was written from the
  VTF format spec but not verified against your actual files - if any
  output image looks wrong (color-shifted, garbled, checkerboarded),
  stop and report exactly which file so the specific bug can be fixed.
"""

import os
import re
import sys
import struct

try:
    import numpy as np
except ImportError:
    print("Missing dependency. Run:  pip install numpy pillow")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing dependency. Run:  pip install numpy pillow")
    sys.exit(1)

JPEG_QUALITY = 92
ALPHA_OPAQUE_THRESHOLD = 250
NORMAL_MAP_HINTS = ('_normal', '_n.', '_n_', 'normal_', '_local', '_bump', 'bump_', '_nrm')

FORMAT_NAMES = {
    0: 'RGBA8888', 1: 'ABGR8888', 2: 'RGB888', 3: 'BGR888', 4: 'RGB565',
    5: 'I8', 6: 'IA88', 7: 'P8', 8: 'A8', 9: 'RGB888_BLUESCREEN',
    10: 'BGR888_BLUESCREEN', 11: 'ARGB8888', 12: 'BGRA8888', 13: 'DXT1',
    14: 'DXT3', 15: 'DXT5', 16: 'BGRX8888', 17: 'BGR565', 18: 'BGRX5551',
    19: 'BGRA4444', 20: 'DXT1_ONEBITALPHA', 21: 'BGRA5551', 22: 'UV88',
    23: 'UVWQ8888', 24: 'RGBA16161616F', 25: 'RGBA16161616', 26: 'UVLX8888',
}

SUPPORTED_UNCOMPRESSED = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 16, 17, 18, 19, 21, 22)


# ---------------------------------------------------------------- helpers --

def block_count(n):
    return (n + 3) // 4


def format_data_size(fmt, w, h):
    if fmt in (13, 20):
        return block_count(w) * block_count(h) * 8
    if fmt in (14, 15):
        return block_count(w) * block_count(h) * 16
    if fmt in (2, 3, 9, 10):
        return w * h * 3
    if fmt in (0, 1, 11, 12, 16, 23, 26):
        return w * h * 4
    if fmt in (4, 17, 18, 19, 21, 22):
        return w * h * 2
    if fmt in (5, 7, 8):
        return w * h
    if fmt == 6:
        return w * h * 2
    if fmt in (24, 25):
        return w * h * 8
    raise ValueError(f"unknown size for format {FORMAT_NAMES.get(fmt, fmt)}")


def rgb565_to_888(c):
    r = ((c >> 11) & 0x1F).astype(np.int32)
    g = ((c >> 5) & 0x3F).astype(np.int32)
    b = (c & 0x1F).astype(np.int32)
    r8 = (r << 3) | (r >> 2)
    g8 = (g << 2) | (g >> 4)
    b8 = (b << 3) | (b >> 2)
    return np.stack([r8, g8, b8], axis=-1)


def assemble_blocks(out_rgb, out_a, bw, bh, w, h):
    out_rgb = out_rgb.reshape(bh, bw, 4, 4, 3)
    out_a = out_a.reshape(bh, bw, 4, 4)
    img_rgb = out_rgb.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    img_a = out_a.transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    img = np.zeros((bh * 4, bw * 4, 4), dtype=np.uint8)
    img[..., :3] = img_rgb
    img[..., 3] = img_a
    return img[:h, :w, :]


def bc_color_part(color_bytes, nblocks, punch_through):
    """Decode the 8-byte color half shared by DXT1/DXT3/DXT5.
    punch_through=True enables DXT1's transparent-4th-color behavior."""
    c0 = color_bytes[:, 0].astype(np.uint16) | (color_bytes[:, 1].astype(np.uint16) << 8)
    c1 = color_bytes[:, 2].astype(np.uint16) | (color_bytes[:, 3].astype(np.uint16) << 8)
    idx = (color_bytes[:, 4].astype(np.uint32) |
           (color_bytes[:, 5].astype(np.uint32) << 8) |
           (color_bytes[:, 6].astype(np.uint32) << 16) |
           (color_bytes[:, 7].astype(np.uint32) << 24))

    rgb0 = rgb565_to_888(c0)
    rgb1 = rgb565_to_888(c1)

    if punch_through:
        opaque = c0 > c1
        c2 = np.where(opaque[:, None], (2 * rgb0 + rgb1) // 3, (rgb0 + rgb1) // 2)
        c3 = np.where(opaque[:, None], (rgb0 + 2 * rgb1) // 3, np.zeros_like(rgb0))
        a3 = np.where(opaque, 255, 0).astype(np.int32)
    else:
        c2 = (2 * rgb0 + rgb1) // 3
        c3 = (rgb0 + 2 * rgb1) // 3
        a3 = np.full(nblocks, 255, dtype=np.int32)

    colors = np.stack([rgb0, rgb1, c2, c3], axis=1)  # (nblocks,4,3)
    arangeN = np.arange(nblocks)
    out_rgb = np.zeros((nblocks, 16, 3), dtype=np.uint8)
    for p in range(16):
        sel = (idx >> (2 * p)) & 0x3
        out_rgb[:, p, :] = colors[arangeN, sel]
    return out_rgb, a3


def decode_dxt1(data, w, h):
    bw, bh = block_count(w), block_count(h)
    nblocks = bw * bh
    blocks = np.frombuffer(data[:nblocks * 8].ljust(nblocks * 8, b'\0'), dtype=np.uint8).reshape(nblocks, 8)
    color_bytes = blocks[:, 0:8]
    out_rgb, a3 = bc_color_part(color_bytes, nblocks, punch_through=True)

    # need per-pixel alpha (255 for idx 0-2, a3 for idx 3) - recompute idx here
    idx = (color_bytes[:, 4].astype(np.uint32) |
           (color_bytes[:, 5].astype(np.uint32) << 8) |
           (color_bytes[:, 6].astype(np.uint32) << 16) |
           (color_bytes[:, 7].astype(np.uint32) << 24))
    alphas = np.stack([np.full(nblocks, 255, dtype=np.int32)] * 3 + [a3], axis=1)
    arangeN = np.arange(nblocks)
    out_a = np.zeros((nblocks, 16), dtype=np.uint8)
    for p in range(16):
        sel = (idx >> (2 * p)) & 0x3
        out_a[:, p] = alphas[arangeN, sel]

    return assemble_blocks(out_rgb, out_a, bw, bh, w, h)


def decode_dxt3(data, w, h):
    bw, bh = block_count(w), block_count(h)
    nblocks = bw * bh
    blocks = np.frombuffer(data[:nblocks * 16].ljust(nblocks * 16, b'\0'), dtype=np.uint8).reshape(nblocks, 16)
    alpha_bytes = blocks[:, 0:8]
    color_bytes = blocks[:, 8:16]

    out_rgb, _ = bc_color_part(color_bytes, nblocks, punch_through=False)

    nibble_low = alpha_bytes & 0x0F
    nibble_high = (alpha_bytes >> 4) & 0x0F
    out_a4 = np.empty((nblocks, 16), dtype=np.uint8)
    out_a4[:, 0::2] = nibble_low
    out_a4[:, 1::2] = nibble_high
    out_a = (out_a4.astype(np.uint16) * 17).astype(np.uint8)

    return assemble_blocks(out_rgb, out_a, bw, bh, w, h)


def decode_dxt5(data, w, h):
    bw, bh = block_count(w), block_count(h)
    nblocks = bw * bh
    blocks = np.frombuffer(data[:nblocks * 16].ljust(nblocks * 16, b'\0'), dtype=np.uint8).reshape(nblocks, 16)
    a0 = blocks[:, 0].astype(np.int64)
    a1 = blocks[:, 1].astype(np.int64)
    abits = blocks[:, 2:8]
    color_bytes = blocks[:, 8:16]

    abits64 = np.zeros(nblocks, dtype=np.uint64)
    for i in range(6):
        abits64 |= (abits[:, i].astype(np.uint64) << np.uint64(8 * i))

    a_gt = a0 > a1
    a2 = np.where(a_gt, (6 * a0 + 1 * a1) // 7, (4 * a0 + 1 * a1) // 5)
    a3 = np.where(a_gt, (5 * a0 + 2 * a1) // 7, (3 * a0 + 2 * a1) // 5)
    a4 = np.where(a_gt, (4 * a0 + 3 * a1) // 7, (2 * a0 + 3 * a1) // 5)
    a5 = np.where(a_gt, (3 * a0 + 4 * a1) // 7, (1 * a0 + 4 * a1) // 5)
    a6 = np.where(a_gt, (2 * a0 + 5 * a1) // 7, np.zeros(nblocks, dtype=np.int64))
    a7 = np.where(a_gt, (1 * a0 + 6 * a1) // 7, np.full(nblocks, 255, dtype=np.int64))

    alpha_table = np.stack([a0, a1, a2, a3, a4, a5, a6, a7], axis=1)
    arangeN = np.arange(nblocks)
    out_a = np.zeros((nblocks, 16), dtype=np.uint8)
    for p in range(16):
        sel = ((abits64 >> np.uint64(3 * p)) & np.uint64(0x7)).astype(np.int64)
        out_a[:, p] = alpha_table[arangeN, sel]

    out_rgb, _ = bc_color_part(color_bytes, nblocks, punch_through=False)
    return assemble_blocks(out_rgb, out_a, bw, bh, w, h)


def decode_uncompressed(data, w, h, fmt):
    name = FORMAT_NAMES[fmt]
    needed = format_data_size(fmt, w, h)
    arr = np.frombuffer(data[:needed].ljust(needed, b'\0'), dtype=np.uint8)

    if name in ('RGBA8888', 'ABGR8888', 'ARGB8888', 'BGRA8888', 'BGRX8888'):
        px = arr.reshape(h, w, 4)
        if name == 'RGBA8888':
            r, g, b, a = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
        elif name == 'ABGR8888':
            a, b, g, r = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
        elif name == 'ARGB8888':
            a, r, g, b = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
        else:  # BGRA8888 / BGRX8888
            b, g, r, a = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
            if name == 'BGRX8888':
                a = np.full_like(a, 255)
        return np.stack([r, g, b, a], axis=-1)

    if name in ('RGB888', 'RGB888_BLUESCREEN'):
        px = arr.reshape(h, w, 3)
        r, g, b = px[..., 0], px[..., 1], px[..., 2]
        a = np.full((h, w), 255, dtype=np.uint8)
        return np.stack([r, g, b, a], axis=-1)

    if name in ('BGR888', 'BGR888_BLUESCREEN'):
        px = arr.reshape(h, w, 3)
        b, g, r = px[..., 0], px[..., 1], px[..., 2]
        a = np.full((h, w), 255, dtype=np.uint8)
        return np.stack([r, g, b, a], axis=-1)

    if name == 'I8':
        px = arr.reshape(h, w)
        return np.stack([px, px, px, np.full_like(px, 255)], axis=-1)

    if name == 'IA88':
        px = arr.reshape(h, w, 2)
        i, a = px[..., 0], px[..., 1]
        return np.stack([i, i, i, a], axis=-1)

    if name == 'A8':
        px = arr.reshape(h, w)
        white = np.full_like(px, 255)
        return np.stack([white, white, white, px], axis=-1)

    if name in ('RGB565', 'BGR565'):
        px16 = arr.view('<u2')[:w * h].reshape(h, w).astype(np.int32)
        top5 = (px16 >> 11) & 0x1F
        mid6 = (px16 >> 5) & 0x3F
        bot5 = px16 & 0x1F
        top8 = ((top5 << 3) | (top5 >> 2)).astype(np.uint8)
        mid8 = ((mid6 << 2) | (mid6 >> 4)).astype(np.uint8)
        bot8 = ((bot5 << 3) | (bot5 >> 2)).astype(np.uint8)
        a = np.full((h, w), 255, dtype=np.uint8)
        if name == 'RGB565':
            return np.stack([top8, mid8, bot8, a], axis=-1)
        return np.stack([bot8, mid8, top8, a], axis=-1)

    if name == 'BGRA4444':
        px16 = arr.view('<u2')[:w * h].reshape(h, w).astype(np.int32)
        b4, g4, r4, a4 = px16 & 0xF, (px16 >> 4) & 0xF, (px16 >> 8) & 0xF, (px16 >> 12) & 0xF
        exp = lambda x: ((x << 4) | x).astype(np.uint8)
        return np.stack([exp(r4), exp(g4), exp(b4), exp(a4)], axis=-1)

    if name in ('BGRA5551', 'BGRX5551'):
        px16 = arr.view('<u2')[:w * h].reshape(h, w).astype(np.int32)
        b5, g5, r5, a1 = px16 & 0x1F, (px16 >> 5) & 0x1F, (px16 >> 10) & 0x1F, (px16 >> 15) & 0x1
        exp5 = lambda x: ((x << 3) | (x >> 2)).astype(np.uint8)
        a = (a1 * 255).astype(np.uint8) if name == 'BGRA5551' else np.full((h, w), 255, dtype=np.uint8)
        return np.stack([exp5(r5), exp5(g5), exp5(b5), a], axis=-1)

    if name == 'UV88':
        px = arr.reshape(h, w, 2)
        u, v = px[..., 0], px[..., 1]
        b = np.full((h, w), 128, dtype=np.uint8)
        a = np.full((h, w), 255, dtype=np.uint8)
        return np.stack([u, v, b, a], axis=-1)

    raise ValueError(f"unsupported uncompressed format {name}")


def decode_pixels(data, w, h, fmt):
    if fmt in (13, 20):
        return decode_dxt1(data, w, h)
    if fmt == 14:
        return decode_dxt3(data, w, h)
    if fmt == 15:
        return decode_dxt5(data, w, h)
    if fmt in SUPPORTED_UNCOMPRESSED:
        return decode_uncompressed(data, w, h, fmt)
    raise ValueError(f"unsupported format {FORMAT_NAMES.get(fmt, fmt)} ({fmt})")


# ------------------------------------------------------------- vtf header --

def parse_vtf_header(data):
    if data[0:4] != b'VTF\x00':
        raise ValueError("bad VTF signature")
    ver_maj, ver_min, header_size = struct.unpack_from('<III', data, 4)
    width, height = struct.unpack_from('<HH', data, 16)
    frames, first_frame = struct.unpack_from('<HH', data, 24)
    high_res_format = struct.unpack_from('<i', data, 52)[0]
    mipmap_count = data[56]
    low_res_format = struct.unpack_from('<i', data, 57)[0]
    low_res_w = data[61]
    low_res_h = data[62]
    version = (ver_maj, ver_min)

    if version >= (7, 3):
        num_resources = struct.unpack_from('<I', data, 68)[0]
        entries_offset = 80
        image_offset = None
        for i in range(num_resources):
            eo = entries_offset + i * 8
            tag = data[eo] | (data[eo + 1] << 8) | (data[eo + 2] << 16)
            resource_value = struct.unpack_from('<I', data, eo + 4)[0]
            if tag == 0x30:  # VTF_LEGACY_RSRC_IMAGE
                image_offset = resource_value
        if image_offset is None:
            raise ValueError("no image-data resource found (7.3+ file)")
        data_start = image_offset
    else:
        thumb_size = 0
        if low_res_format != -1:
            thumb_size = format_data_size(low_res_format, low_res_w, low_res_h)
        data_start = header_size + thumb_size

    return {
        'version': version, 'width': width, 'height': height,
        'frames': frames, 'mipmap_count': mipmap_count,
        'format': high_res_format, 'data_start': data_start,
    }


def extract_highres_image(data, info):
    w, h, fmt = info['width'], info['height'], info['format']
    mipmap_count = max(1, info['mipmap_count'])
    frames = max(1, info['frames'])
    offset = info['data_start']

    for mip in range(mipmap_count - 1, 0, -1):
        mw, mh = max(1, w >> mip), max(1, h >> mip)
        offset += format_data_size(fmt, mw, mh) * frames

    size0 = format_data_size(fmt, w, h)
    chunk = data[offset:offset + size0]
    return decode_pixels(chunk, w, h, fmt)


# --------------------------------------------------------------- pipeline --

def is_normal_map(path):
    name = os.path.basename(path).lower()
    return any(hint in name for hint in NORMAL_MAP_HINTS)


def has_meaningful_alpha(img):
    if img.mode != 'RGBA':
        return False
    alpha = img.getchannel('A')
    lo, hi = alpha.getextrema()
    return lo < ALPHA_OPAQUE_THRESHOLD


SKY_SUFFIX_RE = re.compile(r'^(?P<base>.+?)_?(?P<suffix>bk|dn|ft|lf|rt|up)$', re.IGNORECASE)


def sky_output_rel(rel_noext):
    """If this file lives under a skybox/ folder and its name ends in one
    of the 6 cube-face suffixes, insert an underscore before that suffix
    for the output name.

    Source's own skybox filenames genuinely have no underscore (confirmed
    directly against real content: cxbk.vmt, cxdn.vmt, etc.) - that's not
    a bug in the extractor, it's just what Source actually uses, which is
    why vmf_asset_extractor.py correctly looks for files named that way.
    But SOF2's skyParms convention (and the shader vmt2shader.py generates)
    expects the underscored form: cx_bk, cx_dn, etc. Doing the rename here,
    at .vtf->image conversion time, is the right place for it - by the time
    a shader references these images, the underscored name needs to
    already exist on disk."""
    parts = rel_noext.replace('\\', '/').split('/')
    if 'skybox' not in [p.lower() for p in parts[:-1]]:
        return rel_noext
    base_name = parts[-1]
    m = SKY_SUFFIX_RE.match(base_name)
    if not m:
        return rel_noext
    new_name = '{}_{}'.format(m.group('base'), m.group('suffix').lower())
    parts[-1] = new_name
    return '/'.join(parts)


def convert_file(vtf_path, in_root, out_root, sky_faces=None):
    """Convert one .vtf. A sky-box face is not written here: it is decoded and
    handed to sky_faces (group -> {suffix: face}) so write_sky_groups can
    treat the six faces of one box together (they share one exposure)."""
    rel = os.path.relpath(vtf_path, in_root)
    rel_src = os.path.splitext(rel)[0]
    rel_noext = sky_output_rel(rel_src)

    with open(vtf_path, 'rb') as f:
        data = f.read()

    info = parse_vtf_header(data)
    rgba = extract_highres_image(data, info)

    if sky_faces is not None and rel_noext != rel_src:
        group, suffix = rel_noext.rsplit('_', 1)
        sky_faces.setdefault(group, {})[suffix.lower()] = {
            'rgba': rgba, 'src_rel': rel_src.replace('\\', '/'), 'vtf': vtf_path}
        return 'sky'

    img = Image.fromarray(rgba, 'RGBA')

    use_png = is_normal_map(vtf_path) or has_meaningful_alpha(img)
    ext = '.png' if use_png else '.jpg'
    out_path = os.path.join(out_root, rel_noext + ext)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if use_png:
        img.save(out_path, 'PNG', optimize=True)
    else:
        img.convert('RGB').save(out_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)

    return ext


# ------------------------------------------------------------------ sky box --
#
# What the game does with a sky box, read from the retail SoF2MP.exe (1.03):
#   * ParseSkyParms builds each face name with "%s_%s.tga" and loads it with
#     R_FindImageFile; a face that fails to load becomes tr.defaultImage.
#   * RB_StageIteratorSky draws the outer box only if face 0 (suffix order
#     rt bk lf ft up dn, table at 0x56db68) loaded: if outerbox[0] is null or
#     tr.defaultImage the whole box is skipped and the sky shows whatever was
#     on screen before (the "hall of mirrors").
# So the faces must exist as <farbox>_<suffix>.tga exactly. A .png or .jpg
# under that name is never asked for. They are written as 24-bit TGA: the sky
# stage is drawn opaque, so alpha has no use there.
#
# Two things Source does with sky faces that idTech3 cannot, converted here:
#
# 1. Compressed-HDR faces (CS:GO's "_hdr" skies, Source's
#    sky_hdr_compressed shader). The texel holds a colour normalised to its
#    largest channel and, in alpha, that colour's intensity; the shader
#    multiplies the two. Read as a plain image, every face is a flat pale
#    cyan and the clouds only exist in alpha. A box is treated as compressed
#    HDR when a face VMT names its texture in $hdrcompressedtexture, or,
#    without VMTs, when its pixels have that shape: alpha in use and the
#    largest channel at full scale almost everywhere (an ordinary image's
#    brightness lives in its RGB, so its largest channel varies).
#    Decoded as rgb * alpha, then one exposure for all six faces so they
#    stay seamless: the 99.9th percentile maps to white (only the hottest
#    0.1% - the sun - can clip). Relative brightness between texels is kept
#    exactly; the absolute level is what CS:GO's auto-exposure would choose
#    at run time and has no fixed value to copy.
#
# 2. $basetexturetransform. Valve's skies store the side faces at half
#    height (512x256) and the face VMT says "scale 1 2": the image covers
#    the top half of the face and the clamped last row fills the rest (the
#    horizon, never seen below the world). idTech3 has no per-face
#    transform, so the face is written out already transformed. With no
#    VMT to read, a side face exactly half as tall as it is wide is given
#    Valve's "scale 1 2" - no other reading of a 2:1 cube face exists.

SKY_SIDE_SUFFIXES = ('rt', 'bk', 'lf', 'ft')
SKY_EXPOSURE_PERCENTILE = 99.9
HDR_FULL_CHANNEL = 0.9          # "largest channel at full scale" ...
HDR_FULL_FRACTION = 0.95        # ... for this share of the pixels

_VMT_KV_RE = re.compile(r'"?\$([A-Za-z0-9_]+)"?[ \t]+"([^"\n]*)"|"?\$([A-Za-z0-9_]+)"?[ \t]+([^\s"{}]+)')


def _vmt_params(text):
    out = {}
    text = re.sub(r'//[^\n]*', '', text)
    for m in _VMT_KV_RE.finditer(text):
        key = (m.group(1) or m.group(3)).lower()
        val = m.group(2) if m.group(1) else m.group(4)
        out.setdefault(key, val.strip())
    return out


def _tex_key(v):
    v = v.strip().replace('\\', '/').lower()
    v = re.sub(r'\.vtf$', '', v)
    if v.startswith('materials/'):
        v = v[len('materials/'):]
    return v


def read_sky_vmts(in_root):
    """{texture key: {'transform': str or None, 'hdr_compressed': bool}} for
    every texture named by a VMT in any skybox/ folder under in_root."""
    info = {}
    for dirpath, _dirs, files in os.walk(in_root):
        rel_dir = os.path.relpath(dirpath, in_root).replace('\\', '/').lower()
        if 'skybox' not in rel_dir.split('/'):
            continue
        for fn in files:
            if not fn.lower().endswith('.vmt'):
                continue
            try:
                with open(os.path.join(dirpath, fn), 'r', encoding='utf-8', errors='replace') as f:
                    kv = _vmt_params(f.read())
            except OSError:
                continue
            transform = kv.get('basetexturetransform')
            for key in ('basetexture', 'hdrcompressedtexture', 'hdrbasetexture'):
                if key in kv:
                    e = info.setdefault(_tex_key(kv[key]), {'transform': None, 'hdr_compressed': False})
                    if transform and not e['transform']:
                        e['transform'] = transform
                    if key == 'hdrcompressedtexture':
                        e['hdr_compressed'] = True
    return info


def parse_texture_transform(text):
    """Source's "center cx cy scale sx sy rotate r translate tx ty" ->
    dict of floats; any part left out keeps Source's default."""
    t = {'center': (0.5, 0.5), 'scale': (1.0, 1.0), 'rotate': 0.0, 'translate': (0.0, 0.0)}
    toks = (text or '').split()
    i = 0
    while i < len(toks):
        k = toks[i].lower()
        try:
            if k in ('center', 'scale', 'translate') and i + 2 < len(toks):
                t[k] = (float(toks[i + 1]), float(toks[i + 2]))
                i += 3
                continue
            if k == 'rotate' and i + 1 < len(toks):
                t['rotate'] = float(toks[i + 1])
                i += 2
                continue
        except ValueError:
            pass
        i += 1
    return t


def apply_texture_transform(rgb, t):
    """Bake a (center, scale, translate) transform into the face image, with
    the texture clamped at its edges as Source's sky faces are. Face coord
    u in [0,1] reads texture coord (u - c) * s + c + translate."""
    h, w = rgb.shape[:2]
    sx, sy = t['scale']
    out_w = max(1, int(round(w * abs(sx))))
    out_h = max(1, int(round(h * abs(sy))))
    cx, cy = t['center']
    tx, ty = t['translate']
    u = (np.arange(out_w) + 0.5) / out_w
    v = (np.arange(out_h) + 0.5) / out_h
    tu = (u - cx) * sx + cx + tx
    tv = (v - cy) * sy + cy + ty
    cols = np.clip(np.floor(tu * w).astype(int), 0, w - 1)
    rows = np.clip(np.floor(tv * h).astype(int), 0, h - 1)
    return rgb[rows][:, cols]


def looks_hdr_compressed(rgba):
    a = rgba[..., 3]
    if int(a.min()) >= ALPHA_OPAQUE_THRESHOLD:
        return False
    maxch = rgba[..., :3].max(axis=-1)
    return float((maxch >= HDR_FULL_CHANNEL * 255).mean()) >= HDR_FULL_FRACTION


def write_sky_groups(sky_faces, in_root, out_root, log=print):
    vmt_info = read_sky_vmts(in_root)
    written = 0
    for group, faces in sorted(sky_faces.items()):
        missing = [s for s in ('rt', 'bk', 'lf', 'ft', 'up', 'dn') if s not in faces]
        if missing:
            log("  sky {}: no face for {} - the game skips the whole box "
                "if 'rt' is missing".format(group, ', '.join(missing)))
        infos = {s: vmt_info.get(_tex_key(f['src_rel']), {}) for s, f in faces.items()}
        by_vmt = any(i.get('hdr_compressed') for i in infos.values())
        big = [f['rgba'] for f in faces.values() if f['rgba'].shape[0] * f['rgba'].shape[1] >= 4096]
        by_pixels = bool(big) and all(looks_hdr_compressed(r) for r in big)
        hdr = by_vmt or by_pixels

        decoded = {}
        for s, f in faces.items():
            px = f['rgba'].astype(np.float64) / 255.0
            decoded[s] = px[..., :3] * px[..., 3:4] if hdr else px[..., :3]
        if hdr:
            sample = np.concatenate([d.max(axis=-1).ravel() for s, d in decoded.items()
                                     if d.shape[0] * d.shape[1] >= 4096] or
                                    [d.max(axis=-1).ravel() for d in decoded.values()])
            white = float(np.percentile(sample, SKY_EXPOSURE_PERCENTILE)) or 1.0
            for s in decoded:
                decoded[s] = np.clip(decoded[s] / white, 0.0, 1.0)
            log("  sky {}: compressed HDR ({}), decoded as rgb*alpha, "
                "white point {:.4f}".format(group, 'VMT $hdrcompressedtexture' if by_vmt
                                            else 'detected from pixels', white))

        for s, d in sorted(decoded.items()):
            img = d
            tr_text = infos[s].get('transform')
            h, w = img.shape[:2]
            if tr_text:
                t = parse_texture_transform(tr_text)
                if t['rotate']:
                    log("  sky {}_{}: $basetexturetransform rotate {:g} ignored".format(
                        group, s, t['rotate']))
                if t['scale'] != (1.0, 1.0) or t['translate'] != (0.0, 0.0):
                    img = apply_texture_transform(img, t)
                    log("  sky {}_{}: $basetexturetransform \"{}\" baked in ({}x{} -> {}x{})".format(
                        group, s, tr_text, w, h, img.shape[1], img.shape[0]))
            elif s in SKY_SIDE_SUFFIXES and w == 2 * h:
                img = apply_texture_transform(img, parse_texture_transform('center 0 0 scale 1 2'))
                log("  sky {}_{}: half-height side, no VMT found - Valve's \"scale 1 2\" "
                    "baked in ({}x{} -> {}x{})".format(group, s, w, h, img.shape[1], img.shape[0]))
            out8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
            base = os.path.join(out_root, group + '_' + s)
            os.makedirs(os.path.dirname(base), exist_ok=True)
            Image.fromarray(out8, 'RGB').save(base + '.tga', 'TGA')
            for stale in ('.png', '.jpg', '.jpeg'):
                if os.path.isfile(base + stale):
                    os.remove(base + stale)
            written += 1
    return written


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    in_root, out_root = sys.argv[1], sys.argv[2]
    counts = {'.png': 0, '.jpg': 0, 'sky': 0}
    fail_reasons = {}
    sky_faces = {}

    for dirpath, _, filenames in os.walk(in_root):
        for fn in filenames:
            if not fn.lower().endswith('.vtf'):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, in_root)
            try:
                ext = convert_file(full, in_root, out_root, sky_faces)
                counts[ext] += 1
            except Exception as e:
                reason = str(e)
                fail_reasons.setdefault(reason, []).append(rel)

    sky_written = 0
    if sky_faces:
        print("\nSky boxes (written as .tga, the name the game asks for):")
        try:
            sky_written = write_sky_groups(sky_faces, in_root, out_root)
        except Exception as e:
            fail_reasons.setdefault('sky box: ' + str(e), []).extend(sorted(sky_faces))

    total_fail = sum(len(v) for v in fail_reasons.values())
    print(f"\nDone. PNG: {counts['.png']}  JPG: {counts['.jpg']}  Sky TGA: {sky_written}  Failed: {total_fail}")
    if fail_reasons:
        print("\nFailure breakdown:")
        for reason, files in sorted(fail_reasons.items(), key=lambda x: -len(x[1])):
            print(f"  {reason}: {len(files)} file(s)")
            for f in files[:3]:
                print(f"      e.g. {f}")
    print(f"\nOutput: {out_root}")


if __name__ == '__main__':
    main()
