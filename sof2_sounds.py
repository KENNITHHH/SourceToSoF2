#!/usr/bin/env python3
"""
sof2_sounds.py - Source soundscapes and sound files, for SOF2.

Shared by vmf_asset_extractor.py (which sound files a map needs) and
vmf2map.py (target_speaker entities + the converted sound files).

WHAT SOURCE DOES
----------------
A Source map's ambience is almost all SOUNDSCAPES: env_soundscape entities
name a soundscape from a scripts/soundscapes_*.txt file, and the soundscape
lists
  playlooping   a looping wave, either "ambient" (heard everywhere in the
                soundscape, no position) or at "position" N, which the
                env_soundscape maps to an entity name in its "positionN" key;
  playrandom    one of a list of waves, every "time" "a,b" seconds, again
                ambient or at a position;
  playsoundscape  another soundscape, optionally with all its sounds moved to
                one position ("positionoverride") or only its ambient ones
                ("ambientpositionoverride"), and a volume factor.

WHAT SOF2 HAS
-------------
target_speaker: "noise" (a .wav or .mp3 path), spawnflags 1 = looped and
playing from map start, and "wait"/"random" together = a non-looping sound
that triggers itself every wait +- random seconds. It has no volume key and
no notion of a soundscape region, so:
  - a positioned sound becomes a speaker at that position;
  - an ambient sound becomes a speaker at the env_soundscape itself (heard
    around it, fading with distance - the nearest SOF2 has to "everywhere in
    this area"), and the same wave is not repeated within AMBIENT_MERGE_RANGE
    of another speaker already playing it;
  - the volume is baked into the converted file (a copy per volume used);
  - a list of random waves becomes one speaker per wave, each at N times the
    interval, so the soundscape's overall rate is kept.

SOUND FILES
-----------
SOF2's own sounds are 16-bit PCM mono .wav (22050 Hz) and mono MPEG-1 layer
3 .mp3 (44100 Hz) - checked on files from its own install. idTech3's loader
takes 8/16-bit PCM only, and only mono. Source's .wav files are PCM, MS-ADPCM
(compressed) or MP3 inside a .wav, mono or stereo, and loop through a 'smpl'
or 'cue ' marker rather than as a whole. convert_sound turns any of them into
something SOF2 plays: PCM16 mono .wav (ADPCM decoded, stereo mixed down,
volume applied, a loop trimmed to its loop region), or the MP3 data as an
.mp3 file when the source carried MP3.
"""

import math
import os
import re
import struct

AMBIENT_MERGE_RANGE = 768.0         # map units
SOUND_CHARS = '*#@><^)}$!?&~`+%'    # Source's leading sound-channel characters
SOF2_RATES = (11025, 22050, 44100)
# What SOF2 mixes at. The engine loads a .wav at any of the rates above, but
# resamples anything else to this one as it loads and says so for every file:
#   WARNING: <name> is not a 22kHz wav file
# A map full of those reads like a broken map to whoever is playing it, and the
# engine throws the extra samples away regardless, so the conversion writes
# 22050 Hz and the warning never appears. 44100 is halved by averaging each
# pair (plain decimation would alias); 11025 is interpolated up, which adds no
# detail but costs nothing audible.
SOF2_NATIVE_RATE = 22050


# ------------------------------------------------------------- KeyValues --

_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|(\{)|(\})|([^\s"{}]+)')


def _tokens(text):
    text = re.sub(r'//[^\n]*', '', text)
    for m in _TOKEN_RE.finditer(text):
        if m.group(2):
            yield '{'
        elif m.group(3):
            yield '}'
        else:
            yield m.group(1) if m.group(1) is not None else m.group(4)


def parse_keyvalues(text):
    """Valve KeyValues with repeated keys kept: [(key, str | [pairs])]."""
    toks = list(_tokens(text))
    pos = [0]

    def block():
        out = []
        while pos[0] < len(toks):
            t = toks[pos[0]]
            if t == '}':
                pos[0] += 1
                return out
            if t == '{':
                pos[0] += 1          # stray brace
                continue
            key = t
            pos[0] += 1
            if pos[0] >= len(toks):
                break
            nxt = toks[pos[0]]
            if nxt == '{':
                pos[0] += 1
                out.append((key, block()))
            elif nxt == '}':
                out.append((key, ''))
            else:
                pos[0] += 1
                out.append((key, nxt))
        return out
    return block()


def load_soundscapes(paths):
    """{soundscape name (lower): body} from soundscape script files, in order;
    a later file's definition replaces an earlier one's (pass the map's own
    file last)."""
    defs = {}
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                kv = parse_keyvalues(f.read())
        except OSError:
            continue
        for name, body in kv:
            if isinstance(body, list):
                defs[name.strip().lower()] = body
    return defs


def clean_wave(path):
    """Source wave reference -> 'sound/...' relative path with forward slashes."""
    v = path.strip().lstrip(SOUND_CHARS).replace('\\', '/').lstrip('/')
    if not v.lower().startswith('sound/'):
        v = 'sound/' + v
    return v


def _num(v, default):
    """'0.4' -> 0.4, '10,25' -> (10, 25) mean for scalars."""
    try:
        parts = [float(x) for x in str(v).split(',') if x.strip()]
    except ValueError:
        return default
    if not parts:
        return default
    return sum(parts) / len(parts)


def _range(v, default):
    try:
        parts = [float(x) for x in str(v).split(',') if x.strip()]
    except ValueError:
        return default
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) >= 2:
        return (min(parts[0], parts[1]), max(parts[0], parts[1]))
    return default


def soundscape_items(name, defs, position_override=None, ambient_override=None,
                     volume_scale=1.0, depth=0, missing=None):
    """Flatten one soundscape into
    [{'kind': 'loop'|'random', 'waves': [...], 'volume': v, 'time': (a, b),
      'position': int or None}]."""
    body = defs.get((name or '').strip().lower())
    if body is None:
        if missing is not None:
            missing.add(name)
        return []
    if depth > 8:
        return []
    items = []
    for key, val in body:
        k = key.lower()
        if k in ('playlooping', 'playrandom') and isinstance(val, list):
            d = {}
            waves = []
            for kk, vv in val:
                kl = kk.lower()
                if kl == 'wave' and isinstance(vv, str):
                    waves.append(clean_wave(vv))
                elif kl == 'rndwave' and isinstance(vv, list):
                    waves.extend(clean_wave(w) for wk, w in vv
                                 if wk.lower() == 'wave' and isinstance(w, str))
                elif isinstance(vv, str):
                    d.setdefault(kl, vv)
            if not waves:
                continue
            pos = None
            if 'position' in d:
                try:
                    pos = int(float(d['position']))
                except ValueError:
                    pos = None
            if position_override is not None:
                pos = position_override
            elif pos is None and ambient_override is not None:
                pos = ambient_override
            items.append({'kind': 'loop' if k == 'playlooping' else 'random',
                          'waves': waves,
                          'volume': max(0.0, min(1.0, _num(d.get('volume', '1'), 1.0) * volume_scale)),
                          'time': _range(d.get('time', '10,10'), (10.0, 10.0)),
                          'position': pos})
        elif k == 'playsoundscape' and isinstance(val, list):
            d = {kk.lower(): vv for kk, vv in val if isinstance(vv, str)}
            po = ao = None
            try:
                if 'positionoverride' in d:
                    po = int(float(d['positionoverride']))
                if 'ambientpositionoverride' in d:
                    ao = int(float(d['ambientpositionoverride']))
            except ValueError:
                pass
            if position_override is not None:
                po = position_override
            items.extend(soundscape_items(d.get('name'), defs, po,
                                          ao if ao is not None else ambient_override,
                                          volume_scale * _num(d.get('volume', '1'), 1.0),
                                          depth + 1, missing))
    return items


# ------------------------------------------------- SOF2's own sound sets --
#
# SOF2 (1.03 SDK, g_target.c SP_target_speaker) has a second kind of
# target_speaker: "soundSet" names an ambient set from sound/sound.txt - a
# looped wave and/or random sub-waves, with a fade-out radius - and that
# speaker is EF_PERMANENT, sent to the client once in the game state and
# never in a snapshot (cg_ents.c CG_AddLocalSet plays it client side). So it
# costs nothing against the 256-entity snapshot room target_speaker "noise"
# shares with the networked props.
#
# Only Raven's own sets are used (sound.txt is one shared file; a set missing
# from a player's copy drops them from the server - AS_ParseSets ERR_DROP).
# A Source sound gets one when its waves clearly are that kind of sound;
# anything else stays a CS sound on an ordinary "noise" speaker.
#
# (regex on the Source wave path, Raven localSet, its radius in sound.txt).
# First match wins. Edit freely: a set name here must exist in SOF2's
# sound/sound.txt (all of these are in the 1.03 retail file).
# Empty on purpose: tried on de_overpass_csgo (2026-10), and SOF2's own
# sets did not fit CS maps well enough - every sound is the map's own CS
# sound again. The machinery stays: an entry here, e.g.
#     (r'(^|/)fan\\d*_loop\\.wav$', 'fan', 500),
# turns matching sounds into free "soundSet" speakers.
RAVEN_SET_RULES = []

# Sounds left out altogether - one-off gags that make no sense on a speaker
# repeating forever (de_overpass_csgo's flushing toilet). Matched against the
# Source wave path, case-insensitive.
SOUND_SKIP_KEYWORDS = ('flush', 'toilet')


def raven_set_for(waves):
    """(set name, radius) when every wave in the list matches the same
    RAVEN_SET_RULES entry, else None."""
    found = None
    for w in waves:
        wl = w.lower().replace('\\', '/')
        hit = None
        for rx, name, radius in RAVEN_SET_RULES:
            if re.search(rx, wl):
                hit = (name, radius)
                break
        if hit is None or (found is not None and hit != found):
            return None
        found = hit
    return found


def is_skipped_sound(wave):
    wl = wave.lower()
    return any(k in wl for k in SOUND_SKIP_KEYWORDS)


def soundscape_waves(names, defs):
    """Every wave the given soundscapes play (for the extractor)."""
    out = set()
    for n in names:
        for it in soundscape_items(n, defs):
            out.update(it['waves'])
    return out


# ----------------------------------------------------------- conversion --

_MSADPCM_ADAPT = (230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230)
_MSADPCM_COEF = ((256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232))


def _clamp16(v):
    return -32768 if v < -32768 else (32767 if v > 32767 else v)


def _decode_msadpcm(data, channels, block_align, samples_per_block, coefs):
    """MS-ADPCM (WAVE_FORMAT_ADPCM, 0x0002) -> list of channel sample lists."""
    out = [[] for _ in range(channels)]
    pos = 0
    while pos + 7 * channels <= len(data):
        blk = data[pos:pos + block_align]
        pos += block_align
        if len(blk) < 7 * channels:
            break
        o = 0
        pred = list(blk[o:o + channels])
        o += channels
        delta = list(struct.unpack_from('<%dh' % channels, blk, o))
        o += 2 * channels
        s1 = list(struct.unpack_from('<%dh' % channels, blk, o))
        o += 2 * channels
        s2 = list(struct.unpack_from('<%dh' % channels, blk, o))
        o += 2 * channels
        cf = [coefs[p] if p < len(coefs) else coefs[0] for p in pred]
        for c in range(channels):
            out[c].append(s2[c])
            out[c].append(s1[c])
        produced = 2
        nib_list = []
        for b in blk[o:]:
            nib_list.append(b >> 4)
            nib_list.append(b & 0x0F)
        ch = 0
        for n in nib_list:
            if produced >= samples_per_block:
                break
            sn = n - 16 if n & 8 else n
            p = (s1[ch] * cf[ch][0] + s2[ch] * cf[ch][1]) >> 8
            p = _clamp16(p + sn * delta[ch])
            s2[ch] = s1[ch]
            s1[ch] = p
            delta[ch] = max(16, (_MSADPCM_ADAPT[n] * delta[ch]) >> 8)
            out[ch].append(p)
            ch += 1
            if ch == channels:
                ch = 0
                produced += 1
    return out


def read_wav(data):
    """-> dict(rate, channels=[samples...] (int16), loop=(start, end) or None,
    mp3=bytes or None). Raises ValueError for anything it cannot decode."""
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('not a RIFF/WAVE file')
    i = 12
    fmt = None
    raw = None
    loop = None
    cue_start = None
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        n = struct.unpack_from('<I', data, i + 4)[0]
        body = data[i + 8:i + 8 + n]
        if cid == b'fmt ':
            fmt = body
        elif cid == b'data':
            raw = body
        elif cid == b'smpl' and len(body) >= 36:
            nloops = struct.unpack_from('<I', body, 28)[0]
            if nloops and len(body) >= 36 + 24:
                start, end = struct.unpack_from('<II', body, 36 + 8)
                loop = (start, end)
        elif cid == b'cue ' and len(body) >= 4 + 24:
            if struct.unpack_from('<I', body, 0)[0]:
                cue_start = struct.unpack_from('<I', body, 4 + 20)[0]
        i += 8 + n + (n & 1)
    if fmt is None or raw is None:
        raise ValueError('no fmt/data chunk')
    tag, nch, rate, _bps, align, bits = struct.unpack_from('<HHIIHH', fmt, 0)
    if loop is None and cue_start is not None:
        loop = (cue_start, None)
    if tag == 0x55:                      # MPEG layer 3 inside a .wav
        return {'rate': rate, 'channels': None, 'loop': loop, 'mp3': raw, 'nch': nch}
    if tag == 1 or tag == 0xFFFE:
        if bits == 16:
            cnt = len(raw) // 2
            s = struct.unpack('<%dh' % cnt, raw[:cnt * 2])
        elif bits == 8:
            s = [(b - 128) << 8 for b in raw]
        elif bits == 24:
            s = [struct.unpack('<i', raw[k:k + 3] + (b'\xff' if raw[k + 2] & 0x80 else b'\0'))[0] >> 8
                 for k in range(0, len(raw) - 2, 3)]
        else:
            raise ValueError('PCM with %d bits per sample' % bits)
        chans = [list(s[c::nch]) for c in range(nch)]
    elif tag == 2:
        spb = struct.unpack_from('<H', fmt, 18)[0] if len(fmt) >= 20 else 0
        ncoef = struct.unpack_from('<H', fmt, 20)[0] if len(fmt) >= 22 else 0
        coefs = [struct.unpack_from('<hh', fmt, 22 + 4 * k) for k in range(ncoef)] or list(_MSADPCM_COEF)
        if not spb:
            spb = (align - 7 * nch) * 2 // nch + 2
        chans = _decode_msadpcm(raw, nch, align, spb, coefs)
    else:
        raise ValueError('wave format 0x%04x is not supported' % tag)
    return {'rate': rate, 'channels': chans, 'loop': loop, 'mp3': None, 'nch': nch}


def _resample(s, src, dst):
    if src == dst or not s:
        return s
    if src % dst == 0:
        # Exact integer ratio (44100 -> 22050): average each group instead of
        # dropping samples, so the top octave folds down quietly rather than
        # aliasing into the audible band.
        g = src // dst
        return [sum(s[i:i + g]) // len(s[i:i + g]) for i in range(0, len(s) - g + 1, g)]
    n = max(1, int(round(len(s) * dst / float(src))))
    step = src / float(dst)
    out = []
    for k in range(n):
        x = k * step
        i = int(x)
        f = x - i
        a = s[i] if i < len(s) else s[-1]
        b = s[i + 1] if i + 1 < len(s) else a
        out.append(int(round(a + (b - a) * f)))
    return out


def write_wav16(samples, rate):
    data = struct.pack('<%dh' % len(samples), *samples)
    fmt = struct.pack('<HHIIHH', 1, 1, rate, rate * 2, 2, 16)
    return (b'RIFF' + struct.pack('<I', 4 + 8 + len(fmt) + 8 + len(data)) + b'WAVE' +
            b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(data)) + data)


def convert_sound(data, volume=1.0, looped=False):
    """Source sound bytes -> (bytes, '.wav' | '.mp3', notes). MP3 data is
    passed through (SOF2 plays mono MP3); its volume and loop point cannot be
    applied, which the notes say."""
    notes = []
    if data[:3] == b'ID3' or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        if abs(volume - 1.0) > 1e-3:
            notes.append('mp3: volume not applied')
        return data, '.mp3', notes
    w = read_wav(data)
    if w['mp3'] is not None:
        if w['nch'] != 1:
            notes.append('stereo mp3 inside the .wav - SOF2 may not play it')
        if abs(volume - 1.0) > 1e-3:
            notes.append('mp3: volume not applied')
        return w['mp3'], '.mp3', notes
    chans = w['channels']
    if len(chans) == 1:
        mono = chans[0]
    else:
        n = min(len(c) for c in chans)
        mono = [sum(c[k] for c in chans) // len(chans) for k in range(n)]
        notes.append('stereo mixed to mono')
    if looped and w['loop']:
        a, b = w['loop']
        a = max(0, min(a, len(mono)))
        b = len(mono) if b is None else max(a + 1, min(b + 1, len(mono)))
        if a > 0 or b < len(mono):
            mono = mono[a:b]
            notes.append('trimmed to its loop region')
    rate = w['rate']
    if rate != SOF2_NATIVE_RATE:
        mono = _resample(mono, rate, SOF2_NATIVE_RATE)
        notes.append('resampled %d -> %d Hz' % (rate, SOF2_NATIVE_RATE))
        rate = SOF2_NATIVE_RATE
    if abs(volume - 1.0) > 1e-3:
        mono = [_clamp16(int(round(x * volume))) for x in mono]
    return write_wav16(mono, rate), '.wav', notes


def variant_name(rel_path, volume, looped, ext):
    """sound/a/b.wav at volume 0.4, looped -> sound/a/b_v40_loop.wav"""
    base = os.path.splitext(rel_path)[0]
    if abs(volume - 1.0) > 1e-3:
        base += '_v%02d' % int(round(volume * 100))
    if looped:
        base += '_loop'
    return base + ext
