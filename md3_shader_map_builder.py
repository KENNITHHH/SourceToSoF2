#!/usr/bin/env python3
"""
md3_shader_map_builder.py - Auto-generate the JSON mapping that
md3_shader_tool.py's "rewrite" command needs, by matching each bare shader
name found in your .md3 files against your .vmt material files by filename.

------------------------------------------------------------------------------
WHY THIS WORKS
------------------------------------------------------------------------------
Source models reference their skins by VMT filename (no folder, no
extension) - e.g. a model's skin group might say "metal2X4", which points
at a materials/**/metal2x4.vmt somewhere in the mod's content. That's
exactly the same bare name that ends up baked into the converted .md3
file's shader field (see md3_shader_tool.py's docstring for why that's
broken as-is). So: for each bare name, search your VMT set for a file
whose name (case-insensitive) matches, resolve it (following any
patch/include chains, same as vmt2shader.py) to its real $basetexture, and
that's the correct replacement.

IMPORTANT - MD3 shader names need the FULL runtime path (unlike .map
brush textures, which are compiled by q3map2 and use a different bare
convention - see vmf2map.py's notes on that distinction). MD3s aren't
compiled by anything; the engine loads the shader name exactly as stored,
so it must already be the complete "textures/foo/bar" form matching a
real .shader script name (the same names vmt2shader.py generates).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python3 md3_shader_map_builder.py shader_report.txt materials_vmts/ --output shader_map.json

"shader_report.txt" is the report produced by:
    python3 md3_shader_tool.py scan models/ --report shader_report.txt

"materials_vmts/" is your VMT root (same folder you'd point vmt2shader.py at).

Output is TWO files:
  --output shader_map.json      confident matches only, ready to feed
                                 straight into md3_shader_tool.py rewrite
  --unmatched unmatched.txt     (default: <output>.unmatched.txt) every
                                 bare name that had no matching VMT filename,
                                 or matched more than one (ambiguous) - these
                                 need a manual entry added to the JSON

Re-run this any time you get a new shader_report.txt (e.g. after fixing
some models) or add more VMTs - it's meant to be repeatable, not a
one-off. Manually-added entries in an existing --output file are
preserved and not overwritten by a re-run (pass --merge to enable this;
without it, the file is regenerated from scratch each time).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Reuses vmt2shader.py's VMT parsing/patch-resolution/prefix-normalization
# logic directly rather than duplicating it - keep both files in the same
# folder.
try:
    import vmt2shader as vt
except ImportError:
    print("ERROR: vmt2shader.py must be in the same folder as this script.", file=sys.stderr)
    sys.exit(1)


REPORT_LINE_RE = re.compile(r'^(?P<name>\S+)\s+\(\d+ file\(s\)\)\s*$')


def parse_shader_report(path):
    """Extract the list of unique bare shader names from a
    md3_shader_tool.py scan report (only the top-level 'Name  (N file(s))'
    lines are needed, not the indented file lists under each)."""
    names = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith((' ', '\t')):
                continue
            m = REPORT_LINE_RE.match(line.rstrip('\n'))
            if m:
                names.append(m.group('name'))
    return names


def build_basename_index(materials_dir):
    """Map lowercased VMT basename (no extension) -> list of full paths,
    so we can detect ambiguous matches (same basename in multiple folders)."""
    index = defaultdict(list)
    for root, _dirs, files in os.walk(materials_dir):
        for fname in files:
            if not fname.lower().endswith('.vmt'):
                continue
            base = os.path.splitext(fname)[0].lower()
            index[base].append(os.path.join(root, fname))
    return index


def resolve_basetexture(vmt_path, vmt_index):
    shader_class, body = vt.resolve_vmt(vmt_path, vmt_index)
    basetexture = vt.get_param(body, 'basetexture')
    if not basetexture:
        return None
    return vt.with_textures_prefix(basetexture)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="shader_report.txt produced by md3_shader_tool.py scan")
    ap.add_argument("materials_dir", help="root folder containing your .vmt files")
    ap.add_argument("--output", default="shader_map.json", help="where to write the confident-match JSON (default: shader_map.json)")
    ap.add_argument("--unmatched", help="where to write the list of unresolved names (default: <output>.unmatched.txt)")
    ap.add_argument("--merge", action="store_true",
                     help="if --output already exists, keep its existing entries and only add/refresh matches for names not already present")
    args = ap.parse_args()

    unmatched_path = args.unmatched or (args.output + ".unmatched.txt")

    names = parse_shader_report(args.report)
    if not names:
        print("ERROR: no shader names found in '{}' - is this a report from "
              "md3_shader_tool.py scan?".format(args.report), file=sys.stderr)
        sys.exit(1)

    vmt_basename_index = build_basename_index(args.materials_dir)
    vmt_include_index = vt.build_vmt_index(args.materials_dir)

    existing = {}
    if args.merge and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    resolved = dict(existing)
    ambiguous = []
    no_match = []
    stale_conflicts = []
    newly_resolved = 0

    for name in names:
        candidates = vmt_basename_index.get(name.lower(), [])

        if args.merge and name in existing:
            # Don't just trust an old entry forever - your materials set can
            # gain a new same-basename file between runs (e.g. extracting
            # another content pak later), turning what used to be an
            # unambiguous match into a real conflict. Re-check every
            # existing entry's candidate set and flag it if anything
            # resolves to a different real texture than what's currently
            # stored, rather than silently keeping a now-stale value.
            if len(candidates) > 1:
                resolved_candidates = [(c, resolve_basetexture(c, vmt_include_index)) for c in candidates]
                if any(tex_path != existing[name] for _c, tex_path in resolved_candidates):
                    stale_conflicts.append((name, existing[name], resolved_candidates))
            continue

        if len(candidates) == 0:
            no_match.append(name)
            continue
        if len(candidates) > 1:
            ambiguous.append((name, candidates))
            continue

        tex_path = resolve_basetexture(candidates[0], vmt_include_index)
        if not tex_path:
            no_match.append(name)
            continue

        resolved[name] = tex_path
        newly_resolved += 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(resolved, f, indent=2, sort_keys=True)

    unresolved_lines = []
    if stale_conflicts:
        unresolved_lines.append("--- POSSIBLY STALE: existing mapping conflicts with a candidate found now ---")
        unresolved_lines.append("(your materials set likely gained a new same-basename .vmt since this was set)")
        for name, current_value, candidates in stale_conflicts:
            unresolved_lines.append("{}  (currently mapped to: {}):".format(name, current_value))
            for c, tex_path in candidates:
                rel = os.path.relpath(c, args.materials_dir)
                marker = "  <-- current mapping" if tex_path == current_value else ""
                unresolved_lines.append("    {}  ->  {}{}".format(rel, tex_path, marker))
        unresolved_lines.append("")

    if ambiguous:
        unresolved_lines.append("--- Ambiguous: multiple VMTs share this basename, pick one manually ---")
        for name, candidates in ambiguous:
            unresolved_lines.append(name + ":")
            for c in candidates:
                rel = os.path.relpath(c, args.materials_dir)
                resolved_tex = resolve_basetexture(c, vmt_include_index)
                unresolved_lines.append("    {}  ->  {}".format(rel, resolved_tex))
        unresolved_lines.append("")

    if no_match:
        unresolved_lines.append("--- No matching VMT filename found at all ---")
        for name in no_match:
            unresolved_lines.append("  " + name)

    if unresolved_lines:
        with open(unmatched_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(unresolved_lines) + "\n")

    print("Resolved {} of {} shader name(s) -> {}".format(
        newly_resolved + (len(existing) if args.merge else 0), len(names), args.output), file=sys.stderr)
    if stale_conflicts:
        print("{} existing entry(ies) may be STALE (materials set changed since they were set) -> {}".format(
            len(stale_conflicts), unmatched_path), file=sys.stderr)
    if ambiguous or no_match:
        print("{} ambiguous, {} unmatched -> {}".format(
            len(ambiguous), len(no_match), unmatched_path), file=sys.stderr)
        print("Add manual entries for these directly into {} - keys are the".format(args.output), file=sys.stderr)
        print("exact bare names (case-sensitive), values are the full", file=sys.stderr)
        print("\"textures/...\" path they should point to. Re-run with --merge", file=sys.stderr)
        print("afterward to fill in the rest without losing your manual entries.", file=sys.stderr)


if __name__ == "__main__":
    main()
