#!/usr/bin/env python3
"""
clean_output.py - Tidy the pipeline's output folder into a plain,
ready-to-use structure once every other step has finished:

    maps/       <- .map (+ .map.bak, + any _clips.map)
    models/     <- final .md3 models (renamed from models_fixed/)
    shaders/    <- .shader
    textures/   <- unchanged

Removes the raw materials/ and models/ staging folders (already converted
into shaders/ and models/ by this point), and the working .json/.txt
files (texture-size cache, missing-assets report, any unresolved-shader
report) that were only needed to build the map, not to run it.

USAGE
-----
    python3 clean_output.py <output_dir>
"""

import glob
import os
import shutil
import sys


def move_into(files, dest_dir, output_dir):
    if not files:
        return
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        dest = os.path.join(dest_dir, os.path.basename(f))
        print("  {} -> {}".format(os.path.relpath(f, output_dir), os.path.relpath(dest, output_dir)))
        shutil.move(f, dest)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print("ERROR: output folder not found: {}".format(output_dir), file=sys.stderr)
        sys.exit(1)

    materials_dir = os.path.join(output_dir, "materials")
    models_dir = os.path.join(output_dir, "models")
    models_fixed_dir = os.path.join(output_dir, "models_fixed")

    print("Removing raw staging folders (already converted)...")
    if os.path.isdir(materials_dir):
        print("  removing materials/")
        shutil.rmtree(materials_dir)

    if os.path.isdir(models_fixed_dir):
        # models_fixed/ existing confirms models/ here is genuinely the raw
        # pre-conversion folder that needs clearing before the rename below
        # - NOT an already-cleaned models/ folder from a previous run of
        # this same script, which must never be touched: it's already the
        # final output at that point, not staging.
        if os.path.isdir(models_dir):
            print("  removing models/")
            shutil.rmtree(models_dir)
        print("Renaming models_fixed/ -> models/")
        os.rename(models_fixed_dir, models_dir)
        # any leftover diagnostic report (e.g. _unresolved_shaders.txt)
        # that would otherwise end up inside the final models/ folder
        for txt in glob.glob(os.path.join(models_dir, "*.txt")):
            print("  removing {}".format(os.path.relpath(txt, output_dir)))
            os.remove(txt)
    else:
        print("  (no models_fixed/ folder found - already cleaned, or step 3 didn't run)")

    print("Organizing shaders/maps...")
    move_into(glob.glob(os.path.join(output_dir, "*.shader")),
              os.path.join(output_dir, "shaders"), output_dir)
    move_into(glob.glob(os.path.join(output_dir, "*.map")) +
              glob.glob(os.path.join(output_dir, "*.map.bak")),
              os.path.join(output_dir, "maps"), output_dir)

    # lwo terrain (vmf2map.py --terrain-mode lwo): same relative path the .map's
    # misc_model points at, so this folder mirrors SOF2's base layout
    move_into(glob.glob(os.path.join(output_dir, "*_terrain.lwo")) +
              glob.glob(os.path.join(output_dir, "welded_terrain.lwo")) +
              glob.glob(os.path.join(output_dir, "*_terrain_sky.lwo")) +
              glob.glob(os.path.join(output_dir, "*_terrain_overlays_sky.lwo")) +
              glob.glob(os.path.join(output_dir, "*_terrain_overlays.lwo")) +
              glob.glob(os.path.join(output_dir, "welded_terrain_overlays.lwo")) +
              # parts cut around water volumes (vmf2map.py find_fog_volume_boxes)
              glob.glob(os.path.join(output_dir, "*_terrain_part*.lwo")) +
              glob.glob(os.path.join(output_dir, "*_terrain_overlays_part*.lwo")),
              os.path.join(output_dir, "models", "mapobjects", "terrain"), output_dir)

    print("Removing working files (texture sizes, reports)...")
    for f in glob.glob(os.path.join(output_dir, "*.json")) + \
             glob.glob(os.path.join(output_dir, "*.txt")):
        if os.path.basename(f).lower() == "_missing_assets.txt":
            # kept: the list of files step 1 could not find is what explains
            # a missing texture in game ("Couldn't find image for shader ...")
            print("  keeping  {}".format(os.path.relpath(f, output_dir)))
            continue
        print("  removing {}".format(os.path.relpath(f, output_dir)))
        os.remove(f)

    print("\nDone. Output folder is now:")
    print("  maps/      - .map + .map.bak")
    print("  models/    - .md3 models (+ mapobjects/terrain/*.lwo)")
    print("  shaders/   - .shader")
    print("  textures/  - .jpg/.png")


if __name__ == "__main__":
    main()
