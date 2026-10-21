# Changelog

## v1.1.1

Three fixes found by testing v1.1.0 on de_cbble.

### Fixes
- **Invisible Source volumes no longer cast shadows.** A brush made of Source's
  `nodraw` is solid but invisible, and casts no shadow in Source - the compiled
  Source map has no faces for it at all. Converted, it blocked light and left
  hard black patches: on de_cbble a `func_breakable` over the long roof left a
  640 x 240 unit band of wall and trim pitch black where CS:S has it sunlit.
  Those volumes still stop players and bullets, they just let light through.
- **Terrain in `patch` mode no longer shows the missing-texture grid.** The
  terrain's shader name is built from the map's material, and the four names
  derived from it (`_0`, `_1`, `_0to1`, `.vertex`) ran past the 59 characters
  SoF2 allows, so the shader file shortened two of them while the `.map` kept
  the long ones and nothing matched. The name is shortened up front now, with
  room left for the suffixes. `lwo` terrain was never affected.

### Editor
- **Terrain shows its own ground texture in Radiant** instead of the flat
  `tools/_terrain` tool texture, so a landscape is easier to edit by hand. The
  compile is unchanged.
- **The slabs under displacement terrain draw solid** in Radiant instead of
  see-through, so they are easier to pick and move.

### Notes
- Reconvert the whole map (not the compile alone) after updating: the terrain
  names in the `.map` and in the `.shader` change together, and new alphamap
  `.pcx` files are written.

## v1.1.0

Lighting now matches the lighting Valve baked into the CS:S map. Measured
against Valve's own lightmaps, de_dust2 comes out at 1.00 in sunlight, shade
and lamp-lit areas (the B tunnels were at 0.57 in v1.0.0).

### Lighting
- **Texture lights.** Rooms Source lights with glowing textures (fluorescent
  tubes, light panels, emergency lamps: lights.rad) instead of light entities
  are now lit. Those lights exist only in the map's compiled `.bsp`; the tool
  reads them from it, and the textures glow as they do in CS:S. On de_cbble
  the tube room went from black to 1.05 of CS:S.
- **No more cut-off light.** q3map2 throws away light below a fixed threshold,
  which cut every dim lamp off well before its light reached the floor. The
  converted map is now set up so this no longer happens, with no change to
  light that was already correct. The compile `.bat` passes the matching
  `-brightness` / `-vertexscale` / `-gridscale` values, worked out per map.
- **Removed the flat ambient fill** that stood in for the bounce light q3map2
  used to lose; with the bounce light arriving, it only made dark corners too
  bright.
- **Lamps inside props** (lanterns, light fixtures) light the room around them
  instead of being blocked by their own model, and their glass glows.
- **Window frames and wall details** on props were too bright and are back to
  normal.
- The quick test `.bat` now lights exactly like the full one (only
  visibility is faster), so test builds show the final lighting.

### Fixes
- Windows, door frames and other props near the 3D skybox no longer
  disappear: the skybox area is found by the skybox's own brushes instead of a
  sphere around the sky camera.
- The gap in the terrain on de_cbble (and the void seen through it) is closed:
  terrain edges are no longer snapped to a neighbour that would move them
  sideways.

### Notes
- Use NetRadiant's q3map2 (as before). It is needed for the new compile
  options; the `.bat` warns if the one you picked does not have them.
- Keep the map's own `.bsp` next to the `.vmf` (or under the content folder)
  so texture lights can be read. The GUI shows whether it was found.
- Compiles take longer than before (de_dust2: 484 s -> 666 s on 16 threads);
  big prop-heavy maps take longest in the bounce pass.
- Reconvert maps made with v1.0.0: the lights in the `.map` and the compile
  options in the `.bat` change together.

## v1.0.0

First release.
