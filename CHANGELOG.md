# Changelog

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
