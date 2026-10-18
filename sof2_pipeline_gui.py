#!/usr/bin/env python3
"""
sof2_pipeline_gui.py - One window for the whole CS/HL2 -> SOF2 conversion
pipeline: pick your .vmf, pick your folders, tick the steps you want, hit
Run. A live log at the bottom shows exactly what each step is doing.

------------------------------------------------------------------------------
HOW THIS RUNS THE OTHER SCRIPTS
------------------------------------------------------------------------------
Each pipeline script is imported as a normal Python module and its main()
is called directly (with sys.argv patched to the arguments this GUI builds,
and stdout/stderr redirected into the log pane) - NOT launched as a
subprocess. This matters once this is frozen into a standalone .exe via
PyInstaller: there is no python.exe to shell out to anymore, and
sys.executable just points back at the GUI's own .exe. Calling main()
in-process works identically either way.

------------------------------------------------------------------------------
PACKAGING INTO AN .EXE (you run this part, on your own machine)
------------------------------------------------------------------------------
    pip install pyinstaller numpy pillow
    pyinstaller --onefile --noconsole --name SourceToSoF2 ^
        --add-data "vmf_asset_extractor.py;." ^
        --add-data "vtf_to_png_jpg.py;." ^
        --add-data "mdl_to_md3.py;." ^
        --add-data "md3_shader_autofix.py;." ^
        --add-data "build_texture_sizes.py;." ^
        --add-data "vmt2shader.py;." ^
        --add-data "vmf2map.py;." ^
        --add-data "phy2clip.py;." ^
        --add-data "vpk_reader.py;." ^
        --add-data "bsp_pakfile.py;." ^
        --add-data "sof2_sounds.py;." ^
        --add-data "clean_output.py;." ^
        --add-data "install_to_sof2.py;." ^
        --add-data "bsp_fix_leafbounds.py;." ^
        --add-data "bsp_decal_relight.py;." ^
        --add-data "bsp_shader_count.py;." ^
        --add-data "shaderlist_for_compile.py;." ^
        --add-data "make_pk3.py;." ^
        --add-data "audit_release.py;." ^
        --hidden-import numpy ^
        --hidden-import PIL ^
        --hidden-import PIL.Image ^
        --hidden-import PIL.PngImagePlugin ^
        --hidden-import PIL.JpegImagePlugin ^
        --hidden-import heapq ^
        --hidden-import PIL.TgaImagePlugin ^
        --hidden-import glob ^
        sof2_pipeline_gui.py

The window is plain tkinter (part of Python itself), so nothing else has to
be bundled for it. MakeExe.bat runs exactly this.

Run that from the same folder as all 16 scripts. The resulting
dist\\SourceToSoF2.exe is a single, standalone file - copy it wherever
you like, it doesn't need the .py files sitting next to it anymore (they're
bundled inside).

The --hidden-import flags matter and are easy to miss: PyInstaller finds
dependencies by statically scanning this entry script's own "import ..."
lines. vtf_to_png_jpg.py (numpy + PIL, for alpha-aware image decoding) and
build_texture_sizes.py (PIL, for reading converted image dimensions) both
import numpy/PIL, but ONLY inside functions, wrapped in a try/except that
prints a friendly "pip install ..." message if missing - specifically so
the script still runs standalone without them for anyone who doesn't need
image conversion. That's exactly the pattern PyInstaller's static scanner
can't see: since this GUI loads those scripts dynamically at runtime (via
importlib, not a top-level "import vtf_to_png_jpg"), and even a REAL
top-level import wouldn't help, because the numpy/PIL imports inside those
scripts are themselves deferred until a function actually runs. Without
the --hidden-import flags, the .exe builds successfully and looks fine
right up until step 2 actually runs, where it fails with a "missing
dependency" message that's technically correct but confusing coming from
a supposedly-standalone .exe (confirmed: this exact failure happened on a
real build before these flags were added).

------------------------------------------------------------------------------
FOLDER LAYOUT THIS GUI ASSUMES
------------------------------------------------------------------------------
Given one "project folder" (e.g. E:\\CSS\\Extracted\\de_cache_files), every
step reads/writes fixed subfolder names within it. While steps 1-8 are
running, that looks like:

    materials/       <- vmf_asset_extractor.py output (raw .vmt/.vtf)
    models/           <- vmf_asset_extractor.py output (raw .mdl/.md3/.phy/...)
    textures/         <- vtf_to_png_jpg.py output (.jpg/.png)
    models_fixed/     <- md3_shader_autofix.py output (fixed .md3 + siblings)
    sizes.json        <- build_texture_sizes.py output
    <mapname>.shader   <- vmt2shader.py output
    <mapname>.map      <- vmf2map.py output (phy2clip.py merges into this)

Step 9 (clean_output.py, on by default) then reorganizes all of that into
the plain structure meant to be the actual deliverable:

    maps/       <- .map + .map.bak
    models/     <- final .md3 models (renamed from models_fixed/)
    shaders/    <- .shader
    textures/   <- unchanged

removing the raw materials/models/ staging folders and the working
.json/.txt files along the way, since none of those are needed to
actually run the map.

This matches the exact folder names used throughout real command-line
usage of these scripts earlier in this project - nothing here is a new
convention, just automated wiring of the existing one. One deliberate,
easy-to-get-wrong detail preserved on purpose: vmt2shader.py's --mdl-dir
points at the RAW models/ folder, never models_fixed/ - .mdl files (as
opposed to .md3) only ever exist in the raw extraction output.
"""

import importlib.util
import io
import json
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


def script_search_dirs():
    """Where pipeline scripts are looked for, in order:
      1. next to the .exe (or next to this .py when not frozen) - so an
         updated or newly added script can simply be dropped beside the .exe
         without rebuilding it;
      2. inside the .exe (PyInstaller unpacks --add-data files to
         sys._MEIPASS)."""
    if getattr(sys, 'frozen', False):
        dirs = [os.path.dirname(os.path.abspath(sys.executable))]
    else:
        dirs = [os.path.dirname(os.path.abspath(__file__))]
    bundled = getattr(sys, '_MEIPASS', None)
    if bundled and bundled not in dirs:
        dirs.append(bundled)
    return dirs


# The run log is named after the program itself: SourceToSoF2.exe writes
# SourceToSoF2.txt next to it (a renamed .exe follows its own name).
APP_NAME = "SourceToSoF2"
APP_VERSION = "1.0.0"      # bump for every release: shown in the title and every log
RUN_LOG_NAME = (os.path.splitext(os.path.basename(sys.executable))[0]
                if getattr(sys, 'frozen', False) else APP_NAME) + ".txt"


def run_log_path():
    """Where the finished run's log is saved: next to the .exe (next to
    this .py when not frozen)."""
    return os.path.join(script_search_dirs()[0], RUN_LOG_NAME)


def resource_path(filename):
    """Path of a pipeline script (see script_search_dirs), or None."""
    for d in script_search_dirs():
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


def load_module(filename):
    path = resource_path(filename)
    if path is None:
        raise FileNotFoundError("{} not found in: {}".format(filename, "; ".join(script_search_dirs())))
    name = os.path.splitext(filename)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LogStream(io.TextIOBase):
    """A writable stream that pushes text into a thread-safe queue instead
    of an actual file - lets us redirect a background thread's stdout into
    the GUI's log pane, which must only ever be touched from the main
    thread (tkinter is not thread-safe)."""
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if text:
            self.log_queue.put(text)
        return len(text)

    def flush(self):
        pass


class Tooltip:
    """Standard hover-tooltip pattern - tkinter has no built-in equivalent: shows a small borderless
    window near the cursor on <Enter>, destroys it on <Leave>.

    x_offset/y_offset control where the tooltip appears relative to the
    widget's own top-left corner. Default (below, slightly right) suits a
    single isolated control; a tightly-packed vertical list (like the
    Steps checkboxes) instead wants a big x_offset and a y_offset level
    with the row, so the tooltip appears beside it rather than overlapping
    the rows underneath."""
    def __init__(self, widget, text, wraplength=420, x_offset=20, y_offset=None):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.tipwindow = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + self.x_offset
        if self.y_offset is None:
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        else:
            y = self.widget.winfo_rooty() + self.y_offset
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+{}+{}".format(x, y))
        label = tk.Label(tw, text=self.text, justify='left', background="#2a3140",
                          foreground="#e6e9ef", relief='solid', borderwidth=1,
                          wraplength=self.wraplength, font=("Segoe UI", 10), padx=10, pady=8)
        label.pack()

    def _hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


def find_map_bsp(vmf_path, sources_arg):
    """(path, note) for the Source .bsp that belongs to this .vmf.

    Same rule as vmf_asset_extractor.find_map_bsp, and the GUI passes what it
    finds to step 1 as --map-bsp, so the window and the run can never
    disagree: <map>.bsp next to the .vmf first, then anywhere under the
    content roots (a CS:S install keeps it in cstrike/maps/). It must be a
    Source VBSP - a same-named SOF2 .bsp (RBSP) from compiling this very map
    is reported in `note` and ignored. Returns (None, note) when not found."""
    if not vmf_path:
        return None, ""
    name = os.path.splitext(os.path.basename(vmf_path.replace('\\', '/')))[0].lower()
    if not name:
        return None, ""
    wrong = []

    def check(full):
        try:
            with open(full, 'rb') as f:
                ident = f.read(4)
        except OSError:
            return False
        if ident == b'VBSP':
            return True
        wrong.append((full, ident.decode('latin-1', 'replace')))
        return False

    sibling = os.path.join(os.path.dirname(os.path.abspath(vmf_path)), name + '.bsp')
    if os.path.isfile(sibling) and check(sibling):
        return sibling, ""
    raw = (sources_arg or "").strip()
    roots = [raw] if os.path.isdir(raw) else [r.strip() for r in raw.split(';') if r.strip()]
    for root in roots:
        if os.path.isfile(root):
            if os.path.basename(root).lower() == name + '.bsp' and check(root):
                return root, ""
            continue
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower() == name + '.bsp' and check(os.path.join(dirpath, fn)):
                    return os.path.join(dirpath, fn), ""
    note = ""
    if wrong:
        note = "{} is a {} file, not a Source map - ignored.".format(
            os.path.basename(wrong[0][0]), "SOF2/idTech3" if wrong[0][1] == 'RBSP' else wrong[0][1])
    return None, note


class Step:
    def __init__(self, key, label, module_name, build_args, check_ready, tooltip=""):
        self.key = key
        self.label = label
        self.module_name = module_name
        self.build_args = build_args      # fn(paths) -> list[str] (argv[1:])
        self.check_ready = check_ready    # fn(paths) -> (bool, reason)
        self.tooltip = tooltip
        self.var = None                   # tk.BooleanVar, set once UI exists


class PipelinePaths:
    """Derives every path this GUI wires up from what the user actually
    picks: the .vmf, the source content root(s), and ONE output folder
    that holds everything this pipeline produces - materials/models/
    textures/models_fixed (the staging area) alongside the compiled
    .shader/.map/sizes.json/clips. Earlier versions of this GUI split
    these into two separate folders, matching an old manual command-line
    convention - simplified to one, since a GUI has no reason to make
    someone pick two locations for what's really one project's output."""
    # The defaults below are the values the GUI no longer asks for. Each one is
    # either the only answer that was ever right or the one that measurement
    # settled; the matching CLI flags all still exist for a hand run.
    #   scale / model_scale 1.25  SOF2 is ~2cm per unit against Source's inch
    #       (2.54cm), so ~1.27 - but 1.25 (= 10/8) puts every whole Source unit
    #       exactly on q3map2's 1/8-unit plane grid. At 1.27 almost every plane
    #       has to be snapped, and sloped and flat faces snap by different
    #       amounts, which opens hairline seams where they meet: measured on
    #       de_mirage, 1386 near-miss vertices at 1.27 vs 840 at 1.25 (1027 at
    #       1.0). The 1.6% size difference is not visible in game.
    #   skybox_entity True       the 3D skybox is drawn as a distant backdrop,
    #       the way Source draws it. The leaf-face blowup this used to risk is
    #       what skybox_landscape_brushes exists to control.
    #   skip_3d_skybox False     with the backdrop on, the miniature copy IS
    #       the backdrop - skipping it would delete what gets rendered.
    #   seal_map_box "64"        always seal the map in a box. A leak is not
    #       cosmetic: q3map2 refuses to write the .prt, so -vis cannot run and
    #       the map ships with no visibility culling at all. Six big
    #       axis-aligned caulk brushes cost the BSP almost nothing.
    #   seal_terrain_under "0"   the terrain-following shell was MEASURED as a
    #       net loss on de_dust2 - its 552 thin sloped brushes took leafs from
    #       8127 to 15234 and tiny portals from 67 to 1416, and it still leaked,
    #       because the flood passed underneath it. The box above cannot be
    #       gone around.
    #   lightmap_samplesize "0"  lighting stays as fine as CS:S's: each face's
    #       own Source lightmapscale x map scale, written into the .map as
    #       _lightmapsamplesize. Coarser lightmaps are a MAX_SHADERS lever, and
    #       that key is editable in Radiant after the fact - see step 10.
    #   distance_cull "0"        no far-clip override; the engine default.
    #   drop_terrain_backing_brush True  confirmed in Hammer that a
    #       displacement's flat base brush is editor-only, and keeping it made
    #       it physically interfere with the real terrain above it.
    #   install_assets True      the .pk3 step 10 builds has to contain the
    #       textures and models, or the map loads untextured.
    def __init__(self, vmf_path, sources_arg, output_dir, map_name, scale="1.25",
                 model_scale="1.25", terrain_mode="lwo", drop_terrain_backing_brush=True,
                 sof2_base="", install_assets=True, skybox_entity=True,
                 skybox_models_to_world=True, skip_3d_skybox=False,
                 skybox_landscape_brushes="4", seal_leak_lin="", seal_terrain_under="0",
                 distance_cull="0", seal_map_box="64", lightmap_samplesize="0",
                 map_bsp=None, full_quality=False, q3map2=""):
        self.vmf_path = vmf_path
        self.sources_arg = sources_arg
        self.output_dir = output_dir
        self.scale = scale
        self.model_scale = model_scale
        self.terrain_mode = terrain_mode
        self.drop_terrain_backing_brush = drop_terrain_backing_brush
        self.skybox_entity = skybox_entity
        self.skybox_models_to_world = skybox_models_to_world
        self.skip_3d_skybox = skip_3d_skybox
        self.skybox_landscape_brushes = skybox_landscape_brushes
        self.seal_leak_lin = seal_leak_lin
        self.seal_terrain_under = seal_terrain_under
        self.distance_cull = distance_cull
        self.seal_map_box = seal_map_box
        self.lightmap_samplesize = lightmap_samplesize
        self.sof2_base = sof2_base
        self.install_assets = install_assets
        self.map_bsp = map_bsp
        self.full_quality = full_quality
        self.q3map2 = q3map2

        if map_name:
            self.map_name = map_name
        elif vmf_path:
            # Split on both separators explicitly rather than relying on
            # os.path.basename's host-OS-dependent behavior - this script
            # is developed and tested on a non-Windows sandbox but always
            # runs on Windows in practice, so provably handling both
            # separator styles here is safer than trusting the platform
            # default to behave the same way it would on the real target.
            base = vmf_path.replace('\\', '/').rsplit('/', 1)[-1]
            self.map_name = base[:-4] if base.lower().endswith('.vmf') else base
        else:
            self.map_name = "output"

    @property
    def extract_dir(self):
        return self.output_dir

    @property
    def materials_dir(self):
        return os.path.join(self.output_dir, "materials")

    @property
    def models_dir(self):
        return os.path.join(self.output_dir, "models")

    @property
    def textures_dir(self):
        return os.path.join(self.output_dir, "textures")

    @property
    def models_fixed_dir(self):
        return os.path.join(self.output_dir, "models_fixed")

    @property
    def sizes_json(self):
        return os.path.join(self.output_dir, self.map_name + "_sizes.json")

    @property
    def shader_path(self):
        return os.path.join(self.output_dir, self.map_name + ".shader")

    @property
    def map_path(self):
        return os.path.join(self.output_dir, self.map_name + ".map")

    @property
    def terrain_scales_json(self):
        # Matches vmf2map.py's own naming exactly: splitext(output) + this
        # suffix - written alongside the .map whenever any
        # WorldVertexTransition displacement exists.
        return os.path.join(self.output_dir, self.map_name + "_terrain_scales.json")

    @property
    def terrain_lwo_json(self):
        # vmf2map.py, --terrain-mode lwo: the blended terrainN shaders
        return os.path.join(self.output_dir, self.map_name + "_terrain_lwo_shaders.json")

    @property
    def overlay_json(self):
        # vmf2map.py: materials of flat info_overlays built as patches
        return os.path.join(self.output_dir, self.map_name + "_overlay_materials.json")

    @property
    def nonsolid_json(self):
        # vmf2map.py: shaders needing a non-solid copy (func_brush Solidity 1)
        return os.path.join(self.output_dir, self.map_name + "_nonsolid_materials.json")

    @property
    def overlay_mesh_json(self):
        # vmf2map.py: flat shader names of info_overlays sitting on terrain,
        # built as a dense conforming mesh instead of a patch
        return os.path.join(self.output_dir, self.map_name + "_overlay_mesh_materials.json")

    @property
    def planar_materials_json(self):
        # mdl_to_md3.py (batch): materials of FLAT models, which need
        # polygonOffset so they don't z-fight/hide behind wall brushes
        return os.path.join(self.models_fixed_dir, "md3_planar_materials.json")

    @property
    def model_static_list_json(self):
        # mdl_to_md3.py (batch): small-clutter model paths (bottles, cans,
        # tires) that should spawn as model_static, wherever placed
        return os.path.join(self.models_fixed_dir, "md3_small_models.json")

    @property
    def model_offsets_json(self):
        # mdl_to_md3.py (batch): models moved back into MD3's 512-unit range
        return os.path.join(self.models_fixed_dir, "md3_origin_offsets.json")

    @property
    def clips_path(self):
        return os.path.join(self.output_dir, self.map_name + "_clips.map")


def dir_ready(path, label):
    if not path or not os.path.isdir(path):
        return False, "{} not found: {}".format(label, path)
    return True, ""


def file_ready(path, label):
    if not path or not os.path.isfile(path):
        return False, "{} not found: {}".format(label, path)
    return True, ""


def _build_extract_args(p):
    args = [p.vmf_path, p.sources_arg, p.extract_dir]
    bsp = p.map_bsp or find_map_bsp(p.vmf_path, p.sources_arg)[0]
    if bsp:
        args += ["--map-bsp", bsp]
    return args


def map_lightmap_samplesize(map_path, fallback):
    """worldspawn "_lightmapsamplesize" from a .map vmf2map.py wrote, as a string."""
    import re as _re
    try:
        with open(map_path, 'r', encoding='utf-8', errors='replace') as f:
            head = f.read(65536)
    except OSError:
        return fallback
    m = _re.search(r'"_lightmapsamplesize"\s+"([0-9.]+)"', head)
    if not m:
        return fallback
    try:
        return fallback if float(m.group(1)) <= 0 else m.group(1)
    except ValueError:
        return fallback


def build_steps():
    steps = []

    steps.append(Step(
        "extract", "1. Extract assets used by .vmf",
        "vmf_asset_extractor",
        _build_extract_args,
        lambda p: file_ready(p.vmf_path, ".vmf"),
        tooltip="Scans the .vmf for every material, model, sound, and skybox "
                "it references, then pulls each one out of your source "
                "content. Point it at your game install - counter-strike "
                "source or its cstrike folder both work - and the game's "
                "content is mounted exactly the way its gameinfo.txt does "
                "(cstrike_pak.vpk, the hl2 VPKs that hold most shared props, "
                "then loose files). The map's own "
                ".bsp - shown under the content roots above - is read first, "
                "the way Source mounts it, so anything the map packs wins "
                "over the game's copy. Other maps' .bsp files are ignored. "
                "Re-running only rewrites files whose source changed, and "
                "removes files an earlier run extracted that the map no "
                "longer uses.",
    ))

    steps.append(Step(
        "vtf", "2. Convert .vtf textures to .jpg/.png",
        "vtf_to_png_jpg",
        lambda p: [p.materials_dir, p.textures_dir],
        lambda p: dir_ready(p.materials_dir, "materials/ (run step 1 first)"),
        tooltip="Converts every texture into a plain image format. "
                "Textures with transparency (or an unusual format) become "
                "PNG; everything else becomes JPG to keep file sizes down. "
                "Sky-box faces become TGA, the exact name the game asks "
                "for when it loads a sky.",
    ))

    steps.append(Step(
        "mdltomd3", "3. Convert .mdl models to .md3",
        "mdl_to_md3",
        lambda p: [p.models_dir, p.models_fixed_dir, "--materials-dir", p.materials_dir,
                  "--model-scale", p.model_scale],
        lambda p: dir_ready(p.models_dir, "models/ (run step 1 first)"),
        tooltip="Converts models directly into the game's native format, "
                "with correct materials assigned automatically. A very "
                "small number of unusual models may need the fallback "
                "step below instead.",
    ))

    steps.append(Step(
        "autofix", "4. Fix any remaining .md3 shader names  (optional fallback)",
        "md3_shader_autofix",
        # Reads FROM models_fixed_dir now, not raw models_dir - that's where
        # mdl_to_md3.py's own (already-correct) output lives, plus wherever
        # a manually-converted straggler might get dropped. Safe to run
        # even when nothing needs fixing: it only rewrites shader names that
        # are actually wrong, and copies everything else through unchanged.
        lambda p: ["--md3-dir", p.models_fixed_dir, "--mdl-dir", p.models_dir,
                   "--materials-dir", p.materials_dir, "--output", p.models_fixed_dir],
        lambda p: dir_ready(p.models_fixed_dir, "models_fixed/ (run step 3 first)"),
        tooltip="Only needed for the rare model step 3 couldn't convert on "
                "its own (check the log for 'not yet supported'). Safe to "
                "leave checked even when nothing needs it - it only fixes "
                "names that are actually wrong, and leaves correct ones "
                "untouched.",
    ))

    steps.append(Step(
        "sizes", "5. Build texture size mapping file",
        "build_texture_sizes",
        lambda p: [p.extract_dir, p.sizes_json],
        lambda p: dir_ready(p.extract_dir, "extraction folder"),
        tooltip="Records the pixel dimensions of every texture, needed to "
                "get texture alignment right on brush faces later.",
    ))

    def _build_map_args(p):
        args = [p.vmf_path, p.map_path, "--texture-sizes", p.sizes_json,
                "--materials-dir", p.materials_dir,
                "--scale", str(p.scale), "--model-scale", str(p.model_scale),
                "--terrain-mode", p.terrain_mode]
        if p.drop_terrain_backing_brush:
            args.append("--drop-terrain-backing-brush")
        if p.skybox_entity:
            args.append("--skybox-entity")
        if p.skybox_models_to_world:
            args.append("--skybox-models-to-world")
        if p.skip_3d_skybox:
            args.append("--skip-3d-skybox")
        try:
            cells = int(str(p.skybox_landscape_brushes).strip() or "0")
        except ValueError:
            cells = 0
        if cells > 0:
            args += ["--skybox-landscape-brushes", str(cells)]
        # --distance-cull and --seal-terrain-under are not passed from here any
        # more (see the note on PipelinePaths' defaults): the far clip stays the
        # engine's own, and gaps are sealed by the box below, which measurement
        # showed beats the terrain-following shell outright. Both flags remain
        # on vmf2map.py for a hand run.
        #
        # Neither is --lightmap-samplesize. The .map gets Source's own
        # lightmapscale x map scale per entity instead, which is as fine as
        # CS:S's lighting, and raising it is a MAX_SHADERS lever best pulled
        # after seeing a compile: it is the worldspawn's "_lightmapsamplesize"
        # key, editable in Radiant without reconverting.
        #
        # No lighting flags are passed from here on purpose. Point light
        # brightness, the worldspawn ambient (shadow fill) and the light grid
        # cell size are all derived by vmf2map.py from the map's own
        # light_environment and bounding box, which is the whole point of
        # converting a map rather than re-lighting one. The CLI flags still
        # exist - --ambient, --point-light-scale, --gridsize - for a hand run
        # when a particular map needs pushing; they are just not choices anyone
        # should have to make before the first compile.
        boxmargin = str(p.seal_map_box or "").strip()
        if boxmargin:
            try:
                margin = float(boxmargin)
            except ValueError:
                margin = 0.0
            if margin > 0:
                args += ["--seal-map-box", ("%g" % margin)]
        lin = str(p.seal_leak_lin or "").strip().strip('"')
        if lin and os.path.isfile(lin):
            args += ["--seal-leak-from-lin", lin]
        # Written by step 3 only when some model was too big for MD3's range
        # (e.g. palm_tree_trunk); without it those props sit in the wrong place.
        if os.path.isfile(p.model_offsets_json):
            args += ["--model-offsets", p.model_offsets_json]
        if os.path.isfile(p.model_static_list_json):
            args += ["--model-static-list", p.model_static_list_json]
        if str(p.lightmap_samplesize).strip() not in ('', '0'):
            args += ["--lightmap-samplesize", str(p.lightmap_samplesize).strip()]
        if os.path.isdir(p.models_fixed_dir):
            args += ["--md3-dir", p.models_fixed_dir]
        if p.full_quality:
            # Every prop at Source's own full detail: no simplified copies,
            # nothing dropped. The log still says how big the .bsp will be.
            args += ["--prop-vertex-budget", "0"]
        return args

    def _build_shader_args(p):
        args = [p.materials_dir, p.shader_path, "--mdl-dir", p.models_dir, "--vmf", p.vmf_path,
                # images whose path is too long for idTech3 get a short copy here
                "--textures-dir", p.textures_dir]
        # Each of these only exists if step 6 produced it (depends on terrain
        # mode and on the map having overlays), so pass whichever are there.
        if os.path.isfile(p.terrain_scales_json):
            args += ["--terrain-scales", p.terrain_scales_json]
        if os.path.isfile(p.terrain_lwo_json):
            args += ["--terrain-lwo-shaders", p.terrain_lwo_json]
        if os.path.isfile(p.planar_materials_json):
            args += ["--planar-model-materials", p.planar_materials_json]
        if os.path.isfile(p.overlay_json):
            args += ["--overlay-materials", p.overlay_json]
        if os.path.isfile(p.overlay_mesh_json):
            args += ["--overlay-mesh-materials", p.overlay_mesh_json]
        if os.path.isfile(p.nonsolid_json):
            args += ["--nonsolid-materials", p.nonsolid_json]
        nofog_json = os.path.join(p.output_dir, p.map_name + "_nofog_shaders.json")
        if os.path.isfile(nofog_json):
            args += ["--nofog-shaders", nofog_json]
        # The sun's intensity is solved by vmt2shader.py from this map's own
        # light_environment and sun elevation, so nothing about it is passed
        # from here either. --sun-intensity and --light-scale remain for a hand
        # run.
        #
        # These two are behaviour, not options. Both were defaults once and both
        # were wrong: without --terrain-lightmapped the converted terrain bakes
        # fullbright and casts no shadow, and with --decal-lighting none every
        # decal and overlay renders fullbright over correctly lit brushwork,
        # which reads as glowing stickers. There is no map for which either is
        # the better answer, so neither is a question any more.
        args.append("--terrain-lightmapped")
        # The terrain's luxel size: the one Source lit this map at (its faces'
        # lightmapscale x map scale), which step 6 writes into the .map as
        # worldspawn "_lightmapsamplesize". It used to be a fixed 32 - 1.6x
        # coarser than de_overpass_csgo's own 20 - and its sun shadows on the
        # ground came out as visible 32-unit stair steps. 32 stays the fallback
        # when the .map does not say.
        args += ["--terrain-lightmap-samplesize", map_lightmap_samplesize(p.map_path, "32")]
        return args

    steps.append(Step(
        "map", "6. Convert .vmf to .map",
        "vmf2map",
        # Deliberately runs BEFORE step 7 (shader generation) now, not
        # after: vmt2shader.py's --terrain-scales needs THIS step's own
        # "<map>_terrain_scales.json" output to exist first, for correct
        # per-displacement terrain shader naming (confirmed necessary -
        # without it, terrain shaders silently fall back to one flat,
        # less accurate value shared across a whole material). This does
        # mean --shader-file's own cross-check (comparing against the
        # .shader file) is no longer possible here, since that file can't
        # exist yet at this point - a minor loss (nicer warning text
        # only, not a correctness issue) traded for the terrain-scales
        # dependency, which matters more.
        _build_map_args,
        lambda p: file_ready(p.vmf_path, ".vmf"),
        tooltip="Converts the map geometry, entities, and terrain into "
                "the target map format.",
    ))

    steps.append(Step(
        "shader", "7. Generate .shader",
        "vmt2shader",
        # NOTE: --mdl-dir intentionally points at the RAW models/ folder,
        # not models_fixed/ - .mdl files only ever exist in the raw
        # extraction output, models_fixed/ only ever contains .md3.
        # --terrain-scales points at step 6's own output, which is why
        # step 6 now runs first - see its own comment for why. No
        # --terrain-texel-scale here deliberately - --terrain-scales
        # already covers every real displacement automatically; that flag
        # is only a rarely-used fallback for whatever isn't, so vmt2shader.py's
        # own built-in default handles it without needing a GUI field.
        _build_shader_args,
        lambda p: dir_ready(p.materials_dir, "materials/ (run step 1 first)"),
        tooltip="Converts each material definition, including sky and "
                "lighting setup for the map's own skybox.",
    ))

    steps.append(Step(
        "clip", "8. Generate + merge model collision clips",
        "phy2clip",
        # Reads .phy from the RAW models/ folder, matching proven real
        # usage - not models_fixed/. The .phy bytes are identical either
        # way (autofix only ever copies them, never modifies them), so
        # this isn't a correctness fix, just matching the convention
        # already confirmed to work.
        # --skip-ids: vmf2map.py's list of props the prop budget dropped, so
        # their collision goes too (it writes the file on every run).
        lambda p: [p.vmf_path, p.models_dir, p.clips_path,
                   "--materials-dir", p.materials_dir, "--merge-into", p.map_path,
                   "--scale", str(p.scale), "--model-scale", str(p.model_scale),
                   "--skip-ids", os.path.splitext(p.map_path)[0] + "_dropped_props.json"],
        lambda p: dir_ready(p.models_dir, "models/ (run step 1 first)"),
        tooltip="Generates collision for every model from its own "
                "collision data, and merges it straight into the map "
                "from step 7.",
    ))

    steps.append(Step(
        "clean", "9. Clean up output folder  (recommended)",
        "clean_output",
        lambda p: [p.output_dir],
        lambda p: dir_ready(p.output_dir, "output folder"),
        tooltip="Tidies everything into maps/, models/, shaders/, and "
                "textures/ folders, and removes the in-between working "
                "files that were only needed to build the map, not to "
                "run it.\n\nRecommended to leave on - without it, your "
                "output folder fills up with leftover files you'll never "
                "need again.",
    ))

    def _build_install_args(p):
        # Everything the map needs is packed into <output>/<map>.pk3 and that
        # one file is copied into base, so base gains an archive and the .map
        # and nothing else.
        #
        # --clean-loose is deliberately NOT passed. It moves loose files in
        # base that the new .pk3 also contains into <output>/_moved_out_of_base/
        # - worth knowing about, because q3map2's VFS reads real directories
        # BEFORE mounted .pk3s, so a leftover from an older conversion is what
        # a compile would use, silently. But a .pk3's paths are not that map's
        # private property: CS:S content is shared, so props_urban and
        # cs_office textures sit at identical paths in several maps, and
        # sweeping them for one map pulls them out from under another that is
        # still reading them loose. Running install_to_sof2.py by hand with
        # --clean-loose is there when base needs tidying on purpose.
        args = [p.output_dir, p.map_name, p.sof2_base,
                "--tool-version", "{} v{}".format(APP_NAME, APP_VERSION)]
        if p.q3map2:
            args += ["--q3map2", p.q3map2]
        if getattr(sys, 'frozen', False):
            # the compile .bat runs its helper scripts through this .exe
            # (--run), so nobody compiling needs Python installed
            args += ["--runner", sys.executable]
        return args

    def _install_ready(p):
        if not p.sof2_base:
            return False, "no SOF2 base folder set (field above)"
        return dir_ready(p.sof2_base, "SOF2 base folder")

    steps.append(Step(
        "install", "10. Install into SOF2 base folder",
        "install_to_sof2",
        _build_install_args,
        _install_ready,
        tooltip="First AUDITS the finished map: every texture, shader and model it "
                "names is followed to the file the game (or q3map2, for baked props) "
                "will look for, in this .pk3 or the game's own content. Names too long "
                "for the engine stop the step; missing assets are listed and the .pk3 "
                "is still built. Assets that only resolve through files on this "
                "machine (an old loose install, another converted map) are listed "
                "separately - they would be missing for anyone you give the map to. "
                "Full list in <output>/<map>_audit.txt.\n\n"
                "Then packs the map's shader, every converted texture, every .md3 and the "
                "terrain .lwo into ONE base/<map>.pk3, and copies the .map itself into "
                "base/maps so it can be compiled. VERIFIED that q3map2 reads shaders "
                "and textures straight out of a .pk3 in base (control run with the .pk3 "
                "removed: 'Couldn't find image for shader'), so nothing has to be loose. "
                "The .shader is still added to shaders/shaderlist.txt when that file "
                "exists, because q3map2 - not the game - loads only the shaders listed "
                "there.\n\nThen run the maps/compile_<map>.bat it writes: every stage in "
                "order with the flags this map needs (-rename on -bsp so baked props use "
                "Raven's <name>_bsp shaders, bsp_fix_leafbounds.py between -bsp and -vis, "
                "-nohint, and -gamma <g> -gridscale <s> -bounce 1 on -light, solved from "
                "this map's own light_environment). compile_<map>_quick.bat is the same "
                "build with -vis -fast and no -bounce for testing. Both finish by "
                "printing how close the map is to SOF2's 2048-shader table and by "
                "packing the finished .bsp into maps/<map>_release.pk3 - that single "
                "file is what you give someone else.",
    ))

    return steps


# Files a step needs that are NOT the module it imports - see the check in
# _run_steps. Keep this in step with the --add-data list at the top of this
# file: everything named here has to be in the .exe (or sitting next to it).
STEP_HELPER_FILES = {
    'extract': ('sof2_sounds.py',),
    'map': ('sof2_sounds.py',),
    'clip': ('vmt2shader.py',),
    'install': ('make_pk3.py', 'audit_release.py', 'bsp_fix_leafbounds.py',
                'bsp_decal_relight.py', 'bsp_shader_count.py',
                'shaderlist_for_compile.py'),
}

# Dark theme on plain tkinter/ttk ('clam' recoloured) - no CustomTkinter.
BG = "#1b1f27"
PANEL = "#232833"
FIELD = "#12151b"
FG = "#e6e9ef"
MUTED = "#8a94a6"
ACCENT = "#3b8ede"
OK_COLOR = "#3ecf6e"
WARN_COLOR = "#e8b339"
ERR_COLOR = "#e2574c"


def apply_dark_theme(root):
    root.configure(bg=BG)
    st = ttk.Style(root)
    st.theme_use('clam')
    st.configure('.', background=BG, foreground=FG, fieldbackground=FIELD,
                 bordercolor="#3a4150", lightcolor=BG, darkcolor=BG, troughcolor=FIELD,
                 focuscolor=ACCENT, selectbackground=ACCENT, selectforeground="white",
                 font=("Segoe UI", 9))
    st.configure('TLabelframe', background=BG, bordercolor="#3a4150")
    st.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                 font=("Segoe UI", 10, "bold"))
    st.configure('TLabel', background=BG, foreground=FG)
    st.configure('TFrame', background=BG)
    st.configure('TCheckbutton', background=BG, foreground=FG, indicatorbackground=FIELD,
                 indicatorforeground=FG)
    st.map('TCheckbutton', background=[('active', BG)],
           indicatorbackground=[('selected', ACCENT), ('active', PANEL)])
    st.configure('TButton', background=PANEL, foreground=FG, bordercolor="#3a4150",
                 padding=(10, 3))
    st.map('TButton', background=[('pressed', ACCENT), ('active', "#2e3544"),
                                  ('disabled', BG)],
           foreground=[('disabled', MUTED)])
    st.configure('Accent.TButton', background=ACCENT, foreground="white",
                 font=("Segoe UI", 10, "bold"), padding=(14, 5))
    st.map('Accent.TButton', background=[('pressed', "#2f76bd"), ('active', "#4a9be8"),
                                         ('disabled', PANEL)])
    st.configure('TEntry', fieldbackground=FIELD, foreground=FG, insertcolor=FG)
    st.configure('TCombobox', fieldbackground=FIELD, foreground=FG, background=PANEL,
                 arrowcolor=FG)
    st.map('TCombobox', fieldbackground=[('readonly', FIELD)],
           foreground=[('readonly', FG)], selectbackground=[('readonly', FIELD)])
    root.option_add('*TCombobox*Listbox.background', FIELD)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    st.configure('Vertical.TScrollbar', background=PANEL, arrowcolor=FG, troughcolor=FIELD)


def _pe_bitness(data):
    """32 or 64 from a Windows executable's PE header, or None."""
    if len(data) < 0x40 or data[:2] != b'MZ':
        return None
    off = int.from_bytes(data[0x3C:0x40], 'little')
    if off + 6 > len(data) or data[off:off + 4] != b'PE\0\0':
        return None
    machine = int.from_bytes(data[off + 4:off + 6], 'little')
    return {0x014c: 32, 0x8664: 64, 0xaa64: 64, 0x0200: 64}.get(machine)


class PipelineGUI:
    """Plain tkinter/ttk window (no CustomTkinter): the standard Windows look,
    and nothing extra to bundle into the .exe."""

    def __init__(self, root):
        self.root = root
        root.title("Source to SoF2  v{}".format(APP_VERSION))
        root.geometry("900x900")
        root.minsize(760, 640)
        apply_dark_theme(root)

        self.vmf_var = tk.StringVar()
        self.sources_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mapname_var = tk.StringVar()
        self.terrain_mode_var = tk.StringVar(value="lwo")
        self.skybox_models_to_world_var = tk.BooleanVar(value=True)
        self.skybox_landscape_var = tk.StringVar(value="4")
        self.full_quality_var = tk.BooleanVar(value=False)
        self.q3map2_var = tk.StringVar()
        self.sof2_base_var = tk.StringVar()

        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.steps = build_steps()

        self._build_ui()
        self._load_settings()
        self.vmf_var.trace_add('write', self._schedule_bsp_check)
        self.sources_var.trace_add('write', self._schedule_bsp_check)
        self._schedule_bsp_check()
        self.root.after(100, self._poll_log_queue)

    # --------------------------------------------------------- settings --
    # Field values are remembered between sessions, so the paths don't have
    # to be picked again every time.
    SETTINGS_PATH = os.path.join(os.path.expanduser("~"), APP_NAME + ".json")
    OLD_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".sof2_pipeline_gui.json")

    def _settings_vars(self):
        # Keys that no longer exist here are simply ignored if an older
        # settings file still has them.
        return {"vmf": self.vmf_var, "sources": self.sources_var, "output": self.output_var,
                "map_name": self.mapname_var,
                "terrain_mode": self.terrain_mode_var,
                "sof2_base": self.sof2_base_var,
                "skybox_models_to_world": self.skybox_models_to_world_var,
                "skybox_landscape_brushes": self.skybox_landscape_var,
                "full_quality": self.full_quality_var,
                "q3map2": self.q3map2_var}

    def _load_settings(self):
        data = None
        for path in (self.SETTINGS_PATH, self.OLD_SETTINGS_PATH):   # old name: pre-1.0 builds
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                break
            except (OSError, ValueError):
                continue
        if not isinstance(data, dict):
            return
        for key, var in self._settings_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except (tk.TclError, ValueError):
                    pass
        steps_on = data.get("steps")
        if isinstance(steps_on, dict):
            for step in self.steps:
                if step.key in steps_on:
                    step.var.set(bool(steps_on[step.key]))

    def _save_settings(self):
        data = {key: var.get() for key, var in self._settings_vars().items()}
        data["steps"] = {step.key: bool(step.var.get()) for step in self.steps}
        try:
            with open(self.SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # --- settings ---
        fields = ttk.LabelFrame(self.root, text="Map")
        fields.pack(fill='x', padx=10, pady=(10, 4))
        fields.columnconfigure(1, weight=1)
        self._row = 0

        self._path_row(fields, "Source .vmf", self.vmf_var, self._pick_vmf,
                       "Source map file (.vmf)")
        self._path_row(fields, "Source content root(s)", self.sources_var, self._pick_sources,
                       "Your game install (counter-strike source, or its cstrike folder) - "
                       "or extracted folders / .vpk files")
        self.bsp_status = self._status_row(
            fields, "Map's own .bsp",
            "A Source map can pack its own materials and models into its .bsp. For a "
            "ported or community map that is most of what makes it look like itself. "
            "Searched for next to the .vmf, then under the content roots, and read before "
            "the game's own content, the way Source mounts it.")
        self._path_row(fields, "Output folder", self.output_var, self._pick_output,
                       "Everything this pipeline produces goes here - maps / models / "
                       "shaders / textures")
        self._entry_row(fields, "Map name", self.mapname_var,
                        "Auto-filled from your .vmf - edit if you want a different name")
        self._path_row(fields, "SOF2 base folder", self.sof2_base_var, self._pick_sof2_base,
                       "For step 10 - e.g. ...\\Soldier of Fortune II - MODDING\\base")
        self._path_row(fields, "q3map2.exe", self.q3map2_var, self._pick_q3map2,
                       "The q3map2.exe the compile .bat (written by step 10) uses. Empty = "
                       "looked for next to the SOF2 install.")

        # --- options ---
        opts = ttk.LabelFrame(self.root, text="Options")
        opts.pack(fill='x', padx=10, pady=4)
        opts.columnconfigure(1, weight=1)
        self._row = 0
        self._combo_row(opts, "Terrain mode", self.terrain_mode_var, ["lwo", "patch", "brush"],
                        "How displacements (terrain) are built. 'lwo' (recommended): every "
                        "displacement welded into one seam-free model, built the way Source "
                        "builds it. 'patch' and 'brush' are the older modes.")
        self._check_row(opts, "Skybox props -> real map space", self.skybox_models_to_world_var,
                        "Move 3D-skybox props to the real position they stand for "
                        "(P_world = (P_sky - sky_camera origin) x scale), as ordinary props. "
                        "Leave on unless a map's skybox props land wrongly.")
        self._entry_row(opts, "Skybox landscape brushes", self.skybox_landscape_var,
                        "Cells per displacement for the 3D-skybox backdrop landscape, built "
                        "as coarse brushes so it stays within the skybox surface budget. "
                        "0 = off. Default 4.", width=6)
        self._check_row(opts, "Props at original quality", self.full_quality_var,
                        "Every prop at Source's own full detail: no reduced-detail copies "
                        "and nothing dropped to fit the .bsp size target. The step 6 log "
                        "still says how big the .bsp will be - a very large one may not "
                        "load in SOF2 (it is read into one block of memory), so this is "
                        "for optimising the map by hand afterwards.")

        # --- steps ---
        steps = ttk.LabelFrame(self.root, text="Steps (run top to bottom)")
        steps.pack(fill='x', padx=10, pady=4)
        for step in self.steps:
            step.var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(steps, text=step.label, variable=step.var)
            cb.pack(anchor='w', padx=8, pady=1)
            if step.tooltip:
                Tooltip(cb, step.tooltip, x_offset=330, y_offset=-6)
        btns = ttk.Frame(steps)
        btns.pack(fill='x', padx=8, pady=(4, 6))
        ttk.Button(btns, text="Select All", command=self._select_all).pack(side='left')
        ttk.Button(btns, text="Select None", command=self._select_none).pack(side='left', padx=6)

        # --- run ---
        run_row = ttk.Frame(self.root)
        run_row.pack(fill='x', padx=10, pady=4)
        self.run_button = ttk.Button(run_row, text="Run Selected Steps", style='Accent.TButton',
                                     command=self._run_clicked)
        self.run_button.pack(side='left')
        self.status_label = tk.Label(run_row, text="Ready", fg=OK_COLOR, bg=BG,
                                     font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side='left', padx=12)

        # --- log ---
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        self.log_text = ScrolledText(log_frame, wrap='word', state='disabled', height=12,
                                     font=("Consolas", 9), bg=FIELD, fg="#d0d0d0",
                                     insertbackground=FG, relief='flat', borderwidth=0)
        self.log_text.pack(fill='both', expand=True, padx=6, pady=6)
        try:
            self.log_text.vbar.configure(bg=PANEL, troughcolor=FIELD, activebackground=ACCENT,
                                         relief='flat', borderwidth=0)
        except tk.TclError:
            pass

    def _label(self, parent, title, tooltip):
        lbl = ttk.Label(parent, text=title)
        lbl.grid(row=self._row, column=0, sticky='w', padx=8, pady=3)
        if tooltip:
            Tooltip(lbl, tooltip)
        return lbl

    def _path_row(self, parent, title, var, command, tooltip=None):
        self._label(parent, title, tooltip)
        ttk.Entry(parent, textvariable=var).grid(row=self._row, column=1, sticky='ew', pady=3)
        ttk.Button(parent, text="Browse...", command=command).grid(
            row=self._row, column=2, padx=8, pady=3)
        self._row += 1

    def _entry_row(self, parent, title, var, tooltip=None, width=None):
        self._label(parent, title, tooltip)
        ttk.Entry(parent, textvariable=var, width=width or 20).grid(
            row=self._row, column=1, sticky='w' if width else 'ew', pady=3)
        self._row += 1

    def _check_row(self, parent, title, var, tooltip=None):
        self._label(parent, title, tooltip)
        ttk.Checkbutton(parent, variable=var).grid(row=self._row, column=1, sticky='w', pady=3)
        self._row += 1

    def _combo_row(self, parent, title, var, options, tooltip=None):
        self._label(parent, title, tooltip)
        ttk.Combobox(parent, textvariable=var, values=options, state='readonly',
                     width=10).grid(row=self._row, column=1, sticky='w', pady=3)
        self._row += 1

    def _status_row(self, parent, title, tooltip=None):
        self._label(parent, title, tooltip)
        value = tk.Label(parent, text="", fg=MUTED, bg=BG, anchor='w', justify='left',
                         wraplength=560)
        value.grid(row=self._row, column=1, columnspan=2, sticky='ew', pady=3)
        self._row += 1
        return value

    # --- map .bsp detection -------------------------------------------------
    # Walking a whole game folder takes a moment, so it runs on a worker
    # thread, half a second after the last change to either path, and a
    # result is only shown if the paths have not changed again since.
    def _schedule_bsp_check(self, *_):
        if getattr(self, '_bsp_after', None):
            self.root.after_cancel(self._bsp_after)
        self._bsp_after = self.root.after(500, self._start_bsp_check)

    def _start_bsp_check(self):
        self._bsp_after = None
        vmf, src = self.vmf_var.get().strip(), self.sources_var.get().strip()
        key = (vmf, src)
        self._bsp_key = key
        if not vmf:
            self.bsp_status.configure(text="", fg=MUTED)
            self._bsp_result = (key, None)
            return
        name = os.path.splitext(os.path.basename(vmf.replace('\\', '/')))[0]
        self.bsp_status.configure(text="Looking for {}.bsp ...".format(name), fg=MUTED)

        def work():
            found, note = find_map_bsp(vmf, src)
            self.root.after(0, lambda: self._show_bsp_result(key, name, found, note))
        threading.Thread(target=work, daemon=True).start()

    def _show_bsp_result(self, key, name, found, note):
        if key != getattr(self, '_bsp_key', None):
            return  # the paths changed while this search ran
        self._bsp_result = (key, found)
        if found:
            self.bsp_status.configure(
                text="✓ {} found - its packed content is read first, the way Source "
                     "loads it.\n{}".format(os.path.basename(found), os.path.dirname(found)),
                fg=OK_COLOR)
        else:
            self.bsp_status.configure(
                text="No {}.bsp found next to the .vmf or under your content roots.{}\n"
                     "Fine for a stock map. A ported or community map keeps its custom "
                     "materials and models inside its .bsp, and they will be reported "
                     "missing without it.".format(name, ("  " + note) if note else ""),
                fg=WARN_COLOR)

    def _cached_map_bsp(self):
        """The detected .bsp, if the search finished for the current paths."""
        key = (self.vmf_var.get().strip(), self.sources_var.get().strip())
        res = getattr(self, '_bsp_result', None)
        return res[1] if res and res[0] == key else None

    def _pick_vmf(self):
        path = filedialog.askopenfilename(title="Select .vmf", filetypes=[("VMF files", "*.vmf")])
        if path:
            self.vmf_var.set(path)
            self.mapname_var.set(os.path.splitext(os.path.basename(path))[0])

    def _pick_sources(self):
        path = filedialog.askdirectory(title="Select source content root folder")
        if path:
            self.sources_var.set(path)

    def _pick_output(self):
        path = filedialog.askdirectory(title="Select (or create) your output folder")
        if path:
            self.output_var.set(path)

    def _pick_q3map2(self):
        path = filedialog.askopenfilename(title="Select q3map2.exe",
                                          filetypes=[("q3map2", "q3map2*.exe"), ("Programs", "*.exe")])
        if path:
            self.q3map2_var.set(path)

    def _pick_sof2_base(self):
        path = filedialog.askdirectory(title="Select SOF2's base folder")
        if path:
            self.sof2_base_var.set(path)

    def _preflight(self, selected):
        """Plain-language problems with the fields, for the steps about to
        run - caught here instead of as an error halfway through a step."""
        keys = {s.key for s in selected}
        out = []
        vmf = self.vmf_var.get().strip()
        if keys & {'extract', 'map', 'clip', 'shader'} and not os.path.isfile(vmf):
            out.append("The Source .vmf was not found:\n{}".format(vmf or "(empty)"))
        if 'extract' in keys:
            roots = [r.strip().strip('"') for r in self.sources_var.get().split(';') if r.strip()]
            if not roots:
                out.append("No source content root is set (your CS:S install, its cstrike "
                           "folder, or extracted folders / .vpk files).")
            for r in roots:
                if not os.path.exists(r):
                    out.append("Source content root not found:\n{}".format(r))
        if 'install' in keys:
            base = self.sof2_base_var.get().strip()
            if not os.path.isdir(base):
                out.append("The SOF2 base folder was not found:\n{}".format(base or "(empty)"))
            elif not (os.path.isdir(os.path.join(base, 'maps')) or any(
                    f.lower().endswith('.pk3') for f in os.listdir(base))):
                out.append("The SOF2 base folder has no maps\\ folder and no .pk3 files - "
                           "is it really SOF2's base folder?\n{}".format(base))
            q3 = self.q3map2_var.get().strip()
            if q3 and not os.path.isfile(q3):
                out.append("q3map2.exe was not found:\n{}".format(q3))
            elif not q3:
                out.append("No q3map2.exe set. The compile .bat will look for it next to the "
                           "SOF2 install, and say so if it is not there.")
            else:
                note = self._q3map2_lump_warning(q3)
                if note:
                    out.append(note)
        return out

    @staticmethod
    def _q3map2_lump_warning(q3map2):
        """GtkRadiant's q3map2 keeps the .bsp draw lumps in fixed arrays, so a
        big converted map stops with MAX_MAP_DRAW_INDEXES part way through
        -bsp. Said here, before a run, not after a failed compile."""
        try:
            with open(q3map2, 'rb') as f:
                data = f.read()
        except OSError:
            return None
        notes = []
        if (b'GtkRadiant' in data and not (b'NetRadiant' in data or b'netradiant' in data
                                           or b'-gridscale' in data)):
            notes.append(
                "It keeps the .bsp draw-vertex and draw-index lumps in fixed arrays (524,288 "
                "each), so a large converted map stops part way through -bsp with "
                "\"MAX_MAP_DRAW_INDEXES\" (and at 8192 shaders with \"MAX_SHADER_INFO\").")
        if _pe_bitness(data) == 32:
            notes.append(
                "It is a 32-bit program. -light asks for ONE block of a few hundred MB for the "
                "trace nodes on a map this size (measured: 214.5 MiB), which a 32-bit process "
                "often cannot give it - the compile stops with \"safe_malloc failed on "
                "allocation\" in SetupTraceNodes.")
        if not notes:
            return None
        return ("This q3map2 will not compile a large converted map:\n{}\n\n{}\n\n"
                "NetRadiant's q3map2 (64-bit, growable lumps) compiles the same map "
                "unchanged - use it here. Whichever Radiant you prefer stays fine for "
                "editing.".format(q3map2, "\n\n".join(notes)))

    def _select_all(self):
        for step in self.steps:
            step.var.set(True)

    def _select_none(self):
        for step in self.steps:
            step.var.set(False)

    # ------------------------------------------------------------- run --

    def _run_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "A pipeline run is already in progress.")
            return
        if not self.output_var.get():
            messagebox.showerror("Missing output folder", "Pick an output folder first.")
            return

        selected = [s for s in self.steps if s.var.get()]
        if not selected:
            messagebox.showinfo("Nothing selected", "Tick at least one step to run.")
            return
        problems = self._preflight(selected)
        if problems and not messagebox.askyesno(
                "Check these first",
                "\n\n".join(problems) + "\n\nRun anyway?"):
            return

        paths = PipelinePaths(self.vmf_var.get(), self.sources_var.get(),
                               self.output_var.get(), self.mapname_var.get(),
                               terrain_mode=self.terrain_mode_var.get(),
                               skybox_models_to_world=self.skybox_models_to_world_var.get(),
                               skybox_landscape_brushes=self.skybox_landscape_var.get(),
                               sof2_base=self.sof2_base_var.get(),
                               map_bsp=self._cached_map_bsp(),
                               full_quality=self.full_quality_var.get(),
                               q3map2=self.q3map2_var.get().strip())
        self._save_settings()

        self.run_button.configure(state='disabled')
        self.status_label.configure(text="Running...", fg=WARN_COLOR)
        self._clear_log()
        # The previous run's log file goes as this run starts; this run's is
        # written when it ends (_write_run_log).
        self._run_output_dir = self.output_var.get()
        for old in (run_log_path(), os.path.join(self._run_output_dir, RUN_LOG_NAME)):
            try:
                if old and os.path.isfile(old):
                    os.remove(old)
            except OSError:
                pass

        self.worker_thread = threading.Thread(target=self._run_steps, args=(selected, paths),
                                               daemon=True)
        self.worker_thread.start()

    def _run_steps(self, selected, paths):
        """Runs on the background thread - must never touch tkinter widgets
        directly, only push text onto log_queue (polled on the main thread)."""
        log_stream = LogStream(self.log_queue)
        old_out, old_err, old_argv = sys.stdout, sys.stderr, sys.argv
        sys.stdout, sys.stderr = log_stream, log_stream

        overall_ok = True
        try:
            missing = [s.module_name + ".py" for s in selected if resource_path(s.module_name + ".py") is None]
            # Helpers a step hands on rather than imports (STEP_HELPER_FILES),
            # checked up front with the rest.
            for step in selected:
                for helper in STEP_HELPER_FILES.get(step.key, ()):
                    if resource_path(helper) is None and helper not in missing:
                        missing.append(helper)
            if missing:
                self.log_queue.put(
                    "Can't start - these scripts are missing:\n  {}\n\nPut them next to the "
                    ".exe (looked in: {}), or untick those steps. Nothing was run.\n".format(
                        "\n  ".join(missing), "; ".join(script_search_dirs())))
                overall_ok = False
                return
            for step in selected:
                self.log_queue.put("\n{}\n{}\n{}\n".format("=" * 70, step.label, "=" * 70))

                ready, reason = step.check_ready(paths)
                if not ready:
                    self.log_queue.put("SKIPPED - {}\n".format(reason))
                    overall_ok = False
                    continue

                try:
                    mod = load_module(step.module_name + ".py")
                except Exception:
                    self.log_queue.put("FAILED to load {}.py:\n{}\n".format(
                        step.module_name, traceback.format_exc()))
                    overall_ok = False
                    continue

                args = step.build_args(paths)
                used = resource_path(step.module_name + ".py")
                if not getattr(sys, '_MEIPASS', None) or not used.startswith(sys._MEIPASS):
                    self.log_queue.put("(using {})\n".format(used))
                sys.argv = [step.module_name + ".py"] + [str(a) for a in args]
                self.log_queue.put("$ {}\n".format(" ".join(sys.argv)))

                try:
                    mod.main()
                    self.log_queue.put("\n[{}] completed.\n".format(step.key))
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
                    if code != 0:
                        self.log_queue.put("\n[{}] EXITED with code {} - stopping here.\n"
                                            .format(step.key, code))
                        overall_ok = False
                        break
                    self.log_queue.put("\n[{}] completed.\n".format(step.key))
                except Exception:
                    self.log_queue.put("\n[{}] CRASHED:\n{}\nStopping here.\n".format(
                        step.key, traceback.format_exc()))
                    overall_ok = False
                    break
        finally:
            sys.stdout, sys.stderr, sys.argv = old_out, old_err, old_argv
            self.log_queue.put("\n{}\n".format("ALL DONE" if overall_ok else "STOPPED (see above)"))
            self.log_queue.put(("__DONE_OK__" if overall_ok else "__DONE_FAIL__"))

    # --------------------------------------------------------- log pane --

    def _write_run_log(self, ok):
        """Save the whole log pane to RUN_LOG_NAME next to the .exe (the
        output folder if that one is not writable, e.g. under Program Files)."""
        import datetime
        text = self.log_text.get('1.0', 'end')
        head = "{} v{} - {} - {}\n\n".format(
            APP_NAME, APP_VERSION, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "finished" if ok else "STOPPED")
        for path in (run_log_path(), os.path.join(getattr(self, '_run_output_dir', '') or '',
                                                  RUN_LOG_NAME)):
            if not path or not os.path.isdir(os.path.dirname(os.path.abspath(path))):
                continue
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(head + text)
            except OSError:
                continue
            self._append_log("\nLog saved to {}\n".format(path))
            return

    def _append_log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__DONE_OK__":
                    self.run_button.configure(state='normal')
                    self.status_label.configure(text="Done", fg=OK_COLOR)
                    self._write_run_log(True)
                    continue
                if item == "__DONE_FAIL__":
                    self.run_button.configure(state='normal')
                    self.status_label.configure(text="Stopped - see log", fg=ERR_COLOR)
                    self._write_run_log(False)
                    continue
                self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)


def run_helper(argv):
    """SourceToSoF2.exe --run <script.py> [args...]: run one bundled pipeline
    script the way "python script.py args" would. The compile .bat uses this,
    so the person compiling needs no Python of their own. The .exe has no
    console window of its own (--noconsole), so on Windows it attaches to the
    console of the .bat that started it, for the script's output."""
    if os.name == 'nt':
        try:
            import ctypes
            if ctypes.windll.kernel32.AttachConsole(-1):
                sys.stdout = open('CONOUT$', 'w', buffering=1)
                sys.stderr = sys.stdout
        except Exception:
            pass
    if not argv:
        print("usage: {} --run <script.py> [args...]".format(APP_NAME))
        return 2
    script = os.path.basename(argv[0])
    path = resource_path(script)
    if path is None:
        print("{} --run: {} is not bundled (looked in: {})".format(
            APP_NAME, script, "; ".join(script_search_dirs())))
        return 2
    import runpy
    sys.argv = [path] + list(argv[1:])
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (1 if e.code else 0)
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        sys.exit(run_helper(sys.argv[2:]))
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
