@echo off
rem ---------------------------------------------------------------------------
rem Builds dist\SourceToSoF2.exe. Run it from the folder that holds all the
rem pipeline scripts. The window is plain tkinter (part of Python), so
rem CustomTkinter is no longer needed or bundled.
rem
rem Every script the GUI uses has to be listed as --add-data, because
rem PyInstaller only bundles what it can see: it follows "import" lines, and
rem the GUI never imports these - it loads each one from a file path at run
rem time. Four of them are easy to miss because no step is named after them:
rem   make_pk3.py            builds the .pk3 (step 10 fails without it)
rem   audit_release.py       the release audit step 10 runs before packing
rem   bsp_fix_leafbounds.py  copied next to the compile .bat; runs between -bsp
rem                          and -vis, without it -light leaves black bands
rem                          along wall bottoms
rem   bsp_decal_relight.py   copied next to the compile .bat; runs after -light,
rem                          gives decals the light of the ground under them
rem   bsp_shader_count.py    copied next to the compile .bat; the MAX_SHADERS
rem                          check at the end of a compile
rem   shaderlist_for_compile.py  copied next to the compile .bat; keeps the other
rem                          converted maps' shaders out of q3map2 during the
rem                          compile (MAX_SHADER_INFO), then restores the list
rem The GUI checks for all of them before it runs anything, so a bundle missing
rem one says so up front instead of failing at step 10.
rem ---------------------------------------------------------------------------

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
    --add-data "make_pk3.py;." ^
    --add-data "audit_release.py;." ^
    --add-data "bsp_fix_leafbounds.py;." ^
    --add-data "bsp_decal_relight.py;." ^
    --add-data "bsp_shader_count.py;." ^
    --add-data "shaderlist_for_compile.py;." ^
    --hidden-import numpy ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.PngImagePlugin ^
    --hidden-import PIL.JpegImagePlugin ^
    --hidden-import heapq ^
    --hidden-import PIL.TgaImagePlugin ^
    --hidden-import glob ^
    --exclude-module customtkinter ^
    --exclude-module darkdetect ^
    sof2_pipeline_gui.py
pause
