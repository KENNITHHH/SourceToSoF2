# Building SourceToSoF2.exe

The repository is the Python source. The release is one `.exe` built from it.

## Once

```
pip install pyinstaller numpy pillow
```

Python 3.9 or newer, 64-bit. `tkinter` comes with the standard Windows
installer; nothing else is needed.

## Every build

Run **`MakeExe.bat`** from the folder holding the `.py` files. It writes
`dist\SourceToSoF2.exe` - a single file with the interpreter, the libraries and
all 20 pipeline scripts inside it.

The scripts have to sit next to `MakeExe.bat`: it names each one as
`--add-data`, and the GUI checks at startup that every script a selected step
needs is present. **Adding a script means adding it in three places** -
`MakeExe.bat`, the build command in `sof2_pipeline_gui.py`'s docstring, and
`STEP_HELPER_FILES` in that same file.

## What goes in the release

| File | |
|---|---|
| `SourceToSoF2.exe` | from `dist\` |
| `README.md` | |
| `LICENSE` | |
| `THIRD-PARTY-NOTICES.md` | |

Nothing else: the `.exe` carries its own Python and every helper script.

Bump `APP_VERSION` in `sof2_pipeline_gui.py` before tagging - it shows in the
window title and at the top of every `SourceToSoF2.txt` log, which is how a bug
report says which build it came from.

## Checking a build

1. Open it - the title bar should read the new version.
2. Point **q3map2.exe** at a GtkRadiant build: the warning box should appear
   before anything runs.
3. Convert a small map with all ten steps, then run the compile `.bat`.
4. Load the map with `com_hunkMegs 256`.
