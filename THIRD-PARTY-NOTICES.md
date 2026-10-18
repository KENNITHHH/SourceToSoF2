# Third-party notices

`SourceToSoF2.exe` is a single file built with [PyInstaller](https://pyinstaller.org/),
so it carries a Python interpreter and a few libraries inside it. SourceToSoF2's
own code is MIT (see LICENSE); the bundled components keep their own licenses,
listed here.

| Component | License |
|---|---|
| CPython (the interpreter and its standard library, including tkinter) | Python Software Foundation License 2.0 |
| NumPy | BSD 3-Clause |
| Pillow (PIL fork) | MIT-CMU |
| PyInstaller bootloader (the small launcher at the start of the `.exe`) | GPL 2.0 or later, with the PyInstaller bootloader exception, which allows bundling with software under any license |

Full license texts ship with each project and are installed alongside it
(`pip show -f <package>` locates them).

Nothing from Soldier of Fortune II, Counter-Strike or the Source engine is
bundled in the `.exe`. The tool reads content you already own, on your own
machine, and writes the converted result to your own folders.

## Converted content

Maps, models, textures and sounds converted with this tool stay the property of
their owners - Valve Corporation for Counter-Strike content, Raven Software and
Activision for Soldier of Fortune II. Converting them does not change that, and
none of it is licensed by this project. Share converted maps only where you have
the right to.
