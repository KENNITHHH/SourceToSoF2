#!/usr/bin/env python3
"""
sof2_memmap.py - How big a .bsp can SoF2MP.exe load on THIS machine, right now?

SOF2 reads the whole .bsp into one allocation (Z_Malloc). SoF2MP.exe is a
32-bit program without the large-address-aware flag, so that block has to fit
in the largest FREE hole of its 2 GB address space. That hole depends on what
else is loaded into the process: the GL driver, overlays (Steam, Discord,
GeForce), wrapper/hook DLLs, com_hunkmegs. This script reads the running
game's address space and prints the holes and the biggest DLLs.

USAGE (Windows, with SOF2 running and sitting at the main menu):
    python sof2_memmap.py              (looks for SoF2MP.exe)
    python sof2_memmap.py otherexe.exe

It only reads; it changes nothing in the game.
"""
import ctypes
import ctypes.wintypes as wt
import sys

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
TH32CS_SNAPPROCESS = 0x2
TH32CS_SNAPMODULE = 0x8
TH32CS_SNAPMODULE32 = 0x10
MEM_FREE = 0x10000
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_IMAGE = 0x1000000


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.c_void_p), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', ctypes.c_long), ('dwFlags', wt.DWORD),
                ('szExeFile', ctypes.c_char * 260)]


class MODULEENTRY32(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('th32ModuleID', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('GlblcntUsage', wt.DWORD), ('ProccntUsage', wt.DWORD),
                ('modBaseAddr', ctypes.c_void_p), ('modBaseSize', wt.DWORD),
                ('hModule', ctypes.c_void_p), ('szModule', ctypes.c_char * 256),
                ('szExePath', ctypes.c_char * 260)]


class MBI(ctypes.Structure):
    _fields_ = [('BaseAddress', ctypes.c_void_p), ('AllocationBase', ctypes.c_void_p),
                ('AllocationProtect', wt.DWORD), ('PartitionId', wt.WORD),
                ('RegionSize', ctypes.c_size_t), ('State', wt.DWORD),
                ('Protect', wt.DWORD), ('Type', wt.DWORD)]


k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE
k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t


def find_pid(name):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(pe)
    ok = k32.Process32First(snap, ctypes.byref(pe))
    pid = None
    while ok:
        if pe.szExeFile.decode('latin-1').lower() == name.lower():
            pid = pe.th32ProcessID
            break
        ok = k32.Process32Next(snap, ctypes.byref(pe))
    k32.CloseHandle(snap)
    return pid


def modules(pid):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(me)
    out = []
    ok = k32.Module32First(snap, ctypes.byref(me))
    while ok:
        out.append((me.modBaseAddr or 0, me.modBaseSize, me.szModule.decode('latin-1')))
        ok = k32.Module32Next(snap, ctypes.byref(me))
    k32.CloseHandle(snap)
    return out


def mib(n):
    return n / 1048576.0


def main(argv):
    name = argv[1] if len(argv) > 1 else 'SoF2MP.exe'
    pid = find_pid(name)
    if pid is None:
        print("{} is not running - start SOF2, wait at the main menu, run this again.".format(name))
        return 1
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        print("Could not open {} (pid {}): error {}. Run this from the same user, or as "
              "administrator.".format(name, pid, ctypes.get_last_error()))
        return 1
    free, used = [], {'commit': 0, 'reserve': 0, 'image': 0}
    addr, top = 0x10000, 0x7FFF0000
    mbi = MBI()
    while addr < top:
        if not k32.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        size = mbi.RegionSize
        if mbi.State == MEM_FREE:
            free.append((size, addr))
        elif mbi.Type == MEM_IMAGE:
            used['image'] += size
        elif mbi.State == MEM_COMMIT:
            used['commit'] += size
        elif mbi.State == MEM_RESERVE:
            used['reserve'] += size
        addr += size
    k32.CloseHandle(h)
    free.sort(reverse=True)
    total_free = sum(s for s, _a in free)
    print("{} (pid {}), 2 GB address space:".format(name, pid))
    print("  free in total      {:8.1f} MiB".format(mib(total_free)))
    print("  DLLs/exe (images)  {:8.1f} MiB".format(mib(used['image'])))
    print("  committed memory   {:8.1f} MiB".format(mib(used['commit'])))
    print("  reserved, unused   {:8.1f} MiB".format(mib(used['reserve'])))
    print()
    print("LARGEST FREE HOLES - the .bsp (plus a little) has to fit in the first one:")
    for size, a in free[:8]:
        print("  {:8.1f} MiB  ({:,} bytes) at 0x{:08X}".format(mib(size), size, a))
    print()
    print("BIGGEST MODULES LOADED (what splits the address space):")
    for base, size, mod in sorted(modules(pid), key=lambda m: -m[1])[:15]:
        print("  {:8.1f} MiB  0x{:08X}  {}".format(mib(size), base, mod))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
