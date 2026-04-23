"""Portability shim: make mmasim import on macOS.

mmasim/simulator/arithmetic.py does `ctypes.CDLL("libm.so.6")` at module
load, which is Linux-only. We patch ctypes.CDLL to redirect that one name
to libSystem on Darwin, so the oracle loads unmodified.

Import this BEFORE any `mmasim` import.
"""
import ctypes
import platform


def install() -> None:
    if platform.system() != "Darwin":
        return
    real_cdll = ctypes.CDLL

    def patched_cdll(name, *args, **kwargs):
        if name == "libm.so.6":
            return real_cdll("libSystem.B.dylib", *args, **kwargs)
        return real_cdll(name, *args, **kwargs)

    ctypes.CDLL = patched_cdll  # type: ignore[assignment]


install()
