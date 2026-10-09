"""
The app's name on macOS, where a script run by Python is otherwise called
after the Python executable (`python3.13`).

macOS keeps the name in two places, and both are set here through the
system's own libraries (`ctypes`, no extra package):
- the main bundle's `CFBundleName`, read once at start for the menu bar
  (`name_bundle`, before the `QApplication` exists);
- the name Launch Services shows in the app switcher (Command-Tab) and
  Activity Monitor (`name_process`, once the application has started). This
  uses the private calls Chromium names its processes with, so a macOS
  update could stop it working; it then does nothing.

Both do nothing on other systems, or when a call is missing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

# CFStringCreateWithCString's encoding for UTF-8
UTF8 = 0x08000100

# Launch Services' session for the current user
DEFAULT_SESSION = -2


def _library(name: str):
    path = ctypes.util.find_library(name)

    return None if path is None else ctypes.cdll.LoadLibrary(path)


def _cf_string(foundation, text: str) -> ctypes.c_void_p:
    create = foundation.CFStringCreateWithCString
    create.restype = ctypes.c_void_p
    create.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]

    return create(None, text.encode(), UTF8)


def name_bundle(name: str) -> None:
    """The menu bar's name; call it before the `QApplication` is made."""
    if sys.platform != "darwin":
        return

    try:
        objc = _library("objc")
        foundation = _library("Foundation")
    except OSError:
        return

    if objc is None or foundation is None:
        return

    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(receiver, selector: bytes, *args, returns=ctypes.c_void_p):
        # A function object per signature, since the argument types differ
        function = ctypes.CFUNCTYPE(
            returns, ctypes.c_void_p, ctypes.c_void_p, *[ctypes.c_void_p] * len(args)
        )(("objc_msgSend", objc))

        return function(receiver, objc.sel_registerName(selector), *args)

    bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
    info = send(bundle, b"infoDictionary") if bundle else None
    can_set = objc.sel_registerName(b"setObject:forKey:")

    # Only a mutable dictionary takes the name (an immutable one would abort)
    if info and send(info, b"respondsToSelector:", can_set, returns=ctypes.c_bool):
        send(
            info,
            b"setObject:forKey:",
            _cf_string(foundation, name),
            _cf_string(foundation, "CFBundleName"),
        )


def name_process(name: str) -> None:
    """The app switcher's name; call it once the application has started."""
    if sys.platform != "darwin":
        return

    try:
        services = _library("CoreServices")
        foundation = _library("CoreFoundation")
    except OSError:
        return

    if services is None or foundation is None:
        return

    try:
        current = services._LSGetCurrentApplicationASN
        set_item = services._LSSetApplicationInformationItem
        key = ctypes.c_void_p.in_dll(services, "_kLSDisplayNameKey")
    except (AttributeError, ValueError):
        return

    current.restype = ctypes.c_void_p
    current.argtypes = []
    set_item.restype = ctypes.c_int
    set_item.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    application = current()

    if application and key.value:
        set_item(
            DEFAULT_SESSION, application, key.value, _cf_string(foundation, name), None
        )
