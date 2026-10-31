"""Minimal Win32 focus/hotkeys from the working test.py; no game input injection."""
import ctypes
from ctypes import wintypes
import time


def require_english_keyboard(api,hwnd):
    """Inspect the game's thread layout; do not change the user's language."""
    api.GetWindowThreadProcessId.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    api.GetWindowThreadProcessId.restype = wintypes.DWORD
    api.GetKeyboardLayout.argtypes = [wintypes.DWORD]
    api.GetKeyboardLayout.restype = wintypes.HANDLE
    thread = api.GetWindowThreadProcessId(hwnd,None)
    if not thread:
        raise RuntimeError("Cannot read game keyboard thread; no macro input sent")
    layout = api.GetKeyboardLayout(thread)
    if not layout:
        raise RuntimeError("Cannot read game keyboard layout; no macro input sent")
    language = int(layout) & 0xFFFF
    # PRIMARYLANGID: accept English US/UK/etc., not only one keyboard variant.
    if (language & 0x03FF) != 0x09:
        raise RuntimeError(f"Game keyboard language is 0x{language:04X}, expected EN. "
                           "Switch to English while the game is focused, then press F8 again. No macro input sent.")
    return language


class GameWindow:
    def __init__(self, title="TailsTopiaOnline"):
        self.api = ctypes.windll.user32
        self.api.SetProcessDPIAware()
        self.title = title
        self.hwnd = None
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        signatures = {
            "EnumWindows": ([self.callback_type,wintypes.LPARAM],wintypes.BOOL),
            "GetForegroundWindow": ([],wintypes.HWND),
            "GetWindowTextLengthW": ([wintypes.HWND],ctypes.c_int),
            "GetWindowTextW": ([wintypes.HWND,wintypes.LPWSTR,ctypes.c_int],ctypes.c_int),
            "IsWindowVisible": ([wintypes.HWND],wintypes.BOOL),
            "IsIconic": ([wintypes.HWND],wintypes.BOOL),
            "ShowWindow": ([wintypes.HWND,ctypes.c_int],wintypes.BOOL),
            "SetForegroundWindow": ([wintypes.HWND],wintypes.BOOL),
            "GetAsyncKeyState": ([ctypes.c_int],ctypes.c_short),
            "GetClientRect": ([wintypes.HWND,ctypes.POINTER(wintypes.RECT)],wintypes.BOOL),
            "ClientToScreen": ([wintypes.HWND,ctypes.POINTER(wintypes.POINT)],wintypes.BOOL),
        }
        for name,(arguments,result) in signatures.items():
            function = getattr(self.api,name)
            function.argtypes,function.restype = arguments,result

    def pressed(self, vk):
        return bool(self.api.GetAsyncKeyState(vk) & 0x8000)

    def geometry(self):
        rect, origin = wintypes.RECT(),wintypes.POINT(0,0)
        if not self.api.GetClientRect(self.hwnd,ctypes.byref(rect)) or not self.api.ClientToScreen(self.hwnd,ctypes.byref(origin)):
            raise RuntimeError("Cannot read client geometry")
        return origin.x,origin.y,rect.right,rect.bottom

    def check(self):
        # Import the same exception used by the proven replayer.
        from tailstopia_macro_replay import StopRequested
        if self.pressed(0x78):
            raise StopRequested("F9 pressed")
        if self.api.GetForegroundWindow() != self.hwnd or self.api.IsIconic(self.hwnd):
            raise RuntimeError("Game lost focus or minimized")

    def activate(self):
        matches = []
        @self.callback_type
        def visit(hwnd,_):
            if self.api.IsWindowVisible(hwnd):
                length = self.api.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length+1)
                self.api.GetWindowTextW(hwnd,title,length+1)
                if self.title.casefold() in title.value.casefold():
                    matches.append(hwnd)
            return True
        self.api.EnumWindows(visit,0)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {self.title} window; found {len(matches)}")
        self.hwnd = matches[0]
        if self.api.IsIconic(self.hwnd):
            self.api.ShowWindow(self.hwnd,9)
        self.api.SetForegroundWindow(self.hwnd)
        for _ in range(100):
            if self.pressed(0x78) or self.api.GetForegroundWindow() == self.hwnd:
                break
            time.sleep(0.01)
        self.check()
