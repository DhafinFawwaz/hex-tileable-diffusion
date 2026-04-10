from time import time
from typing import Any

from hex_tileable_diffusion.types import LogLevel

def format_print(v: Any) -> str:
    if isinstance(v, dict): return "\n".join(f"{k}: {v}" for k, v in v.items())
    elif hasattr(v, "__dict__"): return format_print(v.__dict__)
    else: return str(v)

def format_time(t: float) -> str:
    """Format to MM:SS.SS"""
    minutes = int(t // 60)
    seconds = int(t % 60)
    milliseconds = int((t % 1) * 100)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

class HexObserver():
    start_time: float = 0

    def __init__(self):
        self.start_time = 0

    def on_start(self):
        self.start_time = time()
        self.on_log("info", "Hexagonal Seamless & Tileable Texture Diffusion Generation started")

    def on_log(self, level: LogLevel, message: str, values: dict | Any | None = None) -> None:
        t = time() - self.start_time if self.start_time else 0
        print(f"[{level.upper()}] [{format_time(t)}] {message}")
        if values is not None: print(format_print(values))
    
    def on_wrapped_finished(self, rgb_arr, mask_arr, hex_outline_thickness):
        self.on_log("info", "Hexagon image wrapped successfully")

    def on_denoise_step(self):
        self.on_log("info", "Denoise step")
    
    def on_each_denoise_step(self):
        self.on_log("info", "Each denoise step")

    def on_after_inpaint(self, rgb_arr, mask_img, result):
        self.on_log("info", "After inpaint")

    def on_after_unwrap(self, result, R_final):
        self.on_log("info", "After unwrap")

    def on_after_postprocess(self, result):
        self.on_log("info", "After postprocess")
