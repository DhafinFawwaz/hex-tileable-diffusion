import math

from hex_tileable_diffusion.config import HexWrapperConfig
from hex_tileable_diffusion.core.constant import SQRT3
from hex_tileable_diffusion.evaluation.evaluation import HexEvaluation
from hex_tileable_diffusion.evaluation.metric import Metrics
from hex_tileable_diffusion.types import ImageInput
from hex_tileable_diffusion.util.image import load_image


def evaluate_hex_tileable_diffusion_texture(input: ImageInput, output: ImageInput, wrapper_config: HexWrapperConfig, metrics: Metrics | None = None) -> HexEvaluation:
    output_size = wrapper_config.output_size
    ref_arr = load_image(input, output_size)
    gen_arr = load_image(output, output_size)

    return HexEvaluation(
        ref_arr, gen_arr,
        hex_radius=_compute_R_final(wrapper_config),
        metrics=metrics,
    )


def _compute_R_final(cfg: HexWrapperConfig) -> float:
    R_base = cfg.hypotenuse if cfg.hypotenuse is not None else cfg.output_size / 2
    R_cam = R_base + (cfg.outer_margin / SQRT3) if cfg.outer_margin > 0 else R_base
    pad_raw = int(math.ceil(2 * R_cam))
    pad_gen = ((pad_raw + 7) // 8) * 8
    return R_cam * cfg.output_size / pad_gen
