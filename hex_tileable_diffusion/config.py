from dataclasses import dataclass, field
from pathlib import Path

from .types import ControlNetModelId, IPAdapterModelId, IPAdapterWeightName, InpaintModelId, RollMode, SchedulerType, VaeModelId, ImageInput


@dataclass(frozen=True)
class HexWrapperConfig:

    output_size: int = 512 # >= 64
    hypotenuse: float | None = None # >= 1.0, defaults to output_size / 2
    outer_margin: float = 200.0 # >= 0.0
    inner_padding: float = 75.0 # >= 0.0
    gap_padding: float = 50.0 # [0.0, inner_padding]
    feather_width: float = 30.0 # >= 0.0
    horizontal_camera_padding: float = 0.0 # >= 0.0
    vertical_camera_padding: float = 0.0 # >= 0.0
    x_offset: float = 0.0 # [-1.0, 1.0]
    y_offset: float = 0.5 # [-1.0, 1.0]


@dataclass(frozen=True)
class DiffusionConfig:

    model_id: InpaintModelId | str = "runwayml/stable-diffusion-inpainting"
    vae_model: VaeModelId | str | None = None
    vae_fp32: bool = True
    scheduler_type: SchedulerType | None = None

    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 50 # >= 1
    guidance_scale: float = 7.5 # >= 1.0
    strength: float = 0.95 # [0.0, 1.0]
    seed: int = 42
    guidance_schedule: list[float] | None = None # each value >= 1.0

    use_rolling_noise: bool = True
    roll_mode: RollMode = "hex_copy"


@dataclass(frozen=True)
class ControlNetConfig:

    model_id: ControlNetModelId | str
    conditioning_scale: float = 0.8 # >= 0.0


@dataclass(frozen=True)
class IPAdapterConfig:

    model_id: IPAdapterModelId | str
    subfolder: str = "models"
    weight_name: IPAdapterWeightName | str = "ip-adapter_sd15.bin"
    scale: float = 0.85 # >= 0.0 (but recommended <= 1.5)
    reference_image_path: str | Path | None = None
    use_pass1_reference_for_pass2: bool = False
    use_on_pass1: bool = True


@dataclass(frozen=True)
class FinetuneConfig:

    steps: int = 300 # >= 1
    lr: float = 5e-5 # > 0.0
    crop_size: int | None = None # >= 64
    mask_ratio_range: tuple[float, float] = (0.1, 0.4) # each [0.0, 1.0], first <= second
    use_lora: bool = True
    lora_rank: int = 8 # >= 1
    lora_alpha: int = 16 # >= 1
    gradient_accumulation_steps: int = 4 # >= 1
    use_augmentation: bool = True
    use_prompt_conditioning: bool = True
    prompt_dropout_prob: float = 0.1 # [0.0, 1.0]
    log_interval: int = 50 # >= 1
    cleanup_after: bool = False


@dataclass(frozen=True)
class ExteriorPassConfig:

    strength: float = 1.0 # [0.0, 1.0]
    steps: int = 30 # >= 1
    guidance_scale: float = 3.0 # >= 1.0
    seed: int | None = None


@dataclass(frozen=True)
class PostprocessConfig:

    use_latent_color_correction: bool = True
    use_pixel_pasteback: bool = True
    pasteback_feather_px: int = 12 # >= 0
    use_color_postprocess: bool = True


@dataclass(frozen=True)
class VisualizationConfig:

    hex_outline_thickness: int = 3 # >= 1
    reference_image_path: str | Path | None = None
    in_between_preview_count: int = 3 # >= 0


@dataclass(frozen=True)
class HexTileableDiffusionConfig:

    image_path: ImageInput
    output_path: str | Path = "."
    cache_dir: str | Path = ".cache"

    wrapper: HexWrapperConfig = field(default_factory=HexWrapperConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    controlnet: ControlNetConfig | None = None
    ip_adapter: IPAdapterConfig | None = None
    finetune: FinetuneConfig | None = None
    exterior: ExteriorPassConfig | None = None
    postprocess: PostprocessConfig | None = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
