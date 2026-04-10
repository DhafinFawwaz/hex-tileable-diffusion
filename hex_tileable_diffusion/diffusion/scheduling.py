from functools import partial
from typing import Any

from hex_tileable_diffusion.types import SchedulerType

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler

def create_scheduler(scheduler_type: SchedulerType, base_config: dict[str, object]) -> Any:

    map: dict[SchedulerType, Any] = {
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "dpm++_2m": partial(
            DPMSolverMultistepScheduler.from_config,
            algorithm_type="dpmsolver++",
        ),
        "dpm++_2m_karras": partial(
            DPMSolverMultistepScheduler.from_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        ),
        "dpm++_sde_karras": partial(
            DPMSolverSDEScheduler.from_config,
            use_karras_sigmas=True,
        ),
        "ddim": DDIMScheduler,
        "uni_pc": UniPCMultistepScheduler,
    }

    factory = map[scheduler_type]
    if isinstance(factory, partial): return factory(base_config)
    return factory.from_config(base_config)


def interpolate_schedule(schedule: list[float], step: int, total_steps: int) -> float:
    n = len(schedule)
    if n == 1: return schedule[0]
    t = step / max(total_steps - 1, 1)
    pos = t * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return schedule[lo] + (pos - lo) * (schedule[hi] - schedule[lo])
