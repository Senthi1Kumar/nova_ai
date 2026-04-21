import multiprocessing as mp
import multiprocessing.synchronize

from .kws_worker import run_kws_worker
from .llm_worker import run_llm_worker
from .tts_worker import run_tts_worker
from .pvad_worker import run_pvad_worker


def run_stt_worker(
    stt_in_queue: mp.Queue,  # type: ignore[type-arg]
    llm_in_queue: mp.Queue,  # type: ignore[type-arg]
    ws_out_queue: mp.Queue,  # type: ignore[type-arg]
    stop_event: multiprocessing.synchronize.Event,
):
    """
    Dispatch to the backend selected by STT_SETTINGS.active (driven by
    NOVA_STT_VARIANT env). Cross-backend switches at runtime are rejected by
    the workers — restart the process with a different NOVA_STT_VARIANT.
    """
    from .stt_config import STT_VARIANT_REGISTRY, STT_SETTINGS

    active = STT_SETTINGS.active
    backend = (
        STT_VARIANT_REGISTRY[active].backend
        if active in STT_VARIANT_REGISTRY
        else "moonshine"
    )

    if backend == "kyutai":
        from .stt_kyutai_worker import run_stt_worker as _run
    else:
        from .stt_moonshine_worker import run_stt_worker as _run

    _run(stt_in_queue, llm_in_queue, ws_out_queue, stop_event)


__all__ = ["run_stt_worker", "run_kws_worker", "run_llm_worker", "run_tts_worker", "run_pvad_worker"]
