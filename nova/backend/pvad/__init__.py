"""
nova.backend.pvad — Two-Stage Personalized Voice Activity Detection

Stage 1: FireRedGate   — generic voice / silence / music gate (DFSMN, 10ms)
Stage 2: SpeakerGate   — driver identity verification (ECAPA-TDNN, 250ms)

Public API
----------
run_pvad_worker(pvad_in_queue, ws_out_queue, stop_event)
    Multiprocessing entry point. Called by pipeline_mp.

FireRedGate
    Standalone Stage-1 class (importable for testing).

SpeakerGate
    Standalone Stage-2 class (importable for testing).
"""

from .worker       import run_pvad_worker
from .firered_gate import FireRedGate
from .speaker_gate import SpeakerGate

__all__ = ["run_pvad_worker", "FireRedGate", "SpeakerGate"]
