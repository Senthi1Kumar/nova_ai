"""
KWS Engine V2 — VAD-free variant of StreamingKWS.

Identical to kws_engine.py except Silero-VAD gating is completely removed.
Every stride window goes directly through embedding → DTW / cosine / MLP
fusion scoring.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import wave
import time
from pathlib import Path
from scipy.spatial.distance import cdist

class WakeWordClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dims=[64], dropout=0.3):
        super().__init__()
        layers, in_dim = [], embedding_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RingBuffer:
    def __init__(self, size):
        self.data = np.zeros(size, dtype=np.float32)
        self.size = size
        self.pos  = 0
        self.full = False

    def extend(self, chunk):
        n = len(chunk)
        if n >= self.size:
            self.data[:] = chunk[-self.size:]
            self.pos  = 0
            self.full = True
            return
        if self.pos + n <= self.size:
            self.data[self.pos: self.pos + n] = chunk
        else:
            overhead = (self.pos + n) - self.size
            self.data[self.pos:] = chunk[: n - overhead]
            self.data[:overhead] = chunk[n - overhead:]
        self.pos = (self.pos + n) % self.size
        if not self.full and self.pos < n:
            self.full = True

    def get(self):
        if not self.full:
            return self.data[: self.pos]
        return np.roll(self.data, -self.pos)


class GoogleEmbeddingModel:
    def __init__(self, model_path: str):
        # Use CPU — this model is small (~7 MB) and KWS runs in its own process.
        # CUDAExecutionProvider requires cuBLAS matching the onnxruntime-gpu build
        # (e.g. CUDA 12 vs system CUDA 13), causing load errors. CPU avoids this.
        self.session    = ort.InferenceSession(model_path,
        providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, audio_16k: np.ndarray) -> np.ndarray:
        MIN_SAMPLES = 12400
        if len(audio_16k) < MIN_SAMPLES:
            audio_16k = np.pad(audio_16k, (0, MIN_SAMPLES - len(audio_16k)))
        if audio_16k.ndim == 1:
            audio_16k = audio_16k[np.newaxis, :]
        outputs = self.session.run(None, {self.input_name: audio_16k.astype(np.float32)})
        emb = outputs[0]
        if emb.ndim == 4:
            return emb[0, :, 0, :]
        elif emb.ndim == 3:
            return emb[0, :, :]
        return emb # type: ignore


# Distance / similarity helpers

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def dtw_cosine_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    s1, s2   = _normalize_rows(seq1), _normalize_rows(seq2)
    dist_mat = 1.0 - np.dot(s1, s2.T)
    r, c     = dist_mat.shape
    D        = np.full((r, c), np.inf)
    D[0, 0]  = dist_mat[0, 0]
    for i in range(1, r):
        D[i, 0] = D[i-1, 0] + dist_mat[i, 0]
    for j in range(1, c):
        D[0, j] = D[0, j-1] + dist_mat[0, j]
    for i in range(1, r):
        for j in range(1, c):
            D[i, j] = dist_mat[i, j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[-1, -1] / (r + c))


def max_cosine_similarity(test_frames: np.ndarray, ref_frames: np.ndarray) -> float:
    tn = _normalize_rows(test_frames)
    rn = _normalize_rows(ref_frames)
    return float(np.max(np.dot(tn, rn.T)))


def mean_cosine_similarity(test_mean: np.ndarray, ref_mean: np.ndarray) -> float:
    tn = test_mean / (np.linalg.norm(test_mean) + 1e-9)
    rn = ref_mean  / (np.linalg.norm(ref_mean)  + 1e-9)
    return float(np.dot(tn, rn))


def min_euclidean_distance(test_frames: np.ndarray, pooled_ref: np.ndarray) -> float:
    d = cdist(test_frames, pooled_ref, metric='euclidean')
    return float(np.min(d))


def fuse_scores(dtw_dist, cos_sim, euc_dist, mlp_prob,
                w_dtw=0.25, w_cos=0.35, w_euc=0.15, w_mlp=0.25):
    dtw_sim = 1.0 / (1.0 + dtw_dist)
    cos_sim = float(np.clip(cos_sim, 0.0, 1.0))
    euc_sim = 1.0 / (1.0 + euc_dist)
    return w_dtw * dtw_sim + w_cos * cos_sim + w_euc * euc_sim + w_mlp * mlp_prob


# StreamingKWSv2 — NO VAD

class StreamingKWSv2:
    """
    Multi-metric KWS with soft-voting fusion — **without** VAD gating.

    Same trigger logic as V1 (three OR-gates: fusion, hard-AND, sustained)
    but every stride goes directly to the embedding model instead of
    being filtered by Silero-VAD first.
    """

    def __init__(
        self,
        model_path: str = None,
        fusion_threshold: float = 0.60,
        w_dtw: float = 0.25,
        w_cos: float = 0.35,
        w_euc: float = 0.15,
        w_mlp: float = 0.25,
        dtw_hard_threshold: float = 0.15,
        cos_hard_threshold: float = 0.90,
        mlp_hard_threshold: float = 0.1,
        sustained_fusion_threshold: float = 0.58,
        sustained_window: int = 3,
        window_sec: float = 1.5,
        stride_sec: float = 0.2,
        cooldown_sec: float = 2.0,
        debug: bool = True,
    ):
        if model_path:
            self.model = GoogleEmbeddingModel(model_path)
        else:
            self.model = None

        self.fusion_threshold = fusion_threshold
        self.w_dtw, self.w_cos = w_dtw, w_cos
        self.w_euc, self.w_mlp = w_euc, w_mlp
        self.dtw_hard_threshold = dtw_hard_threshold
        self.cos_hard_threshold = cos_hard_threshold
        self.mlp_hard_threshold = mlp_hard_threshold
        self.sustained_fusion_threshold = sustained_fusion_threshold
        self.sustained_window = sustained_window
        self.window_size = int(16000 * window_sec)
        self.stride_size = int(16000 * stride_sec)
        self.cooldown_sec = cooldown_sec
        self.debug = debug

        self.buffer = RingBuffer(self.window_size)
        self.reference_sequences = []
        self.ref_means = []
        self.pooled_ref = None
        self.mlp = None
        self.new_samples = 0
        self._last_trigger_time = 0.0
        self._muted = False
        self._recent_fusions = []

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False
        self._recent_fusions.clear()
        self._last_trigger_time = time.perf_counter()

    def enroll(self, ref_paths: list[str], noise_paths: list[str] = None, version_tag: str = None):
        X, y = [], []
        self.reference_sequences = []
        self.ref_means = []
        embedding_dim = 0

        for p in ref_paths:
            with wave.open(p, "rb") as wf:
                audio = (
                    np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
            audio = self._normalize_audio(audio)
            seq   = self.model.predict(audio)
            self.reference_sequences.append(seq)
            self.ref_means.append(np.mean(seq, axis=0))
            X.append(np.mean(seq, axis=0))
            y.append(1)
            embedding_dim = seq.shape[1]

        self.pooled_ref = np.concatenate(self.reference_sequences, axis=0)

        if noise_paths:
            for p in noise_paths:
                with wave.open(p, "rb") as wf:
                    audio = (
                        np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                audio = self._normalize_audio(audio)
                seq   = self.model.predict(audio)
                X.append(np.mean(seq, axis=0))
                y.append(0)

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = WakeWordClassifier(embedding_dim, hidden_dims=[64], dropout=0.3).to(device)
        X_t = torch.FloatTensor(np.array(X)).to(device)
        y_t = torch.LongTensor(np.array(y)).to(device)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.mlp.train()
        for _ in range(150):
            optimizer.zero_grad()
            loss = criterion(self.mlp(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.mlp.eval()

        save_dir = Path(__file__).parent / "models"
        save_dir.mkdir(exist_ok=True)
        torch.save(self.mlp.state_dict(), save_dir / "mlp_weights.pth")
        np.save(save_dir / "ref_means.npy", np.array(self.ref_means))
        np.savez(save_dir / "ref_sequences.npz", *self.reference_sequences)
        np.save(save_dir / "pooled_ref.npy", self.pooled_ref)

        if version_tag:
            v_dir = save_dir / "versions" / version_tag
            v_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.mlp.state_dict(), v_dir / "mlp_weights.pth")
            np.save(v_dir / "ref_means.npy", np.array(self.ref_means))
            np.savez(v_dir / "ref_sequences.npz", *self.reference_sequences)
            np.save(v_dir / "pooled_ref.npy", self.pooled_ref)
            print(f"[KWSv2] Saved versioned model: {version_tag}")

        print(f"[KWSv2] Enrolled {len(ref_paths)} refs | "
              f"{len(noise_paths or [])} noise | emb_dim={embedding_dim}")

    def load_model(self):
        save_dir = Path(__file__).parent / "models"
        if not (save_dir / "mlp_weights.pth").exists():
            return False
        try:
            self.ref_means = list(np.load(save_dir / "ref_means.npy"))
            npz = np.load(save_dir / "ref_sequences.npz")
            self.reference_sequences = [npz[k] for k in npz.files]
            self.pooled_ref = np.load(save_dir / "pooled_ref.npy")

            embedding_dim = self.ref_means[0].shape[0]
            self.mlp = WakeWordClassifier(embedding_dim, hidden_dims=[64], dropout=0.3).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.mlp.load_state_dict(torch.load(save_dir / "mlp_weights.pth"))
            self.mlp.eval()
            print("[KWSv2] Loaded pre-trained KWS model from disk.")
            return True
        except Exception as e:
            print(f"[KWSv2] Failed to load pre-trained model: {e}")
            return False

    def load_version(self, version_tag: str) -> bool:
        """Activate a previously saved versioned model by copying it to the active slot."""
        import shutil
        v_dir = Path(__file__).parent / "models" / "versions" / version_tag
        if not (v_dir / "mlp_weights.pth").exists():
            print(f"[KWSv2] Version {version_tag} not found.")
            return False
        save_dir = Path(__file__).parent / "models"
        for fname in ["mlp_weights.pth", "ref_means.npy", "ref_sequences.npz", "pooled_ref.npy"]:
            shutil.copy2(v_dir / fname, save_dir / fname)
        print(f"[KWSv2] Activated version {version_tag}.")
        return self.load_model()

    # internals

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-9)

    def _mlp_prob(self, mean_emb: np.ndarray) -> float:
        if self.mlp is None:
            return 0.5
        device = next(self.mlp.parameters()).device
        feat   = torch.FloatTensor(mean_emb).unsqueeze(0).to(device)
        with torch.no_grad():
            return F.softmax(self.mlp(feat), dim=1)[0, 1].item()

    def _compute_all_metrics(self, current_seq: np.ndarray):
        dtw_dists, cos_frame_sims, cos_mean_sims, euc_dists = [], [], [], []
        cur_mean = np.mean(current_seq, axis=0)

        for i, ref_seq in enumerate(self.reference_sequences):
            dtw_dists.append(dtw_cosine_distance(current_seq, ref_seq))
            cos_frame_sims.append(max_cosine_similarity(current_seq, ref_seq))
            cos_mean_sims.append(mean_cosine_similarity(cur_mean, self.ref_means[i]))

        euc_dists.append(min_euclidean_distance(current_seq, self.pooled_ref))

        best_dtw      = min(dtw_dists)
        best_cos_frame = max(cos_frame_sims)
        best_cos_mean  = max(cos_mean_sims)
        min_euc        = min(euc_dists)
        mlp_prob       = self._mlp_prob(cur_mean)

        fusion = fuse_scores(
            best_dtw, best_cos_frame, min_euc, mlp_prob,
            self.w_dtw, self.w_cos, self.w_euc, self.w_mlp
        )

        return best_dtw, best_cos_frame, best_cos_mean, min_euc, mlp_prob, fusion

    # streaming inference (NO VAD)

    def process_chunk(self, audio_16k: np.ndarray):
        try:
            self.buffer.extend(audio_16k)
            if not self.buffer.full:
                return False, 0

            self.new_samples += len(audio_16k)
            if self.new_samples < self.stride_size:
                return False, 0
            self.new_samples = 0

            if self._muted:
                return False, 0

            now = time.perf_counter()
            if now - self._last_trigger_time < self.cooldown_sec:
                return False, 0

            window = self.buffer.get()
            window = self._normalize_audio(window)

            # NO VAD CHECK - go straight to embedding
            t_start     = time.perf_counter()
            current_seq = self.model.predict(window)

            (best_dtw, best_cos_frame, best_cos_mean,
             min_euc, mlp_prob, fusion) = self._compute_all_metrics(current_seq)

            if self.debug and (
                best_dtw < 0.4 or best_cos_frame > 0.6
                or mlp_prob > 0.4 or fusion > 0.45
            ):
                print(
                    f"[KWSv2] DTW={best_dtw:.3f} | "
                    f"CosFrm={best_cos_frame:.3f} | "
                    f"CosMean={best_cos_mean:.3f} | "
                    f"Euc={min_euc:.3f} | "
                    f"MLP={mlp_prob:.3f} | "
                    f"Fusion={fusion:.3f}"
                )

            self._recent_fusions.append(fusion)
            self._recent_fusions = self._recent_fusions[-self.sustained_window:]

            # All gates require MLP agreement to prevent false alarms.
            # The Google speech embedding produces high CosFrm (~0.95) for
            # ANY speech, so DTW+CosFrm alone are not discriminative.
            mlp_ok = mlp_prob >= self.mlp_hard_threshold

            gate_a = fusion >= self.fusion_threshold and mlp_ok
            gate_b = (
                best_dtw       <  self.dtw_hard_threshold
                and best_cos_frame >= self.cos_hard_threshold
                and mlp_ok
            )
            gate_c = (
                len(self._recent_fusions) >= self.sustained_window
                and all(s >= self.sustained_fusion_threshold
                        for s in self._recent_fusions)
                and mlp_ok
            )

            triggered = gate_a or gate_b or gate_c

            if triggered:
                latency = (time.perf_counter() - t_start) * 1000
                self._last_trigger_time = time.perf_counter()
                self._recent_fusions.clear()
                tag = ("FUSION" if gate_a else "HARD-AND" if gate_b else "SUSTAINED")
                print(
                    f"🔥 [KWSv2] VERIFIED/{tag} "
                    f"DTW={best_dtw:.3f} CosFrm={best_cos_frame:.3f} "
                    f"MLP={mlp_prob:.3f} Fusion={fusion:.3f}"
                )
                return True, latency

        except Exception as e:
            print(f"[ERROR] KWSv2 Processing failed: {e}")
        return False, 0
