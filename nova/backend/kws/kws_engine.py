import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import pyaudio
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
        self.session    = ort.InferenceSession(model_path)
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
        return emb


# Distance / similarity helpers
def _normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def dtw_cosine_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Normalized DTW on cosine distance matrix (same as kws_classifier.py)."""
    s1, s2   = _normalize_rows(seq1), _normalize_rows(seq2)
    dist_mat = 1.0 - np.dot(s1, s2.T)          # cosine distance
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
    """Max cosine similarity between any frame pair (mirrors kws_classifier.py)."""
    tn = _normalize_rows(test_frames)
    rn = _normalize_rows(ref_frames)
    return float(np.max(np.dot(tn, rn.T)))


def mean_cosine_similarity(test_mean: np.ndarray, ref_mean: np.ndarray) -> float:
    """Cosine similarity between mean embeddings."""
    tn = test_mean / (np.linalg.norm(test_mean) + 1e-9)
    rn = ref_mean  / (np.linalg.norm(ref_mean)  + 1e-9)
    return float(np.dot(tn, rn))


def min_euclidean_distance(test_frames: np.ndarray, pooled_ref: np.ndarray) -> float:
    """Min Euclidean distance between any frame pair."""
    d = cdist(test_frames, pooled_ref, metric='euclidean')
    return float(np.min(d))


# Fusion score
def fuse_scores(dtw_dist: float, cos_sim: float, euc_dist: float,
                mlp_prob: float,
                w_dtw=0.25, w_cos=0.35, w_euc=0.15, w_mlp=0.25) -> float:
    """
    Combine four signals into one [0, 1] confidence score.

    Conversions to similarity (all → higher = more like wake word):
      DTW  : sim = 1 / (1 + dtw_dist)   — used by kws_classifier.py
      Cos  : already in [-1, 1], clip to [0, 1]
      Euc  : sim = 1 / (1 + euc_dist)
      MLP  : already probability in [0, 1]
    """
    dtw_sim = 1.0 / (1.0 + dtw_dist)
    cos_sim = float(np.clip(cos_sim, 0.0, 1.0))
    euc_sim = 1.0 / (1.0 + euc_dist)
    return w_dtw * dtw_sim + w_cos * cos_sim + w_euc * euc_sim + w_mlp * mlp_prob


class StreamingKWS:
    """
    Multi-metric KWS with soft-voting fusion.

    Trigger logic (OR of three gates so a single strong signal suffices):
      A) Fusion score  >= fusion_threshold                 (default 0.60)
      B) DTW alone     <  dtw_hard_threshold               (default 0.22)
         AND cosine    >= cos_hard_threshold               (default 0.82)
      C) Sustained: N consecutive strides all pass a
         softer fusion threshold                           (default 0.52)

    Weights and thresholds are all __init__ parameters so you can tune them
    from the CLI / experiment loop without editing this file.
    """

    def __init__(
        self,
        model_path: str = None,
        fusion_threshold: float = 0.60,
        w_dtw: float = 0.25,
        w_cos: float = 0.35,
        w_euc: float = 0.15,
        w_mlp: float = 0.25,
        dtw_hard_threshold: float = 0.22,
        cos_hard_threshold: float = 0.82,
        sustained_fusion_threshold: float = 0.52,
        sustained_window: int = 3,
        window_sec: float = 1.5,
        stride_sec: float = 0.2,
        cooldown_sec: float = 2.0,
        debug: bool = True,
    ):
        if model_path:
            self.model     = GoogleEmbeddingModel(model_path)
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
        else:
            self.model     = None
            self.vad_model = None

        self.fusion_threshold = fusion_threshold
        self.w_dtw, self.w_cos = w_dtw, w_cos
        self.w_euc, self.w_mlp = w_euc, w_mlp
        self.dtw_hard_threshold = dtw_hard_threshold
        self.cos_hard_threshold = cos_hard_threshold
        self.sustained_fusion_threshold = sustained_fusion_threshold
        self.sustained_window = sustained_window
        self.window_size = int(16000 * window_sec)
        self.stride_size = int(16000 * stride_sec)
        self.cooldown_sec = cooldown_sec
        self.debug = debug

        self.buffer = RingBuffer(self.window_size)
        self.reference_sequences = []   # list of [T, D] arrays (per enrollment file)
        self.ref_means = []   # list of [D] mean vectors
        self.pooled_ref = None # [N*T, D] concatenated for euclidean
        self.mlp = None
        self.new_samples = 0
        self._last_trigger_time = 0.0
        self._muted = False
        self._recent_fusions = []   # rolling fusion scores for sustained gate

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False
        self._recent_fusions.clear()
        self._last_trigger_time = time.perf_counter()

    def enroll(self, ref_paths: list[str], noise_paths: list[str] = None):
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
            seq   = self.model.predict(audio)  # [T, D]
            self.reference_sequences.append(seq)
            self.ref_means.append(np.mean(seq, axis=0))
            X.append(np.mean(seq, axis=0))
            y.append(1)
            embedding_dim = seq.shape[1]

        # Concatenate all ref frames for Euclidean gate
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
        
        # Save the trained model and reference sequences
        save_dir = Path(__file__).parent / "models"
        save_dir.mkdir(exist_ok=True)
        torch.save(self.mlp.state_dict(), save_dir / "mlp_weights.pth")
        np.save(save_dir / "ref_means.npy", np.array(self.ref_means))
        np.savez(save_dir / "ref_sequences.npz", *self.reference_sequences)
        np.save(save_dir / "pooled_ref.npy", self.pooled_ref)
        
        print(f"[KWS] Enrolled {len(ref_paths)} refs | "
              f"{len(noise_paths or [])} noise | emb_dim={embedding_dim}")

    def load_model(self):
        """Load pre-trained MLP and reference sequences from disk if they exist."""
        save_dir = Path(__file__).parent / "models"
        if not (save_dir / "mlp_weights.pth").exists():
            return False
            
        try:
            # Load arrays
            self.ref_means = list(np.load(save_dir / "ref_means.npy"))
            npz = np.load(save_dir / "ref_sequences.npz")
            self.reference_sequences = [npz[k] for k in npz.files]
            self.pooled_ref = np.load(save_dir / "pooled_ref.npy")
            
            # Initialize and load MLP
            embedding_dim = self.ref_means[0].shape[0]
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            self.mlp.load_state_dict(torch.load(save_dir / "mlp_weights.pth"))
            self.mlp.eval()
            print("[KWS] Loaded pre-trained KWS model from disk.")
            return True
        except Exception as e:
            print(f"[KWS] Failed to load pre-trained model: {e}")
            return False

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
        """
        Returns (best_dtw, best_cos_frame, best_cos_mean, min_euc, mlp_prob, fusion).
        'best' = most wake-word-like value across all reference sequences.
        """
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

        # Use frame-level cosine (stronger signal) for fusion
        fusion = fuse_scores(
            best_dtw, best_cos_frame, min_euc, mlp_prob,
            self.w_dtw, self.w_cos, self.w_euc, self.w_mlp
        )

        return best_dtw, best_cos_frame, best_cos_mean, min_euc, mlp_prob, fusion

    # Streaming Inference
    async def process_chunk(self, audio_16k: np.ndarray):
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

            # VAD
            speech_detected = False
            with torch.no_grad():
                for offset in range(0, len(window) - 512, 4000):
                    chunk = window[offset: offset + 512]
                    if len(chunk) < 512:
                        continue
                    prob = self.vad_model(
                        torch.from_numpy(chunk).unsqueeze(0), 16000
                    ).item()
                    if prob > 0.6:
                        speech_detected = True
                        break
            if not speech_detected:
                return False, 0

            t_start     = time.perf_counter()
            current_seq = self.model.predict(window)

            (best_dtw, best_cos_frame, best_cos_mean,
             min_euc, mlp_prob, fusion) = self._compute_all_metrics(current_seq)

            # Debug print (any metric is close)
            if self.debug and (
                best_dtw < 0.4 or best_cos_frame > 0.6
                or mlp_prob > 0.4 or fusion > 0.45
            ):
                print(
                    f"[KWS] DTW={best_dtw:.3f} | "
                    f"CosFrm={best_cos_frame:.3f} | "
                    f"CosMean={best_cos_mean:.3f} | "
                    f"Euc={min_euc:.3f} | "
                    f"MLP={mlp_prob:.3f} | "
                    f"Fusion={fusion:.3f}"
                )

            # Sustain buffer: always append current fusion score
            self._recent_fusions.append(fusion)
            self._recent_fusions = self._recent_fusions[-self.sustained_window:]

            # Gate A: fusion
            gate_a = fusion >= self.fusion_threshold

            # Gate B: hard-AND (DTW + cosine both strong)
            gate_b = (
                best_dtw      <  self.dtw_hard_threshold
                and best_cos_frame >= self.cos_hard_threshold
            )

            # Gate C: sustained softer fusion
            gate_c = (
                len(self._recent_fusions) >= self.sustained_window
                and all(s >= self.sustained_fusion_threshold
                        for s in self._recent_fusions)
            )

            triggered = gate_a or gate_b or gate_c

            if triggered:
                latency = (time.perf_counter() - t_start) * 1000
                self._last_trigger_time = time.perf_counter()
                self._recent_fusions.clear()
                tag = ("FUSION" if gate_a else "HARD-AND" if gate_b else "SUSTAINED")
                print(
                    f"🔥 [KWS] VERIFIED/{tag} "
                    f"DTW={best_dtw:.3f} CosFrm={best_cos_frame:.3f} "
                    f"MLP={mlp_prob:.3f} Fusion={fusion:.3f}"
                )
                return True, latency

        except Exception as e:
            print(f"[ERROR] KWS Processing failed: {e}")
        return False, 0


# CLI enrollment helper

class EnrollmentManager:
    def __init__(self, ref_dir="refs", sample_rate=16000):
        self.ref_dir     = Path(ref_dir)
        self.ref_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
        self.pa          = pyaudio.PyAudio()

    def record_sample(self, filename: str, prompt: str, duration=2.0):
        print(f"\n[ENROLL] {prompt}")
        input("Press Enter to start recording...")
        stream = self.pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=self.sample_rate, input=True, frames_per_buffer=1024,
        )
        print("Recording...")
        frames = []
        for _ in range(int(self.sample_rate / 1024 * duration)):
            frames.append(stream.read(1024))
        print("Done.")
        stream.stop_stream()
        stream.close()
        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

    def run_full_enrollment(self):
        prompts = [
            "Say 'Hi Nova'", "Say 'Nova'", "Say 'Hey Nova'",
            "Say 'Hello Nova'", "Say 'Nova'",
        ]
        ref_paths = []
        for i, prompt in enumerate(prompts):
            path = self.ref_dir / f"nova_{i+1}.wav"
            self.record_sample(path, prompt)
            ref_paths.append(str(path))

        print("\n[CALIBRATION] Recording 5 samples of room noise — stay silent.")
        noise_paths = []
        for i in range(5):
            path = self.ref_dir / f"noise_{i+1}.wav"
            self.record_sample(path, "Remain silent...", duration=2.0)
            noise_paths.append(str(path))

        return ref_paths, noise_paths