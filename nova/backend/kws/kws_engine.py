import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import pyaudio
import wave
import time
from pathlib import Path


class WakeWordClassifier(nn.Module):
    """Lightweight MLP for wake word detection."""

    def __init__(self, embedding_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        in_dim = embedding_dim
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
        self.pos = 0
        self.full = False

    def extend(self, chunk):
        n = len(chunk)
        if n >= self.size:
            self.data[:] = chunk[-self.size :]
            self.pos = 0
            self.full = True
            return

        if self.pos + n <= self.size:
            self.data[self.pos : self.pos + n] = chunk
        else:
            overhead = (self.pos + n) - self.size
            self.data[self.pos :] = chunk[: n - overhead]
            self.data[:overhead] = chunk[n - overhead :]

        self.pos = (self.pos + n) % self.size
        if not self.full and self.pos < n:
            self.full = True

    def get(self):
        if not self.full:
            return self.data[: self.pos]
        return np.roll(self.data, -self.pos)


class GoogleEmbeddingModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, audio_16k: np.ndarray) -> np.ndarray:
        MIN_SAMPLES = 12400
        if len(audio_16k) < MIN_SAMPLES:
            audio_16k = np.pad(
                audio_16k, (0, MIN_SAMPLES - len(audio_16k)), mode="constant"
            )
        if audio_16k.ndim == 1:
            audio_16k = audio_16k[np.newaxis, :]
        outputs = self.session.run(
            None, {self.input_name: audio_16k.astype(np.float32)}
        )
        emb = outputs[0]
        if emb.ndim == 4:
            return emb[0, :, 0, :]
        elif emb.ndim == 3:
            return emb[0, :, :]
        return emb


class StreamingKWS:
    """DTW + MLP Classifier."""

    def __init__(
        self, model_path: str = None, threshold=0.25, window_sec=1.5, stride_sec=0.2
    ):
        if model_path:
            self.model = GoogleEmbeddingModel(model_path)
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
        else:
            self.model = None
            self.vad_model = None

        self.threshold = threshold
        self.window_size = int(16000 * window_sec)
        self.stride_size = int(16000 * stride_sec)
        self.buffer = RingBuffer(self.window_size)
        self.reference_sequences = []
        self.mlp = None
        self.new_samples = 0

    def enroll(self, ref_paths: list[str], noise_paths: list[str] = None):
        X, y = [], []
        self.reference_sequences = []
        embedding_dim = 0

        for p in ref_paths:
            with wave.open(p, "rb") as wf:
                audio = (
                    np.frombuffer(
                        wf.readframes(wf.getnframes()), dtype=np.int16
                    ).astype(np.float32)
                    / 32768.0
                )
                audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-9)
                seq = self.model.predict(audio)
                self.reference_sequences.append(seq)
                X.append(np.mean(seq, axis=0))
                y.append(1)
                embedding_dim = seq.shape[1]

        if noise_paths:
            for p in noise_paths:
                with wave.open(p, "rb") as wf:
                    audio = (
                        np.frombuffer(
                            wf.readframes(wf.getnframes()), dtype=np.int16
                        ).astype(np.float32)
                        / 32768.0
                    )
                    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-9)
                    seq = self.model.predict(audio)
                    X.append(np.mean(seq, axis=0))
                    y.append(0)

            # Train MLP
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mlp = WakeWordClassifier(embedding_dim).to(device)
            X_t = torch.FloatTensor(np.array(X)).to(device)
            y_t = torch.LongTensor(np.array(y)).to(device)

            optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            self.mlp.train()
            for _ in range(100):
                optimizer.zero_grad()
                outputs = self.mlp(X_t)
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()
            self.mlp.eval()
            print(f"[KWS] MLP trained. Embedded Dim: {embedding_dim}")

    def _dtw_distance(self, seq1, seq2):
        s1 = seq1 / (np.linalg.norm(seq1, axis=1, keepdims=True) + 1e-9)
        s2 = seq2 / (np.linalg.norm(seq2, axis=1, keepdims=True) + 1e-9)
        dist_mat = 1 - np.dot(s1, s2.T)
        r, c = dist_mat.shape
        D = np.zeros((r, c))
        D[0, 0] = dist_mat[0, 0]
        for i in range(1, r):
            D[i, 0] = D[i - 1, 0] + dist_mat[i, 0]
        for j in range(1, c):
            D[0, j] = D[0, j - 1] + dist_mat[0, j]
        for i in range(1, r):
            for j in range(1, c):
                D[i, j] = dist_mat[i, j] + min(
                    D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
                )
        return D[-1, -1] / (r + c)

    async def process_chunk(self, audio_16k: np.ndarray):
        try:
            self.buffer.extend(audio_16k)
            if not self.buffer.full:
                return False, 0
            self.new_samples += len(audio_16k)
            if self.new_samples < self.stride_size:
                return False, 0

            self.new_samples = 0
            window = self.buffer.get()
            window = (window - np.mean(window)) / (np.std(window) + 1e-9)

            speech_detected = False
            with torch.no_grad():
                for offset in [0, 5120, 10240, 15488]:
                    chunk = window[offset : offset + 512]
                    if len(chunk) < 512:
                        continue
                    audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
                    prob = self.vad_model(audio_tensor, 16000).item()
                    if prob > 0.5:
                        speech_detected = True
                        break
            if not speech_detected:
                return False, 0

            t_start = time.perf_counter()
            current_seq = self.model.predict(window)

            distances = [
                self._dtw_distance(current_seq, ref) for ref in self.reference_sequences
            ]
            min_dist = min(distances) if distances else 1.0

            prob_wake = 0.0
            if self.mlp:
                feat = (
                    torch.FloatTensor(np.mean(current_seq, axis=0))
                    .unsqueeze(0)
                    .to(next(self.mlp.parameters()).device)
                )
                with torch.no_grad():
                    logits = self.mlp(feat)
                    prob_wake = F.softmax(logits, dim=1)[0, 1].item()

            if min_dist < 0.4 or prob_wake > 0.3:
                print(f"[DEBUG] KWS Dist: {min_dist:.3f} | MLP Prob: {prob_wake:.3f}")

            if min_dist < self.threshold and prob_wake > 0.6:
                latency = (time.perf_counter() - t_start) * 1000
                print(
                    f"🔥 [KWS] VERIFIED (Dist: {min_dist:.2f}, Prob: {prob_wake:.2f})"
                )
                return True, latency
        except Exception as e:
            print(f"[ERROR] KWS Processing failed: {e}")
        return False, 0


class EnrollmentManager:
    """Handles enrollment including room noise calibration."""

    def __init__(self, ref_dir="refs", sample_rate=16000):
        self.ref_dir = Path(ref_dir)
        self.ref_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate
        self.pa = pyaudio.PyAudio()

    def record_sample(self, filename: str, prompt: str, duration=2.0):
        print(f"\n[ENROLL] {prompt}")
        input("Press Enter to start recording...")
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
        )
        print("Recording...")
        frames = []
        for _ in range(0, int(self.sample_rate / 1024 * duration)):
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
            "Say 'Hi Nova'",
            "Say 'Nova'",
            "Say 'Hey Nova'",
            "Say 'Hello Nova'",
            "Say 'Nova'",
        ]
        ref_paths = []
        for i, prompt in enumerate(prompts):
            path = self.ref_dir / f"nova_{i + 1}.wav"
            self.record_sample(path, prompt)
            ref_paths.append(str(path))

        print("\n[CALIBRATION] Now recording 5 samples of ROOM NOISE (Stay silent).")
        noise_paths = []
        for i in range(5):
            path = self.ref_dir / f"noise_{i + 1}.wav"
            self.record_sample(path, "Remain silent...", duration=2.0)
            noise_paths.append(str(path))

        return ref_paths, noise_paths
