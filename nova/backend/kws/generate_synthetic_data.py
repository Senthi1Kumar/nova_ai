import sys
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import random

# Add parent dir to path so we can import pocket_tts
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from pocket_tts import TTSModel
    import torch
except ImportError:
    print("pocket_tts not available. Skipping synthetic data generation.")
    sys.exit(0)

# We want words that sound similar to "Nova" to train the model to reject them
HARD_NEGATIVES = [
    "No way", "Over", "Noah", "Nava", "Never", "Motor", "Sofa",
    "Rover", "Lova", "Nola", "Boba", "Dova", "Cova", "Mova",
    "Hey there", "Hello", "Okay" "No uh", "No way", "Bixby", "Alexa", "Siri",
    "Hey Google", "Hey Siri", "Hey Alexa", "Hey Bixby"
]

def add_noise(audio, snr_db=10):
    """Add white noise to the audio signal."""
    audio_power = np.mean(audio ** 2)
    sig_db = 10 * np.log10(audio_power + 1e-10)
    noise_db = sig_db - snr_db
    noise_power = 10 ** (noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def change_pitch_speed(audio, speed_factor=1.0):
    """Resample to change pitch and speed simultaneously."""
    from scipy.interpolate import interp1d
    x = np.arange(len(audio))
    f = interp1d(x, audio, kind='linear', fill_value="extrapolate")
    new_len = int(len(audio) / speed_factor)
    new_x = np.linspace(0, len(audio) - 1, new_len)
    return f(new_x)

def generate():
    print("Loading TTS Model for synthetic KWS data generation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTSModel.load_model().to(device)
    
    # Use Alba (default) for voice cloning
    state = model.get_state_for_audio_prompt("alba")
    for module_name, module_state in state.items():
        for key, tensor in module_state.items():
            if isinstance(tensor, torch.Tensor):
                state[module_name][key] = tensor.to(device)
                
    refs_dir = Path(__file__).parent / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Hard Negatives (Noise)
    print(f"Generating 100 hard negative samples...")
    for i in range(100):
        # Pick a random hard negative word
        word = random.choice(HARD_NEGATIVES)
        try:
            chunks = list(model.generate_audio_stream(
                model_state=state, text_to_generate=word, copy_state=True
            ))
            audio_np = torch.cat(chunks, dim=0).cpu().numpy().squeeze()
            
            # Apply some random noise/pitch
            speed = random.uniform(0.85, 1.2)
            audio_np = change_pitch_speed(audio_np, speed)
            audio_np = add_noise(audio_np, snr_db=random.uniform(5, 25))
            
            # Normalize to 16-bit PCM
            audio_i16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            
            filename = refs_dir / f"noise_synth_{i}.wav"
            wavfile.write(filename, int(24000 * speed), audio_i16) 
        except Exception as e:
            print(f"Failed to generate {word}: {e}")

    # 2. Generate Augmented Positives ("Nova")
    print("Generating 100 augmented positive 'Nova' samples...")
    pos_chunks = list(model.generate_audio_stream(
        model_state=state, text_to_generate="Nova.", copy_state=True
    ))
    base_pos = torch.cat(pos_chunks, dim=0).cpu().numpy().squeeze()
    
    for i in range(100):
        try:
            speed = random.uniform(0.85, 1.2) # Vary speed/pitch significantly
            audio_np = change_pitch_speed(base_pos, speed)
            audio_np = add_noise(audio_np, snr_db=random.uniform(10, 25))
            
            audio_i16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            filename = refs_dir / f"nova_synth_{i}.wav"
            wavfile.write(filename, int(24000 * speed), audio_i16)
        except Exception as e:
            print(f"Failed to augment positive {i}: {e}")

    print("Synthetic data generation complete!")

if __name__ == "__main__":
    generate()