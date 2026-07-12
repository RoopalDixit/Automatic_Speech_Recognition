# vad_test.py
import torch
import soundfile as sf
import numpy as np

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, _, _, _) = utils   # skip Silero's own read_audio

# Load audio with soundfile instead (same library you're already using elsewhere)
audio, sample_rate = sf.read('custom_data_fixed/clip_001.wav', dtype='float32')
assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"

wav = torch.from_numpy(audio)
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
print(speech_timestamps)