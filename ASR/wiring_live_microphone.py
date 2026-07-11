import sounddevice as sd
import numpy as np

sample_rate = 16000
chunk_duration = 0.16  # 160ms chunks
chunk_samples = int(sample_rate * chunk_duration)

def audio_callback(indata, frames, time_info, status):
    audio_chunk = indata[:, 0]  # mono
    # feed audio_chunk into your streaming model's process_chunk function
    ...

with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=chunk_samples
):
    print("Listening... Ctrl+C to stop")
    while True:
        pass