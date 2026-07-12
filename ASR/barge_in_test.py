# barge_in_test.py
import torch
import sounddevice as sd
import soundfile as sf
import threading

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, _, _, _) = utils

sample_rate = 16000
chunk_samples = 512
playback_active = threading.Event()
playback_active.set()

def play_response(filepath):
    audio, sr = sf.read(filepath, dtype='float32')
    assert sr == sample_rate, f"Expected {sample_rate}Hz, got {sr}Hz"

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
    stream.start()
    for i in range(0, len(audio), chunk_samples):
        if not playback_active.is_set():
            print("🛑 Playback interrupted")
            break
        chunk = audio[i:i + chunk_samples]
        stream.write(chunk)
    else:
        print("✅ Playback finished naturally")
    stream.stop()
    stream.close()

def monitor_for_interrupt():
    def callback(indata, frames, time_info, status):
        if not playback_active.is_set():
            raise sd.CallbackStop
        audio_chunk = indata[:, 0]
        tensor_chunk = torch.from_numpy(audio_chunk.copy())
        speech_prob = vad_model(tensor_chunk, sample_rate).item()
        if speech_prob > 0.5:
            print(f"🎙️  User interrupt detected (prob={speech_prob:.2f})")
            playback_active.clear()
            raise sd.CallbackStop

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback,
                         blocksize=chunk_samples, dtype='float32'):
        while playback_active.is_set():
            sd.sleep(50)

# Use one of your existing clips as the stand-in "AI response"
response_clip = "custom_data_fixed/clip_001.wav"

monitor_thread = threading.Thread(target=monitor_for_interrupt)
monitor_thread.start()
play_response(response_clip)
monitor_thread.join()

import time

timestamps = []
start = time.time()
# ... run your pipeline on a test utterance ...
first_partial_time = time.time()
timestamps.append(first_partial_time - start)

import numpy as np
print(f"P50 latency: {np.percentile(timestamps, 50)*1000:.0f}ms")
print(f"P95 latency: {np.percentile(timestamps, 95)*1000:.0f}ms")