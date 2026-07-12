# vad_live.py
import torch
import sounddevice as sd
import numpy as np

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, _, _, _) = utils

sample_rate = 16000
chunk_duration = 0.032  # 32ms — Silero VAD expects chunks of this size at 16kHz (512 samples)
chunk_samples = 512

speaking = False

def audio_callback(indata, frames, time_info, status):
    global speaking

    if status:
        print(status)

    audio_chunk = indata[:, 0]  # mono
    tensor_chunk = torch.from_numpy(audio_chunk.copy())

    # Silero VAD's streaming API: pass one chunk at a time, get back a speech probability
    speech_prob = vad_model(tensor_chunk, sample_rate).item()

    is_speech = speech_prob > 0.5  # threshold — tune this based on your mic/environment

    if is_speech and not speaking:
        speaking = True
        print(f"🎙️  Speech started (prob={speech_prob:.2f})")
    elif not is_speech and speaking:
        speaking = False
        print(f"🤫 Speech ended (prob={speech_prob:.2f})")

print("Listening... press Ctrl+C to stop")
with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=chunk_samples,
    dtype='float32'
):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")