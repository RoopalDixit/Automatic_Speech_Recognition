import nemo.collections.asr as nemo_asr

# This downloads the model on first run (~600MB-1GB) and caches it locally
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")

transcript = model.transcribe(["test.wav"])
print(transcript)