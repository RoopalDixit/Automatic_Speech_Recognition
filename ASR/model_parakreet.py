import nemo.collections.asr as nemo_asr

# This downloads the model on first run (~600MB-1GB) and caches it locally
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
import json
from jiwer import wer  # pip install jiwer --break-system-packages

with open("ground_truth.json") as f:
    ground_truth = json.load(f)

predictions = {}
for filename in ground_truth:
    result = model.transcribe([filename])
    predictions[filename] = result[0].text  # .text pulls the string out of the Hypothesis object

total_wer = wer(
    list(ground_truth.values()),
    list(predictions.values())
)
print(f"WER on gaming vocab set: {total_wer:.2%}")