import whisper
from jiwer import wer
import json

whisper_model = whisper.load_model("base")  # "base" is a good speed/accuracy starting point

with open("ground_truth.json") as f:
    ground_truth = json.load(f)

predictions = {}
for filename in ground_truth:
    result = whisper_model.transcribe(filename)
    predictions[filename] = result["text"].strip()

whisper_wer = wer(list(ground_truth.values()), list(predictions.values()))
print(f"Whisper WER: {whisper_wer:.2%}")